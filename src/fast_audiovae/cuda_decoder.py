"""Explicit NVIDIA FP32 decoding with bounded, per-stream CUDA graphs.

Weights are immutable and shared. Each stream owns its histories and graph
storage, and a failed packet never advances its committed history. Returned
NumPy audio is ready on the CPU and never aliases a graph's reusable output.
"""
from __future__ import annotations

from pathlib import Path
import os
import threading

import numpy as np
import torch

from .mps_compiled import TensorStep as _TensorStep


# Graph capture is process-global in CUDA. Serialize this backend's capture
# and execution, including across decoder instances, rather than letting an
# independent stream execute CUDA operations during another stream's capture.
_CUDA_LOCK = threading.RLock()


def _precision_policy():
    for name in ("NVIDIA_TF32_OVERRIDE", "TORCHINDUCTOR_USE_FAST_MATH"):
        if os.environ.get(name) not in (None, "0"):
            raise RuntimeError(name + " must be unset or 0 for CUDA FP32 decoding")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("CUDA FP32 decoding requires matmul and cuDNN TF32 disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("CUDA FP32 decoding requires float32 matmul precision='highest'")
    if torch.backends.cudnn.benchmark:
        raise RuntimeError("CUDA decoding requires cuDNN benchmark=False")


def _configure_precision():
    for name in ("NVIDIA_TF32_OVERRIDE", "TORCHINDUCTOR_USE_FAST_MATH"):
        if os.environ.get(name) not in (None, "0"):
            raise RuntimeError(name + " must be unset or 0 for CUDA FP32 decoding")
    # Explicit GPU selection opts into this backend's strict FP32 policy.
    # Later changes by another library are rejected before the next packet.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    _precision_policy()


class _CheckedStep(torch.nn.Module):
    """Keep finite validation in the compiled graph and use one CPU transfer."""

    def __init__(self, model):
        super().__init__()
        self.step = _TensorStep(model)

    def forward(self, z, *history):
        result = self.step(z, *history)
        values = torch.cat(tuple(value.reshape(-1) for value in result))
        valid = torch.isfinite(values).all().to(torch.float32).reshape(1)
        # A single packed output carries audio and validation status. Validation
        # also covers next history, which tanh-bounded audio alone cannot detect.
        packed = torch.cat((result[0].reshape(-1), valid))
        return (packed, *result[1:])


def _ready(packed, frames):
    if (packed.dtype != torch.float32 or not packed.is_cuda
            or tuple(packed.shape) != (frames * 1920 + 1,)):
        raise RuntimeError("CUDA model returned invalid audio storage")
    ready = packed.detach().to(device="cpu", non_blocking=False).numpy()
    if not np.isfinite(ready).all() or ready[-1] != 1:
        raise RuntimeError("CUDA model returned nonfinite audio or history")
    return ready[:-1].reshape(1, 1, frames * 1920).copy()


class CUDADecoder:
    """Frozen CUDA weights with the same batch/stream API as the CPU decoder."""

    def __init__(self, model, info, mode="streaming"):
        with _CUDA_LOCK:
            _configure_precision()
        if mode not in ("batch", "streaming"):
            raise ValueError("mode must be batch or streaming")
        if (model.sample_rate, model.hop_samples, model.latent_channels) != (48000, 1920, 64):
            raise RuntimeError("Unexpected CUDA sample/latent contract")
        shapes = model.state_shapes
        if (not isinstance(shapes, dict) or not shapes or any(
                not isinstance(name, str) or not name or not isinstance(shape, (tuple, list))
                or len(shape) != 3 or shape[0] != 1
                or any(type(n) is not int or n <= 0 for n in shape)
                for name, shape in shapes.items())):
            raise RuntimeError("Invalid CUDA history specification")
        self._state_shapes = {name: tuple(shape) for name, shape in shapes.items()}
        self._names = tuple(self._state_shapes)
        self._model, self.mode = model, mode
        self._device = torch.device(model.device)
        if self._device.type != "cuda":
            raise RuntimeError("CUDA decoder requires CUDA weights; CPU fallback is disabled")
        self.state_bytes = sum(int(np.prod(s)) * 4 for s in self._state_shapes.values())
        self._lock = threading.RLock()
        self._eager = _CheckedStep(model).eval()
        self._compiled = None
        self._prepared = False
        self._spare = None
        self.info = {
            **info, "mode": mode, "backend": "cuda", "state_device": "cuda",
            "output_device": "cpu", "output_ready": True,
            "state_bytes_per_stream": self.state_bytes,
            "rollback_bytes_per_stream": self.state_bytes,
            "compiled_packet_ms": [40, 80] if mode == "streaming" else [],
            "other_packet_sizes": "eager_cuda", "tf32": False,
        }

    @classmethod
    def from_onnx(cls, path, info=None, mode="streaming"):
        from .mps_decoder import MPSModel
        from .cuda_kernels import optimize_model

        with _CUDA_LOCK, torch.inference_mode():
            _configure_precision()
            # The existing parser verifies both pinned files and every original
            # operator before materializing literal FP32 coefficients.
            model = MPSModel.from_onnx(Path(path), device="cpu").to(device="cuda")
            if mode == "streaming":
                optimize_model(model)
        return cls(model, {} if info is None else info, mode)

    def prepare(self):
        """Compile/capture 40/80 ms shapes without advancing a live stream."""
        if self.mode == "streaming":
            with self._lock, _CUDA_LOCK:
                if not self._prepared:
                    _precision_policy()
                    self._compiled = torch.compile(
                        self._eager, fullgraph=True, dynamic=False,
                        options={"triton.cudagraphs": False},
                    )
                    # Hand this already captured stream to the first caller.
                    self._spare = CUDAStream(self)
                    self._prepared = True
        return self

    def _validate_result(self, result, frames):
        if not isinstance(result, (tuple, list)) or len(result) != len(self._names) + 1:
            raise RuntimeError("CUDA model returned incomplete history")
        for name, value in zip(self._names, result[1:]):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.device != self._device
                    or tuple(value.shape) != self._state_shapes[name]):
                raise RuntimeError("CUDA model returned invalid history: " + name)
        return _ready(result[0], frames)

    def decode(self, latents):
        """Decode an independent complete sequence from zero state."""
        from .gpu import _latents

        if self.mode != "batch":
            raise RuntimeError("Use decoder.stream() for stateful streaming")
        latents = _latents(latents)
        if not latents.shape[2]:
            return np.empty((1, 1, 0), dtype=np.float32)
        with self._lock, _CUDA_LOCK, torch.inference_mode(), torch.cuda.device(self._device):
            _precision_policy()
            with torch.autocast(device_type="cuda", enabled=False):
                z = torch.from_numpy(latents).to(device=self._device)
                histories = tuple(torch.zeros(s, dtype=torch.float32, device=self._device)
                                  for s in self._state_shapes.values())
                result = self._eager(z, *histories)
                return self._validate_result(result, latents.shape[2])

    def stream(self):
        if self.mode != "streaming":
            raise RuntimeError("Load mode='streaming' to create an audio stream")
        self.prepare()
        with self._lock, _CUDA_LOCK:
            if self._spare is not None:
                stream, self._spare = self._spare, None
                return stream
            return CUDAStream(self)


class CUDAStream:
    """Private graph/state storage for one causal utterance."""

    def __init__(self, decoder):
        self._decoder = decoder
        self._lock = threading.Lock()
        self._closed = False
        self.frames_decoded = 0
        self._graphs, self._inputs, self._outputs = {}, {}, {}
        with _CUDA_LOCK, torch.inference_mode(), torch.cuda.device(decoder._device):
            _precision_policy()
            self._arena = torch.zeros(decoder.state_bytes // 4, dtype=torch.float32,
                                      device=decoder._device)
            self._backup = torch.empty_like(self._arena)
            states, offset = [], 0
            for shape in decoder._state_shapes.values():
                length = int(np.prod(shape))
                states.append(self._arena[offset:offset + length].view(shape))
                offset += length
            self._states = tuple(states)
            with torch.autocast(device_type="cuda", enabled=False):
                self._capture()
            self._arena.zero_()
            torch.cuda.synchronize(decoder._device)

    @property
    def _history(self):
        """Diagnostic views only; callers must not mutate these tensors."""
        return dict(zip(self._decoder._names, self._states))

    def _capture(self):
        decoder = self._decoder
        for frames in (1, 2):
            z = torch.zeros((1, 64, frames), dtype=torch.float32, device=decoder._device)
            self._inputs[frames] = z
            stream = torch.cuda.Stream(device=decoder._device)
            stream.wait_stream(torch.cuda.current_stream(decoder._device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    decoder._compiled(z, *self._states)
            torch.cuda.current_stream(decoder._device).wait_stream(stream)
            torch.cuda.synchronize(decoder._device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # One contiguous device copy preserves rollback without 26
                # Python launches. The accepted history commits stay unchanged.
                self._backup.copy_(self._arena)
                result = decoder._compiled(z, *self._states)
                for destination, value in zip(self._states, result[1:]):
                    destination.copy_(value)
            self._graphs[frames] = graph
            self._outputs[frames] = result
            graph.replay()
            decoder._validate_result(result, frames)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Decoder stream is closed; create a new stream")

    def decode_chunk(self, latents):
        from .gpu import _latents

        latents = _latents(latents)
        with self._lock:
            self._check_open()
            frames = latents.shape[2]
            if not frames:
                return np.empty((1, 1, 0), dtype=np.float32)
            decoder = self._decoder
            with decoder._lock, _CUDA_LOCK, torch.inference_mode(), torch.cuda.device(decoder._device):
                _precision_policy()
                with torch.autocast(device_type="cuda", enabled=False):
                    z = torch.from_numpy(latents).to(device=decoder._device)
                    if frames in self._graphs:
                        self._inputs[frames].copy_(z)
                        try:
                            self._graphs[frames].replay()
                            result = self._outputs[frames]
                            # These fixed graph tensors were structurally
                            # validated during capture. Their shapes/devices
                            # cannot change on replay. Values, including every
                            # history, are still checked on every packet.
                            audio = _ready(result[0], frames)
                        except BaseException:
                            self._arena.copy_(self._backup)
                            torch.cuda.synchronize(decoder._device)
                            raise
                    else:
                        # Uncommon sizes remain bounded eager CUDA. Validate
                        # all outputs before committing any streaming state.
                        result = decoder._eager(z, *self._states)
                        audio = decoder._validate_result(result, frames)
                        for destination, value in zip(self._states, result[1:]):
                            destination.copy_(value)
                        torch.cuda.synchronize(decoder._device)
                self.frames_decoded += frames
                return audio

    def reset(self):
        with self._lock:
            self._check_open()
            with self._decoder._lock, _CUDA_LOCK, torch.inference_mode():
                self._arena.zero_()
                torch.cuda.synchronize(self._decoder._device)
            self.frames_decoded = 0

    def flush(self):
        with self._lock:
            self._check_open()
            return np.empty((1, 1, 0), dtype=np.float32)

    def close(self):
        with self._lock, self._decoder._lock, _CUDA_LOCK:
            if not self._closed:
                torch.cuda.synchronize(self._decoder._device)
                self._graphs.clear()
                self._inputs.clear()
                self._outputs.clear()
                self._states = ()
                self._arena = self._backup = None
                self._closed = True

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()


def load_cuda_model(info, mode="streaming"):
    """Create the NVIDIA decoder after the public facade verifies its source."""
    decoder = CUDADecoder.from_onnx(info["model_path"], info=info, mode=mode)
    return decoder.prepare()
