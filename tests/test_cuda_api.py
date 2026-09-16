"""CUDA dispatch contracts on CPU, plus tiny real-CUDA streaming contracts.

The CUDA cases need no downloaded checkpoint and skip on non-CUDA machines.
Full pinned-model quality and timing qualification is a separate release check.
"""
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
import os
import subprocess
import sys

import numpy as np
import pytest

from fast_audiovae import automatic, gpu


@pytest.fixture
def cuda_facade(monkeypatch):
    torch = SimpleNamespace(
        __version__="2.14.0+cu126", version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_capability=lambda: (9, 0),
                             get_device_name=lambda: "test NVIDIA GPU"))
    monkeypatch.setattr(gpu.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu.platform, "machine", lambda: "x86_64")
    original = gpu.importlib.import_module
    monkeypatch.setattr(gpu.importlib, "import_module",
                        lambda name: torch if name == "torch" else original(name))
    return torch


def test_linux_cuda_requires_explicit_gpu_and_supported_runtime(cuda_facade, monkeypatch):
    assert gpu._torch() is cuda_facade
    monkeypatch.setattr(gpu, "_policy", lambda: pytest.fail("MPS policy reached CUDA"))
    assert gpu._torch() is cuda_facade
    cuda_facade.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        gpu._torch()
    cuda_facade.cuda.is_available = lambda: True
    cuda_facade.cuda.get_device_capability = lambda: (7, 5)
    with pytest.raises(RuntimeError, match="compute capability"):
        gpu._torch()
    cuda_facade.cuda.get_device_capability = lambda: (9, 0)
    cuda_facade.__version__ = "2.13.0+cu126"
    with pytest.raises(RuntimeError, match="2.14"):
        gpu._torch()


def test_linux_setup_has_no_inference_and_load_routes_to_cuda(cuda_facade, monkeypatch, tmp_path):
    verified, loaded = [], []
    monkeypatch.setattr(gpu, "verify_model", lambda path: verified.append(path))
    monkeypatch.setattr(gpu, "fetch_model", lambda *a: pytest.fail("Unexpected download"))
    monkeypatch.setattr(automatic, "_build", lambda *a: pytest.fail("CPU build requested"))
    monkeypatch.setitem(sys.modules, "fast_audiovae.mps_decoder", SimpleNamespace(
        MPSModel=SimpleNamespace(from_onnx=lambda *a, **kw: pytest.fail("MPS loader reached CUDA"))))
    sentinel = object()

    def factory(info, *, mode):
        loaded.append((info, mode))
        return sentinel

    monkeypatch.setitem(sys.modules, "fast_audiovae.cuda_decoder", SimpleNamespace(load_cuda_model=factory))
    source = tmp_path / "audio_vae_decoder.onnx"
    both = automatic.setup(device="gpu", mode="both", source=source,
                           cache_dir=tmp_path, offline=True)
    assert not loaded
    assert both["streaming"]["compiled_packet_ms"] == [40, 80]
    assert both["batch"]["compiled_packet_ms"] == []
    assert both["streaming"]["cache_key"] != both["batch"]["cache_key"]
    for info in both.values():
        selected = "torch_cuda_fused" if info["mode"] == "streaming" else "torch_cuda_eager"
        assert info["backend"] == "cuda" and info["selected"] == selected
        assert info["precision"] == "FP32" and info["tf32"] is False
        assert info["fallback"] is False and info["cpu_only"] is False
        assert info["capability"] == [9, 0]
        assert "mps_fast_math" not in info and "mps_cpu_fallback" not in info
    assert automatic.load(device="gpu", source=source, cache_dir=tmp_path, offline=True) is sentinel
    assert loaded[0][1] == "streaming" and loaded[0][0]["threads"] == 1
    assert verified == [source, source]


def test_cpu_import_does_not_import_cuda_or_triton():
    code = """
import importlib.abc
import sys
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'triton') or fullname in (
                'fast_audiovae.cuda_decoder', 'fast_audiovae.cuda_kernels'):
            raise AssertionError('CPU import reached optional CUDA dependency: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
import fast_audiovae
import fast_audiovae.automatic
assert 'torch' not in sys.modules and 'triton' not in sys.modules
"""
    source = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run([sys.executable, "-c", code],
                               env={**os.environ, "PYTHONPATH": source},
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@lru_cache(maxsize=1)
def _tiny_classes(torch):
    # Keep model types stable across fixtures, as they are in the real loader.
    # Recreating Python classes adds artificial Dynamo type specializations.
    class TinyModel(torch.nn.Module):
        sample_rate, hop_samples, latent_channels = 48000, 1920, 64
        state_shapes = {"history": (1, 1, 1)}
        state_bytes = 4

        def __init__(self):
            super().__init__()
            self.register_buffer("marker", torch.ones(1, device="cuda"))
            self.bad = None

        @property
        def device(self):
            return self.marker.device

        def decode(self, z, history):
            prior = history.get("history", z.new_zeros((1, 1, 1)))
            values = z.mean(1, keepdim=True).cumsum(-1) + prior
            audio = values.repeat_interleave(1920, dim=-1)
            next_history = {"history": values[..., -1:].clone()}
            if self.bad == "audio":
                audio = audio * float("nan")
            if self.bad == "history":
                next_history["history"] = next_history["history"] * float("nan")
            if self.bad == "raise":
                raise RuntimeError("injected eager CUDA failure")
            return audio, next_history

    class Step(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.names = tuple(model.state_shapes)

        def forward(self, z, *state):
            audio, next_history = self.model.decode(z, dict(zip(self.names, state)))
            return (audio, *(next_history[name] for name in self.names))

    return TinyModel, Step


@pytest.fixture
def cuda_decoder(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires NVIDIA compute capability 8.0 or newer")
    from fast_audiovae import cuda_decoder as backend
    TinyModel, Step = _tiny_classes(torch)
    monkeypatch.setattr(backend, "_TensorStep", Step)
    model = TinyModel().eval()
    decoder = backend.CUDADecoder(model, {"device": "gpu", "backend": "cuda"}, mode="streaming")
    return torch, model, decoder, backend


def test_many_cuda_decoders_share_the_same_bounded_shape_contract(cuda_decoder):
    torch, _, _, backend = cuda_decoder
    TinyModel, _ = _tiny_classes(torch)
    for _ in range(6):
        decoder = backend.CUDADecoder(TinyModel().eval(), {}, mode="streaming")
        with decoder.stream() as stream:
            np.testing.assert_array_equal(stream.decode_chunk(_packet()),
                                          np.full((1, 1, 1920), 0.125, np.float32))
            assert set(stream._graphs) == {1, 2}


def _packet(length=1, value=0.125):
    return np.full((1, 64, length), value, dtype=np.float32)


def test_cuda_streams_preserve_mixed_packet_history_reset_and_output_ownership(cuda_decoder):
    _, _, decoder, _ = cuda_decoder
    a, b = decoder.stream(), decoder.stream()
    assert set(a._graphs) == set(b._graphs) == {1, 2}
    total = 0
    held = []
    for length in (1, 2, 3, 1, 2):
        output = a.decode_chunk(_packet(length))
        values = np.arange(total + 1, total + length + 1, dtype=np.float32) * 0.125
        expected = np.repeat(values, 1920).reshape(1, 1, -1)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == np.float32 and output.flags.owndata
        held.append((output, output.copy()))
        total += length
        assert a.frames_decoded == total
    # Separate streams must not share capture buffers or advance each other.
    np.testing.assert_array_equal(b.decode_chunk(_packet()), np.full((1, 1, 1920), 0.125, np.float32))
    assert b.frames_decoded == 1 and a.frames_decoded == 9
    for output, saved in held:
        np.testing.assert_array_equal(output, saved)
    a.reset()
    assert a.frames_decoded == 0
    np.testing.assert_array_equal(a.decode_chunk(_packet()), b.decode_chunk(_packet(value=0)))
    a.close()
    b.close()


def test_cuda_graph_validation_failure_rolls_back_and_can_recover(cuda_decoder, monkeypatch):
    _, _, decoder, backend = cuda_decoder
    with decoder.stream() as stream:
        stream.decode_chunk(_packet())
        original = backend._ready

        def fail_copy(*args):
            raise RuntimeError("injected output-copy failure")

        monkeypatch.setattr(backend, "_ready", fail_copy)
        with pytest.raises(RuntimeError, match="output-copy failure"):
            stream.decode_chunk(_packet(2))
        monkeypatch.setattr(backend, "_ready", original)
        assert stream.frames_decoded == 1
        np.testing.assert_array_equal(stream.decode_chunk(_packet()),
                                      np.full((1, 1, 1920), 0.25, np.float32))


def test_cuda_graph_overflow_is_detected_and_history_rolls_back(cuda_decoder):
    _, _, decoder, _ = cuda_decoder
    with decoder.stream() as stream:
        stream.decode_chunk(_packet())
        # Inputs are finite, but summation/cumulative summation overflows FP32.
        with pytest.raises(RuntimeError, match="nonfinite"):
            stream.decode_chunk(_packet(2, value=np.finfo(np.float32).max))
        assert stream.frames_decoded == 1
        np.testing.assert_array_equal(stream.decode_chunk(_packet()),
                                      np.full((1, 1, 1920), 0.25, np.float32))


def test_cuda_prepare_and_policy_rejection_do_not_advance_live_stream(cuda_decoder):
    torch, _, decoder, _ = cuda_decoder
    with decoder.stream() as stream:
        stream.decode_chunk(_packet())
        assert decoder.prepare() is decoder
        assert stream.frames_decoded == 1
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with pytest.raises(RuntimeError, match="TF32"):
                stream.decode_chunk(_packet())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        assert stream.frames_decoded == 1
        np.testing.assert_array_equal(stream.decode_chunk(_packet()),
                                      np.full((1, 1, 1920), 0.25, np.float32))


@pytest.mark.parametrize("invalid", [None, np.zeros((1, 64, 1), np.float64),
                                    np.zeros((2, 64, 1), np.float32),
                                    np.zeros((1, 63, 1), np.float32),
                                    _packet(value=np.nan), _packet(value=np.inf)])
def test_cuda_invalid_input_never_advances_stream(cuda_decoder, invalid):
    _, _, decoder, _ = cuda_decoder
    with decoder.stream() as stream:
        stream.decode_chunk(_packet())
        with pytest.raises(ValueError):
            stream.decode_chunk(invalid)
        assert stream.frames_decoded == 1
        np.testing.assert_array_equal(stream.decode_chunk(_packet()),
                                      np.full((1, 1, 1920), 0.25, np.float32))


@pytest.mark.parametrize("bad", ["audio", "history", "raise"])
def test_cuda_failed_eager_output_never_commits_history(cuda_decoder, bad):
    _, model, decoder, _ = cuda_decoder
    with decoder.stream() as stream:
        stream.decode_chunk(_packet())
        model.bad = bad
        with pytest.raises(RuntimeError):
            # Three frames use the uncompiled CUDA path, allowing a controlled
            # output fault without changing an already-captured graph.
            stream.decode_chunk(_packet(3))
        model.bad = None
        assert stream.frames_decoded == 1
        np.testing.assert_array_equal(stream.decode_chunk(_packet()),
                                      np.full((1, 1, 1920), 0.25, np.float32))


def test_cuda_empty_noncontiguous_inputs_and_closed_stream(cuda_decoder):
    _, _, decoder, _ = cuda_decoder
    stream = decoder.stream()
    empty = np.empty((1, 64, 0), np.float32)
    assert stream.decode_chunk(empty).shape == stream.flush().shape == (1, 1, 0)
    assert stream.frames_decoded == 0
    backing = _packet(4)
    packet = backing[..., ::2]
    assert not packet.flags.c_contiguous
    output = stream.decode_chunk(packet)
    np.testing.assert_array_equal(output, np.repeat([0.125, 0.25], 1920).astype("f").reshape(1, 1, -1))
    np.testing.assert_array_equal(backing, _packet(4))
    stream.close()
    stream.close()
    for operation in (lambda: stream.decode_chunk(empty), stream.flush, stream.reset, stream.__enter__):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_cuda_batch_decodes_independent_sequences_and_rejects_streaming_api(cuda_decoder):
    _, model, streaming, backend = cuda_decoder
    batch = backend.CUDADecoder(model, {"device": "gpu", "backend": "cuda"}, mode="batch")
    with pytest.raises(RuntimeError):
        streaming.decode(_packet())
    with pytest.raises(RuntimeError):
        batch.stream()
    first = batch.decode(_packet(3))
    np.testing.assert_array_equal(first, batch.decode(_packet(3)))
    assert first.flags.owndata and batch.decode(_packet(0)).shape == (1, 1, 0)
