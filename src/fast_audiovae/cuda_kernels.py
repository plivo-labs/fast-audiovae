"""Screened IEEE FP32 kernels for the explicit NVIDIA CUDA decoder.

No weights are quantized. Kernel selection is bounded by the original model
shapes; other packet sizes retain the same eager CUDA operations. This module
is imported only after a caller explicitly chooses GPU inference on NVIDIA.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _exact_n_gemv(W, X, Y, M: tl.constexpr, K: tl.constexpr,
                  N: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
                  ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, BLOCK_K)
    acc0 = tl.full((ROWS, BLOCK_K), 0, tl.float32)
    if N == 2:
        acc1 = tl.full((ROWS, BLOCK_K), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + lane
        w = tl.load(W + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), 0)
        x0 = tl.load(X + k * XS0, k < K, 0)
        acc0 = tl.fma(w, x0[None, :], acc0)
        if N == 2:
            x1 = tl.load(X + k * XS0 + XS1, k < K, 0)
            acc1 = tl.fma(w, x1[None, :], acc1)
    tl.store(Y + rows * N, tl.sum(acc0, axis=1), rows < M)
    if N == 2:
        tl.store(Y + rows * N + 1, tl.sum(acc1, axis=1), rows < M)


def _validate(weight, features):
    if weight.dtype != torch.float32 or features.dtype != torch.float32:
        raise ValueError("Matrix screen requires literal FP32 operands")
    if not weight.is_cuda or features.device != weight.device:
        raise ValueError("Matrix operands must share a CUDA device")
    if weight.shape != (8192, 4096) or not weight.is_contiguous():
        raise ValueError("Only the original contiguous stage-0 packed weight is supported")
    if features.ndim != 2 or features.shape[0] != 4096 or features.shape[1] not in (1, 2):
        raise ValueError("Only exact-N=1 or N=2 stage-0 features are supported")


@torch.library.custom_op("fast_audiovae_cuda::exact_n", mutates_args=())
def exact_n(weight: torch.Tensor, features: torch.Tensor, variant: int) -> torch.Tensor:
    _validate(weight, features)
    if variant != 0:
        raise ValueError("Unsupported CUDA matrix configuration")
    rows, block_k, warps = 4, 1024, 4
    output = torch.empty((weight.shape[0], features.shape[1]),
                         dtype=weight.dtype, device=weight.device)
    _exact_n_gemv[(triton.cdiv(weight.shape[0], rows),)](
        weight, features, output, weight.shape[0], weight.shape[1],
        features.shape[1], features.stride(0), features.stride(1),
        ROWS=rows, BLOCK_K=block_k, num_warps=warps)
    return output


@exact_n.register_fake
def _exact_n_fake(weight, features, variant):
    return weight.new_empty((weight.shape[0], features.shape[1]))


@triton.jit
def _depthwise_post_snake_kernel(
    X, H, W, B, A, R, Y,
    C: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    XC: tl.constexpr, XT: tl.constexpr,
    HC: tl.constexpr, HT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = flat < C * T
    channel = flat // T
    time = flat % T
    halo: tl.constexpr = 6 * D
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for tap in tl.static_range(7):
        joined_time = time + tap * D
        old = tl.load(H + channel * HC + joined_time * HT,
                      valid & (joined_time < halo), other=0.0)
        current = tl.load(X + channel * XC + (joined_time - halo) * XT,
                          valid & (joined_time >= halo), other=0.0)
        value = tl.where(joined_time < halo, old, current)
        weight = tl.load(W + channel * 7 + tap, valid, other=0.0)
        # cuDNN's FP32 convolution uses fused multiply-add. Keep this explicit
        # while disabling implicit fusion below for the literal Snake stages.
        total = tl.fma(value, weight, total)
    bias = tl.load(B + channel, valid, other=0.0)
    convolved = total + bias
    alpha = tl.load(A + channel, valid, other=0.0)
    reciprocal = tl.load(R + channel, valid, other=0.0)
    scaled = alpha * convolved
    sine = libdevice.sin(scaled)
    squared = sine * sine
    correction = reciprocal * squared
    activated = convolved + correction
    tl.store(Y + flat, activated, valid)


@torch.library.custom_op("fast_audiovae_cuda::depthwise_post_snake", mutates_args=())
def depthwise_post_snake(
    x: torch.Tensor, history: torch.Tensor, weight: torch.Tensor,
    bias: torch.Tensor, alpha: torch.Tensor, reciprocal: torch.Tensor,
    dilation: int, block: int,
) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    assert x.ndim == 3 and x.shape[0] == 1
    channels, length = x.shape[1:]
    assert dilation in (1, 3, 9) and block in (128, 256)
    assert history.shape == (1, channels, 6 * dilation)
    assert weight.shape == (channels, 1, 7)
    assert bias.numel() == alpha.numel() == reciprocal.numel() == channels
    for coefficient in (history, weight, bias, alpha, reciprocal):
        assert coefficient.device == x.device and coefficient.dtype == torch.float32
    for coefficient in (weight, bias, alpha, reciprocal):
        assert coefficient.is_contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if length:
        _depthwise_post_snake_kernel[(triton.cdiv(channels * length, block),)](
            x, history, weight, bias, alpha, reciprocal, output,
            channels, length, dilation, x.stride(1), x.stride(2),
            history.stride(1), history.stride(2), block,
            num_warps=4, enable_fp_fusion=False,
        )
    return output


@depthwise_post_snake.register_fake
def _depthwise_post_snake_fake(x, history, weight, bias, alpha, reciprocal, dilation, block):
    return x.new_empty(x.shape)


def fused_decode(x, history, depthwise, after, block=128):
    """Return post-Snake branch and an independent exact activated-input tail."""
    if x.shape[-1] == 0:
        return x.new_empty(x.shape), history.clone()
    branch = depthwise_post_snake(
        x, history, depthwise.weight, depthwise.bias,
        after.alpha, after.reciprocal, depthwise.dilation, block,
    )
    next_history = torch.cat((history, x), dim=-1)[..., -depthwise.halo:].clone()
    return branch, next_history

CONFIGS = {
    0: dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, num_warps=4),
    1: dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4),
}
SUPPORTED_SHAPES = frozenset({
    (128, 480), (128, 960), (64, 960), (64, 1920),
    (32, 1920), (32, 3840),
})


@triton.jit
def _matrix_bias_residual(
    weight, branch, bias, residual, output,
    C: tl.constexpr, T: tl.constexpr,
    WM: tl.constexpr, WK: tl.constexpr,
    XK: tl.constexpr, XN: tl.constexpr,
    RM: tl.constexpr, RN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    accum = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for start in range(tl.cdiv(C, BLOCK_K)):
        inner = start * BLOCK_K + ks
        a = tl.load(weight + rows[:, None] * WM + inner[None, :] * WK,
                    mask=(rows[:, None] < C) & (inner[None, :] < C), other=0)
        b = tl.load(branch + inner[:, None] * XK + cols[None, :] * XN,
                    mask=(inner[:, None] < C) & (cols[None, :] < T), other=0)
        accum = tl.dot(a, b, accum, input_precision="ieee")
    bias_value = tl.load(bias + rows, mask=rows < C, other=0)
    # Keep the source grouping: (matrix result + bias), then residual.
    biased = (accum + bias_value[:, None]).to(tl.float32)
    skip = tl.load(residual + rows[:, None] * RM + cols[None, :] * RN,
                   mask=(rows[:, None] < C) & (cols[None, :] < T), other=0)
    result = (biased + skip).to(tl.float32)
    tl.store(output + rows[:, None] * T + cols[None, :], result,
             mask=(rows[:, None] < C) & (cols[None, :] < T))


@torch.library.custom_op("fast_audiovae_cuda::residual_epilogue", mutates_args=())
def residual_epilogue(
    weight: torch.Tensor, branch: torch.Tensor, bias: torch.Tensor,
    residual: torch.Tensor, config_id: int = 0,
) -> torch.Tensor:
    """Compute pointwise(branch) + residual; arguments are read-only."""
    if branch.ndim != 3 or branch.shape[0] != 1 or residual.shape != branch.shape:
        raise ValueError("Expected equal [1, C, T] branch and residual tensors")
    channels, length = branch.shape[1:]
    if (channels, length) not in SUPPORTED_SHAPES or config_id not in CONFIGS:
        raise ValueError("Unscreened shape or configuration")
    if tuple(weight.shape) != (channels, channels, 1) or tuple(bias.shape) != (channels,):
        raise ValueError("Expected native pointwise coefficients")
    if any(t.dtype != torch.float32 or t.device != branch.device or not t.is_cuda
           for t in (weight, branch, bias, residual)):
        raise ValueError("CUDA FP32 tensors on one device are required")
    if bias.stride(0) != 1:
        raise ValueError("Bias must be contiguous")
    out = torch.empty(branch.shape, dtype=branch.dtype, device=branch.device)
    cfg = CONFIGS[config_id]
    grid = (triton.cdiv(channels, cfg["BLOCK_M"]), triton.cdiv(length, cfg["BLOCK_N"]))
    _matrix_bias_residual[grid](
        weight, branch, bias, residual, out, channels, length,
        weight.stride(0), weight.stride(1), branch.stride(1), branch.stride(2),
        residual.stride(1), residual.stride(2),
        **cfg, enable_fp_fusion=False,
    )
    return out


@residual_epilogue.register_fake
def _fake_residual_epilogue(weight, branch, bias, residual, config_id=0):
    return torch.empty_like(branch, memory_format=torch.contiguous_format)



_EPILOGUES = {
    (32, 1920): 0, (32, 3840): 0,
    (64, 960): 0, (64, 1920): 1,
    (128, 480): 0, (128, 960): 1,
}


class _FusedResidual(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.before, self.depthwise = original.before, original.depthwise
        self.after, self.pointwise = original.after, original.pointwise

    def decode(self, x, history):
        branch, next_history = fused_decode(
            self.before(x), history, self.depthwise, self.after, block=128,
        )
        choice = _EPILOGUES.get((x.shape[1], x.shape[2]))
        if choice is None:
            return x + self.pointwise(branch), next_history
        return residual_epilogue(
            self.pointwise.weight, branch, self.pointwise.bias, x, choice,
        ), next_history


def _paired_transpose_decode(self, x, history):
    length, channels, stride = x.shape[-1], self.weight.shape[1], self.stride
    if length == 0:
        return x.new_empty((1, channels, 0)), history.clone()
    previous = torch.cat((history, x[..., :-1]), dim=-1)
    features = torch.cat((x[0], previous[0]), dim=0)
    projection = (exact_n(self.packed_weight, features, 0)
                  if length in (1, 2) else torch.mm(self.packed_weight, features))
    y = projection.reshape(channels, stride, length)
    y = y.permute(0, 2, 1).reshape(1, channels, length * stride)
    return (y + self.bias[None, :, None]).contiguous(), x[..., -1:].clone()


def optimize_model(model):
    """Install the qualified three-kernel combination on one frozen model."""
    if getattr(model, "_cuda_kernels_installed", False):
        raise ValueError("CUDA kernels are already installed on this model")
    if len(model.stages) != 6 or any(len(stage.residuals) != 3 for stage in model.stages):
        raise ValueError("CUDA kernels require the original six-stage decoder")
    transpose = model.stages[0].transpose
    if not transpose.paired or tuple(transpose.packed_weight.shape) != (8192, 4096):
        raise ValueError("Unexpected first-stage CUDA projection")
    for stage in model.stages:
        for residual in stage.residuals:
            dw = residual.depthwise
            if (dw.groups != dw.channels or dw.weight.shape[1:] != (1, 7)
                    or dw.dilation not in (1, 3, 9)):
                raise ValueError("Unexpected residual CUDA convolution")
    transpose.decode = MethodType(_paired_transpose_decode, transpose)
    for stage in model.stages:
        stage.residuals = nn.ModuleList(_FusedResidual(block) for block in stage.residuals)
    model._cuda_kernels_installed = True
    return model.eval()
