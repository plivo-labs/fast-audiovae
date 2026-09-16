"""Numerical and causal-halo checks for the three shipped CUDA kernels."""
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires NVIDIA CUDA compute capability 8.0 or newer")
    from fast_audiovae import cuda_kernels
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield torch, cuda_kernels
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _assert_close(torch, actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.fixture(scope="module")
def stage0_weight(cuda):
    torch, _ = cuda
    generator = torch.Generator(device="cuda").manual_seed(251)
    return torch.randn((8192, 4096), generator=generator, device="cuda") / 64


@pytest.mark.parametrize("length", [1, 2])
def test_exact_projection_preserves_inputs_and_supports_strided_features(cuda, stage0_weight, length):
    torch, kernels = cuda
    generator = torch.Generator(device="cuda").manual_seed(77 + length)
    backing = torch.randn((4096, length * 2), generator=generator, device="cuda")
    features = backing[:, ::2]
    before = backing.clone()
    result = kernels.exact_n(stage0_weight, features, 0)
    _assert_close(torch, result, stage0_weight @ features)
    assert result.shape == (8192, length) and result.dtype == torch.float32
    assert torch.equal(backing, before)
    with pytest.raises(ValueError, match="configuration"):
        kernels.exact_n(stage0_weight, features, 1)


@pytest.mark.parametrize("dilation,length", [(1, 2), (3, 11), (9, 53)])
def test_depthwise_snake_reads_causal_history_and_returns_independent_tail(cuda, dilation, length):
    torch, kernels = cuda
    generator = torch.Generator(device="cuda").manual_seed(94 + dilation)
    channels, halo = 7, 6 * dilation
    x_backing = torch.randn((1, channels, length * 2), generator=generator, device="cuda") * 0.2
    h_backing = torch.randn((1, channels, halo * 2), generator=generator, device="cuda") * 0.1
    x, history = x_backing[..., ::2], h_backing[..., ::2]
    weight = torch.randn((channels, 1, 7), generator=generator, device="cuda") * 0.1
    bias = torch.randn((channels,), generator=generator, device="cuda") * 0.01
    alpha = torch.linspace(0.4, 2.0, channels, device="cuda").reshape(1, channels, 1)
    # Deliberately not 1 / alpha: the original graph's stored coefficient must
    # be used literally instead of being recomputed inside the fused kernel.
    reciprocal = torch.linspace(0.2, 0.8, channels, device="cuda").reshape(1, channels, 1)
    original_x, original_h = x_backing.clone(), h_backing.clone()
    conv = SimpleNamespace(weight=weight, bias=bias, halo=halo, dilation=dilation)
    snake = SimpleNamespace(alpha=alpha, reciprocal=reciprocal)
    convolved = torch.nn.functional.conv1d(torch.cat((history, x), -1), weight, bias,
                                         groups=channels, dilation=dilation)
    sine = torch.sin(alpha * convolved)
    expected = convolved + reciprocal * (sine * sine)
    actual, next_history = kernels.fused_decode(x, history, conv, snake)
    _assert_close(torch, actual, expected)
    assert torch.equal(next_history, torch.cat((history, x), -1)[..., -halo:])
    assert torch.equal(x_backing, original_x) and torch.equal(h_backing, original_h)
    next_history.zero_()
    assert torch.equal(x_backing, original_x) and torch.equal(h_backing, original_h)
    empty, retained = kernels.fused_decode(x[..., :0], history, conv, snake)
    assert empty.shape == (1, channels, 0) and torch.equal(retained, history)
    assert retained.data_ptr() != history.data_ptr()


@pytest.mark.parametrize("channels,length,config", [(32, 1920, 0), (64, 1920, 1), (128, 480, 0)])
def test_fused_matrix_bias_residual_matches_separate_fp32_operations(cuda, channels, length, config):
    torch, kernels = cuda
    generator = torch.Generator(device="cuda").manual_seed(channels)
    weight = torch.randn((channels, channels, 1), generator=generator, device="cuda") / channels**0.5
    bias = torch.randn((channels,), generator=generator, device="cuda") * 0.02
    branch = torch.randn((1, channels, length * 2), generator=generator, device="cuda")[..., ::2]
    residual = torch.randn((1, channels, length * 2), generator=generator, device="cuda")[..., ::2]
    branch_before, residual_before = branch.clone(), residual.clone()
    expected = (weight[:, :, 0] @ branch[0]).unsqueeze(0) + bias[None, :, None]
    expected = expected + residual
    result = kernels.residual_epilogue(weight, branch, bias, residual, config)
    _assert_close(torch, result, expected)
    assert result.is_contiguous() and result.dtype == torch.float32
    assert torch.equal(branch, branch_before) and torch.equal(residual, residual_before)
