# NVIDIA GPU decoding

Select `load(device="gpu")` on Linux x86-64 to use the CUDA decoder. CPU and streaming remain the defaults. NVIDIA support is included in the refreshed 0.5.0 build 1 wheels.

Requirements: an NVIDIA GPU with compute capability 8.0 or newer, CUDA-enabled PyTorch 2.14.x and its matching Triton package. The measured configuration is an H100 NVL, PyTorch 2.14.0+cu126, CUDA 12.6 and cuDNN 9.25.1. Other supported NVIDIA GPUs compile the same kernels, but their performance has not been measured here.

```python
from fast_audiovae import load

decoder = load(device="gpu")
with decoder.stream() as stream:
    audio = stream.decode_chunk(latents)  # NumPy float32 [1, 64, L]
```

Audio returns as an owned, ready-to-use NumPy array at 48 kHz. One latent gives 40 ms; two give 80 ms. Those sizes use compiled CUDA graphs automatically. Other packet sizes use eager CUDA with the same causal history. Independent full-clip decoding uses `load(device="gpu", mode="batch")` and `decoder.decode(latents)`.

First load compiles the two streaming shapes. Each stream owns its graph buffers and history. Keep a stream open for one utterance; reset it for a new utterance. Separate streams do not share histories. `flush()` returns empty audio because there is no lookahead to drain. Calls are serialized for safe graph execution; this is not a concurrent GPU batching scheduler. `threads=1` is required for GPU mode.

## Implementation and safeguards

The decoder retains the original FP32 weights. It specializes the first upsampler's narrow matrix, fuses the 18 residual depthwise convolutions with their following Snake activations, and fuses nine late pointwise matrices with bias and residual addition. The history-update experiment was excluded because it slowed the combined winner.

TF32 and fast math are disabled. Loading configures PyTorch's FP32 policy; later incompatible changes are rejected. An explicit GPU request never falls back to CPU. Batch mode retains the original eager operations and is identified separately in `decoder.info`.

Every packet validates finite audio and all next-history values. Graph shapes and devices are validated when fixed buffers are created. Failed output validation restores the previous history. Invalid inputs are rejected before a stream advances. This preserves the public stream contract while avoiding repeated structural checks of fixed graph buffers.

## Validation

The installed wheel was checked against the original upstream CUDA decoder using the pinned original weights, with `atol=1e-5, rtol=1e-4`. Numerical parity means agreement within this tolerance, not bit-identical floating-point accumulation.

The qualification covers all 60 cached multilingual recordings, totaling 535.08 seconds, at both 40/80 ms; actual encoded silence and quiet waveforms; a continuous 60.04-second stream with mixed 40/80/120/200 ms calls; independent streams, reset, odd tails, output ownership and invalid inputs. Another 15 held-out recordings cover 66.55 seconds of laughter, screaming, shouting and human whistling, including independent batch decoding.

All 45,894 numerical comparisons passed, including the previous compiled control. The 27 CUDA tests also passed, including overflow rejection with history rollback. The local regression suite passed 566 tests and 1,963 subtests; 24 CUDA or optional-environment tests were skipped on the Mac.

| Output chunk | Previous compiled CUDA RTF | Combined default RTF | Less decoding time |
| --- | ---: | ---: | ---: |
| 40 ms | 0.01623 | **0.01459** | 10.1% |
| 80 ms | 0.00880 | **0.00742** | 15.6% |

RTF includes public API validation, CPU-to-GPU input transfer and completed, independently owned CPU output. Model loading, graph preparation and quality comparisons are outside timing. Two measured rounds reverse execution order. No perceptual score changes are claimed from numerical checks alone. Details and measured results are recorded in [the validation summary](../benchmarks/nvidia-cuda/summary.json).

The earlier bare-kernel screen measured 0.01337/0.00690. The public decoder adds finite checks across audio and histories, rollback storage and API handling, so those lower figures are not its end-to-end latency. No packet missed its 40/80 ms deadline in this run; paced live streaming can behave differently.
