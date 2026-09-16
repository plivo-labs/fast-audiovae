# fast-audiovae

Fast, causal AudioVAE2 decoding on Apple, AMD and Intel CPUs, with optional Apple and NVIDIA GPU support. Kernels are selected automatically. **CPU, streaming and one thread are the defaults.**

## Install

On Linux or macOS, clone the repository:

```sh
git clone https://github.com/plivo-labs/fast-audiovae.git
cd fast-audiovae
```

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install fast-audiovae==0.5.0 --find-links https://github.com/plivo-labs/fast-audiovae/releases/expanded_assets/v0.5.0
```

Or use pip with Python 3.11–3.13:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install fast-audiovae==0.5.0 --find-links https://github.com/plivo-labs/fast-audiovae/releases/expanded_assets/v0.5.0
```

These commands install the release wheels with optimized CPU kernels. For GPU support, use `'fast-audiovae[gpu]==0.5.0'` instead. CPU installation does not require PyTorch.

To refresh an existing 0.5.0 installation to build 1, add `--reinstall-package fast-audiovae` with uv or `--force-reinstall --no-deps` with pip.

Native CPU wheels support Apple ARM on macOS 26.2+ and Intel/AMD Linux with glibc 2.38+. [Apple GPU requirements](docs/apple-gpu.md) · [NVIDIA GPU requirements](docs/nvidia-gpu.md).

## Use

```python
from fast_audiovae import load

decoder = load()
with decoder.stream() as stream:
    audio = stream.decode_chunk(latents)  # NumPy float32 [1, 64, L]
```

Each latent frame produces 40 ms of mono 48 kHz audio; two frames produce 80 ms. Pass only new latents on each call. First load downloads the weights and prepares the backend.

- GPU: `load(device="gpu")`; GPU is never selected automatically.
- More CPU threads: `load(threads=4)`.
- Batch: `load(mode="batch")`, then `decoder.decode(latents)`.

## Speed

**RTF: lower is faster; 1.0 is real time.** Causal streaming, one CPU thread or one GPU host thread. Decoder only; loading, compilation and encoding are excluded. AudioVAE2 outputs 48 kHz; Pocket Mimi outputs 24 kHz.

### CPU

| CPU | Chunk | Original AudioVAE2 | fast-audiovae | Pocket Mimi |
| --- | --- | ---: | ---: | ---: |
| Apple M5 Max | 40 ms | 0.7050* | **0.1263** | n/a |
| Apple M5 Max | 80 ms | 0.4706* | **0.0996** | 0.1019* |
| AMD EPYC 9654 | 40 ms | Not measured | Not measured | n/a |
| AMD EPYC 9654 | 80 ms | 0.6006* | **0.1375** | 0.1230* |
| Intel Xeon 8280 | 40 ms | Not measured | **0.3941** | n/a |
| Intel Xeon 8280 | 80 ms | 1.0041* | **0.2666** | 0.2223* |

### GPU

| GPU | Chunk | Original AudioVAE2 | fast-audiovae | Pocket Mimi |
| --- | --- | ---: | ---: | ---: |
| Apple M5 Max | 40 ms | 0.1696 | **0.0577** | n/a |
| Apple M5 Max | 80 ms | 0.0933 | **0.0321** | 0.0634 |
| NVIDIA H100 NVL | 40 ms | 0.11643* | **0.01459** | n/a |
| NVIDIA H100 NVL | 80 ms | 0.05885* | **0.00742** | 0.02818* |

`n/a`: Mimi requires at least 80 ms. `*`: retained earlier measurements with different workloads or timing procedures from the current package checks; these are an overview, not matched speedup comparisons. Apple GPU rows share one matched test.

[CPU protocols](benchmarks/README.md) · [Apple GPU](experiments/apple-gpu-v6/report.md) · [NVIDIA GPU](docs/nvidia-gpu.md) · [NVIDIA baseline records](benchmarks/nvidia-cuda/baselines.json) · [Batch results](docs/apple-batch-validation.md) · [Multithreaded results](benchmarks/streaming/apple-selected-20260913.json).

## Reconstruction quality

Same 60 FLEURS recordings across ten languages, 80 ms streaming. Higher is better. The optimized row is the Apple CPU default.

| Audio | PESQ-WB | STOI | UTMOS22 | DNSMOS P.835 | DNSMOS P.808 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original recordings | Reference | Reference | 2.321 | 2.775 | 3.428 |
| Original AudioVAE2 | 3.742 | 0.9360 | 2.257 | 2.765 | 3.404 |
| fast-audiovae | 3.741 | 0.9360 | 2.257 | 2.766 | 3.403 |
| Pocket Mimi | 2.130 | 0.8074 | 2.517 | 2.894 | 3.339 |

Scored at 16 kHz; these metrics do not measure high-frequency fidelity. UTMOS and DNSMOS are predictions, not listening ratings. Meta DACVAE is excluded because the tested model requires lookahead. [Quality details](docs/apple-int8.md) · [DNSMOS verification](docs/streaming-baseline.md#dnsmos-revalidation).

[Streaming API](docs/streaming.md) · [Optional encoder](docs/optional-backends.md) · [Kernel details](docs/cpu-kernel-results.md).

[Apache-2.0 license](LICENSE), matching [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM), the source of the architecture and weights. Third-party dependencies retain their own licenses.
