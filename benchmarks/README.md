# Benchmark records

The results below qualify the paths shipped in 0.5.0. Each file records its own hardware, runtime, workload and measurement scope. The repository stores results in JSON and CSV files, not a database.

| Path | Current evidence | Scope |
| --- | --- | --- |
| Apple CPU, one-thread streaming | [Apple INT8](streaming/apple-int8-v1.json), [protocol](../docs/apple-int8.md) | M5 Max, ORT 1.30, 40/80 ms; 79 recordings for waveform checks and confirmation timing, 60 for speech quality |
| AMD CPU, one-thread streaming | [AMD serving](streaming/amd-serving-v1.json), [protocol](../docs/amd-serving.md) | EPYC 9654, 80 ms; short package checks against the accepted selective INT8 recipe |
| Intel CPU, one-thread streaming | [Intel serving](streaming/intel-serving-v1.json), [protocol](../docs/intel-serving.md) | Xeon Platinum 8280 VM, ORT 1.29, 40/80 ms; short package checks against the accepted selective INT8 recipe |
| Apple CPU, four-thread streaming | [Apple comparison](streaming/apple-selected-20260913.json), [protocol](../docs/streaming-baseline.md) | Retained kernels; matched AudioVAE2 and Mimi comparison before the one-thread INT8 update |
| Apple CPU, batch | [Batch comparison](batch/apple-mimi-audit-20260913.json), [protocol](../docs/apple-batch-validation.md) | One thread; independent 40/80/960 ms inputs |
| Apple GPU, streaming | [GPU V6](../experiments/apple-gpu-v6/report.md) | PyTorch 2.14, 40/80 ms; matched short GPU comparison and separate numerical checks |

Version 0.5.0 inherited the 0.4.2 runtime and native binaries unchanged. [Release validation](../docs/releases/v0.5.0-validation.json) records that relationship; it does not represent a new benchmark run.

## Reading comparisons

- Compare timings within the same test session and protocol. Confirmation timing with state checks, native-session timing and end-to-end latency are different measurements.
- The latest Apple CPU timing did not rerun Mimi. Its older Mimi result is a separate comparison.
- The current README quality row belongs to the Apple CPU default. AMD's separate quality panel and Intel/GPU numerical checks do not establish those same perceptual scores on other paths.
- Exact agreement with an accepted INT8 recipe does not mean exact agreement with stock FP32.

## Historical records

[streaming/summary.json](streaming/summary.json), [streaming/one-thread.json](streaming/one-thread.json), the older precision campaigns and full-call reports remain experiment records. They are not a current-results index. Their model paths, runtimes, packet lengths and thread counts can differ from the shipping defaults.

Raw measurements, source hashes and rejected experiments are retained unchanged. A hash-pinned experiment bundle may still contain its original `bevenky/fast-audiovae` URL; current installation and navigation use `plivo-labs/fast-audiovae`.
