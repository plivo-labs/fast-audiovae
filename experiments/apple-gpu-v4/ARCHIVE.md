# Apple GPU numerical repair

[report.md](report.md) explains the divergence and the qualified repair. [selected-candidate.json](selected-candidate.json) records the selected implementation and evidence hashes.

The first upsampler uses a paired matrix projection; later upsamplers retain the V3 projection form. Original weights, state schema, causal dependencies and validation tolerances remain unchanged. Both repaired variants passed the short state/waveform panel. The selected variant adds the 19 pointwise matrix replacements. CPU remains the default. Public-loader integration is separate.

These are retained workspace source snapshots using the V2 and V3 harnesses and verified local model assets. Failed and diagnostic attempts remain included. Interleaved projection has CPU algebra checks only; it and retain_first were not run on the GPU. No audio, model weights or raw state values are included.

The historical [English trace](https://github.com/plivo-labs/fast-audiovae/blob/9c860e1c386365496a55f2fbd8734355e213e8ab/experiments/apple-gpu-v4/trace-english-r1.json) and [expressive trace](https://github.com/plivo-labs/fast-audiovae/blob/9c860e1c386365496a55f2fbd8734355e213e8ab/experiments/apple-gpu-v4/trace-expressive-r1.json) now live in the research archive. These links also replace the original local trace paths in the preserved numerical analysis. To verify the complete original `manifest.json`, use a separate checkout of `archive/research-2026-09-16`; the retained manifest still describes that complete snapshot.
