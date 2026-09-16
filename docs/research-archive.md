# Historical research archive

Historical source snapshots and raw profiling files are preserved at
[`archive/research-2026-09-16`](https://github.com/plivo-labs/fast-audiovae/tree/archive/research-2026-09-16),
commit `9c860e1c386365496a55f2fbd8734355e213e8ab`.
The cleanup removed their contents from `main`; it did not alter their archived
bytes or rewrite Git history.

| Archived material | Files | Location in the archived revision |
| --- | ---: | --- |
| Intel historical source snapshots | 224 | [Intel archive](https://github.com/plivo-labs/fast-audiovae/tree/9c860e1c386365496a55f2fbd8734355e213e8ab/experiments/intel-precision/archive) |
| Apple worker experiments | 52 | [Apple archive](https://github.com/plivo-labs/fast-audiovae/tree/9c860e1c386365496a55f2fbd8734355e213e8ab/experiments/apple-threading/archive) |
| Raw CPU/GPU profiles and traces | 15 | Original paths under `benchmarks/` and `experiments/` in that revision |

The 291 archived files contained 12,942,140 bytes. A small forwarding page remains
at the former Intel archive landing path. Production build resources, dependency
licenses, current benchmark summaries, release validation and console logs remain
in `main`. None of the removed files is in the package's build-resource allowlist.
The [SLEEF license for the archived inline headers](https://github.com/plivo-labs/fast-audiovae/blob/9c860e1c386365496a55f2fbd8734355e213e8ab/experiments/intel-precision/archive/iteration3/SLEEF-LICENSE.txt)
remains alongside those headers in the archived revision.

Raw profiles belong to the AMD precision, Apple precision, Intel precision,
Intel upsampling, Intel streaming-transfer and Apple GPU V3/V4 experiments, plus
`benchmarks/intel-profile.json`. Their summaries and historical conclusions
remain available in the [benchmark index](../benchmarks/README.md).

## Reproduce an archived experiment

Use the complete archived checkout so its relative paths, licenses and original
hash manifests remain together:

```sh
git fetch origin tag archive/research-2026-09-16
git worktree add --detach ../fast-audiovae-research archive/research-2026-09-16
```

Run historical verification scripts from that checkout. In particular, the Intel
publication verifier expects the archived snapshots and raw profiles. Frozen
manifests and source-pinned documents in `main` still describe their original
complete experiment; their hashes were not changed to skip missing evidence.

The installed package is unaffected. This is a source-tree cleanup, not a new
model, kernel, benchmark or wheel release.
