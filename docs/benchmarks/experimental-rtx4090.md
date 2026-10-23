# RTX 4090 measurements — 2026-10-20/21

Hardware: RTX 4090, 24,564 MiB device memory, 32 GiB system RAM. Host Torch
2.14.1+cu130. Other desktop processes were running. All sources are synthetic;
no physical webcam was activated. Research sources remain frozen at the
requested 2026-10-20 cutoff.

Standalone LivePortrait head processing, after four warmup frames, reference
conditioning once, CUDA-synchronized per-frame timings:

| Input/composite resolution | Samples | Native inference FPS | Mean ms | P50 ms | P95 ms | P99 ms | Peak / steady Torch MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640×480 | 30 | 24.33 | 41.10 | 38.69 | 56.84 | 64.74 | 1057.34 / 643.58 |
| 1280×720 | 60 | 21.08 | 47.44 | 47.84 | 55.69 | 58.38 | 1057.34 / 642.95 |
| 1920×1080 | 60 | 18.86 | 53.01 | 52.76 | 62.85 | 66.05 | 1057.34 / 642.95 |

The neural head input is 256×256 at every composite resolution; these are not
native 720p/1080p generative models. Model loading took 1.03–1.62 s; reference
preparation 0.42–1.34 s. Filesystem caches were warm: this does not establish
true cold-start time. PNG/video export and pixel-difference calculations are
excluded from native timing and included in separate wall-rate JSON fields.
Torch allocation excludes driver/display/other-process allocations.

Baseline AlphaFace synthetic 30 FPS pipeline at 720p, short two-second runs:

| FrameGen | Native swap FPS | New output FPS | Capture→sink median ms | Peak Torch MiB |
|---|---:|---:|---:|---:|
| off | 29.6 | 29.8 | 25.8 | 1128.5 |
| RIFE 2x | 29.2 | 56.0 | 74.0 | 1291.3 |
| RIFE 3x | 29.3 | 84.6 | 75.9 | 1347.1 |

Short samples are noisy, and output/native rate windows differ. Frame-generated
FPS is never native neural inference FPS. Head model throughput and AlphaFace
capture→sink pipeline results have different timing scope; do not compare their
latencies as though they were the same experiment.

The shared-pipeline validation additionally exercises real LivePortrait + RIFE,
private original/processed previews, IPC, CPU H.264 recording and 12-frame
offline source-rate preservation (17 FPS). Its raw report is included beside
the standalone timings. Held recording frames maintain wall time under dropped
inference; they are not neural or interpolated frames.

Full-body EditaLive: 52.71 GiB / 412 official files downloaded and SHA-256
verified. Validation was stopped after a host freeze requiring forced reboot.
Previous-boot logs show systemd-oomd terminating terminal-session processes;
memory exhaustion is suspected, without conclusive process attribution.
The local backend and validation script are disabled pending enforced RAM/swap
limits. The isolated environment used Python 3.11 / Torch 2.6.0 CUDA 12.4.
Initial runs exposed NixOS Triton driver lookup and
compiler issues; the conservative profile uses SDPA and disables Dynamo
compilation, including the upstream unconditional FP8-helper decorator.
No full-body FPS/VRAM result is claimed until actual generation completes.

Not measured: identity similarity, expression/pose accuracy, flow-compensated
flicker, autoregressive drift, long-duration stability, walking/hand-occlusion
quality, hybrid benefit, quantized-vs-unquantized parity, quality-preset matrix,
actual camera/PipeWire/OBS/browser latency. PersonaLive/DirectSwap/PixReenact
have no runnable local results. A translated still is a smoke test, not a
representative motion benchmark. No inference-quality improvement from RIFE
is asserted without further comparisons.

Reproduce using [the performance guide](../experimental-performance.md).
