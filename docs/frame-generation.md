# Frame generation (temporal interpolation)

`--frame-gen` raises the **presented** frame rate without running AlphaFace on
every displayed frame. It sits between the face swap and the output:

```
camera ─▶ detect ─▶ AlphaFace ─▶ composite ─▶ [frame generation] ─▶ watermark ─▶ preview / file / FFmpeg / v4l2loopback
```

Inserted frames are **synthesised by an interpolation network at the exact
fractional time of each output slot**, never duplicated. The disclosure
watermark is stamped *after* interpolation, so it stays crisp on every frame.

## Why RIFE (Practical-RIFE v4.25)

Candidates reviewed on 2026-09-22 for this exact job: 20–30 fps swapped
webcam frames, low added latency, stable faces, 720p/1080p on one RTX 4090
that also runs AlphaFace, and a license that permits redistribution of weights.

| Method | License (code / weights) | Fit for this use case |
|---|---|---|
| **RIFE / Practical-RIFE 4.25, 4.25-lite, 4.26** | MIT / MIT | Built for real time; per-frame feature cache means each extra frame costs one IFNet pass. **Measured** below. Chosen. |
| IFRNet | MIT | Also real-time class; unmaintained since 2024 and its weights are only on Google Drive, while Practical-RIFE keeps shipping weights trained for real-world video. |
| EMA-VFI | Apache-2.0 | Strong benchmark quality, but a hybrid CNN/transformer that is substantially heavier than RIFE per frame — too costly next to AlphaFace on one GPU in real time. |
| FILM (Google) | Apache-2.0 | Designed for large motion between photos; far too slow for real time. |
| GMFSS (Fortuna) | MIT | GMFlow-based, tuned for anime, heavy. |
| AMT | CC BY-NC 4.0 | Non-commercial license — excluded. |
| XVFI | research/education only | License excludes; 4K-oriented. |
| ANVIL (arXiv:2603.26835, 2026) | — | Uses motion vectors from a compressed bitstream; our input is decoded, swapped frames, so the prior it relies on does not exist here. |
| Diffusion/flow-matching VFI (e.g. TemporalFlowDiffuser) | — | Multi-step sampling; not real time on a shared GPU. |

Only the RIFE variants were benchmarked on this machine; the other rows are
based on each project's license and published design, not on local runs.

Isolated RIFE cost on the RTX 4090 (fp16, pinned-memory upload/download
included, per generated frame):

| Variant | 1280×720 | 1920×1080 |
|---|---|---|
| 4.25 | 5.5 ms | 12.7 ms (flow at full res) · 11.3 ms (half-res flow, default ≥1080p) |
| 4.25-lite | 4.8 ms | 11.3 ms |
| 4.26 | 5.5 ms | 11.1 ms |

Peak extra VRAM: ~0.3 GB at 720p, ~0.5–1.0 GB at 1080p.

## How it works

* **Backend abstraction** – `deepfake.framegen.FrameGenerationBackend`
  (`capabilities`, `initialize`, `push`, `interpolate`, `benchmark`,
  `shutdown`). RIFE is registered as `rife`; new backends register in
  `framegen/registry.py`.
* **Once per source frame** – upload through a pinned buffer, run the RIFE
  feature head on its own CUDA stream and cache it; each generated frame is
  one IFNet pass plus an async download.
* **Timeline** – output slots sit on a constant-rate grid
  `T_j = t0 + j / output_fps` in the camera's clock. When swapped keyframe `k`
  (captured at `t_k`) arrives, each slot in `(t_{k-1}, t_k]` becomes either the
  real keyframe (if within half a slot of `t_k`) or a RIFE frame at
  `s = (T_j − t_{k−1}) / (t_k − t_{k−1})`. Irregular swap timing therefore
  produces correctly timed in-betweens, and timestamps are strictly
  increasing.
* **Pacer** – presents slot `j` at `T_j + delay`. The delay adapts: it rises
  to exactly what a late frame needed (idempotent, capped by the preset's
  latency budget) and decays when frames arrive with steady slack. Stalls
  beyond the cap are treated as outages and dropped, not absorbed as latency.
* **Constant-rate sinks** – if a slot's frame is late, the previous frame is
  re-sent so files, FFmpeg and v4l2loopback keep valid CFR timestamps. Those
  frames are counted as **held**, never as generated, and the `output fps`
  statistic counts **new** frames only.
* **Degradation** – when more than 15 % of slots arrive late in a second, the
  worker sheds work: level 1 keeps only the in-between closest to each
  interval's midpoint, level 2 generates nothing (keyframes only). It steps
  back after 3 s of headroom. The swap stage keeps camera rate throughout.
* **Bounded buffers everywhere** – camera slot holds 1 frame, swap→frame-gen
  queue 2 keyframes, pacer queue ≈ output_fps × latency budget.

Other pipeline changes that came with this work (all measured):

* The ResNet-50 identity encoder ran on every frame; the identity code is now
  computed once per source (−1.3 ms/frame).
* `--swap-precision bf16` (default for latency/balanced): AlphaFace
  26.5 → 15.9 ms. fp16 was rejected: mean absolute error 0.11–0.13 vs fp32
  (bf16: 0.016).
* Haar detection runs on a ≤480 px copy (33 → 15 ms per call at 720p) and,
  in live mode, on its own thread against the newest frame.
* Temporal face smoothing now fades out with head motion; the old fixed blend
  produced a double face whenever the head moved.

## Measured results (RTX 4090, 2026-09-22)

`deepfake benchmark` — synthetic 30 fps camera of a bundled *fictional* face,
real AlphaFace + RIFE 4.25, 10 s per row, torch 2.6.0+cu124, driver 595.99.
Raw data: [`benchmarks/rtx4090-2026-09-22.json`](benchmarks/rtx4090-2026-09-22.json).

| Res | Mode | Swapped fps | Output fps (new frames) | Held | Swap latency | Capture→sink latency | Peak VRAM (torch) |
|---|---|---|---|---|---|---|---|
| 720p | off | 30.0 | 30.0 | – | 25.6 ms | 28.2 ms | 1.39 GB |
| 720p | 2x | 30.2 | 59.5 | 1.7 % | 26.1 ms | 82.3 ms | 1.68 GB |
| 720p | 3x | 30.0 | 89.7 | 1.2 % | 27.3 ms | 87.0 ms | 1.70 GB |
| 1080p | off | 30.0 | 30.0 | – | 27.8 ms | 30.9 ms | 1.39 GB |
| 1080p | 2x | 27.9 | 59.4 | 2.3 % | 34.1 ms | 128.3 ms | 1.87 GB |
| 1080p | 3x | 27.6 | 55.7 | 39 % | 35.9 ms | 207.5 ms | 1.87 GB — overloaded, shedding engaged |

Recommendation printed by the tool: **2x** at 720p (+54 ms) and 1080p
(+97 ms). 3x is fine at 720p, not at 1080p on this GPU.

Real USB webcam (1280×720 @ 30): 30.0 fps in, swap 30.0 fps at
22 ms, **60.6 fps out**, RIFE 4.9 ms/frame, steady-state capture→sink latency
≈ 51 ms.

The added latency is inherent to interpolation: an in-between cannot be shown
before the *next* swapped frame exists, so frame generation costs at least one
source interval (33 ms at 30 fps) plus inference. That is why it stays **off
by default**; use `deepfake benchmark` or `--frame-gen auto` to decide.

## Output verification

* Offline: 90 frames @ 30 fps → `--frame-gen 2x` → 179 frames @ 60/1,
  strictly increasing PTS (ffprobe).
* Live → FFmpeg `h264_nvenc` @ 60: 10 s run → 601 frames, `r_frame_rate` and
  `avg_frame_rate` 60/1, duration 10.016 s, no gaps.
* v4l2loopback could not be exercised here: the module is not built into this
  kernel configuration (see README).

## Attribution

RIFE: Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou,
"Real-Time Intermediate Flow Estimation for Video Frame Interpolation",
ECCV 2022, arXiv:2011.06294 — https://github.com/hzwer/Practical-RIFE (MIT).
The IFNet definition and weight mirror come via
https://github.com/HolyWu/vs-rife (MIT). See `NOTICE`.
