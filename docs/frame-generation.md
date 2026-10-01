# Frame generation

Deepfake synthesises intermediate frames **after** AlphaFace swap so the
presented stream can run at 60 / 90 / 120 FPS even when the swap path is slower.

## Architecture

```
Camera / decode
    ↓
Face detect → AlphaFace swap → composite
    ↓
Frame generation
  ├─ Maxine VFG   (SDK detection only; bindings unavailable)
  ├─ NVIDIA Optical Flow (NVOFA)   ← default on RTX Ada
  └─ passthrough
    ↓
Frame pacer / scheduler
    ↓
Preview · v4l2loopback · NVENC / file
```

## Default backend: NVIDIA Optical Flow

On this workstation the driver exposes `libnvidia-opticalflow.so.1`. Deepfake
binds it via ctypes (headers vendored from
[NVIDIAOpticalFlowSDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK), BSD-3),
estimates forward/backward flow on the Optical Flow Accelerator, then warps and
blends on CUDA with PyTorch.

RIFE (Practical-RIFE) has been **removed**. Old configs with
`frame_gen_backend=rife` are migrated to `auto` → NvOF.

## CLI

```bash
deepfake webcam                         # FrameGen auto (NvOF)
deepfake webcam --target-fps 120
deepfake webcam --framegen-mode latency
deepfake webcam --frame-gen off
deepfake webcam --frame-gen-backend nvof
deepfake doctor                         # shows Optical Flow / Maxine status
deepfake benchmark
```

## Modes

| Mode | NvOF perf level | Latency budget |
|------|-----------------|----------------|
| latency | FAST | ~80 ms |
| balanced | MEDIUM | ~160 ms |
| quality | SLOW | ~250 ms |

## Scene cuts

Large luma MAD between keyframes disables interpolation for that interval
(emit the newest real frame) and turns off temporal hints for the next OF call.

## Maxine VFG

When the Maxine VFX SDK + `nvvfxvideoframegeneration` feature are installed
(NGC), set `DEEPFAKE_MAXINE_ROOT` and `--frame-gen-backend maxine`. The Python
path currently detects the SDK but cannot execute it. Explicit selection reports
unavailable; auto selects NvOF. No Maxine performance result is claimed.


## Face sharpness

NvOF FrameGen uses 1×1 optical-flow vectors on Ada GPUs, forward–backward
occlusion checks, and nearer-keyframe bias so interpolated faces are not
soft-averaged. Prefer `--framegen-mode balanced` (default) or `quality` for
webcam/virtualcam; `latency` trades a little detail for lower gen time.

## Visibility-aware interpolation (0.5.0)

Swapped pixels, target pixels, visibility and paste region now share the same
warp. Hidden foreground is restored after sharpening. Flow signs and backward
consistency coordinates are covered by CUDA regressions. The final file frame
interval is held explicitly, so a 60-frame / 30 FPS input produces 120 frames at
60 FPS rather than losing the last interval. Original audio is muxed after video
EOF and bounded by actual written frames; `-shortest` cannot discard the last frame.

Earlier September figures below predate occlusion handling. Use the
[October benchmark](benchmarks/2026-10-01.md) for the current pipeline and explicit
held/late counts. Quality mode is offline and is not promised to render at playback
speed. These frames come from hardware motion interpolation, not a diffusion model.
