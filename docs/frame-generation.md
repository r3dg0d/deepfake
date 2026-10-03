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
  ├─ Maxine VFG   (preferred when SDK installed)
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
path currently detects the SDK and falls through to NvOF until the Maxine
bindings are completed.


## Face sharpness

NvOF FrameGen uses 1×1 optical-flow vectors on Ada GPUs, forward–backward
occlusion checks, and nearer-keyframe bias so interpolated faces are not
soft-averaged. Prefer `--framegen-mode balanced` (default) or `quality` for
webcam/virtualcam; `latency` trades a little detail for lower gen time.
