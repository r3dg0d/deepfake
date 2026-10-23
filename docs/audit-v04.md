# v0.4 audit — 2026-10-20

Baseline: origin/main c67a425; local changes to detect.py, models.py,
widget.py, run.sh and untracked noqs.sh were present before this work.
They are preserved and excluded from feature commits where possible.

`nix develop --command python -m pytest -q`: 52 passed, 1 GPU test deselected.
STATUS.md was absent. pyproject version 0.3.0; flake package version stale (0.2.1).
GPU: NVIDIA RTX 4090, 24,564 MiB, approximately 2,542 MiB already used.

Existing architecture: FaceSwapPipeline owns face detection, identity encoding,
swap, compositing. realtime.py owns bounded capture, FrameGen and pacing.
DeepfakeSession owns FrameGen policy, JSON IPC and widget lifetime. Sinks cover
preview files, FFmpeg, GStreamer and V4L2. Model manager uses XDG cache paths.

Installed launcher: galaxy-deepfake.desktop → ~/.local/bin/deepfake-widget →
quickshell -c deepfake. Installed QML lives at ~/.config/quickshell/deepfake.
The CLI wrapper invokes the Nix system package plus a Python 3.13 CUDA environment.

Confirmed defects: cleanup_orphans kills launcher-owned windows unconditionally;
PreviewSink encodes preview.png.tmp with no recognized image extension and
falls back to undeclared Pillow; no-FG video path resizes/re-times to preset;
session construction overwrites live session state before checking ownership.
