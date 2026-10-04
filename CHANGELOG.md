# Changelog

## Unreleased

### Added
- SAM 2.1 Hiera-Tiny runs in the existing pipx deepfake env (torch 2.6.0+cu124, not upgraded). A CPU box prompt on bench_face.jpg keeps the face center and rejects a flat cheek patch. Hands are not labeled. Pytest still does not require sam2.

- SAM 2.1 Hiera-Tiny checkpoint is catalogued (official URL, 156008466 bytes, sha256). Doctor reports when the sam2 package is missing and the webcam stays on BiSeNet+XSeg. A box-prompt hole is applied only if it cannot eat the face. No hand mask was demonstrated.

- Optional cached XSeg matte (`xseg_2.onnx`) subtracts BiSeNet skin it does not accept as a face, on an interval. Missing weights keep today's BiSeNet-only mask. No SAM download.

- Finished video files get a C2PA manifest (face replacement, plus frame interpolation when frame generation emitted frames) via `c2patool`, then a verify step. `deepfake provenance inspect` reports only what verified. The on-screen watermark is unchanged. SynthID is not embedded.

- Visible-face masks are smoothed across frames: one empty parse keeps the last mask, a new hole closes with alpha 0.9, and the hole opens back with alpha 0.35.
- Live and offline swap now ask OcclusionEngine for a visible-face mask. BiSeNet runs through optional onnxruntime when `bisenet_resnet_18.onnx` is already cached and the graph is 19-class CelebAMask-HQ; otherwise the ellipse is unchanged and doctor/stderr say occlusion is inactive.
- Occlusion mask math: visible face = face region minus occluder, and `paste_face` can intersect the ellipse with that mask. No parser runs yet; without executing weights the engine keeps the ellipse and reports the parser unavailable. `webcam` / `virtualcam` / `video` gain no flags.

### Security
- The AlphaFace checkpoint is now loaded with `torch.load(..., weights_only=True)`, like the
  other checkpoints, so a tampered `.pt` cannot execute code when unpickled. Verified against
  the real `alphaface_demo.pt`.

### Fixed
- Ruff import ordering so CI lint passes.

## 0.4.0 — 2026-09-22

- **Replace RIFE** with NVIDIA Optical Flow (NVOFA) frame generation as the default backend.
- Vendored Optical Flow SDK headers; ctypes binding to `libnvidia-opticalflow.so.1`.
- Backend auto-select: Maxine (when installed) → NvOF → passthrough.
- Legacy `frame_gen_backend=rife` migrates to NvOF.
- Removed RIFE weights download / IFNet code / safetensors models from the install path.
- `--framegen-mode latency|balanced|quality`, `--target-fps`, `--frame-gen-backend nvof|maxine|passthrough`.

## 0.3.0 — 2026-09-22

- **FrameGen on by default** (`auto`) for `webcam`, `virtualcam`, and `video`.
- **Quickshell widget auto-starts** with every processing session (IPC via `~/.local/state/deepfake/session.json`); `--no-widget` to disable.
- Shared `DeepfakeSession` owns FrameGen policy, metrics IPC, widget lifecycle, and cleanup.
- Graceful FrameGen fallback (warn + continue with native AlphaFace).
- CLI simplified: `-f` face, `video -i/-o`, `doctor`, `config`; default video name `*-deepfake.mp4`.
- Video encode via ffmpeg (NVENC when available) with audio mux from the source.
- Fixed RIFE `4.25.lite` channel width so latency preset FrameGen loads correctly.
- Richer widget metrics: Source / AlphaFace / FrameGen / Output FPS, latency, VRAM, progress.

## 0.2.0 — 2026-09-22

### Added
- AI frame generation (`--frame-gen [2x|3x|4x|auto]`, `--output-fps`, `--no-frame-gen`,
  `--frame-gen-backend`, `--frame-gen-model`) using RIFE / Practical-RIFE 4.25, 4.25-lite, 4.26
  behind a swappable `FrameGenerationBackend` interface.
- Timeline-based pacing: output slots on a constant-rate grid, interpolation at each slot's exact
  time, adaptive bounded delay, held/late accounting, overload shedding.
- Threaded live pipeline (capture / swap / frame-gen / pacer) with per-second stats: input, swap and
  output fps (new frames only), latencies, generated/held/late, queue depth, GPU util, VRAM.
- `deepfake benchmark` rewritten: real AlphaFace + RIFE runs at 720p/1080p, with and without frame
  generation, `--json`, and a measured recommendation.
- `deepfake models install rife --yes` (SHA-256 pinned, converted to safetensors).
- Presets `latency` / `quality` aliases; `--swap-precision auto|fp32|bf16`.

### Changed
- AlphaFace identity code computed once per source instead of every frame.
- Haar detection on a ≤480 px copy; asynchronous detection in live mode.
- Temporal face smoothing fades out with head motion (fixes double-face ghosting).
- Watermark label is ASCII (Hershey fonts rendered the em dash as `???`).
- Upstream AlphaFace prints go to stderr so `--json` output stays clean.

## Unreleased (pre-0.2.0 notes)

- Wire AlphaFace Swapper end-to-end (CUDA torch, real Drive weights path).
- Fix face paste geometry: crop and paste share the same square box.
- Oval soft mask + LAB color match; face-crop identity before ArcFace.
- Quickshell matrix overlay under overlays/deepfake-preview.

# Changelog

## 0.2.1 — 2026-09-22

### Changed
- No flags needed: the consent notice is shown once and remembered; `--consent-ack` is hidden
  (kept for scripts, as is `DEEPFAKE_CONSENT_ACK=1`).
- `--source` is optional after first use (last face remembered); interactive prompt otherwise.
- `deepfake help [command]`.
- `virtualcam` auto-detects the v4l2loopback device, checks it is writable, and explains the
  NixOS/other-distro setup when missing.

### Fixed
- Virtual camera advertised 30 fps while receiving 60 fps: the sink now calls
  `v4l2loopback-ctl set-fps`, so consumers read 60/1 with monotonic timestamps; output is yuv420p.
- Loopback devices were listed as `capture` in `deepfake devices`.

## 0.1.0 — 2026-09-22

- Initial Linux CLI: `webcam`, `video`, `virtualcam`, `devices`, `benchmark`, `models list|install`.
- Presets: `low-latency` | `balanced` | `high-quality`.
- Pipeline skeleton: OpenCV capture → Haar detect → AlphaFace/inswapper/passthrough → composite → watermark → preview/file/ffmpeg/gstreamer/v4l2loopback.
- Consent gate + synthetic-media watermark (default on).
- Model manager with explicit `--yes` ack; AlphaFace Drive weights not redistributed; InsightFace NC fallback.
- Optional fakeperson identity source path.
