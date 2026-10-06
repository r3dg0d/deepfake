# Changelog

## 0.5.1rc4

- Use area downsampling for face crops and cubic enlargement for pasted faces to reduce avoidable eye softness.
- Add bounded, noise-gated luminance detail on freshly parsed generated eyes; target eyes are never copied.
- Intersect eye regions with current visible alpha so foreground protection and mouth preservation remain active.


## 0.5.1rc3

- Smooth inward mask feathering removes the clipped-Gaussian opacity step without expanding foreground support.
- Preserve freshly parsed target lips and oral detail; refresh semantic parsing on every frame in mouth-aware adapters.
- Mask-normalized local lighting/chroma transfer excludes protected mouths and occluders.
- Add disabled, numerically tested partial research operators for September 2026 shadow harmonisation and RefGAP.
- Virtualcam remains free of the visible demo label by default.


## 0.5.1rc2 — 2026-10-05

- Virtualcam starts without the synthetic-demo corner label. `--visible-watermark` adds it explicitly.
- Close the live session/widget when sink initialization fails.
- Report failed/timed-out virtual-camera FFmpeg writers instead of silently treating shutdown as successful.
- Publish final live metrics before shutting down, including short headless virtualcam runs.


## 0.5.1rc1 — 2026-10-05

- Detect lower-confidence foreground profiles with YuNet (0.6 threshold); quality mode selects the largest face unless `--multi-face` is requested.
- Avoid accidental 3× FrameGen for 29.999/29.97 FPS input targeting 60 FPS.
- Refresh heuristic FrameGen choices instead of reusing stale source rates; preserve measured benchmark decisions.
- Preserve local target chroma within visible support to reduce cyan/magenta casts on strongly lit profiles.
- Add hidden local `compare` command for original/before/after clips and recomputed mask diagnostics.
- Preserve and package the existing 0.5.0 mask-aware FrameGen, verified dual provenance and pending Nix CUDA-shell fixes. Nix runtime tools are wrapped into PATH.
- Add real talking/profile/hand clip validation, final codec stress results and documented coverage limits. This is a prerelease pending broader live-object validation.

## 0.5.0 — 2026-10-01

- Automatic visible-face compositing using pinned BiSeNet and XSeg, conservative foreground margins, per-track mask flow and fail-original handling.
- Stable YuNet/LK tracks; correct detector confidence column, offscreen paste geometry and per-track color statistics. Preserve strong profiles and full occlusions.
- Transport target pixels and visibility through the same NVOFA warp as generated frames; fix flow direction and matched-coordinate consistency. Count held frames separately.
- Warm up actual face inference before capture. Quality video keeps original geometry and timing, including the final frame interval and bounded audio mux.
- Real post-encode C2PA signing and verification; private development signer, explicit certificate trust, atomic publication and tamper checks.
- Optional pinned TrustMark Q disclosure, safe weight loading, final-encoded verification and documented codec-stress failures; visible disclosure for raw live outputs.
- Extended doctor, categorized model metadata, measured mask/track/backend metrics, debug masks, occlusion comparison and GPU regressions.
- Package Quickshell assets in wheels, detect widget startup failure, restore automatic virtualcam widget, correct loopback rate after FFmpeg format initialization.
- Research matrix, architecture, provenance and benchmark documentation; Python 3.12 Nix dependencies and package tests. No unvalidated diffusion cinematic mode or SynthID claim.

## 0.4.0 (previously local, unreleased) — 2026-09-22

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
