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
