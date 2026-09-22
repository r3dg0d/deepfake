## Unreleased

- Wire AlphaFace Swapper end-to-end (CUDA torch, real Drive weights path).
- Fix face paste geometry: crop and paste share the same square box.
- Oval soft mask + LAB color match; face-crop identity before ArcFace.
- Quickshell matrix overlay under overlays/deepfake-preview.

# Changelog

## 0.1.0 — 2026-09-22

- Initial Linux CLI: `webcam`, `video`, `virtualcam`, `devices`, `benchmark`, `models list|install`.
- Presets: `low-latency` | `balanced` | `high-quality`.
- Pipeline skeleton: OpenCV capture → Haar detect → AlphaFace/inswapper/passthrough → composite → watermark → preview/file/ffmpeg/gstreamer/v4l2loopback.
- Consent gate + synthetic-media watermark (default on).
- Model manager with explicit `--yes` ack; AlphaFace Drive weights not redistributed; InsightFace NC fallback.
- Optional fakeperson identity source path.
