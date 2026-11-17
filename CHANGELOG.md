# Changelog

## 0.5.0.dev0 — unreleased

Development version: everything below is covered by the CPU test suite, but none of it has run
on a GPU, with a camera or under a real Quickshell yet. See "Validation" at the end.

### Added
- **Real-time rotoscoping for the LivePortrait head engine.** The generated head is cut out of its
  square crop with a person matte (PP-HumanSeg, OpenCV Zoo, Apache-2.0, CPU via `cv2.dnn`) instead of
  being pasted together with the reference image's background. The matte is refined against the image
  (guided filter), sharpened and smoothed over time. Where the character is smaller than the real
  head, a clean plate learned from the camera hides what would stick out. On by default once the
  weights are installed: `deepfake models install pphumanseg --yes`; `--rotoscope/--no-rotoscope`.
- **Background replacement** for the same engine: `--background blur|green|blue|black|white|#rrggbb|IMAGE`,
  with `--matte-scope person` (keep the real body below the chin, default) or `head` (only the
  generated head). Frames without a detected face still get the background treatment.
- `deepfake models repair NAME --yes`: re-hash every file, download only the missing or damaged ones.
- `deepfake experimental enable|disable --backend NAME [--memory-limit-gb N]`.
- Matte time, rotoscope state and background are part of the live session state.

### Fixed
- **PersonaLive can be downloaded again.** It was blocked because `lambdalabs/sd-image-variations-diffusers`
  is now an empty repository: the Lambda organisation was renamed and the old name re-registered by
  someone else. The manifest pins the original repository, `lambda/sd-image-variations-diffusers@42bc0ee`,
  and `stabilityai/sd-vae-ft-mse@31f26fd` (13 files, 14.47 GiB, each with a SHA-256).
- **Reinstall and repair work for experimental models** (PersonaLive, EditaLive, LivePortrait), from the
  CLI and the control center. They were limited to the core models. A forced reinstall replaces each
  file only after its download passed the checksum.
- Experimental downloads resume automatically after a dropped connection (5 attempts) instead of
  failing the whole job, and the disk-space check counts only what still has to be downloaded.
- **EditaLive could not be loaded from the control center.** Four causes: every render re-hashed the
  whole 52.7 GiB bundle first (a full check now leaves a stamp, and unchanged files are trusted); the
  session looked dead while that hash ran; failures were only written to a log nobody was shown; and
  the checkout, worker Python and the disabled flag could only be changed in a terminal. All four are
  addressed, see "Control center".
- **Host memory guard for isolated workers.** The EditaLive worker previously exhausted host RAM and
  froze the desktop. Workers now run under a ceiling (total RAM minus a desktop reserve, or
  `--memory-limit-gb`): a systemd user scope with `MemoryMax` and no swap where available, and always
  a watchdog that stops the worker when it exceeds the ceiling or the host runs low. A backend that
  was disabled after such an incident stays disabled until it is explicitly enabled.
- A model job whose process died (crash, reboot) was shown as running forever, which disabled every
  model button. It is now reported as interrupted and can be resumed.
- The LivePortrait engine constructed the Haar detector directly and could not load on OpenCV 5,
  which ships no Haar cascades. It uses the shared detector factory (YuNet first).
- PP-HumanSeg is read with OpenCV's classic DNN engine: the new graph engine of OpenCV 5.0 returns a
  matte shifted by about 8 % of the image for this network. The model's load check tests for it.
- Preferences written by another version (unknown keys, removed values) no longer block Start.
- Re-running `experimental configure` no longer clears a backend's disabled flag.
- The snapshot polled by the control center no longer carries EditaLive's 412 per-file rows.

### Control center (UI overhaul)
- Six pages instead of eight: **Studio, Characters, Models, Render, Performance, Settings**. Camera
  settings moved to Settings, engine selection into the Studio.
- **Studio**: preview on the left, an inspector on the right with one engine choice (Face swap, Head,
  Portrait, Full body) that shows each engine's state, the selected character, the matte and
  background controls, quality, output and GPU.
- **Readiness checklist** in place of a silently disabled Start button: each missing piece (consent,
  character, weights, worker setup, disabled backend, matte model) is listed with the button that fixes it.
- **Models**: core and experimental groups; Install / Resume, Verify, Repair, Reinstall, Remove; overall
  download progress in bytes; worker setup (source checkout, worker Python) and "Enable with memory
  guard" for experimental engines.
- **Render**: input, output, encoder and prompt; live stage, worker RAM against its ceiling, and the
  worker log while an isolated render runs; the error of the last render stays visible.
- Settings save as you edit (no Save buttons). Requests are queued instead of dropped with "please
  wait". Results and errors appear as toasts; session errors are shown, not only logged.
- Lists no longer rebuild on every poll, so hover states and half-typed fields survive.
- Ctrl+1 … 6 switch pages. The old page names in `DEEPFAKE_UI_PAGE` still work.

### Validation
- About 130 CPU tests, including the matting compositor against the real PP-HumanSeg network and the
  installer against a local HTTP server (resume, retry, repair, reinstall).
- The control center was rendered page by page with plain Qt and stand-ins for the Quickshell types
  (`tests/quickshell_shim`), not under `qs`.
- **Not run:** LivePortrait with the matte on a GPU, a camera or virtual camera, the PersonaLive and
  EditaLive workers, the cgroup limit under a real systemd user manager, and the large downloads
  themselves (their digests come from Hugging Face's LFS metadata; the small files were downloaded
  and checked).

## 0.4.0.dev0 (development snapshot, never released)

### v0.4 development
- Shared transformation-engine interface and AlphaFace adapter preserve the original inference implementation and CLI defaults.
- Genuine LivePortrait head animation, pinned official weights, cached reference features and single-subject guards; measured on RTX 4090.
- Isolated PersonaLive/EditaLive offline workers with integrity checks, cancellation, shared output sinks and disclosure. PersonaLive remains blocked by an unavailable required base checkpoint; full-body release status is explicit in STATUS.md.
- Quickshell control center: real original/processed previews, engine controls, private character library, graphical model operations, recording/offline export, virtual-camera startup, measured performance history and themes.
- IPC v1, session ownership lock, scoped stop controls, atomic private state and persistent GUI lifetime.
- Fixed PNG preview encoding, offline source-rate/order preservation, startup resource cleanup, and live recording duration under dropped inference frames.
- Package QML/icon/launcher assets in wheels and Nix; derive the Nix version from pyproject.toml. Development version is 0.4.0.dev0 pending full release acceptance.

### Security
- The AlphaFace checkpoint is now loaded with `torch.load(..., weights_only=True)`, like the
  other checkpoints, so a tampered `.pt` cannot execute code when unpickled. Verified against
  the real `alphaface_demo.pt`.

### Fixed
- Ruff import ordering so CI lint passes.

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
