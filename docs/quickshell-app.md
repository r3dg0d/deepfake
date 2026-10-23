# Quickshell control center

Run `deepfake app`, or use the existing **Deepfake** app-launcher entry after:

```bash
deepfake desktop install --executable "$(command -v deepfake)"
```

The installer keeps the `galaxy-deepfake.desktop` identity, makes a backup under
the private state directory and updates the existing Quickshell config.
Nix packages ship QML, the same desktop entry and its icon. No terminal or second
frontend is opened. Quickshell must be installed separately (`qs` on PATH).

1. In Settings, acknowledge the existing consent/disclosure notice if needed.
2. In Characters, browse/import a local reference, enter a name and provenance,
   then select Use. Only supported image files are accepted; video references
   and separate head/body reference assignments are not implemented.
3. In Engines, choose AlphaFace or an installed/configured experimental backend.
   Unsupported live combinations disable Start; offline adapters are labeled.
4. In Camera, choose an index/device, size, FPS and optional virtual output path.
   Save preferences, then Start. The camera is never activated on application open.
5. Dashboard shows real processed/original preview images, side-by-side mode,
   mirroring, expanded preview and measured rates. Preview is capped at 20 Hz.
6. Models shows sizes, terms, download confirmation, progress/cancellation,
   verification and removal through the same Python model manager.
7. Recording selects a new file and encoder (auto, CPU H.264 or NVENC), then
   starts a recording session. Stop finalizes it. Use an input video to render
   offline with the selected engine. Live microphone audio is not captured;
   video input audio is muxed where supported by the existing sink.
8. Start virtual camera routes through existing v4l2loopback; select it in OBS
   or a browser. Stop a session before changing engines or outputs.

Performance shows camera/native/output FPS, median/P95 capture-to-sink latency
when available, GPU/VRAM, queue and drop/generated counters plus a native-FPS
history graph. Missing measurements show `—`; no mock metrics exist.

Dark, light and Matrix themes are available. Ctrl+S saves session defaults;
F11 expands preview; Esc returns; Ctrl+Q requests stop and closes the window.
Closing the application requests graceful stop/finalization. A session stopping
does not destroy the app. Use Backend log for startup/CUDA/device errors.

Protocol v1 uses validated JSON requests passed as argv, atomic private state
files, scoped stop commands and a process lock. Preview pixels never pass
through JSON. Existing CLI/TOML/legacy IPC remain supported. Preferences specific
to the GUI live in `desktop.json`; the original configuration is preserved.

Current limits: no before/after slider, interactive diagnostics overlays,
microphone capture, mid-session output toggling, CPU usage sample, startup-at-login
setting or automatic experimental environment provisioning. Real-device camera,
OBS and browser UX needs further testing; synthetic pipeline and Wayland window
loading are tested separately.
