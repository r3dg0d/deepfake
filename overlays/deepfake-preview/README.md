# deepfake-preview

Purple-galaxy Quickshell status/control widget for Deepfake.

Launched automatically by the Deepfake session (`deepfake webcam` / `virtualcam` / `video`).
Reads structured state from `~/.local/state/deepfake/session.json` (and preview frames from `preview.png`).

- Live preview with resolution / uptime / frame-gen badges, 8 stat tiles, 60 s FPS and latency graphs
- **Start webcam** / **Stop webcam**, **Start / Stop virtual cam** (restarts the session with
  `--output-device <v4l2loopback>`), virtual-cam status pill, snapshot to `~/Pictures/deepfake`
- Stop goes through the session-scoped `control.json` IPC (`quickshell/control.py`);
  start/restart go through `quickshell/run.sh` (log: `~/.cache/deepfake-widget.log`)
- Fonts: Cellforge (labels), Zerodot (digits), JetBrainsMono Nerd Font (glyphs)

```bash
deepfake webcam -f face.jpg     # widget starts with the session
deepfake webcam --no-widget     # CLI only
DEEPFAKE_QS_PREVIEW=0 deepfake webcam   # also disables widget
quickshell/run.sh start virtual # start a session that feeds the virtual cam
```

Install as a desktop app (optional):

```bash
cp -r quickshell ~/.config/quickshell/deepfake
install -Dm644 desktop/galaxy-deepfake.svg ~/.local/share/icons/hicolor/scalable/apps/galaxy-deepfake.svg
install -Dm644 desktop/galaxy-deepfake.desktop ~/.local/share/applications/galaxy-deepfake.desktop
```

The virtual cam needs v4l2loopback, e.g. on NixOS:
`boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];`,
`boot.kernelModules = [ "v4l2loopback" ];` and
`options v4l2loopback devices=1 video_nr=10 card_label="Galaxy Deepfake" exclusive_caps=1`.

Manual launch (debug):

```bash
deepfake-preview
# or: qs -p overlays/deepfake-preview/quickshell
```
