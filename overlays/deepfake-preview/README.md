# deepfake-preview

Matrix-themed Quickshell status/control widget for Deepfake.

Launched automatically by the Deepfake session (`deepfake webcam` / `virtualcam` / `video`).
Reads structured state from `~/.local/state/deepfake/session.json` (and preview frames from `preview.png`).

```bash
deepfake webcam -f face.jpg     # widget starts with the session
deepfake webcam --no-widget     # CLI only
DEEPFAKE_QS_PREVIEW=0 deepfake webcam   # also disables widget
```

Manual launch (debug):

```bash
deepfake-preview
# or: qs -p overlays/deepfake-preview/quickshell
```
