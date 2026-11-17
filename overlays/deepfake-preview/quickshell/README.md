# Galaxy Deepfake control center

Quickshell shell for Deepfake in the purple Galaxy theme. `deepfake app` opens it; see
`docs/quickshell-app.md` for the pages and `docs/galaxy-theme.md` for the design tokens.

- `shell.qml` → `ControlCenter.qml`: backend state, request queue, readiness rules, dialogs, toasts,
  window frame and sidebar.
- `StudioPage.qml`, `CharactersPage.qml`, `ModelsPage.qml`, `RenderPage.qml`, `PerformancePage.qml`,
  `SettingsPage.qml`: one file per page; each receives the shell as `app`.
- `Galaxy*.qml`, `Starfield.qml`: the design system shared with the aivoice widget.
- `Card`, `Segmented`, `ChoiceRow`, `CheckRow`, `KeyValue`, `PathField`, `LogView`, `PreviewPane`,
  `TextLabel`, `TextHint`: components of this app only.

Everything shown comes from `deepfake desktop request` (set `DEEPFAKE_CLI` to choose the
executable). No camera opens when the window does. The preview refreshes at 10 Hz, only while a
session is live and the Studio is visible.

Review without a desktop: `DEEPFAKE_UI_PAGE=Models DEEPFAKE_UI_SIZE=1240x840
DEEPFAKE_UI_SCREENSHOT=/tmp/models.png DEEPFAKE_UI_SCREENSHOT_QUIT=1 qs -p .` or, without
Quickshell, `python tests/quickshell_shim/run.py overlays/deepfake-preview/quickshell`.

`run.sh`, `control.py` and `noqs.sh` are the older launcher helpers and are kept for existing setups.
