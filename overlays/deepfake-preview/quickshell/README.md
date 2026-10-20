# Galaxy Deepfake

Purple galaxy Quickshell webcam status panel. Reads Deepfake session.json and
preview.png under $XDG_STATE_HOME/deepfake (default ~/.local/state/deepfake).
Displays live preview, source, resolution, input, frame-generation status,
FPS, latency, GPU use and VRAM. Stats older than 15 seconds show offline.

Drag the header to move; use the compositor's resize gesture to resize the
floating window. X hides only the panel. Stop webcam sends a stop command
through Deepfake's existing control.json IPC. Preview updates at 10 FPS to
limit UI overhead; session output frame generation is unaffected.

The ~/.local/bin/deepfake wrapper sets DEEPFAKE_WIDGET_DIR so `deepfake webcam`
starts this widget automatically. --no-widget still works. Run
`~/.local/bin/deepfake-widget` to reopen a hidden panel. No camera opens when
launching the widget by itself. No unsupported runtime face/framegen toggles
are exposed. Deepfake currently only records those pending requests.
