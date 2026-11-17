# Quickshell control center

Run `deepfake app`, or use the **Deepfake** app-launcher entry after:

```bash
deepfake desktop install --executable "$(command -v deepfake)"
```

The installer keeps the `galaxy-deepfake.desktop` identity, makes a backup under the private state
directory and copies every QML file of the shell to `~/.config/quickshell/deepfake`. Run it again
after upgrading: 0.5 added files. Quickshell must be installed separately (`qs` on PATH). Opening
the window never opens a camera.

## Pages

**Studio** (Ctrl+1). The preview on the left; on the right, everything that shapes the next session:

- *Engine*: Face swap (AlphaFace, live), Head (LivePortrait, live, rotoscoped), Portrait (PersonaLive,
  renders files), Full body (EditaLive, renders files). Each row shows whether that engine is ready
  and, if not, why.
- *Character*: the selected reference; Change opens Characters.
- *Matte* (Head engine): rotoscope Auto / On / Off, background Keep / Blur / colours / custom colour /
  image, and whether your body below the chin is kept. See [rotoscoping](rotoscoping.md).
- *Quality*: preset and frame generation. *Output*: Start, or Start with the virtual camera.
- *Before you start*: what is still missing (consent, character, weights, worker setup, a disabled
  backend, the matte model), each with the button that fixes it. Start stays disabled until the
  blocking items are done, and its tooltip names the first one.

**Characters** (Ctrl+2). Import a reference image with a name and its provenance; pick the one in use.
Images stay in the private local library.

**Models** (Ctrl+3). Core models and experimental engines. Install (or Resume a partial download),
Verify, Repair (re-check everything, download only what is missing or damaged), Reinstall (download
everything again; installed files stay until their replacements pass) and Remove. One operation runs
at a time, with overall progress and Cancel; partial downloads are kept. Experimental engines also
have a setup block: the source checkout at the pinned revision and, for PersonaLive and EditaLive,
the Python of their isolated environment. A backend that was disabled shows the reason and
**Enable with memory guard…**.

**Render** (Ctrl+4). Input video, a new output file, encoder, and for EditaLive an appearance prompt.
*Render* uses the engine selected in the Studio. While an isolated worker runs, the page shows its
stage (the worker's own last output line, not an invented percentage), its host RAM against the
ceiling, and its log. A failed render leaves its error on the page. *Record the webcam* starts a
live session that also writes the output file.

**Performance** (Ctrl+5). Camera, native inference and output rates, latency p50/p95, matte time,
frame-generation cost, GPU load and 60-second graphs. Anything the backend did not report shows `—`.

**Settings** (Ctrl+6). Consent notice, capture device and size, virtual camera device, shortcuts,
version and the backend log.

Settings are saved as you edit them. F11 shows the preview fullscreen, Esc leaves it, Ctrl+Q stops
the session and closes the window. Closing with × only hides the window; a session keeps running.

## Protocol

Protocol v1 uses validated JSON requests passed as argv, atomic private state files, session-scoped
stop commands and a process lock. Preview pixels never pass through JSON. Actions added in 0.5:
`configure_backend`, `enable_backend`, `disable_backend`, `repair_model`, and `reinstall_model` for
experimental models; `log` takes `which` = `backend`, `worker` or `model`. GUI preferences live in
`desktop.json`; keys from another version are ignored.

## Limits

No before/after slider, microphone capture, mid-session change of engine, matte or output, startup
at login, or automatic creation of the worker environments. The 0.5 interface was rendered and
checked page by page with plain Qt and stand-in Quickshell types; it has not yet been run under
`qs` on a desktop, with a camera, OBS or a browser.
