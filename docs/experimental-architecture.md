# Shared engine architecture

`engines/base.py` defines capabilities, frame context and a lifecycle interface.
`AlphaFaceEngine` delegates to the original `FaceSwapPipeline`, including its
identity cache, asynchronous detection, compositing and precision settings.
`LivePortraitEngine` loads official model classes under a private package alias,
prepares the reference once and reuses feature/keypoint tensors. It shares
`realtime.py`, FrameGen, disclosure and every existing output sink.

The capture slot retains only the latest live frame; inference→RIFE has a bounded
two-keyframe queue; the pacer bounds presentation delay. Offline per-frame
processing retains source order, dimensions and rate. Startup and inference
failures release engines, sinks, captures and FrameGen using an ExitStack.

PersonaLive/EditaLive run as isolated offline subprocesses. Their dependencies
never enter the base environment. The parent verifies weights and source
revision, disables implicit Hub networking, polls cancellation, and exports
generated video through the existing sinks with disclosure. A research worker
cannot open a camera through this adapter. Temporary unwatermarked intermediates
stay in the temporary render directory and are deleted at the end.

`DeepfakeSession` remains the session owner. SessionBus holds a nonblocking
filesystem lock before publishing state; a second CLI or GUI launch fails
before opening capture. IPC v1 adds session-scoped controls and private atomic
files while accepting existing legacy stop controls. Quickshell survives session
changes. No JSON message contains image bytes: bounded-size original and
processed preview PNGs are updated atomically, at most 20 Hz.

`desktop.py` validates structured requests, persists GUI preferences separately
from the existing TOML settings, invokes the same CLI using argv arrays and
provides character/model operations through the existing model manager. All
storage is local by default; no listener, remote worker or telemetry is added.

The design does not yet implement separate preprocessing/compositing workers,
stable multi-person IDs, hardware adaptive quality, optical-flow stabilization,
pose/segmentation overlays or hybrid output. Capability reporting and GUI live
controls must not imply those features exist.
