# Experimental mode (development snapshot)

AlphaFace stays the default; existing `--backend` selects its legacy swapper.
Experimental `--model-backend` selects an independent character engine.

```bash
deepfake webcam -f self.png                       # unchanged AlphaFace default
deepfake webcam --engine alphaface -f self.png
deepfake models info liveportrait
deepfake models install liveportrait --yes        # 0.486 GiB, explicit download
pip install -e '.[experimental-head]'            # CUDA torch installed separately
git clone https://github.com/KlingAIResearch/LivePortrait
git -C LivePortrait checkout 9b294b3d0536135442ea73cb01e6cb3ca7029dd3
deepfake experimental configure --backend liveportrait --checkout "$PWD/LivePortrait"
deepfake models verify liveportrait
deepfake experimental doctor
deepfake webcam --engine experimental --mode head --reference avatar.png --frame-gen off
deepfake virtualcam --engine experimental --mode head --reference avatar.png
deepfake video -i input.mp4 -o head.mp4 --engine experimental --mode head --reference avatar.png
```

LivePortrait runs in the host Python environment. It animates a reference head
crop with relative motion and expressions and composites it onto the driving scene.
With the matte model installed (`deepfake models install pphumanseg --yes`) the head
is rotoscoped into the frame and the background can be replaced; see
[rotoscoping](rotoscoping.md). Without it the complete crop is blended as a soft
rectangle, so hair and background inside the crop overwrite nearby content. No neck
reconstruction, eye gaze guarantee, lighting match or hand-occlusion handling is
provided. Keep one frontal, well-lit subject visible. Multiple faces or large
tracking jumps stop processing rather than silently selecting a different subject.

The existing RIFE stage is available independently; use `--frame-gen off` to
evaluate native inference. More output FPS does not mean faster neural inference.

Full-body setup, once the worker environment and RAM are adequate:

```bash
deepfake models info editalive                  # review 52.71 GiB + licenses first
deepfake models install editalive --yes
git clone https://github.com/GVCLab/EditaLive
git -C EditaLive checkout 9fe7fdb4ce3ff69305f026e8dec265aaf7a4196f
# In a separate Python 3.10 environment, follow the pinned upstream requirements.
deepfake experimental configure --backend editalive --checkout "$PWD/EditaLive" --python /path/to/worker/bin/python
deepfake models verify editalive
deepfake video -i input.mp4 -o body.mp4 --engine experimental --mode fullbody \
  --model-backend editalive --reference character.png --frame-gen off \
  --prompt 'Match the reference character while preserving motion and background.'
```

This full-body adapter has not produced validated output in this development
run. Its fixed conservative worker profile requests 384×672, FP8, CPU T5,
weight/KV offloading and no compile/fast decoder. Upstream may resample or pad
the timeline; complete source-rate preservation is guaranteed only by the
shared per-frame engines, not the research body worker. Inspect worker output
and audio synchronization before using a render.

PersonaLive (portrait, offline) installs with `deepfake models install personalive --yes`
(14.47 GiB) and needs its own Python 3.10 environment with upstream's
`requirements_base.txt`, configured like EditaLive above. Its worker has not been run here.

Isolated workers run under a host memory guard. The ceiling defaults to total RAM
minus a desktop reserve (15 %, at least 6 GiB); the worker is stopped above it or when
the host runs low. Where the systemd user manager can create a scope, `MemoryMax` with
swap disabled enforces the same ceiling in the kernel.

```bash
deepfake experimental doctor                                   # status, paths, memory ceiling
deepfake experimental enable --backend editalive               # clear a disabled flag
deepfake experimental enable --backend editalive --memory-limit-gb 40
deepfake experimental disable --backend editalive --reason "not enough RAM"
```

A stopped worker ends the render with a message; raise the ceiling only if the machine
really has the memory. PixReenact, DirectSwap and hybrid mode are not registered as
working backends. No remote processing exists.

GUI: `deepfake app` opens Quickshell. Select/import a character, choose an engine,
save settings and press Start. Recording can start a new camera session or render
an input video to a new filename. See [Quickshell instructions](quickshell-app.md).
