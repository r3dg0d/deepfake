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
crop with relative motion and expressions, then blends that complete crop onto
the driving scene. It is experimental: hair/background inside the rectangular
crop can overwrite nearby content. No segmentation, neck reconstruction, eye
gaze guarantee, lighting match or occlusion guarantee is provided. Keep one
frontal, well-lit subject visible. Multiple faces or large tracking jumps stop
processing rather than silently selecting a different subject.

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

PersonaLive's offline adapter is blocked by an unavailable required official
base checkpoint; installation gives that reason. PixReenact, DirectSwap and
hybrid mode are not registered as working backends. No remote processing exists.

GUI: `deepfake app` opens Quickshell. Select/import a character, choose an engine,
save settings and press Start. Recording can start a new camera session or render
an input video to a new filename. See [Quickshell instructions](quickshell-app.md).
