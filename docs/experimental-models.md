# Models and dependency isolation

| Backend | Installation | Local validation | Environment |
|---|---|---|---|
| AlphaFace | Existing model manager, explicit license acknowledgement | Baseline GPU pipeline measured | Existing host torch/CUDA |
| LivePortrait | Five official human-animation weights, 0.486 GiB, pinned SHA-256 | Neural head generation measured | Host torch, scipy, PyYAML; InsightFace not used |
| PersonaLive | 13 pinned files, 14.47 GiB: six PersonaLive weights plus the base UNet/image encoder from `lambda/sd-image-variations-diffusers@42bc0ee` and the VAE from `stabilityai/sd-vae-ft-mse@31f26fd` | Worker not run | Separate Python 3.10; upstream `requirements_base.txt` |
| PP-HumanSeg | One ONNX file, 5.9 MB, OpenCV Zoo, pinned SHA-256 | Matte checked on the CPU against the real network | Host OpenCV (`cv2.dnn`) |
| EditaLive | 412 pinned files, 52.71 GiB including Wan-Animate + three compatible LoRAs | Offline command adapter unvalidated | Separate Python 3.10, Torch 2.6.0/CUDA 12.4; upstream dependencies |

```bash
deepfake models list --experimental
deepfake models info liveportrait
deepfake models install liveportrait --yes
deepfake models verify liveportrait
deepfake models repair personalive --yes      # re-hash, download only missing or damaged files
deepfake models reinstall personalive --yes   # download everything again
deepfake models remove liveportrait --yes
```

PersonaLive's base checkpoints: the Lambda organisation on Hugging Face was renamed from
`lambdalabs` to `lambda`. The name `lambdalabs/sd-image-variations-diffusers` now belongs to an
unrelated account and holds an empty repository created in October 2026, which is why 0.4 could
not find the files. The manifest pins the original repository (verified organisation, history from
2022) at an immutable revision. Its weights are under CreativeML OpenRAIL-M, the VAE under MIT.

Downloads use immutable official revisions, byte sizes and SHA-256 checks;
partial files resume only when HTTP range responses match. Corrupt downloads
are not promoted to installed files, and a dropped connection is resumed
automatically. Experimental inference verifies the bundle before model loading:
files whose size and modification time are unchanged since their last SHA-256
check are trusted, anything else is hashed again. `models verify` always re-hashes
everything. Torch checkpoint loading remains restricted;
research workers also receive `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`.

`models verify` for legacy models requires recorded integrity hashes. AlphaFace
hashes are locally recorded, not signatures establishing upstream authenticity.
Older RIFE/InsightFace installs without such a record report verification as
unavailable; they are not silently described as verified.

Code and weight licenses are separate. PersonaLive/EditaLive repository LICENSE
files and model cards say Apache-2.0 while their READMEs include an academic-use
notice. Resolve the intended use with the authors before commercial deployment.
No weights are distributed in this repository. Upstream source is configured
explicitly at the documented pinned revision; it is not silently cloned/imported.

`nix develop` remains CPU-friendly. Install a CUDA torch build appropriate to
your NVIDIA driver; the tested host used Torch 2.14.1+cu130. CUDA driver and
library availability must be checked with `deepfake doctor`. LivePortrait's
optional `experimental-head` extra does not force a torch/CUDA version.
PersonaLive and EditaLive should not be installed into the host Python 3.13
environment. Docker is not required. Full-body offloading also consumes system
RAM; a 24 GiB GPU alone does not establish feasibility.
