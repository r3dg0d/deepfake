# Models and dependency isolation

| Backend | Installation | Local validation | Environment |
|---|---|---|---|
| AlphaFace | Existing model manager, explicit license acknowledgement | Baseline GPU pipeline measured | Existing host torch/CUDA |
| LivePortrait | Five official human-animation weights, 0.486 GiB, pinned SHA-256 | Neural head generation measured | Host torch, scipy, PyYAML; InsightFace not used |
| PersonaLive | Disabled: required official base checkpoint missing | Not runnable | Separate Python 3.10; upstream requirements |
| EditaLive | 412 pinned files, 52.71 GiB including Wan-Animate + three compatible LoRAs | Offline command adapter unvalidated | Separate Python 3.10, Torch 2.6.0/CUDA 12.4; upstream dependencies |

```bash
deepfake models list --experimental
deepfake models info liveportrait
deepfake models install liveportrait --yes
deepfake models verify liveportrait
deepfake models remove liveportrait --yes
```

Downloads use immutable official revisions, byte sizes and SHA-256 checks;
partial files resume only when HTTP range responses match. Corrupt downloads
are not promoted to installed files. Experimental inference verifies the
bundle before model loading. Torch checkpoint loading remains restricted;
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
