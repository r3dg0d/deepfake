"""Maintainer utility: generate pinned manifests from audited Hub API metadata.

Only small non-LFS configuration files are fetched to compute SHA-256. Never weights.
"""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
metadata["KlingTeam/LivePortrait"] = json.loads(Path(sys.argv[2]).read_text())


def specs(repo, selector, prefix="", strip=""):
    info = metadata[repo]
    rows = []
    for f in info["siblings"]:
        filename = f["rfilename"]
        if not selector(filename):
            continue
        rows.append({"path": prefix + filename.removeprefix(strip),
                     "url": f"https://huggingface.co/{repo}/resolve/{info['sha']}/{filename}",
                     "size": f["size"], "sha256": f.get("lfs", {}).get("sha256")})
    return rows


def hash_small(row):
    if row["sha256"] is None:
        if row["size"] > 2**20:
            raise ValueError(f"Refusing to fetch large non-LFS file {row['path']}")
        content = urllib.request.urlopen(row["url"], timeout=30).read()
        if len(content) != row["size"]:
            raise ValueError("Size mismatch")
        row["sha256"] = hashlib.sha256(content).hexdigest()
    return row


# The Lambda organisation was renamed from "lambdalabs"; the repository that now answers at
# lambdalabs/sd-image-variations-diffusers is an unrelated, empty re-registration. Pin the original.
SD_BASE = "lambda/sd-image-variations-diffusers"
SD_VAE = "stabilityai/sd-vae-ft-mse"
SD_BASE_FILES = ("model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.bin",
                 "image_encoder/config.json", "image_encoder/pytorch_model.bin")

common = {"experimental": True, "streaming": False, "offline": True, "minimum_vram_gb": None,
          "precision": ["fp16"], "notes": "Official source adapter; GPU integration must be validated locally."}
out = {
    "liveportrait": {**common, "name": "liveportrait", "source": "https://github.com/KlingAIResearch/LivePortrait",
                     "revision": "9b294b3d0536135442ea73cb01e6cb3ca7029dd3", "code_license": "MIT",
                     "weights_license": "MIT (official human animation weights; InsightFace weights are excluded)",
                     "model_type": "head", "streaming": True, "dependencies": ["torch", "scipy", "pyyaml"],
                     "files": specs("KlingTeam/LivePortrait", lambda p: p.startswith("liveportrait/") and p.endswith(".pth"))},
    "personalive": {**common, "name": "personalive", "source": "https://github.com/GVCLab/PersonaLive",
                    "revision": "abdd112e01dcf7d89122c2e5efa29fcff0669740", "code_license": "Apache-2.0; README academic-research-only notice",
                    "weights_license": "Apache-2.0 model-card metadata with an upstream research-use notice (PersonaLive); "
                                       "CreativeML OpenRAIL-M (lambda/sd-image-variations-diffusers base UNet and image encoder); "
                                       "MIT (stabilityai/sd-vae-ft-mse)",
                    "model_type": "head", "dependencies": ["isolated Python 3.10", "upstream requirements_base.txt"],
                    "notes": "Official source adapter; GPU integration must be validated locally. Base components come from "
                             "lambda/sd-image-variations-diffusers@42bc0ee (the verified Lambda organisation: 'lambdalabs' was renamed, "
                             "and the repository now at lambdalabs/sd-image-variations-diffusers is an unrelated empty re-registration) "
                             "and stabilityai/sd-vae-ft-mse@31f26fd.",
                    "components": [
                        {"repo": "huaichang/PersonaLive", "revision": metadata["huaichang/PersonaLive"]["sha"], "license": "Apache-2.0 (research-use notice)"},
                        {"repo": SD_BASE, "revision": metadata[SD_BASE]["sha"], "license": "CreativeML OpenRAIL-M"},
                        {"repo": SD_VAE, "revision": metadata[SD_VAE]["sha"], "license": "MIT"}],
                    "files": specs("huaichang/PersonaLive", lambda p: p.startswith("pretrained_weights/personalive/") and p.endswith(".pth"))
                    # upstream tools/download_weights.py names exactly these base files
                    + specs(SD_BASE, lambda p: p in SD_BASE_FILES, prefix="pretrained_weights/sd-image-variations-diffusers/")
                    + specs(SD_VAE, lambda p: p in ("config.json", "diffusion_pytorch_model.bin"), prefix="pretrained_weights/sd-vae-ft-mse/")},
    "editalive": {**common, "name": "editalive", "source": "https://github.com/GVCLab/EditaLive",
                  "revision": "9fe7fdb4ce3ff69305f026e8dec265aaf7a4196f", "code_license": "Apache-2.0; README academic-research-only notice",
                  "weights_license": "Apache-2.0 model cards (EditaLive and Wan-Animate); respect upstream research-use notice and component terms",
                  "model_type": "fullbody", "precision": ["bf16", "fp8"],
                  "dependencies": ["isolated Python 3.10", "torch 2.6.0", "upstream requirements.txt"],
                  "files": specs("huaichang/EditaLive", lambda p: p.startswith("weights/") and p.endswith(".safetensors"), prefix="EditaLive/", strip="weights/")
                  + specs("Wan-AI/Wan2.2-Animate-14B", lambda p: not p.startswith(("assets/", ".", "relighting_lora", "process_checkpoint/sam2/", "xlm-roberta-large/")) and p != "README.md", prefix="Wan-Animate/")},
}
with ThreadPoolExecutor(max_workers=8) as pool:
    for entry in out.values():
        entry["files"] = list(pool.map(hash_small, entry["files"]))
Path("src/deepfake/assets/experimental-models.json").write_text(json.dumps(out, indent=2) + "\n")
for name, entry in out.items():
    print(name, len(entry["files"]), sum(f["size"] for f in entry["files"]) / 2**30, "GiB")
