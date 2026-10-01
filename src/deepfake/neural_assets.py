"""Pinned, separately licensed ONNX assets; never downloaded during inference."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from .paths import models_dir

ASSETS = {
    "bisenet": {
        "file": "bisenet_resnet_18.onnx",
        "category": "FACE PARSING",
        "version": "models-3.1.0",
        "bytes": 53205356,
        "sha256": "2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9",
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/bisenet_resnet_18.onnx",
        "license": "MIT (yakhyo); training dataset terms apply",
        "upstream": "https://github.com/yakhyo/face-parsing",
    },
    "xseg": {
        "file": "xseg_2.onnx",
        "category": "OCCLUSION",
        "version": "models-3.1.0",
        "bytes": 70324286,
        "sha256": "cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042",
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_2.onnx",
        "license": "GPL-3.0 (DeepFaceLab model; not bundled)",
        "upstream": "https://docs.facefusion.io/introduction/licenses",
    },
    "yunet": {
        "file": "yunet.onnx",
        "category": "FACE DETECTION",
        "version": "2023mar",
        "bytes": 232589,
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "license": "MIT (OpenCV YuNet model)",
        "upstream": "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    },
}


ASSETS["trustmark-config"] = {
    "bytes": 2009,
    "sha256": "43f37103f92efa8bd6b1c5902bb537cc12a981dc699fca19d9bb7de8c62d03d9",
    "url": "https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_Q.yaml",
    "file": "trustmark_Q.yaml",
    "category": "PROVENANCE",
    "version": "TrustMark-Q-0.9.2",
    "license": "MIT (Adobe code AND models)",
    "upstream": "https://github.com/adobe/trustmark",
}
ASSETS["trustmark-encoder"] = {
    "bytes": 17302074,
    "sha256": "dc382c3f6b4fd568b27d6fbb763d6ffc2d2587126d84afe9d5ee95b4c5d99826",
    "url": "https://cai-watermark.adobe.net/watermarking/trustmark-models/encoder_Q.ckpt",
    "file": "encoder_Q.ckpt",
    "category": "PROVENANCE",
    "version": "TrustMark-Q-0.9.2",
    "license": "MIT (Adobe code AND models)",
    "upstream": "https://github.com/adobe/trustmark",
}
ASSETS["trustmark-decoder"] = {
    "bytes": 47652460,
    "sha256": "e3d9cea5406a26590735719f8f15cb10802b11852ae69047eaf4cf17214df781",
    "url": "https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.ckpt",
    "file": "decoder_Q.ckpt",
    "category": "PROVENANCE",
    "version": "TrustMark-Q-0.9.2",
    "license": "MIT (Adobe code AND models)",
    "upstream": "https://github.com/adobe/trustmark",
}


def asset_path(name: str) -> Path:
    root = Path(os.environ.get("DEEPFAKE_MASK_MODELS", str(models_dir() / "vision")))
    return root / ASSETS[name]["file"]


def verify_asset(name: str) -> bool:
    path = asset_path(name)
    if not path.is_file() or path.stat().st_size != ASSETS[name]["bytes"]:
        return False
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return digest == ASSETS[name]["sha256"]


def install_asset(name: str, *, yes: bool = False) -> str:
    meta = ASSETS[name]
    if not yes:
        raise RuntimeError(f"Review {meta['license']} and {meta['bytes'] / 1e6:.1f} MB download; use --yes")
    dest = asset_path(name)
    if verify_asset(name):
        return f"{name}: verified {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".download-", dir=dest.parent)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(meta["url"], timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > meta["bytes"]:
                    raise RuntimeError("model exceeds pinned download size")
                digest.update(chunk)
                out.write(chunk)
        if size != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
            raise RuntimeError("model size / SHA-256 mismatch; installed model was not changed")
        os.replace(tmp, dest)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return f"{name}: installed and SHA-256 verified {dest}"
