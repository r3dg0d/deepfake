"""Manifest-driven experimental installation, used by the existing model manager.

Files are pinned to immutable official revisions and checked against SHA-256.
Downloads resume safely only when the server confirms the byte range.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import PurePosixPath

from .ipc import write_json_atomic
from .paths import models_dir


def manifests():
    path = files("deepfake") / "assets" / "experimental-models.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def validate_manifest(manifest):
    for key in ("name", "source", "revision", "code_license", "weights_license", "files", "precision"):
        if key not in manifest:
            raise ValueError(f"Manifest missing {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Source revision must be pinned")
    seen = set()
    for f in manifest["files"]:
        p = PurePosixPath(f["path"])
        if p.is_absolute() or ".." in p.parts or str(p) in seen:
            raise ValueError("Unsafe or duplicate model path")
        seen.add(str(p))
        if not f["url"].startswith("https://huggingface.co/") or not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
            raise ValueError("Official HTTPS URL and SHA-256 required")
        if f["size"] < 0:
            raise ValueError("Invalid download size")


def download_file(spec, target, progress=None):
    from .models import file_sha256

    if target.is_file() and target.stat().st_size == spec["size"] and file_sha256(target) == spec["sha256"]:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > spec["size"]:
        part.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(spec["url"], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Server returned an unexpected download range")
            done = offset if resumed else 0
            with part.open("ab" if resumed else "wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(spec["path"], done, spec["size"])
    except urllib.error.HTTPError as e:
        if e.code != 416 or offset != spec["size"]:
            raise RuntimeError(f"Download failed for {spec['path']}: HTTP {e.code}") from e
    if part.stat().st_size != spec["size"] or file_sha256(part) != spec["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Size or SHA-256 mismatch: {spec['path']}")
    os.chmod(part, 0o600)
    part.replace(target)


def install(name, *, yes=False, progress=None):
    from .models import _mark_installed

    manifest = manifests()[name]
    validate_manifest(manifest)
    size = sum(f["size"] for f in manifest["files"])
    if not yes:
        raise RuntimeError(f"{name}: {size / 2**30:.2f} GiB download. Code: {manifest['code_license']}; "
                           f"weights: {manifest['weights_license']}. Review models info {name}, then install --yes.")
    if manifest.get("blocked"):
        raise RuntimeError(manifest["blocked"])
    root = models_dir() / name
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < size:
        raise RuntimeError(f"Need {size / 2**30:.1f} GiB free for installation")
    for spec in manifest["files"]:
        download_file(spec, root / spec["path"], progress)
    write_json_atomic(root / "manifest.json", manifest)
    _mark_installed(name, {"manifest_revision": manifest["revision"]})
    return f"{name}: all {len(manifest['files'])} files verified. Configure an isolated worker before inference."


def verify(name, progress=None):
    from .models import file_sha256

    manifest = manifests()[name]
    validate_manifest(manifest)
    missing, corrupt = [], []
    total = sum(f["size"] for f in manifest["files"])
    done = 0
    for spec in manifest["files"]:
        if progress:
            progress(spec["path"], done, total)
        path = models_dir() / name / spec["path"]
        if not path.is_file():
            missing.append(spec["path"])
        elif path.stat().st_size != spec["size"] or file_sha256(path) != spec["sha256"]:
            corrupt.append(spec["path"])
        done += spec["size"]
    return {"name": name, "ok": not missing and not corrupt and not manifest.get("blocked"),
            "missing": missing, "corrupt": corrupt, "blocked": manifest.get("blocked")}
