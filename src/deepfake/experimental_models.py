"""Manifest-driven experimental installation, used by the existing model manager.

Files are pinned to immutable official revisions and checked against SHA-256.
Downloads resume safely only when the server confirms the byte range, and retry
on dropped connections. A full verification leaves a stamp (size + mtime per
file) so starting a render does not have to re-hash tens of gigabytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from functools import lru_cache
from importlib.resources import files
from pathlib import Path, PurePosixPath

from .ipc import read_json, write_json_atomic
from .paths import models_dir

STAMP = ".verified.json"
CHUNK = 4 * 1024 * 1024
RETRIES = 5


@lru_cache(maxsize=1)
def _bundled() -> str:
    path = files("deepfake") / "assets" / "experimental-models.json"
    return path.read_text() if path.is_file() else "{}"


def manifests():
    return json.loads(_bundled())


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


def total_size(manifest) -> int:
    return sum(f["size"] for f in manifest["files"])


def hash_file(path: Path, tick=None) -> str:
    """SHA-256 of ``path``; ``tick(bytes_so_far)`` is called per chunk so long hashes stay observable."""
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
            done += len(chunk)
            if tick:
                tick(done)
    return digest.hexdigest()


# --------------------------------------------------------------------------- verification stamp


def _manifest_digest(manifest) -> str:
    rows = sorted((f["path"], f["size"], f["sha256"]) for f in manifest["files"])
    return hashlib.sha256(json.dumps([manifest.get("revision"), rows]).encode()).hexdigest()


def _fingerprint(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _read_stamp(name: str, manifest) -> dict[str, list[int]]:
    data = read_json(models_dir() / name / STAMP)
    if data.get("manifest_sha256") != _manifest_digest(manifest) or not isinstance(data.get("files"), dict):
        return {}
    return data["files"]


def _write_stamp(name: str, manifest, verified: dict[str, list[int]]) -> None:
    root = models_dir() / name
    if root.is_dir():
        write_json_atomic(root / STAMP, {"manifest_sha256": _manifest_digest(manifest), "files": verified,
                                         "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def integrity(name: str) -> str:
    """``verified`` when every file still matches the stamp of its last SHA-256 check (no hashing here)."""
    manifest = manifests()[name]
    root = models_dir() / name
    stamp = _read_stamp(name, manifest)
    try:
        ok = bool(manifest["files"]) and all(stamp.get(f["path"]) == _fingerprint(root / f["path"])
                                             and stamp[f["path"]][0] == f["size"] for f in manifest["files"])
    except OSError:
        ok = False
    return "verified" if ok else "unverified"


# --------------------------------------------------------------------------- download


def _matches(path: Path, spec, tick=None) -> bool:
    return path.is_file() and path.stat().st_size == spec["size"] and hash_file(path, tick) == spec["sha256"]


def _fetch(spec, part: Path, progress) -> None:
    """One download attempt into ``part``, resuming it when the server honours the byte range."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > spec["size"]:
        part.unlink()
        offset = 0
    if offset == spec["size"]:
        return
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "deepfake-model-store/1", **headers})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - validated https://huggingface.co/
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Server returned an unexpected download range")
            done = offset if resumed else 0
            with part.open("ab" if resumed else "wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done)
    except urllib.error.HTTPError as e:
        if e.code == 416 and offset == spec["size"]:
            return
        if e.code in (401, 403, 404):  # not transient: retrying cannot help
            raise RuntimeError(f"Download failed for {spec['path']}: HTTP {e.code} from {spec['url']}") from e
        raise OSError(f"HTTP {e.code}") from e


def download_file(spec, target: Path, progress=None, *, force: bool = False, retries: int = RETRIES, sleep=time.sleep):
    """Fetch one pinned file. ``progress(path, bytes, size)``; returns True when bytes were downloaded.

    An existing file that already matches is kept unless ``force``. A forced download goes to
    ``<name>.part`` first, so the installed file stays usable until its replacement verified.
    """
    def tick(done):
        if progress:
            progress(spec["path"], done, spec["size"])

    if not force and _matches(target, spec, tick):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        if attempt:
            sleep(min(30.0, 2.0 ** attempt))
        try:
            _fetch(spec, part, tick)
        except (OSError, urllib.error.URLError, TimeoutError) as e:  # dropped connection: resume from .part
            last = e
            continue
        if part.stat().st_size == spec["size"]:
            last = None
            break
        last = OSError(f"incomplete response ({part.stat().st_size} of {spec['size']} bytes)")
    if last is not None:
        kept = part.stat().st_size if part.exists() else 0
        raise RuntimeError(f"Download interrupted for {spec['path']} after {retries} attempts ({last}). "
                           f"{kept / 2**20:.0f} MiB kept; run the install again to resume.")
    if hash_file(part, tick) != spec["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {spec['path']}: the download was discarded, the installed file is untouched")
    os.chmod(part, 0o600)
    part.replace(target)
    return True


# --------------------------------------------------------------------------- install / verify


def install(name, *, yes=False, progress=None, force=False, stage=None):
    """Install, repair or (``force``) fully redownload an experimental model.

    Without ``force`` every present file is re-hashed and only missing or corrupt files are fetched,
    which is what "repair" means. ``progress(path, bytes_overall, total_overall)``; ``stage(text)``.
    """
    from .models import _mark_installed

    manifest = manifests()[name]
    validate_manifest(manifest)
    size = total_size(manifest)
    if not yes:
        raise RuntimeError(f"{name}: {size / 2**30:.2f} GiB download. Code: {manifest['code_license']}; "
                           f"weights: {manifest['weights_license']}. Review models info {name}, then install --yes.")
    if manifest.get("blocked"):
        raise RuntimeError(manifest["blocked"])
    root = models_dir() / name
    root.mkdir(parents=True, exist_ok=True)
    specs = manifest["files"]

    def present(spec):
        p = root / spec["path"]
        return p.is_file() and p.stat().st_size == spec["size"]

    # A forced file is replaced one at a time, so only the largest one is needed on top of what is missing.
    need = sum(s["size"] for s in specs if not present(s))
    if force:
        need += max((s["size"] for s in specs if present(s)), default=0)
    need -= sum(p.stat().st_size for s in specs if (p := (root / (s["path"] + ".part"))).is_file())
    free = shutil.disk_usage(root).free
    if free < need:
        raise RuntimeError(f"{name}: need {need / 2**30:.1f} GiB free under {root}, {free / 2**30:.1f} GiB available")

    base = downloaded = 0
    verified: dict[str, list[int]] = {}
    for spec in specs:
        if stage:
            stage(("Downloading " if force or not present(spec) else "Checking ") + spec["path"])
        target = root / spec["path"]
        relay = (lambda path, done, _size, base=base: progress(path, base + done, size)) if progress else None
        downloaded += bool(download_file(spec, target, relay, force=force))
        verified[spec["path"]] = _fingerprint(target)
        base += spec["size"]
    write_json_atomic(root / "manifest.json", manifest)
    _write_stamp(name, manifest, verified)
    _mark_installed(name, {"manifest_revision": manifest["revision"], "integrity": "verified",
                           "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return (f"{name}: all {len(specs)} files verified ({downloaded} downloaded). "
            "Configure its isolated worker environment before inference.")


def verify(name, progress=None, *, quick=False):
    """Check every file against its pinned SHA-256. ``quick`` trusts files unchanged since their last check.

    ``progress(path, bytes_overall, total_overall)`` advances while hashing, not only between files.
    """
    manifest = manifests()[name]
    validate_manifest(manifest)
    root = models_dir() / name
    stamp = _read_stamp(name, manifest) if quick else {}
    missing, corrupt, verified = [], [], {}
    total = total_size(manifest)
    done = hashed = trusted = 0
    for spec in manifest["files"]:
        path = root / spec["path"]
        if progress:
            progress(spec["path"], done, total)
        if not path.is_file():
            missing.append(spec["path"])
        elif path.stat().st_size != spec["size"]:
            corrupt.append(spec["path"])
        elif stamp.get(spec["path"]) == _fingerprint(path):
            verified[spec["path"]] = stamp[spec["path"]]
            trusted += 1
        else:
            tick = (lambda n, path=spec["path"], done=done: progress(path, done + n, total)) if progress else None
            hashed += 1
            if hash_file(path, tick) == spec["sha256"]:
                verified[spec["path"]] = _fingerprint(path)
            else:
                corrupt.append(spec["path"])
        done += spec["size"]
    _write_stamp(name, manifest, verified)
    return {"name": name, "ok": bool(manifest["files"]) and not missing and not corrupt and not manifest.get("blocked"),
            "missing": missing, "corrupt": corrupt, "blocked": manifest.get("blocked"),
            "hashed": hashed, "trusted_from_stamp": trusted}
