"""Staged, verified (re)installation of the core face-swap and frame-generation weights.

Every core model is described by a manifest of files with their official source URL, expected
size and SHA-256. Installation never writes into the live model directory directly:

1. files are downloaded (resumable, HTTP Range) into ``models/.staging/<name>/``;
2. each file is checked for size, SHA-256, HTML error pages and its container format;
3. post-processing (RIFE pickle -> safetensors, buffalo_l zip extraction) happens in staging;
4. the live directory is moved to ``models/.rollback/<name>`` and staging takes its place
   with an atomic rename;
5. an optional load check runs the real model on the GPU; if it fails the previous
   directory is restored;
6. only then is ``installed.json`` updated and the rollback copy removed.

A file's existence or an ``installed.json`` entry is never treated as proof of integrity:
``verify`` re-hashes every file against the pinned (or install-time recorded) digest.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .paths import models_dir

Progress = Callable[[str, int, int], None]

DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
INSIGHTFACE_RELEASE = "https://github.com/deepinsight/insightface/releases/download/model-zoo"
RIFE_RELEASE = "https://github.com/HolyWu/vs-rife/releases/download/model"
ALPHAFACE_REPO = "https://github.com/andrewyu90/Alphaface_Official.git"
ALPHAFACE_COMMIT = "d41fbd4974ed3a68d9a48b79019f9be297726c30"
ALLOWED_HOSTS = ("https://drive.usercontent.google.com/", "https://github.com/", "https://huggingface.co/")
USER_AGENT = "deepfake-model-store/1"

# trust: "upstream" = digest published by the upstream host (GitHub release asset digest);
#        "pinned" = digest pinned by this project from an earlier verified download (AlphaFace: first
#                   download 2026-10-22, both files load and swap on CUDA; upstream publishes no digest);
#        "recorded" = digest recorded on first verified download of this file (upstream publishes none).
CORE: dict[str, dict[str, Any]] = {
    "rife": {
        "files": [
            {"path": f"flownet_v{v}.pkl", "url": f"{RIFE_RELEASE}/flownet_v{v}.pkl", "size": size, "sha256": digest,
             "trust": "pinned", "format": "torch-pickle", "convert": "rife-safetensors", "keep": False}
            for v, size, digest in (
                ("4.25", 24636301, "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"),
                ("4.25.lite", 24467405, "81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1"),
                ("4.26", 24636301, "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"),
            )
        ],
    },
    "inswapper": {
        "files": [
            {"path": "inswapper_128.onnx", "url": f"{INSIGHTFACE_RELEASE}/inswapper_128.onnx", "size": 554253681,
             "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af", "trust": "upstream",
             "format": "onnx"},
            {"path": "buffalo_l.zip", "url": f"{INSIGHTFACE_RELEASE}/buffalo_l.zip", "size": 288621354,
             "sha256": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", "trust": "upstream",
             "format": "zip", "extract_to": "models/buffalo_l", "keep": False},
        ],
    },
    "yunet": {
        "files": [
            {"path": "face_detection_yunet_2023mar.onnx",
             "url": "https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/"
                    "face_detection_yunet_2023mar.onnx",
             "size": 232589, "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
             "trust": "upstream", "format": "onnx"},
        ],
    },
    "alphaface": {
        "files": [
            {"path": "alphaface_demo.pt", "url": DRIVE.format("18ZOQB3WmIFnMwi1GqBroFFEOuSNKWpZQ"),
             "size": 1932185842, "sha256": "4848bb3ccf8adf68438be335470518bab93e8d32c6de5c6866ab0263c00db757",
             "trust": "pinned", "format": "torch-zip"},
            {"path": "arcface.pt", "url": DRIVE.format("1qc4s6eRQPluma72WFibUnw74GPMAYRtY"),
             "size": 174451754, "sha256": "b4e759baca1760f539d0b81d3614deb789c8354b337b44a78f822be6cfbc5e2b",
             "trust": "pinned", "format": "torch-zip"},
        ],
        "vendor": {"repo": ALPHAFACE_REPO, "commit": ALPHAFACE_COMMIT},
    },
}

BUFFALO_L_MEMBERS = {"1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"}


class ModelStoreError(RuntimeError):
    pass


def staging_dir(name: str) -> Path:
    return models_dir() / ".staging" / name


def rollback_dir(name: str) -> Path:
    return models_dir() / ".rollback" / name


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(path: str) -> PurePosixPath:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ModelStoreError(f"unsafe model path {path!r}")
    return p


# --------------------------------------------------------------------------- download


def looks_like_html(head: bytes) -> bool:
    h = head.lstrip()[:64].lower()
    return h.startswith((b"<!doctype html", b"<html", b"<head", b"<?xml")) or b"<html" in h


def download(spec: dict[str, Any], dest: Path, progress: Progress | None = None, *, timeout: float = 60.0) -> Path:
    """Download ``spec['url']`` to ``dest`` via ``dest.part`` with HTTP Range resume; verify size + SHA-256."""
    url = spec["url"]
    if not url.startswith(ALLOWED_HOSTS):
        raise ModelStoreError(f"refusing download from non-allowlisted host: {url}")
    size, digest = spec.get("size"), spec.get("sha256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (size is None or dest.stat().st_size == size) and (digest is None or sha256(dest) == digest):
        return dest
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    if size is not None and offset > size:
        part.unlink()
        offset = 0
    if size is not None and offset == size:
        pass  # complete .part from an interrupted run: fall through to verification
    else:
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - allowlisted https host
                ctype = resp.headers.get("Content-Type", "")
                if ctype.startswith("text/html"):
                    raise ModelStoreError(f"{spec['path']}: server returned an HTML page instead of the file "
                                          f"(quota, permission or removed upload?) from {url}")
                resumed = offset > 0 and resp.status == 206
                if resumed and not resp.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ModelStoreError(f"{spec['path']}: server returned an unexpected byte range")
                done = offset if resumed else 0
                total = size or int(resp.headers.get("Content-Length") or 0) + done
                with part.open("ab" if resumed else "wb") as out:
                    first = True
                    while chunk := resp.read(1024 * 1024):
                        if first and not resumed and looks_like_html(chunk[:512]):
                            raise ModelStoreError(f"{spec['path']}: download is an HTML page, not a model file ({url})")
                        first = False
                        out.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(spec["path"], done, total)
        except urllib.error.HTTPError as e:
            if not (e.code == 416 and size is not None and offset == size):
                raise ModelStoreError(f"{spec['path']}: HTTP {e.code} from {url}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ModelStoreError(f"{spec['path']}: download interrupted ({e}); re-run to resume from "
                                  f"{part.stat().st_size if part.exists() else 0} bytes") from e
    got = part.stat().st_size
    if size is not None and got != size:
        raise ModelStoreError(f"{spec['path']}: incomplete download ({got} of {size} bytes); re-run to resume")
    with part.open("rb") as f:
        if looks_like_html(f.read(512)):
            part.unlink()
            raise ModelStoreError(f"{spec['path']}: downloaded file is an HTML page, not a model")
    if digest is not None:
        actual = sha256(part)
        if actual != digest:
            part.unlink()
            raise ModelStoreError(f"{spec['path']}: SHA-256 mismatch (expected {digest}, got {actual}); "
                                  "the partial download was discarded")
    os.chmod(part, 0o600)
    part.replace(dest)
    return dest


# --------------------------------------------------------------------------- format checks


def validate_format(path: Path, fmt: str) -> str:
    """Cheap structural check of a weights container (no code execution). Returns a short description."""
    with path.open("rb") as f:
        head = f.read(16)
    if looks_like_html(head):
        raise ModelStoreError(f"{path.name}: is an HTML page")
    if fmt in ("torch-zip", "zip"):
        if not zipfile.is_zipfile(path):
            raise ModelStoreError(f"{path.name}: not a zip container")
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if fmt == "torch-zip" and not any(n.endswith("/data.pkl") or n == "data.pkl" for n in names):
                raise ModelStoreError(f"{path.name}: zip without data.pkl is not a torch checkpoint")
            bad = z.testzip()
            if bad is not None:
                raise ModelStoreError(f"{path.name}: corrupt member {bad}")
        return f"{fmt} ({len(names)} members, CRC ok)"
    if fmt == "torch-pickle":
        if head[:1] != b"\x80" and not head.startswith(b"PK"):
            raise ModelStoreError(f"{path.name}: not a pickle/torch file")
        return "torch pickle (loaded weights_only during conversion)"
    if fmt == "onnx":
        try:
            import onnx

            m = onnx.load_model(str(path), load_external_data=False)
            return f"onnx ir={m.ir_version} opset={[o.version for o in m.opset_import]} graph nodes={len(m.graph.node)}"
        except ImportError:
            if head[:1] != b"\x08":  # protobuf field 1 (ir_version), varint
                raise ModelStoreError(f"{path.name}: does not start like an ONNX ModelProto") from None
            return "onnx (header only; install onnx for a full parse)"
        except Exception as e:  # noqa: BLE001
            raise ModelStoreError(f"{path.name}: ONNX parse failed: {e}") from e
    if fmt == "safetensors":
        n = int.from_bytes(head[:8], "little")
        if not 0 < n < 100_000_000:
            raise ModelStoreError(f"{path.name}: bad safetensors header")
        with path.open("rb") as f:
            f.seek(8)
            json.loads(f.read(n))
        return "safetensors"
    raise ModelStoreError(f"unknown format {fmt}")


def _safe_extract(zpath: Path, dest: Path, expected: set[str] | None = None) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ModelStoreError(f"{zpath.name}: unsafe member path {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ModelStoreError(f"{zpath.name}: refusing symlink member {info.filename}")
            target = dest / rel.name  # flatten: buffalo_l.zip members are top-level onnx files
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            os.chmod(target, 0o600)
            out.append(rel.name)
    if expected is not None and not expected <= set(out):
        raise ModelStoreError(f"{zpath.name}: missing members {sorted(expected - set(out))}")
    return out


def _rife_to_safetensors(pkl: Path, spec: dict[str, Any]) -> Path:
    import torch
    from safetensors.torch import save_file

    sd = torch.load(str(pkl), map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v.contiguous() for k, v in sd.items() if k.startswith("module.")}
    if not sd:
        raise ModelStoreError(f"{pkl.name}: no 'module.' tensors; not a Practical-RIFE flownet")
    st = pkl.with_suffix(".safetensors")
    save_file(sd, str(st), metadata={"source": spec["url"], "sha256_of_source": spec["sha256"] or "", "license": "MIT"})
    os.chmod(st, 0o600)
    return st


# --------------------------------------------------------------------------- vendor code


def ensure_vendor(name: str) -> dict[str, Any] | None:
    v = CORE[name].get("vendor")
    if not v:
        return None
    import subprocess

    from .paths import vendor_dir

    dest = vendor_dir()
    git = shutil.which("git")
    if git is None:
        raise ModelStoreError("git is required to fetch the AlphaFace source checkout")
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([git, "clone", "--quiet", v["repo"], str(dest)], check=True, capture_output=True, text=True)
    head = subprocess.run([git, "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != v["commit"]:
        subprocess.run([git, "-C", str(dest), "fetch", "--quiet", "origin", v["commit"]], capture_output=True, text=True)
        r = subprocess.run([git, "-C", str(dest), "checkout", "--quiet", v["commit"]], capture_output=True, text=True)
        if r.returncode != 0:
            raise ModelStoreError(f"could not check out pinned AlphaFace commit {v['commit']}: {r.stderr.strip()}")
    return {"repo": v["repo"], "commit": v["commit"], "path": str(dest)}


# --------------------------------------------------------------------------- install / verify


def _recorded(name: str) -> dict[str, Any]:
    from .models import _load_installed

    return _load_installed().get(name, {})


def expected_files(name: str) -> list[dict[str, Any]]:
    """Files that must exist in the live directory, with the digest to verify them against."""
    rec = {f["path"]: f for f in _recorded(name).get("files", [])}
    out = []
    for spec in CORE[name]["files"]:
        if spec.get("convert") == "rife-safetensors":
            p = spec["path"].replace(".pkl", ".safetensors")
            out.append({"path": p, "sha256": rec.get(p, {}).get("sha256"), "trust": f"derived from {spec['trust']} {spec['path']}",
                        "format": "safetensors"})
        elif spec.get("extract_to"):
            for m in sorted(BUFFALO_L_MEMBERS):
                p = f"{spec['extract_to']}/{m}"
                out.append({"path": p, "sha256": rec.get(p, {}).get("sha256"), "trust": f"extracted from {spec['trust']} {spec['path']}",
                            "format": "onnx"})
        else:
            digest = spec["sha256"] or rec.get(spec["path"], {}).get("sha256")
            out.append({"path": spec["path"], "sha256": digest, "size": spec.get("size"),
                        "trust": spec["trust"] if spec["sha256"] else "recorded", "format": spec["format"]})
    return out


def verify(name: str, *, progress: Progress | None = None) -> dict[str, Any]:
    if name not in CORE:
        raise KeyError(name)
    root = models_dir() / name
    results, ok = [], True
    files = expected_files(name)
    total = len(files)
    for i, f in enumerate(files):
        if progress:
            progress(f["path"], i, total)
        p = root / f["path"]
        row: dict[str, Any] = {"path": f["path"], "trust": f["trust"]}
        if not p.is_file():
            row.update(ok=False, reason="missing")
        elif f.get("size") is not None and p.stat().st_size != f["size"]:
            row.update(ok=False, reason=f"size {p.stat().st_size} != {f['size']}")
        elif f["sha256"] is None:
            row.update(ok=False, reason="no digest pinned or recorded; reinstall to record one", sha256=sha256(p))
        else:
            actual = sha256(p)
            row.update(sha256=actual, ok=actual == f["sha256"],
                       reason=None if actual == f["sha256"] else f"SHA-256 mismatch (expected {f['sha256']})")
            if row["ok"]:
                try:
                    row["format"] = validate_format(p, f["format"])
                except ModelStoreError as e:
                    row.update(ok=False, reason=str(e))
        ok &= bool(row["ok"])
        results.append(row)
    return {"name": name, "ok": ok, "files": results, "trust": sorted({r["trust"] for r in results})}


def install(name: str, *, force: bool = False, progress: Progress | None = None, load_check: bool = True,
            keep_rollback: bool = False, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """(Re)install a core model through staging; returns the installed.json record."""
    from .models import CATALOG, _mark_installed

    say = log or (lambda m: None)
    if name not in CORE:
        raise KeyError(name)
    live = models_dir() / name
    if not force and live.is_dir():
        current = verify(name)
        if current["ok"]:
            say(f"{name}: installed files verify; nothing to download (use reinstall to force)")
            return _recorded(name) or {"name": name, "files": current["files"]}
    from .paths import ensure_dirs

    ensure_dirs()
    specs = CORE[name]["files"]
    need = sum(s.get("size") or 0 for s in specs) * 2
    if shutil.disk_usage(models_dir()).free < need:
        raise ModelStoreError(f"{name}: need about {need / 2**30:.1f} GiB free under {models_dir()}")

    stage = staging_dir(name)
    stage.mkdir(parents=True, exist_ok=True)
    os.chmod(stage, 0o700)
    vendor = ensure_vendor(name)

    records: list[dict[str, Any]] = []
    for spec in specs:
        rel = _safe_rel(spec["path"])
        dest = stage / rel
        say(f"{name}: downloading {spec['path']} from {spec['url']}")
        download(spec, dest, progress)
        say(f"{name}: {spec['path']}: {validate_format(dest, spec['format'])}")
        digest = spec["sha256"] or sha256(dest)
        base = {"source_url": spec["url"], "trust": spec["trust"], "source_sha256": digest, "source_size": dest.stat().st_size}
        if spec.get("convert") == "rife-safetensors":
            st = _rife_to_safetensors(dest, spec)
            validate_format(st, "safetensors")
            records.append({"path": st.name, "sha256": sha256(st), "size": st.stat().st_size, **base,
                            "derived_from": spec["path"]})
        elif spec.get("extract_to"):
            members = _safe_extract(dest, stage / spec["extract_to"], BUFFALO_L_MEMBERS)
            for m in members:
                p = stage / spec["extract_to"] / m
                validate_format(p, "onnx")
                records.append({"path": f"{spec['extract_to']}/{m}", "sha256": sha256(p), "size": p.stat().st_size,
                                **base, "derived_from": spec["path"]})
        if spec.get("keep", True):
            records.append({"path": spec["path"], "sha256": digest, "size": dest.stat().st_size, **base})
        else:
            dest.unlink()
    meta = CATALOG.get(name, {})
    (stage / "LICENSE_NOTES.txt").write_text(
        f"code license: {meta.get('code_license')}\nweights license: {meta.get('weights_license')}\n"
        + "".join(f"source: {s['url']}\n" for s in specs), encoding="utf-8")

    # swap: live -> rollback, staging -> live
    rb = rollback_dir(name)
    rb.parent.mkdir(parents=True, exist_ok=True)
    if rb.exists():
        shutil.rmtree(rb)
    had_live = live.exists()
    if live.is_symlink():
        raise ModelStoreError(f"refusing to replace symlinked model directory {live}")
    if had_live:
        os.replace(live, rb)
    os.replace(stage, live)
    say(f"{name}: new files in place" + (f"; previous copy kept at {rb}" if had_live else ""))

    record = {"files": records, "vendor": vendor, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "integrity": "verified"}
    check: dict[str, Any] | None = None
    if load_check:
        from .model_checks import run_check

        try:
            check = run_check(name)
        except Exception as e:  # noqa: BLE001
            check = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if not check.get("ok"):
            failed = models_dir() / ".failed" / name
            failed.parent.mkdir(parents=True, exist_ok=True)
            if failed.exists():
                shutil.rmtree(failed)
            os.replace(live, failed)
            if had_live:
                os.replace(rb, live)
            raise ModelStoreError(f"{name}: load check failed ({check.get('error')}); "
                                  f"{'previous files restored' if had_live else 'nothing installed'}; "
                                  f"new files kept for inspection at {failed}")
        record["load_check"] = check
    _mark_installed(name, record)
    if had_live and not keep_rollback:
        shutil.rmtree(rb, ignore_errors=True)
    return {"name": name, **record}


__all__ = ["CORE", "ModelStoreError", "download", "expected_files", "install", "looks_like_html", "sha256",
           "validate_format", "verify"]
