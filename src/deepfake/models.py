"""Model catalog + explicit install (never silent giant downloads)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import ensure_dirs, models_dir

# Download sources, sizes and digests live in model_store.CORE; this catalog is the human-facing
# description (licenses, notes). Weights are never redistributed by this project.
ALPHAFACE_MAIN_DRIVE_ID = "18ZOQB3WmIFnMwi1GqBroFFEOuSNKWpZQ"
ALPHAFACE_ARCFACE_DRIVE_ID = "1qc4s6eRQPluma72WFibUnw74GPMAYRtY"
ALPHAFACE_REPO = "https://github.com/andrewyu90/Alphaface_Official.git"

# Practical-RIFE weights, mirrored as GitHub release assets by vs-rife (MIT).
# SHA-256 pinned from the files verified on 2026-09-22; a mismatch aborts install.
RIFE_RELEASE = "https://github.com/HolyWu/vs-rife/releases/download/model"
RIFE_WEIGHTS: dict[str, str] = {
    "4.25": "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",
    "4.25.lite": "81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1",
    "4.26": "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",
}

CATALOG: dict[str, dict[str, Any]] = {
    "alphaface": {
        "name": "alphaface",
        "backend": "alphaface",
        "source": "Google Drive links in the AlphaFace README + Alphaface_Official @ d41fbd4 (pinned)",
        "paper": "arXiv:2601.16429",
        "code_license": "MIT (Andrewyu90 / Jongmin Andrew Yu)",
        "weights_license": (
            "UNDOCUMENTED — research weights on Google Drive only. "
            "Do NOT redistribute from this wrapper. User downloads with explicit --yes ack."
        ),
        "size_hint": "2.1 GB (alphaface_demo.pt 1.93 GB + arcface 174 MB)",
        "checksum": "sizes pinned; SHA-256 recorded on first verified download (upstream publishes none)",
        "install": "staged https (Drive) + pinned git checkout",
        "notes": (
            "Upstream is still/batch-only; this CLI adds detect→swap→composite. "
            "Weights are NOT vendored in the git repo."
        ),
        "files": {
            "alphaface_demo.pt": ALPHAFACE_MAIN_DRIVE_ID,
            "arcface.pt": ALPHAFACE_ARCFACE_DRIVE_ID,
        },
    },
    "rife": {
        "name": "rife",
        "backend": "framegen",
        "source": f"{RIFE_RELEASE}/flownet_v{{4.25,4.25.lite,4.26}}.pkl (Practical-RIFE weights)",
        "paper": "arXiv:2011.06294 (ECCV 2022)",
        "code_license": "MIT (hzwer/Practical-RIFE; HolyWu/vs-rife)",
        "weights_license": "MIT (released with Practical-RIFE)",
        "size_hint": "~25 MB per variant (3 variants, ~74 MB)",
        "checksum": RIFE_WEIGHTS,
        "install": "https+sha256 → safetensors",
        "notes": (
            "Real-time frame interpolation for --frame-gen. Downloads are SHA-256 "
            "verified, loaded with torch weights_only and re-saved as safetensors."
        ),
    },
    "inswapper": {
        "name": "inswapper",
        "backend": "insightface",
        "source": "deepinsight/insightface GitHub release 'model-zoo': inswapper_128.onnx + buffalo_l.zip",
        "code_license": "MIT (InsightFace code)",
        "weights_license": (
            "NON-COMMERCIAL research only unless you obtain InsightFace commercial license. "
            "https://github.com/deepinsight/insightface"
        ),
        "size_hint": "843 MB (inswapper 554 MB + buffalo_l 289 MB)",
        "checksum": "SHA-256 published by GitHub for the upstream release assets (pinned)",
        "install": "staged https + sha256; buffalo_l extracted under models/inswapper/models/buffalo_l",
        "notes": "Fallback when AlphaFace weights are unavailable. Needs the insightface + onnxruntime-gpu "
                 "packages. NON-COMMERCIAL model terms apply.",
    },
}


@dataclass
class ModelInfo:
    name: str
    installed: bool
    meta: dict[str, Any]
    path: Path | None = None


def _installed_marker() -> Path:
    return models_dir() / "installed.json"


def _load_installed() -> dict[str, Any]:
    marker = _installed_marker()
    if marker.is_file():
        try:
            return json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _mark_installed(name: str, extra: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    data = _load_installed()
    data[name] = {"name": name, **(extra or {})}
    from .ipc import write_json_atomic

    write_json_atomic(_installed_marker(), data)


def list_models() -> list[ModelInfo]:
    ensure_dirs()
    installed = _load_installed()
    out: list[ModelInfo] = []
    for name, meta in CATALOG.items():
        path = models_dir() / name
        is_in = _looks_installed(name, path)
        rec = installed.get(name, {})
        integrity = rec.get("integrity") if is_in and rec.get("files") else ("unverified" if is_in else "not installed")
        out.append(
            ModelInfo(name=name, installed=bool(is_in), path=path if path.exists() else None,
                      meta={**meta, "integrity": integrity, "load_check_ok": (rec.get("load_check") or {}).get("ok")})
        )
    from .experimental_models import manifests

    for name, meta in manifests().items():
        path = models_dir() / name
        present = bool(meta["files"]) and not meta.get("blocked") and all(
            (path / f["path"]).is_file() and (path / f["path"]).stat().st_size == f["size"] for f in meta["files"])
        out.append(ModelInfo(name=name, installed=present, meta={**meta, "experimental": True,
                            "size_hint": f"{sum(f['size'] for f in meta['files']) / 2**30:.2f} GiB"},
                             path=path if path.exists() else None))
    return out


def _looks_installed(name: str, path: Path) -> bool:
    """Every file the live directory must contain is present (integrity is checked by ``verify``).

    A bare ``installed.json`` entry, a ``READY`` marker or "some .pt file" no longer count:
    the files that the backend actually loads must be there.
    """
    if not path.is_dir():
        return False
    from .model_store import CORE, expected_files

    if name in CORE:
        return all((path / f["path"]).is_file() and (f.get("size") is None or (path / f["path"]).stat().st_size == f["size"])
                   for f in expected_files(name))
    return any(path.iterdir())


def is_model_ready(name: str) -> bool:
    return any(m.name == name and m.installed for m in list_models())


def active_backend() -> str | None:
    """Prefer alphaface, else inswapper, else None."""
    if is_model_ready("alphaface"):
        return "alphaface"
    if is_model_ready("inswapper"):
        return "inswapper"
    return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_model(name: str, *, yes: bool = False, force: bool = False, progress=None, load_check: bool = True) -> str:
    from .experimental_models import install, manifests

    if name in manifests():
        return install(name, yes=yes, progress=progress)
    if name not in CATALOG:
        raise KeyError(f"unknown model '{name}'. Known: {', '.join(sorted(CATALOG))}")
    meta = CATALOG[name]
    ensure_dirs()

    if not yes:
        raise RuntimeError(
            f"Refusing to download/install '{name}' without --yes. "
            f"Review license notes:\n  weights: {meta.get('weights_license')}\n"
            f"  code: {meta.get('code_license')}\n"
            f"  size: {meta.get('size_hint')}\n"
            f"Re-run: deepfake models install {name} --yes"
        )

    return _install_core(name, force=force, progress=progress, load_check=load_check)


def _install_core(name: str, *, force: bool, progress=None, load_check: bool = True) -> str:
    from .model_store import ModelStoreError, install

    lines: list[str] = []
    try:
        rec = install(name, force=force, progress=progress, load_check=load_check, log=lines.append)
    except ModelStoreError as e:
        raise RuntimeError(str(e)) from e
    check = rec.get("load_check") or {}
    lines.append(f"{name}: {len(rec.get('files', []))} files verified"
                 + (f"; load check {'passed' if check.get('ok') else 'FAILED'}" if check else "; load check skipped"))
    return "\n".join(lines)


def verify_model(name: str, progress=None, *, load: bool = False) -> dict[str, Any]:
    """Re-hash every installed file against its pinned or install-time digest; optionally load + infer."""
    from .experimental_models import manifests, verify

    if name in manifests():
        return verify(name, progress=progress)
    if name not in CATALOG:
        raise KeyError(f"Unknown model {name}")
    from .model_store import verify as store_verify

    result = store_verify(name, progress=progress)
    if load and result["ok"]:
        from .model_checks import run_check

        result["load_check"] = run_check(name)
        result["ok"] = bool(result["load_check"].get("ok"))
        rec = _load_installed().get(name)
        if rec is not None:  # keep the manifest in step with the latest real check
            import time

            rec.pop("name", None)
            _mark_installed(name, {**rec, "load_check": result["load_check"],
                                   "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return result


def remove_model(name: str, *, yes: bool = False) -> str:
    known = {m.name for m in list_models()}
    if name not in known:
        raise KeyError(f"Unknown model {name}")
    if not yes:
        raise RuntimeError(f"Removing {name} requires --yes")
    target = models_dir() / name
    if target.is_symlink():
        raise RuntimeError("Refusing to remove a symlinked model directory")
    shutil.rmtree(target, ignore_errors=False) if target.exists() else None
    data = _load_installed()
    data.pop(name, None)
    from .ipc import write_json_atomic

    write_json_atomic(_installed_marker(), data)
    return f"Removed {name}"
