"""Model catalog + explicit install (never silent giant downloads)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import ensure_dirs, models_dir, vendor_dir

# Drive IDs from AlphaFace README (may change — verify before packaging).
# Weights license is undocumented — we refuse to redistribute; user downloads with ack.
ALPHAFACE_MAIN_DRIVE_ID = "18ZOQB3WmIFnMwi1GqBroFFEOuSNKWpZQ"
ALPHAFACE_ARCFACE_DRIVE_ID = "1qc4s6eRQPluma72WFibUnw74GPMAYRtY"
ALPHAFACE_REPO = "https://github.com/andrewyu90/Alphaface_Official.git"

# Checksums unknown until first verified download — stored after user ack install.
# Placeholder None means "record checksum on first successful install".

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
        "source": "Google Drive (AlphaFace README) + git clone Alphaface_Official",
        "paper": "arXiv:2601.16429",
        "code_license": "MIT (Andrewyu90 / Jongmin Andrew Yu)",
        "weights_license": (
            "UNDOCUMENTED — research weights on Google Drive only. "
            "Do NOT redistribute from this wrapper. User downloads with explicit --yes ack."
        ),
        "size_hint": "~hundreds of MB (Drive; not pinned here)",
        "checksum": None,
        "install": "drive+git",
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
        "source": "InsightFace inswapper_128.onnx (research)",
        "code_license": "MIT (InsightFace code)",
        "weights_license": (
            "NON-COMMERCIAL research only unless you obtain InsightFace commercial license. "
            "https://github.com/deepinsight/insightface"
        ),
        "size_hint": "~500MB class (onnx + buffalo detectors)",
        "checksum": None,
        "install": "insightface",
        "notes": "Fallback when AlphaFace weights unavailable. Clear NC warning on install.",
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
    _installed_marker().write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def list_models() -> list[ModelInfo]:
    ensure_dirs()
    installed = _load_installed()
    out: list[ModelInfo] = []
    for name, meta in CATALOG.items():
        path = models_dir() / name
        is_in = name in installed or _looks_installed(name, path)
        out.append(
            ModelInfo(name=name, installed=bool(is_in), meta=meta, path=path if path.exists() else None)
        )
    return out


def _looks_installed(name: str, path: Path) -> bool:
    if not path.is_dir():
        return False
    if name == "alphaface":
        return (path / "alphaface_demo.pt").is_file() or any(path.glob("*.pt"))
    if name == "inswapper":
        return any(path.glob("*.onnx")) or (path / "READY").is_file()
    if name == "rife":
        return (path / "flownet_v4.25.safetensors").is_file() or (path / "flownet_v4.25.pkl").is_file()
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


def _gdown_or_curl(file_id: str, dest: Path) -> None:
    """Resumable-ish download via gdown if present, else urllib (no resume)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if shutil.which("gdown"):
        cmd = ["gdown", "--id", file_id, "-O", str(part)]
        subprocess.run(cmd, check=True)
        part.rename(dest)
        return
    # Fallback: Google Drive confirm-token dance is fragile; instruct user.
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        urllib.request.urlretrieve(url, part)  # noqa: S310 — user-acked research download
        part.rename(dest)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Drive id={file_id} to {dest}: {e}. "
            "Install `gdown` (pip install gdown) and retry, or download manually from "
            "https://github.com/andrewyu90/Alphaface_Official README and place files under "
            f"{models_dir() / 'alphaface'}/"
        ) from e


def install_model(name: str, *, yes: bool = False) -> str:
    if name not in CATALOG:
        raise KeyError(f"unknown model '{name}'. Known: {', '.join(sorted(CATALOG))}")
    meta = CATALOG[name]
    ensure_dirs()
    target = models_dir() / name
    target.mkdir(parents=True, exist_ok=True)

    if not yes:
        raise RuntimeError(
            f"Refusing to download/install '{name}' without --yes. "
            f"Review license notes:\n  weights: {meta.get('weights_license')}\n"
            f"  code: {meta.get('code_license')}\n"
            f"  size: {meta.get('size_hint')}\n"
            f"Re-run: deepfake models install {name} --yes"
        )

    if name == "alphaface":
        return _install_alphaface(target, meta)
    if name == "inswapper":
        return _install_inswapper(target, meta)
    if name == "rife":
        return _install_rife(target)
    raise RuntimeError(f"no installer for {name}")


def _install_alphaface(target: Path, meta: dict[str, Any]) -> str:
    # Clone vendor code (MIT) for optional import path — not required for placeholder.
    vendor = vendor_dir()
    if not (vendor / ".git").exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", ALPHAFACE_REPO, str(vendor)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            (target / "VENDOR_CLONE_FAILED.txt").write_text(
                f"git clone failed: {e}\n"
                "You can still place .pt weights manually; swapper import may be limited.\n",
                encoding="utf-8",
            )

    messages: list[str] = []
    files = meta.get("files") or {}
    checksums: dict[str, str] = {}
    for fname, drive_id in files.items():
        dest = target / fname
        if dest.is_file() and dest.stat().st_size > 1024:
            messages.append(f"exists: {dest}")
            checksums[fname] = file_sha256(dest)
            continue
        messages.append(f"downloading {fname} (Drive id={drive_id}) …")
        _gdown_or_curl(drive_id, dest)
        checksums[fname] = file_sha256(dest)
        messages.append(f"sha256({fname})={checksums[fname]}")

    (target / "LICENSE_NOTES.txt").write_text(
        "AlphaFace code: MIT.\n"
        "Weights: license undocumented on Drive — research use; do not redistribute via this project.\n"
        f"Paper: arXiv:2601.16429\nRepo: {ALPHAFACE_REPO}\n",
        encoding="utf-8",
    )
    _mark_installed(
        "alphaface",
        {"checksums": checksums, "vendor": str(vendor) if vendor.exists() else None},
    )
    return "AlphaFace install steps complete:\n" + "\n".join(messages)


def _install_inswapper(target: Path, meta: dict[str, Any]) -> str:
    (target / "LICENSE_WARNING.txt").write_text(
        "InsightFace pretrained models are NON-COMMERCIAL research unless commercially licensed.\n"
        "Code: MIT. See https://github.com/deepinsight/insightface\n"
        f"Catalog note: {meta.get('weights_license')}\n",
        encoding="utf-8",
    )
    # Do not silently pull onnx; instruct via insightface package when available.
    try:
        import insightface  # type: ignore  # noqa: F401

        (target / "READY").write_text(
            "insightface Python package present. "
            "Models download on first use into insightface cache — review their terms.\n",
            encoding="utf-8",
        )
        _mark_installed("inswapper", {"via": "insightface-package"})
        return (
            "inswapper marked ready (insightface package detected). "
            "Weights may download on first inference into InsightFace's cache — "
            "NON-COMMERCIAL research terms apply."
        )
    except ImportError as err:
        (target / "INSTALL_HINT.txt").write_text(
            "pip install insightface onnxruntime-gpu  # or onnxruntime for CPU\n"
            "Then re-run: deepfake models install inswapper --yes\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "insightface package not installed. "
            "Install with: pip install insightface onnxruntime  (GPU: onnxruntime-gpu). "
            "Models are NON-COMMERCIAL research. Then re-run with --yes."
        ) from err


def _install_rife(target: Path) -> str:
    messages: list[str] = []
    for variant, digest in RIFE_WEIGHTS.items():
        st = target / f"flownet_v{variant}.safetensors"
        if st.is_file():
            messages.append(f"exists: {st.name}")
            continue
        pkl = target / f"flownet_v{variant}.pkl"
        part = pkl.with_suffix(".pkl.part")
        url = f"{RIFE_RELEASE}/flownet_v{variant}.pkl"
        messages.append(f"downloading {url}")
        urllib.request.urlretrieve(url, part)  # noqa: S310 — fixed https URL, verified below
        got = file_sha256(part)
        if got != digest:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for RIFE {variant}: expected {digest}, got {got}")
        part.rename(pkl)
        try:
            import torch
            from safetensors.torch import save_file

            sd = torch.load(str(pkl), map_location="cpu", weights_only=True)
            sd = {k.replace("module.", ""): v.contiguous() for k, v in sd.items() if k.startswith("module.")}
            save_file(sd, str(st), metadata={"source": url, "sha256_of_source": digest, "license": "MIT"})
            pkl.unlink()
            messages.append(f"verified + converted → {st.name}")
        except ImportError:
            messages.append(f"verified {pkl.name} (install safetensors to convert; loaded weights_only)")
    (target / "LICENSE_NOTES.txt").write_text(
        "RIFE / Practical-RIFE — MIT, Copyright (c) 2021 hzwer. https://github.com/hzwer/Practical-RIFE\n"
        "Release mirror + refactored IFNet: vs-rife — MIT, Copyright (c) 2021 HolyWu.\n"
        "Paper: Huang et al., Real-Time Intermediate Flow Estimation for Video Frame Interpolation, "
        "ECCV 2022 (arXiv:2011.06294)\n",
        encoding="utf-8",
    )
    _mark_installed("rife", {"variants": sorted(RIFE_WEIGHTS), "sha256": RIFE_WEIGHTS})
    return "RIFE frame-generation weights ready:\n" + "\n".join(messages)
