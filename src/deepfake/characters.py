"""Private, local reference library with content-addressed image storage."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .ipc import write_json_atomic
from .models import file_sha256
from .paths import data_home


def library_dir():
    path = data_home() / "characters"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def list_characters():
    rows = []
    for path in sorted(library_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and Path(data.get("path", "")).is_file():
                rows.append(data)
        except (OSError, ValueError, TypeError):
            continue
    return rows


def import_character(path: Path, name: str, provenance: str):
    import cv2

    path = path.expanduser().resolve(strict=True)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or path.stat().st_size > 50 * 2**20:
        raise ValueError("Select a PNG, JPEG or WebP image under 50 MiB")
    if not name.strip() or len(name) > 120 or not provenance.strip():
        raise ValueError("A name and consent/provenance note are required")
    image = cv2.imread(str(path))
    if image is None or min(image.shape[:2]) < 64:
        raise ValueError("Reference must be a decodable image at least 64×64")
    digest = file_sha256(path)
    root = library_dir()
    dest = root / (digest + path.suffix.lower())
    shutil.copyfile(path, dest)
    dest.chmod(0o600)
    data = {"id": digest, "name": name.strip(), "path": str(dest), "sha256": digest,
            "width": image.shape[1], "height": image.shape[0], "provenance": provenance,
            "kind": "image", "conditioning": "prepared by selected engine on load"}
    write_json_atomic(root / (digest + ".json"), data)
    return data


def remove_character(identifier):
    row = next((r for r in list_characters() if r["id"] == identifier), None)
    if row is None:
        raise ValueError("Unknown character")
    path = Path(row["path"])
    if path.parent.resolve() != library_dir().resolve():
        raise ValueError("Reference is outside the private character library")
    path.unlink(missing_ok=True)
    (library_dir() / f"{identifier}.json").unlink(missing_ok=True)
