import hashlib

import pytest

from deepfake import experimental_models, models


@pytest.fixture
def experimental_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    content = b"model weights"
    spec = {"path": "weights.bin", "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "url": "https://huggingface.co/example/model/resolve/" + "a" * 40 + "/weights.bin"}
    manifest = {"name": "fixture", "source": "example", "revision": "a" * 40,
                "code_license": "MIT", "weights_license": "MIT", "precision": "fp16", "files": [spec]}
    monkeypatch.setattr(experimental_models, "manifests", lambda: {"fixture": manifest})
    path = models.models_dir() / "fixture" / "weights.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path, manifest


def state():
    return next(m for m in models.list_models() if m.name == "fixture")


def test_verified_result_is_visible_and_invalidated_after_change(experimental_fixture):
    path, manifest = experimental_fixture
    assert state().meta["integrity"] == "unverified"
    assert models.verify_model("fixture")["ok"]
    assert state().meta["integrity"] == "verified"
    path.write_bytes(b"changed bytes")
    assert state().meta["integrity"] == "unverified"
    result = models.verify_model("fixture")
    assert not result["ok"] and result["corrupt"] == ["weights.bin"]
    assert state().meta["integrity"] == "unverified"


def test_revision_change_invalidates_verification(experimental_fixture):
    _, manifest = experimental_fixture
    assert models.verify_model("fixture")["ok"]
    manifest["revision"] = "b" * 40
    assert state().meta["integrity"] == "unverified"
