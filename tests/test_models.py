import pytest

from deepfake.models import CATALOG, install_model, list_models


def test_catalog_has_alphaface_and_inswapper():
    assert "alphaface" in CATALOG
    assert "inswapper" in CATALOG
    assert "UNDOCUMENTED" in CATALOG["alphaface"]["weights_license"]
    assert "NON-COMMERCIAL" in CATALOG["inswapper"]["weights_license"]


def test_list_models(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    rows = list_models()
    names = {r.name for r in rows}
    assert names == {"alphaface", "inswapper", "rife"}
    assert all(r.installed is False or r.installed is True for r in rows)


def test_install_refuses_without_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="--yes"):
        install_model("alphaface", yes=False)


def test_unknown_model():
    with pytest.raises(KeyError):
        install_model("nope", yes=True)
