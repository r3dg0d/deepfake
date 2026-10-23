import pytest


@pytest.fixture(autouse=True)
def isolated_xdg(request, tmp_path, monkeypatch):
    real_models = request.node.get_closest_marker("gpu") is not None
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        if real_models and var == "XDG_CACHE_HOME":
            continue  # gpu tests exercise the installed weights in the real model cache
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    monkeypatch.delenv("DEEPFAKE_CONSENT_ACK", raising=False)
