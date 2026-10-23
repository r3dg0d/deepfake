"""Real-weights checks (run with: pytest -m gpu). Never mocked: they load each installed model on CUDA."""

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("name", ["rife", "inswapper", "alphaface"])
def test_installed_model_verifies_and_runs(name):
    from deepfake.model_store import verify
    from deepfake.models import verify_model

    if not verify(name)["ok"]:
        pytest.skip(f"{name} not installed/verified; run deepfake models install {name} --yes")
    r = verify_model(name, load=True)
    assert r["ok"], r
    check = r["load_check"]
    if name == "inswapper":
        assert check["on_gpu"], check
    if name == "alphaface":
        assert check["device"].startswith("cuda"), check
