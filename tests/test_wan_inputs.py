import numpy as np
import pytest

from deepfake.experimental.wan_inputs import load_controls, prepare_controls


def raw_controls(path):
    # Portrait with distinct top/bottom detail: center-cropping would lose both.
    ref = np.zeros((128, 32, 3), np.uint8)
    ref[:32] = (255, 0, 0)
    ref[-32:] = (0, 0, 255)
    background = np.full((1, 64, 64, 3), 170, np.uint8)
    mask = np.zeros((1, 64, 64), np.float32)
    mask[:, 20:44, 20:44] = 1
    np.savez(path, reference=ref, background=background, pose=background, face=background, mask=mask)


def test_preserves_entire_reference_and_removes_target_appearance(tmp_path):
    raw, out = tmp_path / "raw.npz", tmp_path / "prepared.npz"
    raw_controls(raw)
    prepare_controls(raw, out, width=128, height=64)
    data = load_controls(out)
    ref = data["reference"]
    assert ref[:16, 56:72, 0].max() == 255
    assert ref[-16:, 56:72, 2].max() == 255
    mask = data["mask"] > 0
    assert mask[0, 20:44, 20:44].all()
    assert np.all(data["background"][mask] == 0)
    assert np.all(data["background"][~mask] == 170)
    assert np.all(load_controls(raw)["background"] == 170)


def test_invalid_mask_is_rejected(tmp_path):
    raw = tmp_path / "raw.npz"
    raw_controls(raw)
    data = load_controls(raw)
    data["mask"][0, 0, 0] = np.nan
    np.savez(raw, **data)
    with pytest.raises(ValueError, match="finite"):
        load_controls(raw)


def test_original_controls_cannot_be_overwritten(tmp_path):
    raw = tmp_path / "raw.npz"
    raw_controls(raw)
    with pytest.raises(ValueError, match="separate"):
        prepare_controls(raw, raw)
