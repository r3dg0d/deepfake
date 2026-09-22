import numpy as np
import pytest

from deepfake.composite import paste_face
from deepfake.detect import FaceBox
from deepfake.swap.placeholder import MissingModelSwapper, PassthroughSwapper
from deepfake.watermark import apply_watermark


def test_passthrough_swap():
    s = PassthroughSwapper()
    src = np.zeros((64, 64, 3), dtype=np.uint8)
    tgt = np.ones((64, 64, 3), dtype=np.uint8) * 10
    s.set_source(src)
    r = s.swap(tgt)
    assert r.backend == "passthrough"
    assert r.face_bgr.shape == tgt.shape


def test_missing_model_errors():
    s = MissingModelSwapper()
    with pytest.raises(RuntimeError, match="models install"):
        s.set_source(np.zeros((8, 8, 3), dtype=np.uint8))


def test_watermark_and_paste():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    face = np.ones((80, 80, 3), dtype=np.uint8) * 200
    box = FaceBox(20, 20, 80, 80)
    out = paste_face(frame, face, box, feather=4, color_match=False)
    assert out.shape == frame.shape
    marked = apply_watermark(out, enabled=True)
    assert marked.shape == out.shape
