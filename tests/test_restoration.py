import numpy as np
import pytest

from deepfake.restoration import FFHQ, EyeRestorer


class Session:
    calls = 0

    def run(self, _, inputs):
        self.calls += 1
        assert inputs["input"].shape == (1, 3, 512, 512)
        return [np.ones((1, 3, 512, 512), np.float32)]


def restorer():
    model = object.__new__(EyeRestorer)
    model.session = Session()
    model.strength = 0.65
    model.latency_ms = 0
    return model


def test_learned_detail_never_changes_occluder_or_non_eye_pixels():
    image = np.full((256, 256, 3), 120, np.uint8)
    eyes = np.zeros((256, 256), np.float32)
    eyes[110:130, 85:170] = 1
    visible = np.ones_like(eyes)
    visible[:, 128:] = 0  # glasses / hand covers the right eye
    model = restorer()
    output = model.restore(image, FFHQ / 2, eyes, visible)
    assert model.session.calls == 1
    assert np.any(output != image)
    assert np.array_equal(output[:, 128:], image[:, 128:])
    assert np.array_equal(output[eyes == 0], image[eyes == 0])
    assert np.max(np.abs(output.astype(int) - image.astype(int))) <= 21


@pytest.mark.parametrize("condition", ["closed", "missing", "bad-landmarks", "bad-mask", "hidden"])
def test_uncertain_or_absent_eyes_skip_inference(condition):
    image = np.full((256, 256, 3), 120, np.uint8)
    eyes = np.ones(image.shape[:2], np.float32)
    visible = eyes.copy()
    points = FFHQ / 2
    if condition == "closed":
        eyes[:] = 0
    elif condition == "missing":
        eyes = None
    elif condition == "bad-landmarks":
        points = np.full((5, 2), np.nan)
    elif condition == "bad-mask":
        eyes[:] = np.nan
    else:
        visible[:] = 0
    model = restorer()
    assert np.array_equal(model.restore(image, points, eyes, visible), image)
    assert model.session.calls == 0


def test_cuda_allocation_failure_retries_on_cpu_without_changing_hidden_pixels(monkeypatch):
    import onnxruntime as ort

    class Exhausted(Session):
        def run(self, *args):
            raise RuntimeError("Failed to allocate memory for requested buffer")

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    cpu = Session()
    factories = []

    def factory(path, providers):
        factories.append(providers)
        return cpu

    monkeypatch.setattr(ort, "InferenceSession", factory)
    model = restorer()
    model.session = Exhausted()
    image = np.full((256, 256, 3), 120, np.uint8)
    eyes = np.ones(image.shape[:2], np.float32)
    visible = eyes.copy()
    visible[:, 128:] = 0
    output = model.restore(image, FFHQ / 2, eyes, visible)
    assert factories == [["CPUExecutionProvider"]]
    assert cpu.calls == 1
    assert np.array_equal(output[:, 128:], image[:, 128:])


def test_non_memory_model_error_is_not_silently_retried():
    class Broken(Session):
        def run(self, *args):
            raise ValueError("bad model input")

    model = restorer()
    model.session = Broken()
    image = np.full((256, 256, 3), 120, np.uint8)
    with pytest.raises(ValueError, match="bad model input"):
        model.restore(image, FFHQ / 2, np.ones(image.shape[:2]), np.ones(image.shape[:2]))
