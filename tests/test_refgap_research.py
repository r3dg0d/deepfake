import numpy as np

from deepfake.experimental.refgap import correct_reference_logits


def test_two_sided_attention_preserves_within_group_ratios():
    logits = np.random.default_rng(7).normal(size=(2, 6, 8))
    reference = np.array([True, True, False, False, False, False, False, False])
    edit = np.array([True, True, True, False, False, False])
    corrected = correct_reference_logits(logits, reference, edit, edit_strength=0.5, keep_strength=0.3)

    def probability(a):
        e = np.exp(a - a.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    before, after = probability(logits), probability(corrected)
    assert np.all(after[:, edit, :2].sum(-1) > before[:, edit, :2].sum(-1))
    assert np.all(after[:, ~edit, :2].sum(-1) < before[:, ~edit, :2].sum(-1))
    assert np.allclose(after[..., 0] / after[..., 1], before[..., 0] / before[..., 1])
    assert np.array_equal(corrected[..., 2:], logits[..., 2:])
    disabled = correct_reference_logits(logits, reference, edit, edit_strength=0, keep_strength=0)
    assert np.array_equal(disabled, logits)
