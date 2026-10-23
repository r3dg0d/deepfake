import numpy as np
import pytest


def test_sdpa_padding_matches_explicit_attention():
    torch = pytest.importorskip("torch")
    from deepfake.engines.experimental.attention_sdpa import sdpa_attention

    generator = torch.Generator().manual_seed(4)
    q, k, v = [torch.randn(1, n, 2, 4, generator=generator, dtype=torch.bfloat16) for n in (3, 5, 5)]
    result = sdpa_attention(q, k, v, q_lens=torch.tensor([2]), k_lens=torch.tensor([3])).float().numpy()
    qn, kn, vn = [x.float().numpy()[0] for x in (q, k, v)]
    for head in range(2):
        logits = qn[:2, head] @ kn[:3, head].T / 2
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected = probabilities @ vn[:3, head]
        np.testing.assert_allclose(result[0, :2, head], expected, atol=0.015, rtol=0.015)
    np.testing.assert_array_equal(result[0, 2], np.zeros((2, 4)))
    with pytest.raises(ValueError, match="non-causal"):
        sdpa_attention(q, k, v, causal=True)
