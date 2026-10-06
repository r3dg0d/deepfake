"""Standalone RefGAP equations 1, 2, 4, 5 (arXiv:2609.35708).

Numerical research prototype only. AlphaFace has no compatible joint reference
attention hook. Layer profiling/selection and diffusion integration are absent.
"""

import numpy as np


def correct_reference_logits(logits, reference_keys, edit_queries, *, edit_strength, keep_strength):
    """Correct one layer's [heads, queries, keys] logits in float64.

    Strengths must be explicitly provided; no unverified model calibration is
    implied. Every query here must be a content query, not a reference query.
    """
    scores = np.asarray(logits, np.float64)
    reference = np.asarray(reference_keys, bool)
    edit = np.asarray(edit_queries, bool)
    if scores.ndim != 3 or reference.shape != (scores.shape[2],) or edit.shape != (scores.shape[1],):
        raise ValueError("expected matching attention logits and token masks")
    if not np.isfinite(scores).all() or not np.isfinite([edit_strength, keep_strength]).all():
        raise ValueError("finite logits and strengths required")
    if min(edit_strength, keep_strength) < 0 or not reference.any() or reference.all():
        raise ValueError("nonnegative strengths and both reference/rest keys required")
    probability = np.exp(scores - scores.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    mass = probability[..., reference].sum(-1)
    shifts = np.zeros(scores.shape[1], np.float64)
    for region, strength, sign in ((edit, edit_strength, 1), (~edit, keep_strength, -1)):
        if region.any():
            mean_mass = np.clip(mass[:, region].mean(), 1e-8, 1 - 1e-8)
            shifts[region] = sign * strength * np.log(1 / mean_mass)
    return scores + shifts[None, :, None] * reference[None, None, :]
