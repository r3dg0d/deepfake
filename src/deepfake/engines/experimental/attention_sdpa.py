"""Dense SDPA implementation for the FlashAttention contract used by EditaLive.

Keeps padded keys out of softmax and zeros padded queries. No checkpoint,
conditioning or LoRA changes. Windowed/causal variants are deliberately rejected;
EditaLive's causal video cache and sparse branches are handled by its own code.
"""


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                   softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    if causal or tuple(window_size) != (-1, -1):
        raise ValueError("This EditaLive SDPA adapter requires dense non-causal attention")
    output_dtype = q.dtype
    dtype = dtype or torch.bfloat16
    half = (torch.float16, torch.bfloat16)
    q = q if q.dtype in half else q.to(dtype)
    k = k if k.dtype in half else k.to(dtype)
    v = v if v.dtype in half else v.to(dtype)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None and not bool(torch.all(k_lens == k.shape[1])):
        lengths = k_lens.to(device=k.device)
        mask = (torch.arange(k.shape[1], device=k.device)[None, :] < lengths[:, None])[:, None, None, :]
    result = scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale,
                                          enable_gqa=q.shape[2] != k.shape[2]).transpose(1, 2)
    if q_lens is not None:
        valid = torch.arange(q.shape[1], device=q.device)[None, :] < q_lens.to(q.device)[:, None]
        result = result * valid[:, :, None, None]
    return result.to(output_dtype)
