from __future__ import annotations

from typing import Optional

_NATIVE_AVAILABLE = False
_BACKEND = "python"


def _try_load_native() -> bool:
    global _NATIVE_AVAILABLE, _BACKEND
    try:
        from navine.utils.brand import is_python_only

        if is_python_only():
            _NATIVE_AVAILABLE = False
            _BACKEND = "python"
            return False
    except Exception:
        pass
    try:
        from navine.engine.kernels import _attention_kernel  # noqa: F401
        from navine.engine.kernels import _fast_ops  # noqa: F401
        _NATIVE_AVAILABLE = True
        _BACKEND = "mixed"
        return True
    except ImportError:
        _NATIVE_AVAILABLE = False
        _BACKEND = "python"
        return False


def native_available() -> bool:
    return _NATIVE_AVAILABLE


def get_backend() -> str:
    return _BACKEND


def fused_attention(
    q, k, v,
    scale: Optional[float] = None,
    causal: bool = True,
):
    if _NATIVE_AVAILABLE:
        from navine.engine.kernels._attention_kernel import fused_sdpa
        return fused_sdpa(q, k, v, scale=scale, causal=causal)
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def fast_rope(x, cos, sin):
    if _NATIVE_AVAILABLE:
        from navine.engine.kernels._fast_ops import apply_rope
        return apply_rope(x, cos, sin)
    x1, x2 = x[..., ::2], x[..., 1::2]
    import torch
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def fast_rms_norm(x, weight, eps: float = 1e-6):
    if _NATIVE_AVAILABLE:
        from navine.engine.kernels._fast_ops import rms_norm
        return rms_norm(x, weight, eps)
    import torch
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm * weight


_try_load_native()
