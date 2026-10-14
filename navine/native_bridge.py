"""Optional native accelerators with pure-Python fallbacks."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

_navops = None
_navcuda = None
_navops_error: Optional[str] = None
_navcuda_error: Optional[str] = None


def _try_import_navops() -> Any:
    global _navops, _navops_error
    if _navops is not None or _navops_error is not None:
        return _navops
    try:
        from navine.utils.brand import is_python_only

        if is_python_only():
            _navops_error = "disabled (python-only site)"
            return None
    except Exception:
        pass
    try:
        import navops as mod  # type: ignore

        _navops = mod
        return _navops
    except Exception as exc:
        _navops_error = str(exc)
        return None


def _try_import_navcuda() -> Any:
    global _navcuda, _navcuda_error
    if _navcuda is not None or _navcuda_error is not None:
        return _navcuda
    try:
        from navine.utils.brand import is_python_only

        if is_python_only():
            _navcuda_error = "disabled (python-only site)"
            return None
    except Exception:
        pass
    try:
        import navcuda as mod  # type: ignore

        _navcuda = mod
        return _navcuda
    except Exception as exc:
        _navcuda_error = str(exc)
        return None


def native_status() -> dict:
    ops = _try_import_navops()
    cuda = _try_import_navcuda()
    return {
        "navops": bool(ops),
        "navops_error": _navops_error,
        "navcuda": bool(cuda),
        "navcuda_error": _navcuda_error,
    }


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    ops = _try_import_navops()
    if ops is not None and hasattr(ops, "matmul"):
        return ops.matmul(a, b)
    if not a or not b:
        return []
    cols = len(b[0])
    out: List[List[float]] = []
    for row in a:
        out_row = []
        for j in range(cols):
            s = 0.0
            for k, av in enumerate(row):
                s += float(av) * float(b[k][j])
            out_row.append(s)
        out.append(out_row)
    return out


class KvCache:
    def __init__(self, capacity: int = 256) -> None:
        ops = _try_import_navops()
        if ops is not None and hasattr(ops, "KvCache"):
            self._impl = ops.KvCache(capacity)
            self._py = None
        else:
            self._impl = None
            self._py: List[Tuple[Any, Any]] = []
            self._capacity = max(1, int(capacity))

    def push(self, key: Any, value: Any) -> None:
        if self._impl is not None:
            self._impl.push(key, value)
            return
        assert self._py is not None
        self._py.append((key, value))
        if len(self._py) > self._capacity:
            self._py = self._py[-self._capacity :]

    def len(self) -> int:
        if self._impl is not None:
            return int(self._impl.len())
        return len(self._py or [])

    def last(self) -> Optional[Tuple[Any, Any]]:
        if self._impl is not None:
            return self._impl.last()
        if not self._py:
            return None
        return self._py[-1]


def make_betas(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    schedule: str = "linear",
) -> List[float]:
    cuda = _try_import_navcuda()
    if cuda is not None and hasattr(cuda, "make_betas"):
        return list(cuda.make_betas(int(timesteps), float(beta_start), float(beta_end), str(schedule)))
    import math

    t = max(1, int(timesteps))
    schedule = (schedule or "linear").lower()
    if schedule == "cosine":
        s = 0.008
        def alpha_bar(u: float) -> float:
            return math.cos((u + s) / (1.0 + s) * math.pi * 0.5) ** 2
        betas = []
        for i in range(t):
            t1 = i / t
            t2 = (i + 1) / t
            a1 = max(alpha_bar(t1), 1e-8)
            a2 = max(alpha_bar(t2), 1e-8)
            betas.append(min(1.0 - a2 / a1, 0.999))
        return betas
    if schedule == "sigmoid":
        xs = [i / max(t - 1, 1) for i in range(t)]
        def sig(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
        lo, hi = sig(0.0), sig(1.0)
        return [
            float(beta_start + (beta_end - beta_start) * ((sig(x) - lo) / (hi - lo + 1e-12)))
            for x in xs
        ]
    if t == 1:
        return [float(beta_start)]
    return [float(beta_start + (beta_end - beta_start) * i / (t - 1)) for i in range(t)]


def ddim_step_tensors(
    x,
    alpha_bar_t,
    alpha_bar_prev,
    pred_noise,
):
    cuda = _try_import_navcuda()
    if cuda is not None and hasattr(cuda, "ddim_step"):
        try:
            return cuda.ddim_step(x, alpha_bar_t, alpha_bar_prev, pred_noise)
        except Exception:
            pass
    import torch

    ab_t = alpha_bar_t if isinstance(alpha_bar_t, torch.Tensor) else torch.tensor(alpha_bar_t, device=x.device, dtype=x.dtype)
    ab_p = alpha_bar_prev if isinstance(alpha_bar_prev, torch.Tensor) else torch.tensor(alpha_bar_prev, device=x.device, dtype=x.dtype)
    pred_x0 = (x - torch.sqrt(1.0 - ab_t) * pred_noise) / torch.sqrt(ab_t + 1e-8)
    pred_x0 = pred_x0.clamp(-1.0, 1.0)
    return torch.sqrt(ab_p) * pred_x0 + torch.sqrt(1.0 - ab_p) * pred_noise
