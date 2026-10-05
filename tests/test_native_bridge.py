import math

import torch

from navine.image.schedules import make_beta_schedule, schedule_names
from navine.native_bridge import KvCache, ddim_step_tensors, make_betas, matmul, native_status


def test_schedule_names():
    assert "linear" in schedule_names()
    assert "cosine" in schedule_names()
    assert "sigmoid" in schedule_names()


def test_make_betas_python():
    for name in ("linear", "cosine", "sigmoid"):
        b = make_betas(16, 1e-4, 0.02, name)
        assert len(b) == 16
        assert all(0 < x < 1 for x in b)


def test_make_beta_schedule_tensor():
    t = make_beta_schedule(8, schedule="cosine")
    assert t.shape[0] == 8
    assert torch.isfinite(t).all()


def test_matmul_fallback():
    out = matmul([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
    assert out[0][0] == 19.0
    assert out[1][1] == 50.0


def test_kv_cache():
    cache = KvCache(2)
    cache.push("a", 1)
    cache.push("b", 2)
    cache.push("c", 3)
    assert cache.len() == 2
    assert cache.last()[0] == "c"


def test_ddim_step_shapes():
    x = torch.randn(1, 3, 8, 8)
    noise = torch.randn_like(x)
    y = ddim_step_tensors(x, 0.5, 0.4, noise)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_native_status_dict():
    status = native_status()
    assert "navops" in status
    assert "navcuda" in status

