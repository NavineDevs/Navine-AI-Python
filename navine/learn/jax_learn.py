from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_project_root


def jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except Exception:
        return False


def jax_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "installed": jax_available(),
        "devices": [],
        "version": None,
        "backend": None,
    }
    if not info["installed"]:
        info["hint"] = "pip install -U jax jaxlib"
        return info
    import jax

    info["version"] = str(getattr(jax, "__version__", "unknown"))
    try:
        info["devices"] = [str(d) for d in jax.devices()]
        info["backend"] = str(jax.default_backend())
    except Exception as exc:
        info["error"] = str(exc)
    return info


def install_jax() -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pip", "install", "-U", "jax", "jaxlib"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "ok": proc.returncode == 0 and jax_available(),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
        "status": jax_status(),
    }


def demo_jax() -> Dict[str, Any]:
    if not jax_available():
        return {"ok": False, "error": "JAX not installed", "status": jax_status()}
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (4, 3))
    x = jnp.ones((3,))
    y = jnp.tanh(w @ x)

    def loss_fn(weights):
        pred = jnp.tanh(weights @ x)
        return jnp.mean((pred - 0.5) ** 2)

    grad = jax.grad(loss_fn)(w)
    return {
        "ok": True,
        "backend": str(jax.default_backend()),
        "matmul_out": [float(v) for v in y],
        "grad_norm": float(jnp.linalg.norm(grad)),
        "note": "Navine can run JAX matmul/grad demos and use JAX in the racing AI net when installed.",
    }


def jax_corpus_blocks() -> List[str]:
    return [
        "### User: what is JAX\n"
        "### Assistant: JAX is a NumPy-style library for high-performance numerical computing with "
        "automatic differentiation and JIT compilation. Navine can install and use it for experiments "
        "alongside PyTorch.",
        "### User: how do I use JAX with Navine\n"
        "### Assistant: Run: python -m navine.cli learn jax install. Then learn jax demo. "
        "The 2D racing AI uses JAX for network math when jax is installed, otherwise NumPy.",
        "### User: jax vs pytorch for navine\n"
        "### Assistant: PyTorch is Navine's main training stack for text/image/video. "
        "JAX is supported for research-style autodiff, small models, and the racing genetic AI forward pass.",
        "### User: write a tiny jax neural forward pass\n"
        "### Assistant: import jax.numpy as jnp\n"
        "def forward(w1, b1, w2, b2, x):\n"
        "    h = jnp.tanh(x @ w1 + b1)\n"
        "    return jnp.tanh(h @ w2 + b2)",
    ]


def ingest_jax_learning() -> Dict[str, Any]:
    root = get_project_root()
    train_dir = root / "data" / "train" / "coding"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / "jax_navine.txt"
    train_path.write_text("\n\n".join(jax_corpus_blocks()) + "\n", encoding="utf-8")

    pages = root / "data" / "learn" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    page = {
        "url": "navine://learn/jax",
        "title": "Navine JAX learning guide",
        "text": (
            "JAX provides numpy APIs, grad, jit, and vmap. "
            "Navine primary training stays on PyTorch, but JAX is available for demos, "
            "racing AI matrix math, and coding practice. "
            "Install with pip install jax jaxlib. "
            "Commands: navine learn jax status|install|demo|ingest."
        ),
        "content_type": "navine/jax",
        "source": "jax_learn",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "navine_modality": "code",
        "helpful_for_navine": "high",
        "study_notes": "JAX autodiff/JIT complements Navine coding and experimental AI paths.",
    }
    page_path = pages / "jax_navine_guide.json"
    page_path.write_text(json.dumps(page, indent=2), encoding="utf-8")

    rag_id = None
    try:
        from navine.learn.rag import index_document

        rag_id = index_document(
            page["text"],
            source="jax_learn",
            title=page["title"],
            url=page["url"],
        )
    except Exception:
        pass

    return {
        "train_path": str(train_path),
        "page_path": str(page_path),
        "rag_id": rag_id,
        "status": jax_status(),
    }


def study_jax(reindex: bool = True) -> Dict[str, Any]:
    result = ingest_jax_learning()
    if reindex:
        try:
            from navine.learn.rag import rebuild_index

            result["rag_docs"] = rebuild_index()
        except Exception as exc:
            result["rag_error"] = str(exc)
    demo = demo_jax() if jax_available() else {"ok": False, "skipped": True}
    result["demo"] = demo
    return result
