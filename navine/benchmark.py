import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.device_manager import device_summary, get_device


def _logs_path() -> Path:
    from navine.utils.paths import get_project_root

    root = get_project_root()
    path = root / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / "benchmark.jsonl"


def benchmark_text(tokens: int = 64) -> Dict[str, Any]:
    from navine.text.infer import generate

    start = time.perf_counter()
    try:
        generate("Navine AI - Python benchmark prompt.", max_new_tokens=tokens)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return {"module": "text", "tokens": tokens, "seconds": elapsed, "tokens_per_sec": tokens / elapsed}
    except Exception as exc:
        return {"module": "text", "error": str(exc)}


def benchmark_image() -> Dict[str, Any]:
    from navine.image.infer import generate

    start = time.perf_counter()
    try:
        out = generate("benchmark test shape", force_procedural=True, auto_ingest=False)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return {"module": "image", "seconds": elapsed, "images_per_min": 60.0 / elapsed, "output": str(out)}
    except Exception as exc:
        return {"module": "image", "error": str(exc)}


def run_benchmark(modules: Optional[str] = None) -> Dict[str, Any]:
    dev, summary = device_summary()
    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(dev),
        "hardware": summary.get("hardware"),
        "results": [],
    }
    selected = modules or "all"
    if selected in ("all", "text"):
        results["results"].append(benchmark_text())
    if selected in ("all", "image"):
        results["results"].append(benchmark_image())
    path = _logs_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(results) + "\n")
    return results
