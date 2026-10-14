from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.image.procedural import is_bland_like, is_noise_like


def score_image_path(image_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(image_path),
        "score": 0.0,
        "bland": True,
        "noise": True,
        "std": 0.0,
        "sharpness": 0.0,
        "passes": False,
    }
    if not image_path.exists():
        return result
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        if arr.size == 0:
            return result
        std = float(arr.std())
        mean = float(arr.mean())
        bland = is_bland_like(image_path)
        noise = is_noise_like(image_path)
        gray = abs(mean - 127.5) < 8.0 and std < 12.0
        gray_arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        lap_k = (
            -4.0 * gray_arr[1:-1, 1:-1]
            + gray_arr[:-2, 1:-1]
            + gray_arr[2:, 1:-1]
            + gray_arr[1:-1, :-2]
            + gray_arr[1:-1, 2:]
        )
        lap_var = float(lap_k.var()) if lap_k.size else 0.0
        lap = arr[1:, :, :] - arr[:-1, :, :]
        lap_v = float(np.abs(lap).mean())
        sharpness = lap_v
        h, w = gray_arr.shape
        quads = [
            gray_arr[: h // 2, : w // 2],
            gray_arr[: h // 2, w // 2 :],
            gray_arr[h // 2 :, : w // 2],
            gray_arr[h // 2 :, w // 2 :],
        ]
        quad_stds = [float(q.std()) for q in quads if q.size]
        structure = float(np.std(quad_stds)) if len(quad_stds) >= 4 else 0.0
        center = arr[arr.shape[0] // 4: 3 * arr.shape[0] // 4, arr.shape[1] // 4: 3 * arr.shape[1] // 4]
        center_std = float(center.std()) if center.size else 0.0
        hist, _ = np.histogram(arr.reshape(-1, 3).mean(axis=1), bins=32)
        entropy = float(-np.sum((hist / max(hist.sum(), 1)) * np.log((hist / max(hist.sum(), 1)) + 1e-8)))

        score = 0.0
        if not bland and not gray:
            score += 0.22
        if not noise:
            score += 0.18
        if 22.0 <= std <= 75.0:
            score += 0.14
        elif std > 18.0:
            score += 0.06
        if center_std >= 18.0:
            score += 0.12
        if 6.0 <= sharpness <= 55.0:
            score += 0.08
        if entropy > 2.0:
            score += 0.06
        if lap_var >= 35.0:
            score += 0.12
        elif lap_var >= 18.0:
            score += 0.05
        if structure >= 4.5:
            score += 0.12
        elif structure >= 2.0:
            score += 0.04
        score = max(0.0, min(1.0, score))
        blob_like = (
            lap_var < 18.0
            or (lap_var < 25.0 and sharpness < 2.5)
            or (lap_var < 30.0 and structure < 3.0 and sharpness < 3.5)
        )
        passes = (
            score >= 0.68
            and not bland
            and not gray
            and not blob_like
            and std >= 20.0
            and lap_var >= 20.0
            and structure >= 3.0
        )
        result.update(
            {
                "score": round(score, 4),
                "bland": bland,
                "noise": noise,
                "grey": gray,
                "std": round(std, 3),
                "mean": round(mean, 3),
                "sharpness": round(sharpness, 3),
                "center_std": round(center_std, 3),
                "entropy": round(entropy, 3),
                "lap_var": round(lap_var, 3),
                "structure": round(structure, 3),
                "blob_like": blob_like,
                "passes": passes,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def category_quality_pass(metric: Dict[str, Any], category: str) -> bool:
    cat = str(category).lower()
    if cat in ("hentai", "anime"):
        if metric.get("bland") or metric.get("blob_like"):
            return False
        if float(metric.get("score", 0)) < 0.65:
            return False
        if float(metric.get("lap_var", 0)) < 18.0:
            return False
        if float(metric.get("structure", 0)) < 3.0:
            return False
        if float(metric.get("sharpness", 0)) < 2.0:
            return False
        if float(metric.get("std", 0)) < 18.0:
            return False
        return True
    if cat in ("real", "real_human", "human", "porn"):
        if not metric.get("passes"):
            return False
        if metric.get("bland") or metric.get("blob_like"):
            return False
        if float(metric.get("score", 0)) < 0.68:
            return False
        if float(metric.get("lap_var", 0)) < 20.0:
            return False
        if float(metric.get("std", 0)) < 20.0:
            return False
        return True
    return bool(metric.get("passes"))


def score_batch(paths: List[Path]) -> Dict[str, Any]:
    scores = [score_image_path(path) for path in paths]
    avg = sum(s["score"] for s in scores) / max(len(scores), 1)
    return {
        "average_score": round(avg, 4),
        "all_pass": bool(scores) and all(s.get("passes") for s in scores),
        "results": scores,
    }
