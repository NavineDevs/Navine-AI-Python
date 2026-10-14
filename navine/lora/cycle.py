from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.config import load_config
from navine.utils.lora_core import smoke_score
from navine.utils.paths import get_project_root


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    log_path = get_project_root() / "logs" / "lora_cycle.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def _merge_cfg(modality: str) -> dict:
    name = "image_lora" if modality == "image" else "video_lora"
    cfg = load_config(name)
    return dict(cfg.get("merge") or {})


def _should_merge(
    base_results: dict,
    lora_results: dict,
    merge_cfg: dict,
) -> bool:
    min_ratio = float(merge_cfg.get("min_valid_ratio") or 0.5)
    require_improvement = bool(merge_cfg.get("require_improvement", True))
    min_delta = float(merge_cfg.get("min_improvement_delta") or 0.0)
    base_score, base_valid, total = smoke_score(base_results)
    lora_score, lora_valid, _ = smoke_score(lora_results)
    _log(
        f"eval base={base_score:.2f} ({base_valid}/{total}) "
        f"lora={lora_score:.2f} ({lora_valid}/{total})"
    )
    if lora_score < min_ratio:
        _log(f"skip merge: lora score {lora_score:.2f} < min {min_ratio:.2f}")
        return False
    if require_improvement and lora_score + min_delta <= base_score:
        _log(
            f"skip merge: lora {lora_score:.2f} did not beat base {base_score:.2f} "
            f"(delta={min_delta:.2f})"
        )
        return False
    return True


def run_image_lora_cycle(steps: Optional[int] = None, force_merge: bool = False) -> Dict[str, Any]:
    from navine.device_manager import get_device
    from navine.image.lora_custom import (
        adapter_exists,
        evaluate_with_lora,
        merge_lora_into_base,
        remove_adapter,
        train_custom_lora,
    )
    from navine.image.smoke_eval import run_image_smoke_eval
    from navine.utils.tier import load_modality_config

    merge_cfg = _merge_cfg("image")
    config = load_modality_config("image")
    device = get_device()
    _log("IMAGE_LORA_CYCLE_BEGIN")
    base_results = run_image_smoke_eval(config=config, device=device)
    report = train_custom_lora(steps=steps)
    if not adapter_exists():
        _log("IMAGE_LORA_CYCLE_FAIL no adapter after train")
        return {"ok": False, "stage": "train", "train": report}
    lora_results = evaluate_with_lora(config=config, device=device)
    if force_merge or _should_merge(base_results, lora_results, merge_cfg):
        merged = merge_lora_into_base()
        post_results = run_image_smoke_eval(config=config, device=device)
        post_score, _, _ = smoke_score(post_results)
        min_ratio = float(merge_cfg.get("min_valid_ratio") or 0.5)
        if post_score >= min_ratio:
            remove_adapter()
            _log(f"IMAGE_LORA_MERGED score={post_score:.2f} adapter removed")
            return {
                "ok": True,
                "merged": True,
                "adapter_removed": True,
                "post_score": post_score,
                "train": report,
                "merge": merged,
            }
        _log(f"IMAGE_LORA_MERGE_ROLLBACK post_score={post_score:.2f} < {min_ratio:.2f}")
        backup = merged.get("backup_path")
        if backup:
            shutil.copy2(backup, merged.get("checkpoint_path"))
        return {
            "ok": False,
            "merged": False,
            "reason": "post_merge_quality_low",
            "post_score": post_score,
            "train": report,
        }
    _log("IMAGE_LORA_CYCLE_KEEP adapter kept for more training")
    return {
        "ok": True,
        "merged": False,
        "adapter_removed": False,
        "train": report,
        "base_results": base_results,
        "lora_results": lora_results,
    }


def run_video_lora_cycle(steps: Optional[int] = None, force_merge: bool = False) -> Dict[str, Any]:
    from navine.device_manager import get_device
    from navine.video.lora_custom import (
        adapter_exists,
        evaluate_with_lora,
        merge_lora_into_base,
        remove_adapter,
        train_video_lora,
    )
    from navine.video.smoke_eval import run_video_smoke_eval
    from navine.utils.tier import load_modality_config

    merge_cfg = _merge_cfg("video")
    config = load_modality_config("video")
    device = get_device()
    _log("VIDEO_LORA_CYCLE_BEGIN")
    base_results = run_video_smoke_eval(config=config)
    report = train_video_lora(steps=steps)
    if not adapter_exists():
        _log("VIDEO_LORA_CYCLE_FAIL no adapter after train")
        return {"ok": False, "stage": "train", "train": report}
    lora_results = evaluate_with_lora(config=config, device=device)
    if force_merge or _should_merge(base_results, lora_results, merge_cfg):
        merged = merge_lora_into_base()
        post_results = run_video_smoke_eval(config=config)
        post_score, _, _ = smoke_score(post_results)
        min_ratio = float(merge_cfg.get("min_valid_ratio") or 0.5)
        if post_score >= min_ratio:
            remove_adapter()
            _log(f"VIDEO_LORA_MERGED score={post_score:.2f} adapter removed")
            return {
                "ok": True,
                "merged": True,
                "adapter_removed": True,
                "post_score": post_score,
                "train": report,
                "merge": merged,
            }
        _log(f"VIDEO_LORA_MERGE_ROLLBACK post_score={post_score:.2f} < {min_ratio:.2f}")
        backup = merged.get("backup_path")
        if backup:
            shutil.copy2(backup, merged.get("checkpoint_path"))
        return {
            "ok": False,
            "merged": False,
            "reason": "post_merge_quality_low",
            "post_score": post_score,
            "train": report,
        }
    _log("VIDEO_LORA_CYCLE_KEEP adapter kept for more training")
    return {
        "ok": True,
        "merged": False,
        "adapter_removed": False,
        "train": report,
        "base_results": base_results,
        "lora_results": lora_results,
    }


def merge_image_lora(force_merge: bool = False) -> Dict[str, Any]:
    from navine.device_manager import get_device
    from navine.image.lora_custom import (
        adapter_exists,
        evaluate_with_lora,
        merge_lora_into_base,
        remove_adapter,
    )
    from navine.image.smoke_eval import run_image_smoke_eval
    from navine.utils.tier import load_modality_config

    if not adapter_exists():
        raise RuntimeError("No image LoRA adapter found.")
    merge_cfg = _merge_cfg("image")
    config = load_modality_config("image")
    device = get_device()
    base_results = run_image_smoke_eval(config=config, device=device)
    lora_results = evaluate_with_lora(config=config, device=device)
    if not force_merge and not _should_merge(base_results, lora_results, merge_cfg):
        return {
            "ok": False,
            "merged": False,
            "reason": "quality_gate",
            "base_results": base_results,
            "lora_results": lora_results,
        }
    merged = merge_lora_into_base()
    post_results = run_image_smoke_eval(config=config, device=device)
    post_score, _, _ = smoke_score(post_results)
    min_ratio = float(merge_cfg.get("min_valid_ratio") or 0.5)
    if post_score >= min_ratio or force_merge:
        remove_adapter()
        return {
            "ok": True,
            "merged": True,
            "adapter_removed": True,
            "post_score": post_score,
            "merge": merged,
        }
    backup = merged.get("backup_path")
    if backup:
        shutil.copy2(backup, merged.get("checkpoint_path"))
    return {"ok": False, "merged": False, "reason": "post_merge_quality_low", "post_score": post_score}


def merge_video_lora(force_merge: bool = False) -> Dict[str, Any]:
    from navine.device_manager import get_device
    from navine.video.lora_custom import (
        adapter_exists,
        evaluate_with_lora,
        merge_lora_into_base,
        remove_adapter,
    )
    from navine.video.smoke_eval import run_video_smoke_eval
    from navine.utils.tier import load_modality_config

    if not adapter_exists():
        raise RuntimeError("No video LoRA adapter found.")
    merge_cfg = _merge_cfg("video")
    config = load_modality_config("video")
    device = get_device()
    base_results = run_video_smoke_eval(config=config)
    lora_results = evaluate_with_lora(config=config, device=device)
    if not force_merge and not _should_merge(base_results, lora_results, merge_cfg):
        return {
            "ok": False,
            "merged": False,
            "reason": "quality_gate",
            "base_results": base_results,
            "lora_results": lora_results,
        }
    merged = merge_lora_into_base()
    post_results = run_video_smoke_eval(config=config)
    post_score, _, _ = smoke_score(post_results)
    min_ratio = float(merge_cfg.get("min_valid_ratio") or 0.34)
    if post_score >= min_ratio or force_merge:
        remove_adapter()
        return {
            "ok": True,
            "merged": True,
            "adapter_removed": True,
            "post_score": post_score,
            "merge": merged,
        }
    backup = merged.get("backup_path")
    if backup:
        shutil.copy2(backup, merged.get("checkpoint_path"))
    return {"ok": False, "merged": False, "reason": "post_merge_quality_low", "post_score": post_score}


def run_lora_cycle(modality: str, steps: Optional[int] = None, force_merge: bool = False) -> Dict[str, Any]:
    key = str(modality).strip().lower()
    if key in ("image", "img"):
        return run_image_lora_cycle(steps=steps, force_merge=force_merge)
    if key in ("video", "vid"):
        return run_video_lora_cycle(steps=steps, force_merge=force_merge)
    raise ValueError(f"Unknown modality: {modality}")
