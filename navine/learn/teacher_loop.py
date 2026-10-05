import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from navine.image.quality import score_image_path
from navine.learn.from_external import (
    DEFAULT_PROMPTS,
    batch_generate,
    finetune_from_external,
    ingest_external_for_training,
)
from navine.utils.config import load_config
from navine.utils.paths import get_output_dir, get_project_root
from navine.utils.tier import resolve_checkpoint_dir
from navine.utils.training_lock import (
    clear_stale_lock,
    training_lock,
    training_lock_active,
    wait_for_training_unlock,
)

ProgressCallback = Optional[Callable[[str], None]]

IMAGE_CONFIG = "image_enterprise"
VIDEO_CONFIG = "video_enterprise"
DEFAULT_SCORE_GAP = 0.08
DEFAULT_LAP_GAP = 15.0
DEFAULT_MIN_CUSTOM_SCORE = 0.65
EVAL_PROMPT_COUNT = 4


def _log_dir() -> Path:
    path = get_project_root() / "logs" / "teacher_loop"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(msg: str, progress: ProgressCallback = None) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    with (_log_dir() / "teacher_loop.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if progress:
        progress(line)


def _eval_prompts() -> List[str]:
    prompts = list(DEFAULT_PROMPTS)
    real = [p for p in prompts if "hentai" not in p.lower() and "anime" not in p.lower()]
    anime = [p for p in prompts if "hentai" in p.lower() or "anime" in p.lower()]
    selected: List[str] = []
    for group in (real, anime):
        for item in group:
            if item not in selected:
                selected.append(item)
            if len(selected) >= EVAL_PROMPT_COUNT:
                return selected[:EVAL_PROMPT_COUNT]
    return selected[:EVAL_PROMPT_COUNT] if selected else prompts[:EVAL_PROMPT_COUNT]


def _image_ckpt_path() -> Path:
    cfg = load_config(IMAGE_CONFIG)
    return resolve_checkpoint_dir("image", cfg) / "latest.pt"


def _generate_custom(prompt: str, out_path: Path, seed: Optional[int] = None) -> bool:
    from navine.image.infer import generate

    ckpt = _image_ckpt_path()
    ckpt_arg = str(ckpt) if ckpt.exists() else None
    try:
        generate(
            prompt,
            output_path=str(out_path),
            enhance=True,
            auto_ingest=False,
            config_name=IMAGE_CONFIG,
            checkpoint=ckpt_arg,
            seed=seed,
        )
        return out_path.exists()
    except Exception:
        return False


def _generate_teacher(prompt: str, out_path: Path, seed: Optional[int] = None, model: Optional[str] = None) -> bool:
    from navine.image.backends import generate_from_diffusers
    from navine.image.external import apply_teacher_model, load_external_config
    from navine.image.prompts import enhance_prompt

    config = apply_teacher_model(load_external_config(), profile_name=model)
    effective = enhance_prompt(prompt.strip())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return generate_from_diffusers(effective, out_path, config, seed=seed)


def compare_custom_vs_teacher(
    prompts: Optional[List[str]] = None,
    model: Optional[str] = None,
    seed_base: int = 42,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    items = prompts or _eval_prompts()
    out_dir = get_output_dir("image") / "teacher_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(items):
        seed = seed_base + idx
        teacher_path = out_dir / f"teacher_{idx:02d}.png"
        custom_path = out_dir / f"custom_{idx:02d}.png"
        if progress:
            progress(f"Compare {idx + 1}/{len(items)}: teacher")
        teacher_ok = _generate_teacher(prompt, teacher_path, seed=seed, model=model)
        if progress:
            progress(f"Compare {idx + 1}/{len(items)}: custom")
        custom_ok = _generate_custom(prompt, custom_path, seed=seed)
        teacher_metric = score_image_path(teacher_path) if teacher_ok else {"score": 0.0, "lap_var": 0.0}
        custom_metric = score_image_path(custom_path) if custom_ok else {"score": 0.0, "lap_var": 0.0}
        score_gap = float(teacher_metric.get("score", 0)) - float(custom_metric.get("score", 0))
        lap_gap = float(teacher_metric.get("lap_var", 0)) - float(custom_metric.get("lap_var", 0))
        entry = {
            "prompt": prompt,
            "seed": seed,
            "teacher_path": str(teacher_path) if teacher_ok else None,
            "custom_path": str(custom_path) if custom_ok else None,
            "teacher_score": teacher_metric.get("score", 0),
            "custom_score": custom_metric.get("score", 0),
            "teacher_lap_var": teacher_metric.get("lap_var", 0),
            "custom_lap_var": custom_metric.get("lap_var", 0),
            "score_gap": round(score_gap, 4),
            "lap_var_gap": round(lap_gap, 4),
        }
        pairs.append(entry)
    valid = [p for p in pairs if p.get("teacher_path") and p.get("custom_path")]
    avg_score_gap = sum(abs(p["score_gap"]) for p in valid) / max(len(valid), 1)
    avg_lap_gap = sum(abs(p["lap_var_gap"]) for p in valid) / max(len(valid), 1)
    avg_custom_score = sum(float(p["custom_score"]) for p in valid) / max(len(valid), 1)
    return {
        "pairs": pairs,
        "average_score_gap": round(avg_score_gap, 4),
        "average_lap_var_gap": round(avg_lap_gap, 4),
        "average_custom_score": round(avg_custom_score, 4),
        "prompt_count": len(items),
        "valid_pairs": len(valid),
    }


def _converged(
    comparison: Dict[str, Any],
    score_gap_threshold: float,
    lap_gap_threshold: float,
    min_custom_score: float,
) -> bool:
    if comparison.get("valid_pairs", 0) < 1:
        return False
    return (
        float(comparison.get("average_score_gap", 1.0)) <= score_gap_threshold
        and float(comparison.get("average_lap_var_gap", 999.0)) <= lap_gap_threshold
        and float(comparison.get("average_custom_score", 0.0)) >= min_custom_score
    )


def _wait_for_unlocks(progress: ProgressCallback = None) -> None:
    for name in ("image", "video"):
        clear_stale_lock(name)
        if training_lock_active(name):
            _log(f"Waiting for {name} training lock", progress)
            wait_for_training_unlock(name, poll_seconds=60, progress=progress)


def _finetune_video(steps: Optional[int] = None, progress: ProgressCallback = None) -> None:
    video_learn = get_project_root() / "data" / "learn" / "video" / "external"
    if not any(video_learn.glob("*.mp4")):
        return
    _wait_for_unlocks(progress)
    with training_lock("video"):
        from navine.video.train import train as train_video

        cfg = load_config(VIDEO_CONFIG)
        finetune_steps = steps or int(cfg.get("training", {}).get("finetune_steps", 300))
        _log(f"Video finetune {finetune_steps} steps", progress)
        train_video(config_path=VIDEO_CONFIG, finetune=True, finetune_steps=finetune_steps)


def run_teacher_loop(
    count: int = 5,
    max_cycles: int = 20,
    model: Optional[str] = None,
    include_video: bool = False,
    finetune_steps: Optional[int] = None,
    score_gap_threshold: float = DEFAULT_SCORE_GAP,
    lap_gap_threshold: float = DEFAULT_LAP_GAP,
    min_custom_score: float = DEFAULT_MIN_CUSTOM_SCORE,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = load_config(IMAGE_CONFIG)
    steps = finetune_steps or int((cfg.get("training") or {}).get("finetune_steps", 400))
    report: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "count_per_cycle": count,
        "max_cycles": max_cycles,
        "model": model,
        "include_video": include_video,
        "finetune_steps": steps,
        "score_gap_threshold": score_gap_threshold,
        "lap_gap_threshold": lap_gap_threshold,
        "cycles": [],
    }
    _log(
        f"Navine AI - Python teacher loop start count={count} max_cycles={max_cycles} model={model or 'rotation'}",
        progress,
    )
    for cycle in range(1, max_cycles + 1):
        cycle_info: Dict[str, Any] = {"cycle": cycle}
        _wait_for_unlocks(progress)
        _log(f"Cycle {cycle} phase A: generate {count} teacher samples", progress)
        cycle_info["phase_a"] = batch_generate(
            count=count,
            include_video=include_video,
            model=model,
            cycle_index=cycle - 1,
            progress=progress,
        )
        _log(f"Cycle {cycle} phase B: ingest teacher data", progress)
        ingest_path = ingest_external_for_training()
        cycle_info["phase_b"] = {"ingest_dir": ingest_path}
        _wait_for_unlocks(progress)
        _log(f"Cycle {cycle} phase C: finetune image_enterprise ({steps} steps)", progress)
        with training_lock("image"):
            finetune_from_external(finetune_steps=steps, config_path=IMAGE_CONFIG)
        cycle_info["phase_c"] = {"image_finetune_steps": steps}
        if include_video:
            _finetune_video(progress=progress)
            cycle_info["phase_c"]["video_finetune"] = True
        _log(f"Cycle {cycle} phase D: compare custom vs teacher", progress)
        comparison = compare_custom_vs_teacher(model=model, progress=progress)
        cycle_info["phase_d"] = comparison
        cycle_info["converged"] = _converged(
            comparison, score_gap_threshold, lap_gap_threshold, min_custom_score
        )
        report["cycles"].append(cycle_info)
        report_path = _log_dir() / f"cycle_{cycle:02d}.json"
        report_path.write_text(json.dumps(cycle_info, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(
            f"Cycle {cycle} gap score={comparison.get('average_score_gap')} "
            f"lap_var={comparison.get('average_lap_var_gap')} "
            f"custom_avg={comparison.get('average_custom_score')}",
            progress,
        )
        if cycle_info["converged"]:
            report["stopped"] = "converged"
            _log(f"Teacher loop converged at cycle {cycle}", progress)
            break
        if cycle >= max_cycles:
            report["stopped"] = "max_cycles"
            _log(f"Teacher loop reached max_cycles={max_cycles}", progress)
            break
        time.sleep(2)
    report["finished"] = datetime.now(timezone.utc).isoformat()
    final_path = _log_dir() / "teacher_loop_report.json"
    final_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Teacher loop report: {final_path}", progress)
    return report


if __name__ == "__main__":
    run_teacher_loop()
