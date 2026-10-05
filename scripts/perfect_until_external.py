import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navine.image.external import (
    DEFAULT_HENTAI_PROMPTS,
    DEFAULT_PORN_PROMPTS,
    DEFAULT_REAL_PROMPTS,
    batch_generate_nsfw_images,
    diffusers_available,
)
from navine.learn.from_external import finetune_from_external, ingest_external_for_training
from navine.learn.teacher_loop import (
    DEFAULT_LAP_GAP,
    DEFAULT_MIN_CUSTOM_SCORE,
    DEFAULT_SCORE_GAP,
    IMAGE_CONFIG,
    _converged,
    _finetune_video,
    _wait_for_unlocks,
    compare_custom_vs_teacher,
)
from navine.utils.config import load_config
from navine.utils.paths import get_project_root
from navine.utils.training_lock import training_lock
from navine.video.external import batch_generate_nsfw_videos

ProgressCallback = Optional[Callable[[str], None]]

DEFAULT_MODEL = "sd_turbo"
DEFAULT_MAX_CYCLES = 50
DEFAULT_COUNT_PER_CATEGORY = 2
DEFAULT_VIDEO_COUNT = 1
EVAL_PER_CATEGORY = 2


def _log_dir() -> Path:
    path = get_project_root() / "logs" / "perfect_loop"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(msg: str, progress: ProgressCallback = None) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    with (_log_dir() / "perfect_loop.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if progress:
        progress(line)
    else:
        print(line)


def _eval_prompts_for_category(category: str, count: int = EVAL_PER_CATEGORY) -> List[str]:
    pools = {
        "hentai": list(DEFAULT_HENTAI_PROMPTS),
        "porn": list(DEFAULT_PORN_PROMPTS),
        "real": list(DEFAULT_REAL_PROMPTS),
    }
    items = pools.get(category) or pools["real"]
    if count <= len(items):
        return items[:count]
    expanded: List[str] = []
    idx = 0
    while len(expanded) < count:
        expanded.append(items[idx % len(items)])
        idx += 1
    return expanded


def _category_converged(
    comparison: Dict[str, Any],
    score_gap_threshold: float,
    lap_gap_threshold: float,
    min_custom_score: float,
) -> bool:
    return _converged(comparison, score_gap_threshold, lap_gap_threshold, min_custom_score)


def _generate_nsfw_teachers(
    count_per_category: int,
    video_count: int,
    model: str,
    cycle: int,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    base_seed = 1000 + cycle * 97
    result: Dict[str, Any] = {"model": model, "cycle": cycle}
    for category in ("hentai", "porn"):
        if progress:
            progress(f"Teacher images [{category}] x{count_per_category} ({model})")
        paths = batch_generate_nsfw_images(
            category,
            count=count_per_category,
            save_for_learning=True,
            seed=base_seed + hash(category) % 500,
            model_profile=model,
            cycle_index=cycle,
        )
        result[f"{category}_images"] = len(paths)
    for category in ("hentai", "porn"):
        if progress:
            progress(f"Teacher videos [{category}] x{video_count} ({model})")
        paths = batch_generate_nsfw_videos(
            category,
            count=video_count,
            save_for_learning=True,
            seed=base_seed + 200 + hash(category) % 500,
            num_frames=16,
            fps=12,
            model_profile=model,
            cycle_index=cycle,
        )
        result[f"{category}_videos"] = len(paths)
    return result


def _compare_nsfw_categories(
    model: str,
    seed_base: int,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    categories = ("hentai", "porn", "real")
    by_category: Dict[str, Any] = {}
    for cat in categories:
        prompts = _eval_prompts_for_category(cat)
        if progress:
            progress(f"Compare category={cat} prompts={len(prompts)}")
        by_category[cat] = compare_custom_vs_teacher(
            prompts=prompts,
            model=model,
            seed_base=seed_base + hash(cat) % 1000,
            progress=progress,
        )
    hentai = by_category["hentai"]
    porn = by_category["porn"]
    return {
        "by_category": by_category,
        "hentai_score_gap": hentai.get("average_score_gap"),
        "hentai_lap_gap": hentai.get("average_lap_var_gap"),
        "hentai_custom_score": hentai.get("average_custom_score"),
        "porn_score_gap": porn.get("average_score_gap"),
        "porn_lap_gap": porn.get("average_lap_var_gap"),
        "porn_custom_score": porn.get("average_custom_score"),
    }


def _all_nsfw_converged(
    comparison: Dict[str, Any],
    score_gap_threshold: float,
    lap_gap_threshold: float,
    min_custom_score: float,
) -> bool:
    by_cat = comparison.get("by_category") or {}
    for key in ("hentai", "porn"):
        if not _category_converged(
            by_cat.get(key) or {},
            score_gap_threshold,
            lap_gap_threshold,
            min_custom_score,
        ):
            return False
    return True


def run_perfect_loop(
    count_per_category: int = DEFAULT_COUNT_PER_CATEGORY,
    video_count: int = DEFAULT_VIDEO_COUNT,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    model: str = DEFAULT_MODEL,
    finetune_steps: Optional[int] = None,
    score_gap_threshold: float = DEFAULT_SCORE_GAP,
    lap_gap_threshold: float = DEFAULT_LAP_GAP,
    min_custom_score: float = DEFAULT_MIN_CUSTOM_SCORE,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    ok, detail = diffusers_available()
    if not ok:
        raise RuntimeError(f"diffusers unavailable: {detail}")

    cfg = load_config(IMAGE_CONFIG)
    steps = finetune_steps or int((cfg.get("training") or {}).get("finetune_steps", 400))
    report: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "count_per_category": count_per_category,
        "video_count": video_count,
        "max_cycles": max_cycles,
        "finetune_steps": steps,
        "score_gap_threshold": score_gap_threshold,
        "lap_gap_threshold": lap_gap_threshold,
        "min_custom_score": min_custom_score,
        "note": (
            "Matching SD Turbo on CPU may take many cycles (hours to days). "
            "Stopping at max_cycles is expected; review logs/perfect_loop/."
        ),
        "cycles": [],
    }
    _log(
        f"Navine AI perfect loop start model={model} max_cycles={max_cycles} "
        f"hentai+porn images={count_per_category} videos={video_count}",
        progress,
    )
    for cycle in range(1, max_cycles + 1):
        cycle_info: Dict[str, Any] = {"cycle": cycle}
        _log(f"Cycle {cycle} phase A: NSFW teacher batch (no lock)", progress)
        cycle_info["phase_a"] = _generate_nsfw_teachers(
            count_per_category,
            video_count,
            model,
            cycle,
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
        _finetune_video(steps=steps, progress=progress)
        cycle_info["phase_c"]["video_finetune"] = True
        _log(f"Cycle {cycle} phase D: compare hentai+porn+real vs teacher", progress)
        comparison = _compare_nsfw_categories(model, seed_base=42 + cycle * 11, progress=progress)
        cycle_info["phase_d"] = comparison
        cycle_info["converged"] = _all_nsfw_converged(
            comparison,
            score_gap_threshold,
            lap_gap_threshold,
            min_custom_score,
        )
        report["cycles"].append(cycle_info)
        cycle_path = _log_dir() / f"cycle_{cycle:02d}.json"
        cycle_path.write_text(json.dumps(cycle_info, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(
            f"Cycle {cycle} hentai gap score={comparison.get('hentai_score_gap')} "
            f"lap={comparison.get('hentai_lap_gap')} | "
            f"porn gap score={comparison.get('porn_score_gap')} "
            f"lap={comparison.get('porn_lap_gap')}",
            progress,
        )
        if cycle_info["converged"]:
            report["stopped"] = "converged"
            _log(f"Perfect loop converged at cycle {cycle}", progress)
            break
        if cycle >= max_cycles:
            report["stopped"] = "max_cycles"
            _log(f"Perfect loop reached max_cycles={max_cycles}", progress)
            break
        time.sleep(2)
    report["finished"] = datetime.now(timezone.utc).isoformat()
    final_path = _log_dir() / "perfect_loop_report.json"
    final_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Perfect loop report: {final_path}", progress)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Navine AI: loop until custom NSFW quality matches external diffusers teacher."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT_PER_CATEGORY)
    parser.add_argument("--video-count", type=int, default=DEFAULT_VIDEO_COUNT)
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--score-gap", type=float, default=DEFAULT_SCORE_GAP)
    parser.add_argument("--lap-gap", type=float, default=DEFAULT_LAP_GAP)
    parser.add_argument("--min-custom-score", type=float, default=DEFAULT_MIN_CUSTOM_SCORE)
    args = parser.parse_args()
    try:
        report = run_perfect_loop(
            count_per_category=args.count,
            video_count=args.video_count,
            max_cycles=args.max_cycles,
            model=args.model,
            finetune_steps=args.steps,
            score_gap_threshold=args.score_gap,
            lap_gap_threshold=args.lap_gap,
            min_custom_score=args.min_custom_score,
        )
        stopped = report.get("stopped", "unknown")
        cycles = len(report.get("cycles", []))
        print(f"Navine AI perfect loop finished: {stopped} after {cycles} cycle(s)")
        print(f"Report: logs/perfect_loop/perfect_loop_report.json")
        return 0
    except Exception as exc:
        print(f"Perfect loop failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
