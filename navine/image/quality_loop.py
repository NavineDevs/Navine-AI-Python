import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.image.quality import score_batch, score_image_path
from navine.utils.paths import get_output_dir, get_project_root


TEST_PROMPTS = [
    "photorealistic hentai porn explicit nude woman realistic photo natural lighting",
    "photorealistic real amateur nude woman explicit photo",
    "hentai explicit woman photorealistic realistic skin detailed",
]


def _log_dir() -> Path:
    path = get_project_root() / "logs" / "quality_loop"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_prompts(
    prompts: Optional[List[str]] = None,
    output_prefix: str = "quality_check",
) -> Dict[str, Any]:
    from navine.image.infer import generate as gen_image

    prompts = prompts or TEST_PROMPTS
    out_dir = get_output_dir("image")
    paths: List[Path] = []
    entries: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        out_path = out_dir / f"{output_prefix}_{idx}.png"
        gen_image(prompt, output_path=str(out_path), enhance=True, auto_ingest=False)
        meta_path = out_path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        paths.append(out_path)
        metric = score_image_path(out_path)
        metric["prompt"] = prompt
        metric["generation_mode"] = meta.get("generation_mode")
        metric["reference_used"] = meta.get("reference_used")
        entries.append(metric)
    batch = score_batch(paths)
    batch["entries"] = entries
    return batch


def run_quality_loop(
    max_cycles: int = 6,
    finetune_steps: int = 400,
    quality_target: float = 0.62,
    video_every: int = 2,
) -> Dict[str, Any]:
    from navine.utils.training_lock import training_lock

    with training_lock("image"):
        return _run_quality_loop_impl(max_cycles, finetune_steps, quality_target, video_every)


def _run_quality_loop_impl(
    max_cycles: int = 6,
    finetune_steps: int = 400,
    quality_target: float = 0.62,
    video_every: int = 2,
) -> Dict[str, Any]:
    log_dir = _log_dir()
    report: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "cycles": [],
        "quality_target": quality_target,
        "finetune_steps": finetune_steps,
        "max_cycles": max_cycles,
    }
    best_score = 0.0
    for cycle in range(1, max_cycles + 1):
        cycle_info: Dict[str, Any] = {"cycle": cycle, "phase": "evaluate"}
        print(f"Navine AI - Python quality loop cycle {cycle}/{max_cycles}: evaluate")
        eval_result = evaluate_prompts(output_prefix=f"loop_c{cycle}")
        cycle_info["evaluation"] = eval_result
        avg = float(eval_result.get("average_score", 0.0))
        best_score = max(best_score, avg)
        cycle_info["average_score"] = avg
        cycle_info["all_pass"] = bool(eval_result.get("all_pass"))
        (log_dir / f"cycle_{cycle:02d}_eval.json").write_text(
            json.dumps(cycle_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if eval_result.get("all_pass") and avg >= quality_target:
            cycle_info["stopped"] = "quality_met"
            report["cycles"].append(cycle_info)
            print(f"Navine AI - Python quality target met at cycle {cycle} (avg={avg:.3f})")
            break
        if cycle >= max_cycles:
            cycle_info["stopped"] = "max_cycles"
            report["cycles"].append(cycle_info)
            break
        cycle_info["phase"] = "train"
        print(f"Navine AI - Python quality loop cycle {cycle}/{max_cycles}: finetune image ({finetune_steps} steps)")
        from navine.image.train import train

        train(finetune=True, finetune_steps=finetune_steps)
        cycle_info["image_finetune_steps"] = finetune_steps
        if cycle % video_every == 0:
            print(f"Navine AI - Python quality loop cycle {cycle}/{max_cycles}: finetune video")
            from navine.video.train import train as train_video

            train_video(finetune=True)
            cycle_info["video_finetune"] = True
        if cycle % 3 == 0:
            try:
                from navine.autolearn.sources.nsfw import fetch_full_cycle
                from navine.nsfw.config import load_nsfw_config

                print(f"Navine AI - Python quality loop cycle {cycle}: fetch media only")
                cycle_info["media_fetch"] = fetch_full_cycle(
                    progress=print,
                    config=load_nsfw_config(),
                    max_items=80,
                )
            except Exception as exc:
                cycle_info["media_fetch_error"] = str(exc)
        report["cycles"].append(cycle_info)
    print("Navine AI - Python quality loop: final generation")
    from navine.image.infer import generate as gen_image
    from navine.video.infer import generate as gen_video

    final_prompt = TEST_PROMPTS[0]
    img_path = gen_image(final_prompt, output_path="outputs/image/hentai_porn_real.png")
    vid_path = gen_video(final_prompt, output_path="outputs/video/hentai_porn_real.mp4")
    final_img = score_image_path(Path(img_path))
    report["final"] = {
        "image": str(img_path),
        "video": str(vid_path),
        "image_score": final_img,
        "best_average_score": best_score,
    }
    report["finished"] = datetime.now(timezone.utc).isoformat()
    out_report = log_dir / "quality_loop_report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Navine AI - Python quality loop complete. Report: {out_report}")
    return report


if __name__ == "__main__":
    run_quality_loop()
