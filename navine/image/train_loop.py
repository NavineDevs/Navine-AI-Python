import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from navine.image.quality import category_quality_pass, score_image_path
from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir, get_output_dir, get_project_root

CATEGORY_PROMPTS = {
    "real_human": "photorealistic real amateur nude woman explicit photo natural skin texture portrait",
    "hentai": "hentai explicit nude anime woman detailed anatomy anime illustration vibrant colors ecchi",
}

DEFAULT_QUALITY_TARGET = 0.62
MIN_CHECKPOINT_STD = 22.0


def _log_dir() -> Path:
    path = get_project_root() / "logs" / "train_loop"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ckpt_dir() -> Path:
    return get_checkpoint_dir("image")


def ensure_best_checkpoint() -> Path:
    ckpt_dir = _ckpt_dir()
    latest = ckpt_dir / "latest.pt"
    good = ckpt_dir / "latest_good.pt"
    collapsed = ckpt_dir / "latest_collapsed.pt"
    for candidate in (good, collapsed, latest):
        if candidate.exists():
            if candidate != latest:
                shutil.copy2(candidate, latest)
                print(f"Navine AI - Python: restored checkpoint from {candidate.name}")
            return latest
    raise FileNotFoundError("No image checkpoint found to train from.")


def _checkpoint_raw_std(prompt: str, seed: int = 42) -> float:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.utils import save_image

    from navine.device_manager import get_device
    from navine.image.model import NavineDiffusionModel

    config = load_config("image")
    device = get_device()
    ckpt = _ckpt_dir() / "latest.pt"
    model = NavineDiffusionModel.load_checkpoint(ckpt, config, device).to(device)
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    infer = config["inference"]
    sample = model.sample(
        batch_size=1,
        text=prompt,
        device=device,
        num_steps=int(infer.get("num_steps", 200)),
        guidance_scale=float(infer.get("guidance_scale", 1.5)),
    )
    tmp = get_output_dir("image") / "_ckpt_probe.png"
    save_image((sample + 1) / 2, tmp)
    return float(np.array(Image.open(tmp).convert("RGB"), dtype=np.float32).std())


def _save_checkpoint_backup(name: str) -> Optional[Path]:
    latest = _ckpt_dir() / "latest.pt"
    if not latest.exists():
        return None
    dest = _ckpt_dir() / name
    shutil.copy2(latest, dest)
    return dest


def finetune_with_guard(steps: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {"steps": steps, "rolled_back": False}
    _save_checkpoint_backup("latest_pre_finetune.pt")
    from navine.image.train import train

    train(finetune=True, finetune_steps=steps)
    std = _checkpoint_raw_std(CATEGORY_PROMPTS["real_human"])
    info["post_std"] = std
    if std < MIN_CHECKPOINT_STD:
        pre = _ckpt_dir() / "latest_pre_finetune.pt"
        if pre.exists():
            shutil.copy2(pre, _ckpt_dir() / "latest.pt")
            info["rolled_back"] = True
            print(f"Navine AI - Python: finetune rolled back (std={std:.2f} < {MIN_CHECKPOINT_STD})")
    else:
        shutil.copy2(_ckpt_dir() / "latest.pt", _ckpt_dir() / "latest_good.pt")
        info["saved_good"] = True
        print(f"Navine AI - Python: finetune kept (std={std:.2f})")
    return info


def evaluate_categories(
    quality_target: float = DEFAULT_QUALITY_TARGET,
    output_prefix: str = "train_loop",
) -> Dict[str, Any]:
    from navine.image.infer import generate as gen_image

    out_dir = get_output_dir("image")
    categories: Dict[str, Any] = {}
    for key, prompt in CATEGORY_PROMPTS.items():
        out_path = out_dir / f"{output_prefix}_{key}.png"
        gen_image(prompt, output_path=str(out_path), enhance=True, auto_ingest=False)
        meta_path = out_path.with_suffix(".json")
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        metric = score_image_path(out_path)
        metric["prompt"] = prompt
        metric["generation_mode"] = meta.get("generation_mode")
        metric["reference_used"] = meta.get("reference_used")
        metric["reference_category"] = meta.get("reference_category")
        metric["category_pass"] = category_quality_pass(metric, key)
        categories[key] = metric
    scores = [float(categories[k].get("score", 0)) for k in CATEGORY_PROMPTS]
    avg = sum(scores) / max(len(scores), 1)
    all_pass = all(categories[k].get("category_pass") for k in CATEGORY_PROMPTS)
    return {
        "average_score": round(avg, 4),
        "all_pass": all_pass,
        "categories": categories,
    }


def run_train_loop(
    max_cycles: int = 12,
    finetune_steps: int = 250,
    quality_target: float = DEFAULT_QUALITY_TARGET,
    video_every: int = 3,
) -> Dict[str, Any]:
    from navine.utils.training_lock import training_lock

    with training_lock("image"):
        return _run_train_loop_impl(max_cycles, finetune_steps, quality_target, video_every)


def _run_train_loop_impl(
    max_cycles: int,
    finetune_steps: int,
    quality_target: float,
    video_every: int,
) -> Dict[str, Any]:
    log_dir = _log_dir()
    ensure_best_checkpoint()
    report: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "quality_target": quality_target,
        "finetune_steps": finetune_steps,
        "max_cycles": max_cycles,
        "cycles": [],
    }
    best_avg = 0.0
    for cycle in range(1, max_cycles + 1):
        print(f"Navine AI - Python train loop cycle {cycle}/{max_cycles}: evaluate real + hentai")
        evaluation = evaluate_categories(quality_target=quality_target, output_prefix=f"loop_c{cycle}")
        avg = float(evaluation.get("average_score", 0.0))
        best_avg = max(best_avg, avg)
        cycle_info: Dict[str, Any] = {
            "cycle": cycle,
            "evaluation": evaluation,
            "average_score": avg,
            "all_pass": evaluation.get("all_pass"),
        }
        (log_dir / f"cycle_{cycle:02d}.json").write_text(
            json.dumps(cycle_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if evaluation.get("all_pass"):
            cycle_info["stopped"] = "real_and_hentai_met"
            report["cycles"].append(cycle_info)
            print(f"Navine AI - Python: both real human and hentai passed at cycle {cycle}")
            break
        if cycle >= max_cycles:
            cycle_info["stopped"] = "max_cycles"
            report["cycles"].append(cycle_info)
            break
        print(f"Navine AI - Python train loop cycle {cycle}: finetune ({finetune_steps} steps, guarded)")
        cycle_info["finetune"] = finetune_with_guard(finetune_steps)
        if cycle % video_every == 0:
            from navine.video.train import train as train_video

            print(f"Navine AI - Python train loop cycle {cycle}: video finetune")
            train_video(finetune=True)
            cycle_info["video_finetune"] = True
        if cycle % 4 == 0:
            try:
                from navine.autolearn.sources.nsfw import fetch_full_cycle
                from navine.nsfw.config import load_nsfw_config

                cycle_info["media_fetch"] = fetch_full_cycle(
                    progress=print,
                    config=load_nsfw_config(),
                    max_items=60,
                )
            except Exception as exc:
                cycle_info["media_fetch_error"] = str(exc)
        report["cycles"].append(cycle_info)
    print("Navine AI - Python train loop: final outputs")
    from navine.image.infer import generate as gen_image
    from navine.video.infer import generate as gen_video

    real_img = gen_image(
        CATEGORY_PROMPTS["real_human"],
        output_path="outputs/image/real_human.png",
    )
    hentai_img = gen_image(
        CATEGORY_PROMPTS["hentai"],
        output_path="outputs/image/hentai_output.png",
    )
    vid = gen_video(
        CATEGORY_PROMPTS["real_human"],
        output_path="outputs/video/real_human.mp4",
    )
    report["final"] = {
        "real_human_image": str(real_img),
        "hentai_image": str(hentai_img),
        "real_human_video": str(vid),
        "real_human_score": score_image_path(Path(real_img)),
        "hentai_score": score_image_path(Path(hentai_img)),
        "best_average_score": best_avg,
    }
    report["finished"] = datetime.now(timezone.utc).isoformat()
    out = log_dir / "train_loop_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Navine AI - Python train loop complete: {out}")
    return report


if __name__ == "__main__":
    run_train_loop()
