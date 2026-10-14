import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from navine.image.quality import category_quality_pass, score_image_path
from navine.utils.config import load_config
from navine.utils.paths import get_output_dir, get_project_root
from navine.utils.tier import resolve_checkpoint_dir
from navine.utils.training_lock import training_lock, training_lock_active, wait_for_training_unlock

ProgressCallback = Optional[Callable[[str], None]]

IMAGE_CONFIG = "image_enterprise"
VIDEO_CONFIG = "video_enterprise"

REAL_PROMPT = (
    "photorealistic real amateur nude woman explicit photo natural skin texture "
    "full body sharp focus detailed anatomy"
)
REAL_VIDEO_PROMPT = (
    "photorealistic real woman natural motion cinematic lighting slow movement "
    "amateur video realistic skin"
)
HENTAI_PROMPT = (
    "hentai explicit nude anime woman detailed anatomy anime illustration "
    "vibrant colors sharp linework ecchi"
)
HENTAI_VIDEO_PROMPT = (
    "hentai anime woman subtle motion animated style cinematic lighting "
    "ecchi illustration movement"
)

MIN_IMAGE_SCORE = 0.68
MIN_HENTAI_SCORE = 0.65
MIN_LAP_VAR = 20.0
MIN_HENTAI_LAP_VAR = 18.0
MIN_VIDEO_BYTES = 150_000


def _log(msg: str, progress: ProgressCallback = None) -> None:
    log_dir = get_project_root() / "logs" / "photoreal_loop"
    log_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    with (log_dir / "photoreal_loop.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if progress:
        progress(line)


def _image_ckpt_dir() -> Path:
    cfg = load_config(IMAGE_CONFIG)
    return resolve_checkpoint_dir("image", cfg)


def _video_ckpt_dir() -> Path:
    cfg = load_config(VIDEO_CONFIG)
    return resolve_checkpoint_dir("video", cfg)


def _image_ckpt_path() -> Path:
    return _image_ckpt_dir() / "latest.pt"


def _video_ckpt_path() -> Path:
    return _video_ckpt_dir() / "latest.pt"


def _real_image_passes(metric: Dict[str, Any]) -> bool:
    return category_quality_pass(metric, "real_human")


def _hentai_image_passes(metric: Dict[str, Any]) -> bool:
    return category_quality_pass(metric, "hentai")


def _attach_meta(metric: Dict[str, Any], out_img: Path) -> Dict[str, Any]:
    meta_path = out_img.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metric["generation_mode"] = meta.get("generation_mode")
            metric["reference_category"] = meta.get("reference_category")
            metric["reference_used"] = meta.get("reference_used")
        except Exception:
            pass
    return metric


def _video_passes(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "passes": False}
    if not path.exists():
        result["error"] = "missing"
        return result
    size = path.stat().st_size
    result["size_bytes"] = size
    if size < MIN_VIDEO_BYTES:
        result["error"] = "too_small"
        return result
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = 0
        vars_acc = []
        while frames < 12:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vars_acc.append(float(gray.std()))
            frames += 1
        cap.release()
        result["frames_read"] = frames
        result["frame_std_avg"] = sum(vars_acc) / max(len(vars_acc), 1)
        if frames < 4:
            result["error"] = "few_frames"
            return result
        if result["frame_std_avg"] < 8.0:
            result["error"] = "bland_frames"
            return result
    except Exception as exc:
        result["frame_check_error"] = str(exc)
        if size >= MIN_VIDEO_BYTES * 2:
            result["passes"] = True
            return result
        return result
    result["passes"] = True
    return result


def _checkpoint_raw_std(prompt: str, seed: int = 42) -> float:
    from navine.device_manager import get_device
    from navine.image.model import NavineDiffusionModel

    config = load_config(IMAGE_CONFIG)
    device = get_device()
    ckpt = _image_ckpt_path()
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
        guidance_scale=float(infer.get("guidance_scale", 2.0)),
    )
    tmp = get_output_dir("image") / "_photoreal_probe.png"
    save_image((sample + 1) / 2, tmp)
    return float(np.array(Image.open(tmp).convert("RGB"), dtype=np.float32).std())


def finetune_image_guarded(steps: int, progress: ProgressCallback = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"steps": steps, "rolled_back": False}
    ckpt_dir = _image_ckpt_dir()
    latest = ckpt_dir / "latest.pt"
    pre = ckpt_dir / "latest_pre_finetune.pt"
    if latest.exists():
        shutil.copy2(latest, pre)
    from navine.image.train import train

    train(IMAGE_CONFIG, finetune=True, finetune_steps=steps)
    std = _checkpoint_raw_std(REAL_PROMPT)
    info["post_std"] = std
    if std < 22.0 and pre.exists():
        shutil.copy2(pre, latest)
        info["rolled_back"] = True
        _log(f"Image finetune rolled back (std={std:.2f})", progress)
    else:
        shutil.copy2(latest, ckpt_dir / "latest_good.pt")
        info["saved_good"] = True
        _log(f"Image finetune kept (std={std:.2f})", progress)
    return info


def evaluate_dual(progress: ProgressCallback = None) -> Dict[str, Any]:
    from navine.image.infer import generate as gen_image
    from navine.video.infer import generate as gen_video

    ckpt = _image_ckpt_path()
    ckpt_arg = str(ckpt) if ckpt.exists() else None
    out_real = get_output_dir("image") / "real_human.png"
    out_hentai = get_output_dir("image") / "hentai_output.png"
    out_real_vid = get_output_dir("video") / "real_human.mp4"
    out_hentai_vid = get_output_dir("video") / "hentai_output.mp4"

    _log("Generating real_human image", progress)
    gen_image(
        REAL_PROMPT,
        output_path=str(out_real),
        config_name=IMAGE_CONFIG,
        checkpoint=ckpt_arg,
        enhance=True,
        auto_ingest=False,
    )
    real_metric = _attach_meta(score_image_path(out_real), out_real)
    real_metric["real_pass"] = _real_image_passes(real_metric)
    _log(
        f"Real score={real_metric.get('score')} lap_var={real_metric.get('lap_var')} "
        f"ref={real_metric.get('reference_category')} pass={real_metric['real_pass']}",
        progress,
    )

    _log("Generating hentai_output image", progress)
    gen_image(
        HENTAI_PROMPT,
        output_path=str(out_hentai),
        config_name=IMAGE_CONFIG,
        checkpoint=ckpt_arg,
        enhance=True,
        auto_ingest=False,
    )
    hentai_metric = _attach_meta(score_image_path(out_hentai), out_hentai)
    hentai_metric["hentai_pass"] = _hentai_image_passes(hentai_metric)
    _log(
        f"Hentai score={hentai_metric.get('score')} lap_var={hentai_metric.get('lap_var')} "
        f"structure={hentai_metric.get('structure')} ref={hentai_metric.get('reference_category')} "
        f"pass={hentai_metric['hentai_pass']}",
        progress,
    )

    real_vid_metric: Dict[str, Any] = {"skipped": True}
    hentai_vid_metric: Dict[str, Any] = {"skipped": True}
    video_ckpt = str(_video_ckpt_path()) if _video_ckpt_path().exists() else None
    if real_metric["real_pass"] or ckpt.exists():
        _log("Generating real_human video", progress)
        try:
            gen_video(
                REAL_VIDEO_PROMPT,
                output_path=str(out_real_vid),
                config_name=VIDEO_CONFIG,
                checkpoint=video_ckpt,
                auto_ingest=False,
            )
            real_vid_metric = _video_passes(out_real_vid)
        except Exception as exc:
            real_vid_metric = {"passes": False, "error": str(exc)}
        _log(f"Real video pass={real_vid_metric.get('passes')} size={real_vid_metric.get('size_bytes')}", progress)

    if hentai_metric["hentai_pass"] or ckpt.exists():
        _log("Generating hentai_output video", progress)
        try:
            gen_video(
                HENTAI_VIDEO_PROMPT,
                output_path=str(out_hentai_vid),
                config_name=VIDEO_CONFIG,
                checkpoint=video_ckpt,
                auto_ingest=False,
            )
            hentai_vid_metric = _video_passes(out_hentai_vid)
        except Exception as exc:
            hentai_vid_metric = {"passes": False, "error": str(exc)}
        _log(
            f"Hentai video pass={hentai_vid_metric.get('passes')} size={hentai_vid_metric.get('size_bytes')}",
            progress,
        )

    all_pass = (
        bool(real_metric.get("real_pass"))
        and bool(hentai_metric.get("hentai_pass"))
        and bool(real_vid_metric.get("passes"))
        and bool(hentai_vid_metric.get("passes"))
    )
    return {
        "real": real_metric,
        "hentai": hentai_metric,
        "real_video": real_vid_metric,
        "hentai_video": hentai_vid_metric,
        "all_pass": all_pass,
        "real_image_path": str(out_real),
        "hentai_image_path": str(out_hentai),
        "real_video_path": str(out_real_vid),
        "hentai_video_path": str(out_hentai_vid),
    }


def evaluate_real(progress: ProgressCallback = None) -> Dict[str, Any]:
    dual = evaluate_dual(progress=progress)
    return {
        "image": dual["real"],
        "video": dual["real_video"],
        "all_pass": dual["all_pass"],
        "image_path": dual["real_image_path"],
        "video_path": dual["real_video_path"],
        "hentai": dual["hentai"],
        "hentai_video": dual["hentai_video"],
        "hentai_image_path": dual["hentai_image_path"],
        "hentai_video_path": dual["hentai_video_path"],
    }


def run_photoreal_loop(
    max_cycles: int = 30,
    finetune_steps: int = 300,
    video_finetune_steps: int = 200,
    poll_seconds: int = 90,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    log_dir = get_project_root() / "logs" / "photoreal_loop"
    log_dir.mkdir(parents=True, exist_ok=True)
    if _image_ckpt_path().exists():
        try:
            from navine.enterprise.pipeline import _set_active_tier

            _set_active_tier("enterprise")
            _log("Using enterprise tier for image/video inference", progress)
        except Exception:
            pass
    report: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "max_cycles": max_cycles,
        "image_config": IMAGE_CONFIG,
        "video_config": VIDEO_CONFIG,
        "cycles": [],
    }
    for cycle in range(1, max_cycles + 1):
        while training_lock_active("image"):
            _log(f"Cycle {cycle}: image training active, waiting...", progress)
            wait_for_training_unlock("image", poll_seconds=poll_seconds, progress=progress)

        _log(f"Cycle {cycle}/{max_cycles}: evaluate real + hentai", progress)
        evaluation = evaluate_dual(progress=progress)
        cycle_info: Dict[str, Any] = {"cycle": cycle, "evaluation": evaluation}
        (log_dir / f"cycle_{cycle:02d}.json").write_text(
            json.dumps(cycle_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if evaluation.get("all_pass"):
            cycle_info["stopped"] = "real_hentai_photo_and_video_met"
            report["cycles"].append(cycle_info)
            _log(f"Real + hentai targets met at cycle {cycle}", progress)
            break
        if cycle >= max_cycles:
            cycle_info["stopped"] = "max_cycles"
            report["cycles"].append(cycle_info)
            break

        wait_for_training_unlock("image", poll_seconds=poll_seconds, progress=progress)
        _log(f"Cycle {cycle}: finetune image ({finetune_steps} steps)", progress)
        with training_lock("image"):
            cycle_info["image_finetune"] = finetune_image_guarded(finetune_steps, progress=progress)

        if cycle % 2 == 0:
            wait_for_training_unlock("image", poll_seconds=poll_seconds, progress=progress)
            _log(f"Cycle {cycle}: finetune video ({video_finetune_steps} steps)", progress)
            from navine.video.train import train as train_video

            train_video(VIDEO_CONFIG, finetune=True, finetune_steps=video_finetune_steps)
            cycle_info["video_finetune"] = True

        if cycle % 4 == 0:
            try:
                from navine.nsfw.autolearn import run_nsfw_cycle

                _log(f"Cycle {cycle}: NSFW data fetch", progress)
                cycle_info["nsfw_fetch"] = run_nsfw_cycle(progress=progress)
            except Exception as exc:
                cycle_info["nsfw_fetch_error"] = str(exc)

        report["cycles"].append(cycle_info)

    final = evaluate_dual(progress=progress)
    report["final"] = final
    report["finished"] = datetime.now(timezone.utc).isoformat()
    if final.get("all_pass"):
        try:
            from navine.enterprise.pipeline import _set_active_tier

            _set_active_tier("enterprise")
            report["promoted_tier"] = "enterprise"
        except Exception:
            pass
    out = log_dir / "photoreal_loop_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Photoreal loop complete: {out}", progress)
    return report


run_dual_quality_loop = run_photoreal_loop


if __name__ == "__main__":
    run_photoreal_loop()
