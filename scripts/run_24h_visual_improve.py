"""24-hour continuous visual improve loop for image + video (aggressive)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log_dir() -> Path:
    path = ROOT / "logs" / "visual_improve_24h"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with (_log_dir() / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _py() -> str:
    venv = ROOT / "venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def _run(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    _log("RUN " + " ".join(cmd))
    try:
        completed = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": completed.returncode, "ok": completed.returncode == 0}
    except subprocess.TimeoutExpired as exc:
        _log(f"CMD timeout after {timeout}s")
        return {"cmd": cmd, "returncode": -1, "ok": False, "error": f"timeout:{exc}"}
    except Exception as exc:
        _log(f"CMD failed: {exc}")
        return {"cmd": cmd, "returncode": -1, "ok": False, "error": str(exc)}


def _bytes_removed(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    if path.is_file():
        total = int(path.stat().st_size)
        path.unlink(missing_ok=True)
        return total
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += int(child.stat().st_size)
            except OSError:
                pass
    import shutil

    shutil.rmtree(path, ignore_errors=True)
    return total


def _cap_files(dir_path: Path, keep: int) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    files = sorted(
        [p for p in dir_path.iterdir() if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for path in files[max(0, keep) :]:
        try:
            removed += int(path.stat().st_size)
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return removed


def _cap_dirs(dir_path: Path, keep: int) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    dirs = sorted(
        [p for p in dir_path.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for path in dirs[max(0, keep) :]:
        removed += _bytes_removed(path)
    return removed


def _disk_hygiene() -> Dict[str, Any]:
    removed = 0
    for rel in (
        "data/nsfw/images/infini-atomic",
        "data/nsfw/images/waifu",
        "data/nsfw/images/local/general",
        "outputs/image/enterprise_eval",
        "outputs/image/teacher_compare",
        "outputs/image/external",
        "outputs/video/batch",
        "outputs/video/external_keyframes",
    ):
        removed += _bytes_removed(ROOT / rel)
    removed += _cap_files(ROOT / "data" / "image" / "learned", keep=400)
    removed += _cap_dirs(ROOT / "data" / "video" / "learned", keep=40)
    removed += _cap_files(ROOT / "data" / "image" / "samples", keep=100)
    removed += _cap_files(ROOT / "data" / "video" / "samples", keep=40)
    log_dir = _log_dir()
    cycles = sorted(log_dir.glob("cycle_*.json"))
    for path in cycles[:-5]:
        removed += _bytes_removed(path)
    for path in log_dir.glob("run_prev_*.log"):
        removed += _bytes_removed(path)
    console = log_dir / "console.log"
    if console.exists() and console.stat().st_size > 2_000_000:
        removed += _bytes_removed(console)
        console.write_text("", encoding="utf-8")
    _log(f"Disk hygiene removed_mb={removed / (1024 * 1024):.1f}")
    return {"removed_bytes": removed}


def _fetch_refs(max_refs: int = 80, cycle: int = 1) -> Dict[str, Any]:
    py = _py()
    results: Dict[str, Any] = {}
    results["reference_fetch"] = _run(
        [py, "-m", "navine.cli", "reference", "fetch", "hentai", "--max", str(max_refs)],
        timeout=600,
    )
    results["skipped_bulk_nsfw_packs"] = True
    results["cycle"] = cycle
    results["hygiene"] = _disk_hygiene()
    results["wikipedia"] = _run(
        [py, "-m", "navine.cli", "learn", "wikipedia", "--max", "25"],
        timeout=240,
    )
    if cycle % 2 == 1:
        results["hf_code"] = _run(
            [py, "-m", "navine.cli", "learn", "hf-code", "--max", "800"],
            timeout=600,
        )
    else:
        results["arxiv"] = _run(
            [py, "-m", "navine.cli", "learn", "arxiv", "--max", "30"],
            timeout=240,
        )
    return results


def _teacher_cycle(count: int = 4, steps: int = 300, include_video: bool = False) -> Dict[str, Any]:
    py = _py()
    cmd = [
        py, "-m", "navine.cli", "learn", "teacher-loop",
        "--count", str(count), "--max-cycles", "1", "--steps", str(steps),
    ]
    if include_video:
        cmd.append("--video")
    return _run(cmd, timeout=5400)


def _ingest() -> Dict[str, Any]:
    from navine.learn.image_learn import ingest_learned_images, ingest_nsfw_images
    from navine.learn.video_learn import ingest_learned_video

    report: Dict[str, Any] = {}
    try:
        report["learned_images"] = ingest_learned_images()
    except Exception as exc:
        report["learned_images_error"] = str(exc)
    try:
        report["nsfw_images"] = ingest_nsfw_images()
    except Exception as exc:
        report["nsfw_images_error"] = str(exc)
    try:
        from navine.learn.from_external import ingest_external_for_training
        report["external"] = ingest_external_for_training()
    except Exception as exc:
        report["external_error"] = str(exc)
    try:
        report["video"] = ingest_learned_video()
    except Exception as exc:
        report["video_error"] = str(exc)
    return report


def _train_text(steps: int, require_cuda: bool) -> Dict[str, Any]:
    from navine.train.base import finetune_texts
    from navine.train.detective import load_detective_texts
    from navine.train.general import load_general_texts
    from navine.train.thinking import load_thinking_texts
    from navine.train.unrestricted import load_unrestricted_texts

    _log(f"Text enterprise finetune {steps} steps")
    try:
        texts = (
            load_general_texts()
            + load_unrestricted_texts()
            + load_thinking_texts()
            + load_detective_texts()
        )
        if not texts:
            return {"steps": steps, "ok": False, "error": "no text training data"}
        finetune_texts(
            "unrestricted",
            texts,
            desc="Navine AI - Python text_enterprise (general+unrestricted)",
            max_steps=steps,
            require_cuda=require_cuda,
        )
        return {"steps": steps, "ok": True, "samples": len(texts), "checkpoint": "text_enterprise"}
    except Exception as exc:
        _log(f"Text train failed: {exc}")
        return {"steps": steps, "ok": False, "error": str(exc)}


def _train_text_code(steps: int, require_cuda: bool) -> Dict[str, Any]:
    import shutil

    from navine.train.base import finetune_texts
    from navine.train.coding import load_coding_texts
    from navine.utils.paths import get_checkpoint_dir

    _log(f"Text-code specialist finetune {steps} steps")
    try:
        code_dir = get_checkpoint_dir("text_code")
        ent = get_checkpoint_dir("text_enterprise")
        code_dir.mkdir(parents=True, exist_ok=True)
        for name in ("latest.pt", "best.pt", "tokenizer.json"):
            src = ent / name
            if src.exists():
                shutil.copy2(src, code_dir / name)
        texts = load_coding_texts(language=None)
        try:
            from navine.train.detective import load_detective_cipher_texts

            texts = texts + load_detective_cipher_texts()
        except Exception:
            pass
        if not texts:
            return {"steps": steps, "ok": False, "error": "no coding training data"}
        finetune_texts(
            "coding",
            texts,
            desc="Navine AI - Python text_code specialist",
            max_steps=steps,
            checkpoint_dir=code_dir,
            require_cuda=require_cuda,
        )
        return {"steps": steps, "ok": True, "samples": len(texts), "checkpoint": "text_code"}
    except Exception as exc:
        _log(f"Text-code train failed: {exc}")
        return {"steps": steps, "ok": False, "error": str(exc)}


def _train_hitboyx23(steps: int, require_cuda: bool) -> Dict[str, Any]:
    import shutil

    from navine.train.base import finetune_texts
    from navine.train.hitboyx23 import load_hitboyx23_texts
    from navine.utils.paths import get_checkpoint_dir

    _log(f"HitBoyXx23 AI finetune {steps} steps")
    try:
        hb_dir = get_checkpoint_dir("hitboyx23_ai")
        ent = get_checkpoint_dir("text_enterprise")
        hb_dir.mkdir(parents=True, exist_ok=True)
        for name in ("latest.pt", "best.pt", "tokenizer.json"):
            src = ent / name
            dst = hb_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        texts = load_hitboyx23_texts()
        if not texts:
            return {"steps": steps, "ok": False, "error": "no hitboyx23 training data"}
        finetune_texts(
            "hitboyx23",
            texts,
            desc="HitBoyXx23 AI (all languages)",
            max_steps=steps,
            checkpoint_dir=hb_dir,
            require_cuda=require_cuda,
        )
        return {"steps": steps, "ok": True, "samples": len(texts), "checkpoint": "hitboyx23_ai"}
    except Exception as exc:
        _log(f"HitBoyXx23 train failed: {exc}")
        return {"steps": steps, "ok": False, "error": str(exc)}


def _train_hitboyx23_python(steps: int, require_cuda: bool) -> Dict[str, Any]:
    import shutil

    from navine.train.base import finetune_texts
    from navine.train.hitboyx23 import load_hitboyx23_python_texts
    from navine.utils.paths import get_checkpoint_dir

    _log(f"HitBoyXx23 AI Python finetune {steps} steps")
    try:
        hb_dir = get_checkpoint_dir("hitboyx23_ai_python")
        ent = get_checkpoint_dir("text_enterprise")
        hb_dir.mkdir(parents=True, exist_ok=True)
        for name in ("latest.pt", "best.pt", "tokenizer.json"):
            src = ent / name
            dst = hb_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        texts = load_hitboyx23_python_texts()
        if not texts:
            return {"steps": steps, "ok": False, "error": "no hitboyx23 python training data"}
        finetune_texts(
            "hitboyx23_python",
            texts,
            desc="HitBoyXx23 AI (Python only)",
            max_steps=steps,
            checkpoint_dir=hb_dir,
            require_cuda=require_cuda,
        )
        return {"steps": steps, "ok": True, "samples": len(texts), "checkpoint": "hitboyx23_ai_python"}
    except Exception as exc:
        _log(f"HitBoyXx23 Python train failed: {exc}")
        return {"steps": steps, "ok": False, "error": str(exc)}


def _train_image(steps: int, require_cuda: bool) -> Dict[str, Any]:
    from navine.image.train import train
    _log(f"Image finetune {steps} steps")
    try:
        train(config_path="image_enterprise", finetune=True, finetune_steps=steps, require_cuda=require_cuda)
        return {"steps": steps, "ok": True, "checkpoint": "image_enterprise"}
    except Exception as exc:
        _log(f"Image train failed: {exc}")
        return {"steps": steps, "ok": False, "error": str(exc)}


def _train_video(steps: int, require_cuda: bool) -> Dict[str, Any]:
    from navine.video.train import train
    _log(f"Video finetune {steps} steps")
    try:
        train(config_path="video_enterprise", finetune=True, finetune_steps=steps, require_cuda=require_cuda)
        return {"steps": steps, "ok": True, "checkpoint": "video_enterprise"}
    except Exception as exc:
        _log(f"Video train failed: {exc}")
        return {"steps": steps, "ok": False, "error": str(exc)}


def _prune_old_cycle_dirs(review_root: Path, keep_name: str, keep_last: int = 1) -> None:
    import shutil

    if not review_root.exists():
        return
    cycles = sorted(
        [p for p in review_root.iterdir() if p.is_dir() and p.name.startswith("cycle_")],
        key=lambda p: p.name,
    )
    retain = {keep_name}
    for path in cycles[-max(1, keep_last) :]:
        retain.add(path.name)
    for path in cycles:
        if path.name in retain:
            continue
        shutil.rmtree(path, ignore_errors=True)
        _log(f"Pruned old review dir {path}")


def _seed0_galleries(require_cuda: bool, cycle: int) -> Dict[str, Any]:
    import shutil
    from navine.image.batch_domain_generate import run_review_gallery as image_review
    from navine.utils.paths import get_output_dir
    from navine.video.batch_domain_generate import run_review_gallery as video_review

    img_dir = get_output_dir("image") / "review" / f"cycle_{cycle:03d}"
    vid_dir = get_output_dir("video") / "review" / f"cycle_{cycle:03d}"
    latest_img = get_output_dir("image") / "review"
    latest_vid = get_output_dir("video") / "review"
    try:
        image = image_review(seed=0, require_cuda=require_cuda, review_dir=img_dir)
    except Exception as exc:
        _log(f"Image review failed: {exc}")
        image = {"ok": False, "error": str(exc), "items": {}}
    try:
        video = video_review(seed=0, require_cuda=require_cuda, review_dir=vid_dir)
    except Exception as exc:
        _log(f"Video review failed: {exc}")
        video = {"ok": False, "error": str(exc), "items": {}}
    latest_img.mkdir(parents=True, exist_ok=True)
    latest_vid.mkdir(parents=True, exist_ok=True)
    if img_dir.exists():
        for src in img_dir.glob("*"):
            if src.is_file():
                shutil.copy2(src, latest_img / src.name)
    if vid_dir.exists():
        for src in vid_dir.glob("*"):
            if src.is_file():
                shutil.copy2(src, latest_vid / src.name)
    _prune_old_cycle_dirs(latest_img, keep_name=img_dir.name, keep_last=1)
    _prune_old_cycle_dirs(latest_vid, keep_name=vid_dir.name, keep_last=1)
    return {"image": image, "video": video, "image_dir": str(img_dir), "video_dir": str(vid_dir)}


def _gate_scores(gallery: Dict[str, Any]) -> Dict[str, Any]:
    image_items = (gallery.get("image") or {}).get("items") or {}
    video_items = (gallery.get("video") or {}).get("items") or {}
    image_ok = all(
        bool(v.get("passes") or (v.get("valid") and not v.get("noise")))
        for v in image_items.values()
    ) if image_items else False
    video_ok = all(bool(v.get("valid")) for v in video_items.values()) if video_items else False
    return {
        "image_ok": image_ok,
        "video_ok": video_ok,
        "all_ok": bool(image_ok and video_ok),
        "image_scores": {k: v.get("score") for k, v in image_items.items()},
        "video_variance": {k: v.get("frame_variance") for k, v in video_items.items()},
    }


def _write_final_summary(cycles: List[Dict[str, Any]], hours: float, started: str) -> Path:
    path = _log_dir() / "FINAL_SUMMARY.md"
    best = None
    for row in cycles:
        gates = row.get("gates") or {}
        score = sum(float(x or 0) for x in (gates.get("image_scores") or {}).values())
        if best is None or score >= best[0]:
            best = (score, row)
    lines = [
        "# 24h all-4 AI improve final summary",
        "",
        f"- Started: `{started}`",
        f"- Ended: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Hours requested: `{hours}`",
        f"- Cycles completed: `{len(cycles)}`",
        "",
        "## Trained checkpoints each cycle",
        "",
        "- `text_enterprise`",
        "- `text_code`",
        "- `hitboyx23_ai`",
        "- `hitboyx23_ai_python`",
        "- `image_enterprise`",
        "- `video_enterprise`",
        "",
        "## Latest seed-0 review galleries",
        "",
        "- Images: `outputs/image/review/{sfw,human,hentai,object,think,detective}_seed0.png`",
        "- Videos: `outputs/video/review/{sfw,human,hentai,object,think,detective}_seed0.mp4`",
        "",
        "## Pause for visual approval",
        "",
        "Review the seed-0 galleries above. Say continue for another round if quality is still weak.",
        "",
    ]
    if best:
        cyc = best[1]
        lines.extend([
            f"## Best cycle by image score sum: `{cyc.get('cycle')}`",
            "",
            f"- Image dir: `{((cyc.get('gallery') or {}).get('image_dir'))}`",
            f"- Video dir: `{((cyc.get('gallery') or {}).get('video_dir'))}`",
            f"- Gates: `{json.dumps(cyc.get('gates') or {})}`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    (_log_dir() / "FINAL_SUMMARY.json").write_text(
        json.dumps({"started": started, "ended": datetime.now(timezone.utc).isoformat(), "hours": hours, "cycles": cycles}, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def run_loop(
    hours: float = 24.0,
    require_cuda: bool = True,
    image_steps: int = 1500,
    video_steps: int = 900,
    text_steps: int = 1000,
    text_code_steps: int = 700,
    teacher_count: int = 4,
    teacher_steps: int = 300,
    max_refs: int = 40,
    skip_teacher: bool = False,
    skip_text: bool = False,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    start_ts = time.time()
    deadline = start_ts + max(0.1, float(hours)) * 3600.0
    cycles: List[Dict[str, Any]] = []
    cycle = 0
    _log(
        f"All-4 AI improve loop start hours={hours} require_cuda={require_cuda} "
        f"text_steps={text_steps} text_code_steps={text_code_steps} "
        f"image_steps={image_steps} video_steps={video_steps}"
    )
    while time.time() < deadline:
        cycle += 1
        remaining_h = (deadline - time.time()) / 3600.0
        _log(f"=== Cycle {cycle} remaining_h={remaining_h:.2f} ===")
        info: Dict[str, Any] = {"cycle": cycle, "started": datetime.now(timezone.utc).isoformat(), "remaining_hours": remaining_h}
        try:
            info["fetch"] = _fetch_refs(max_refs=max_refs, cycle=cycle)
            if not skip_teacher and remaining_h > 0.5:
                try:
                    info["teacher"] = _teacher_cycle(count=teacher_count, steps=teacher_steps, include_video=False)
                except Exception as exc:
                    info["teacher_error"] = str(exc)
                    _log(f"Teacher skipped: {exc}")
            info["ingest"] = _ingest()
            if not skip_text:
                info["text_train"] = _train_text(text_steps, require_cuda=require_cuda)
                info["text_code_train"] = _train_text_code(text_code_steps, require_cuda=require_cuda)
                info["hitboyx23_train"] = _train_hitboyx23(text_steps, require_cuda=require_cuda)
                info["hitboyx23_python_train"] = _train_hitboyx23_python(text_code_steps, require_cuda=require_cuda)
            info["image_train"] = _train_image(image_steps, require_cuda=require_cuda)
            info["video_train"] = _train_video(video_steps, require_cuda=require_cuda)
            try:
                from navine.llm import pack_all

                info["llm_pack"] = pack_all()
                _log("Packed runnable LLM files for text_enterprise, text_code, image_enterprise, video_enterprise")
            except Exception as exc:
                info["llm_pack_error"] = str(exc)
                _log(f"LLM pack failed: {exc}")
            try:
                info["gallery"] = _seed0_galleries(require_cuda=require_cuda, cycle=cycle)
                info["gates"] = _gate_scores(info["gallery"])
                trains_ok = all(
                    bool((info.get(k) or {}).get("ok", True))
                    for k in ("text_train", "text_code_train", "image_train", "video_train")
                    if k in info
                )
                info["ok"] = bool((info.get("gates") or {}).get("all_ok") and trains_ok)
            except Exception as exc:
                info["gallery_error"] = str(exc)
                info["ok"] = False
                _log(f"Gallery/gates failed: {exc}")
        except Exception as exc:
            info["error"] = str(exc)
            info["traceback"] = traceback.format_exc()
            _log(f"Cycle {cycle} error: {exc}")
        info["ended"] = datetime.now(timezone.utc).isoformat()
        cycles.append(info)
        cycle_path = _log_dir() / f"cycle_{cycle:03d}.json"
        cycle_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        _log(f"Cycle {cycle} saved -> {cycle_path} gates={info.get('gates')}")
        if time.time() >= deadline:
            break
        time.sleep(2)
    summary = _write_final_summary(cycles, hours=hours, started=started)
    _log(f"Loop complete. Summary: {summary}")
    return {"cycles": len(cycles), "summary": str(summary), "started": started}


def main() -> None:
    parser = argparse.ArgumentParser(description="24h continuous all-4 AI improve loop")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--image-steps", type=int, default=1500)
    parser.add_argument("--video-steps", type=int, default=900)
    parser.add_argument("--text-steps", type=int, default=1000)
    parser.add_argument("--text-code-steps", type=int, default=700)
    parser.add_argument("--teacher-count", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--max-refs", type=int, default=40)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--comprehensive", action="store_true", help="Run full data-source training before the visual loop")
    args = parser.parse_args()
    if args.comprehensive:
        comp = ROOT / "scripts" / "train_comprehensive.py"
        if comp.exists():
            _run([_py(), str(comp), "--mode", "full", "--hours", str(max(1.0, min(float(args.hours), 8.0)))])
    run_loop(
        hours=args.hours,
        require_cuda=bool(args.require_cuda),
        image_steps=args.image_steps,
        video_steps=args.video_steps,
        text_steps=args.text_steps,
        text_code_steps=args.text_code_steps,
        teacher_count=args.teacher_count,
        teacher_steps=args.teacher_steps,
        max_refs=args.max_refs,
        skip_teacher=bool(args.skip_teacher),
        skip_text=bool(args.skip_text),
    )


if __name__ == "__main__":
    main()
