"""Batch custom-checkpoint video generation per domain tag."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_output_dir, get_project_root
from navine.utils.tier import resolve_checkpoint_dir
from navine.video.smoke_eval import _frame_variance
from navine.video.train import VIDEO_PROMPTS

DEFAULT_SEED_START = 42


def run_batch_domain_generate(
    count: int = 16,
    seed_start: int = DEFAULT_SEED_START,
    tags: Optional[List[str]] = None,
    config: Optional[dict] = None,
    output_root: Optional[Path] = None,
    require_cuda: bool = False,
) -> Dict[str, Any]:
    from navine.device_manager import print_train_device_banner
    from navine.utils.tier import load_modality_config
    from navine.video.infer import generate as generate_video

    if require_cuda:
        print_train_device_banner(require_cuda=True)
    config = config or load_modality_config("video")
    ckpt_path = resolve_checkpoint_dir("video", config) / "latest.pt"
    batch_root = Path(output_root) if output_root else get_output_dir("video") / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    selected = {k: v for k, v in VIDEO_PROMPTS.items() if not tags or k in tags}
    seeds = list(range(seed_start, seed_start + max(1, int(count))))
    min_var = float(config.get("inference", {}).get("min_frame_variance", 0.001))
    results: Dict[str, Any] = {"ok": True, "count_per_tag": count, "tags": {}}
    total = 0
    for tag, prompt in selected.items():
        tag_dir = batch_root / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        tag_stats = {"valid": 0, "items": []}
        for seed in seeds:
            path = tag_dir / f"{tag}_{seed:03d}.mp4"
            generate_video(
                prompt,
                output_path=str(path),
                config_name=None,
                seed=seed,
                auto_ingest=False,
            )
            exists = path.exists()
            size_ok = exists and path.stat().st_size > 4096
            variance = _frame_variance(path) if exists else 0.0
            valid = exists and size_ok and variance >= min_var
            meta = {
                "prompt": prompt,
                "generation_mode": "video_checkpoint",
                "reference_used": False,
                "custom_trained_only": True,
                "checkpoint": str(ckpt_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output": str(path),
                "valid": valid,
                "frame_variance": variance,
                "category_tag": tag,
                "seed": seed,
            }
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            tag_stats["items"].append({"seed": seed, "path": str(path), "valid": valid, "variance": variance})
            if valid:
                tag_stats["valid"] += 1
            total += 1
            print(f"batch {tag} seed={seed}: valid={valid} variance={variance:.5f} -> {path}")
        results["tags"][tag] = tag_stats
        if tag_stats["valid"] < count // 2:
            results["ok"] = False
    results["total"] = total
    report = get_project_root() / "logs" / "video_batch_domain.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Video batch domain ok={results['ok']} total={total} report={report}")
    return results


def run_review_gallery(
    seed: int = 0,
    tags: Optional[List[str]] = None,
    require_cuda: bool = False,
    review_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from navine.device_manager import print_train_device_banner
    from navine.utils.tier import load_modality_config
    from navine.video.infer import generate as generate_video

    if require_cuda:
        print_train_device_banner(require_cuda=True)
    config = load_modality_config("video")
    ckpt_path = resolve_checkpoint_dir("video", config) / "latest.pt"
    out_root = Path(review_dir) if review_dir else get_output_dir("video") / "review"
    out_root.mkdir(parents=True, exist_ok=True)
    selected = {k: v for k, v in VIDEO_PROMPTS.items() if not tags or k in tags}
    min_var = float(config.get("inference", {}).get("min_frame_variance", 0.001))
    results: Dict[str, Any] = {"ok": True, "seed": seed, "items": {}}
    for tag, prompt in selected.items():
        path = out_root / f"{tag}_seed{seed}.mp4"
        generate_video(
            prompt,
            output_path=str(path),
            config_name=None,
            seed=seed,
            auto_ingest=False,
        )
        exists = path.exists()
        size_ok = exists and path.stat().st_size > 4096
        variance = _frame_variance(path) if exists else 0.0
        valid = exists and size_ok and variance >= min_var
        meta = {
            "prompt": prompt,
            "seed": seed,
            "category_tag": tag,
            "valid": valid,
            "frame_variance": variance,
            "size_bytes": path.stat().st_size if exists else 0,
            "checkpoint": str(ckpt_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output": str(path),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results["items"][tag] = meta
        if not valid:
            results["ok"] = False
        print(f"review {tag} seed={seed}: valid={valid} variance={variance:.5f} -> {path}")
    report = get_project_root() / "logs" / "video_review_seed0.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Video review gallery ok={results['ok']} report={report}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate custom domain videos")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for review gallery (e.g. 0)")
    parser.add_argument("--review-dir", type=str, default="")
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or None
    if args.seed is not None or args.review_dir:
        run_review_gallery(
            seed=0 if args.seed is None else int(args.seed),
            tags=tags,
            require_cuda=args.require_cuda,
            review_dir=Path(args.review_dir) if args.review_dir else None,
        )
        return
    run_batch_domain_generate(
        count=args.count,
        seed_start=args.seed_start,
        tags=tags,
        require_cuda=args.require_cuda,
    )


if __name__ == "__main__":
    main()
