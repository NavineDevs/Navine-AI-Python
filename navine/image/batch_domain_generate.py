"""Batch custom-checkpoint image generation per domain tag."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torchvision.utils import save_image

from navine.image.procedural import is_noise_like, is_valid_generated_image
from navine.image.smoke_eval import SMOKE_PROMPTS
from navine.utils.paths import get_output_dir, get_project_root
from navine.utils.tier import resolve_checkpoint_dir

DEFAULT_SEED_START = 42


def _seeds_for_count(count: int, seed_start: int = DEFAULT_SEED_START) -> List[int]:
    return list(range(seed_start, seed_start + max(1, int(count))))


def run_batch_domain_generate(
    count: int = 16,
    seed_start: int = DEFAULT_SEED_START,
    tags: Optional[List[str]] = None,
    model=None,
    device=None,
    config: Optional[dict] = None,
    output_root: Optional[Path] = None,
    require_cuda: bool = False,
) -> Dict[str, Any]:
    from navine.device_manager import get_device, print_train_device_banner
    from navine.image.model import NavineDiffusionModel
    from navine.utils.tier import load_modality_config

    if require_cuda:
        print_train_device_banner(require_cuda=True)
    config = config or load_modality_config("image")
    device = device or get_device()
    if model is None:
        ckpt = resolve_checkpoint_dir("image", config) / "latest.pt"
        model = NavineDiffusionModel.load_checkpoint(ckpt, config, device=device).to(device)
    model.eval()
    ckpt_path = resolve_checkpoint_dir("image", config) / "latest.pt"
    batch_root = Path(output_root) if output_root else get_output_dir("image") / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    selected = {k: v for k, v in SMOKE_PROMPTS.items() if not tags or k in tags}
    seeds = _seeds_for_count(count, seed_start)
    num_steps = int(config.get("inference", {}).get("num_steps", 200))
    guidance = float(config.get("inference", {}).get("guidance_scale", 1.5))
    scheduler = str(config.get("inference", {}).get("scheduler") or "ddpm")
    out_size = int(config.get("inference", {}).get("output_size", 384))
    train_size = int(config.get("model", {}).get("image_size", 128))
    results: Dict[str, Any] = {"ok": True, "count_per_tag": count, "tags": {}}
    total = 0
    with torch.no_grad():
        for tag, prompt in selected.items():
            tag_dir = batch_root / tag
            tag_dir.mkdir(parents=True, exist_ok=True)
            tag_stats = {"valid": 0, "noise": 0, "items": []}
            for seed in seeds:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                sample = model.sample(
                    batch_size=1,
                    text=prompt,
                    device=device,
                    num_steps=num_steps,
                    guidance_scale=guidance,
                    scheduler=scheduler,
                )
                path = tag_dir / f"{tag}_{seed:03d}.png"
                save_image((sample + 1) / 2, path)
                if out_size != train_size:
                    from PIL import Image

                    Image.open(path).convert("RGB").resize((out_size, out_size)).save(path)
                valid = is_valid_generated_image(path)
                noisy = is_noise_like(path)
                meta = {
                    "prompt": prompt,
                    "enhanced_prompt": prompt,
                    "generation_mode": "diffusion",
                    "reference_used": False,
                    "procedural": False,
                    "checkpoint": str(ckpt_path),
                    "custom_trained_only": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "output": str(path),
                    "valid": valid,
                    "noise": noisy,
                    "category_tag": tag,
                    "seed": seed,
                }
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                tag_stats["items"].append({"seed": seed, "path": str(path), "valid": valid, "noise": noisy})
                if valid:
                    tag_stats["valid"] += 1
                if noisy:
                    tag_stats["noise"] += 1
                total += 1
                print(f"batch {tag} seed={seed}: valid={valid} noise={noisy} -> {path}")
            results["tags"][tag] = tag_stats
            if tag_stats["noise"] > count // 2:
                results["ok"] = False
    results["total"] = total
    report = get_project_root() / "logs" / "image_batch_domain.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Image batch domain ok={results['ok']} total={total} report={report}")
    return results


def run_review_gallery(
    seed: int = 0,
    tags: Optional[List[str]] = None,
    require_cuda: bool = False,
    review_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from navine.device_manager import print_train_device_banner
    from navine.image.infer import generate as generate_image
    from navine.image.quality import score_image_path
    from navine.utils.tier import load_modality_config

    if require_cuda:
        print_train_device_banner(require_cuda=True)
    config = load_modality_config("image")
    out_root = Path(review_dir) if review_dir else get_output_dir("image") / "review"
    out_root.mkdir(parents=True, exist_ok=True)
    selected = {k: v for k, v in SMOKE_PROMPTS.items() if not tags or k in tags}
    results: Dict[str, Any] = {"ok": True, "seed": seed, "items": {}}
    for tag, prompt in selected.items():
        path = out_root / f"{tag}_seed{seed}.png"
        generate_image(
            prompt,
            output_path=str(path),
            seed=seed,
            enhance=False,
            auto_ingest=False,
        )
        metrics = score_image_path(path)
        valid = is_valid_generated_image(path)
        noisy = is_noise_like(path)
        meta = {
            "prompt": prompt,
            "seed": seed,
            "category_tag": tag,
            "valid": valid,
            "noise": noisy,
            "score": metrics.get("score"),
            "lap_var": metrics.get("lap_var"),
            "std": metrics.get("std"),
            "passes": metrics.get("passes"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output": str(path),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results["items"][tag] = meta
        if not valid or noisy or not metrics.get("passes"):
            results["ok"] = False
        print(
            f"review {tag} seed={seed}: score={metrics.get('score')} "
            f"valid={valid} noise={noisy} -> {path}"
        )
    report = get_project_root() / "logs" / "image_review_seed0.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Image review gallery ok={results['ok']} report={report}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate custom domain images")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for review gallery (e.g. 0)")
    parser.add_argument("--review-dir", type=str, default="", help="Write seed review gallery here")
    parser.add_argument("--tags", type=str, default="", help="Comma-separated tags (default: all)")
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
