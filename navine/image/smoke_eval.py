"""Post-train smoke eval for custom-only image diffusion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torchvision.utils import save_image

from navine.image.procedural import is_noise_like, is_valid_generated_image
from navine.utils.paths import get_output_dir, get_project_root
from navine.utils.tier import resolve_checkpoint_dir

SMOKE_PROMPTS = {
    "sfw": "sfw clothed young adult woman portrait, natural light",
    "human": "photoreal adult man standing outdoors",
    "hentai": "hentai anime girl detailed face body",
    "object": "red ceramic mug on wooden table",
    "think": "abstract visualization of deep reasoning, glowing neural pathways connecting puzzle pieces, dark blue background",
    "detective": "noir detective office at night, cork board with red string clues, encrypted symbols, magnifying glass, cinematic mood",
}


def run_image_smoke_eval(
    model=None,
    device=None,
    config: Optional[dict] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from navine.device_manager import get_device
    from navine.image.model import NavineDiffusionModel
    from navine.utils.tier import load_modality_config

    config = config or load_modality_config("image")
    device = device or get_device()
    if model is None:
        ckpt = resolve_checkpoint_dir("image", config) / "latest.pt"
        model = NavineDiffusionModel.load_checkpoint(ckpt, config, device=device).to(device)
    model.eval()
    out_root = Path(output_dir) if output_dir else get_output_dir("image")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_checkpoint_dir("image", config) / "latest.pt"
    results: Dict[str, Any] = {"ok": True, "items": {}}
    with torch.no_grad():
        for tag, prompt in SMOKE_PROMPTS.items():
            torch.manual_seed(42 + hash(tag) % 1000)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(42 + hash(tag) % 1000)
            sample = model.sample(
                batch_size=1,
                text=prompt,
                device=device,
                num_steps=int(config.get("inference", {}).get("num_steps", 200)),
                guidance_scale=float(config.get("inference", {}).get("guidance_scale", 1.5)),
                scheduler=str(config.get("inference", {}).get("scheduler") or "ddpm"),
            )
            path = out_root / f"generated_{tag}.png"
            save_image((sample + 1) / 2, path)
            size = int(config.get("inference", {}).get("output_size", 384))
            if size != int(config.get("model", {}).get("image_size", 128)):
                from PIL import Image

                Image.open(path).convert("RGB").resize((size, size)).save(path)
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
            }
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            results["items"][tag] = {"path": str(path), "valid": valid, "noise": noisy}
            if not valid or noisy:
                results["ok"] = False
            print(f"smoke {tag}: valid={valid} noise={noisy} -> {path}")
    report = get_project_root() / "logs" / "image_smoke_eval.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Image smoke eval ok={results['ok']} report={report}")
    return results


if __name__ == "__main__":
    run_image_smoke_eval()
