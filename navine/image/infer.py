import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from navine.image.backends import generate_image
from navine.image.procedural import parse_prompt, prompt_seed
from navine.image.prompts import enhance_prompt
from navine.utils.config import load_config
from navine.utils.tier import load_modality_config, resolve_checkpoint_dir
from navine.utils.paths import get_output_dir


def generate(
    prompt: str,
    output_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    enhance: bool = True,
    force_procedural: bool = False,
    auto_ingest: bool = True,
    model_profile: Optional[str] = None,
) -> Path:
    if auto_ingest:
        try:
            from navine.nsfw.ingest import auto_ingest_before_generate

            auto_ingest_before_generate()
        except Exception:
            pass

    config = load_config(config_name) if config_name else load_modality_config("image")
    raw_prompt = prompt.strip()
    effective_prompt = enhance_prompt(raw_prompt, profile=model_profile) if enhance else raw_prompt
    out_dir = get_output_dir("image")
    out_path = Path(output_path) if output_path else out_dir / "generated.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    infer_cfg = config["inference"]
    steps = num_steps if num_steps is not None else infer_cfg["num_steps"]
    scheduler = str(infer_cfg.get("scheduler") or "ddpm").lower()
    if scheduler == "dpm":
        steps = max(40, int(steps * 0.75))
        scale = guidance_scale if guidance_scale is not None else infer_cfg["guidance_scale"]
        if guidance_scale is None:
            scale = float(scale) * 1.05
    else:
        scale = guidance_scale if guidance_scale is not None else infer_cfg["guidance_scale"]

    if seed is None:
        seed = prompt_seed(raw_prompt)

    gen_meta = generate_image(
        raw_prompt,
        out_path,
        effective_prompt,
        config=config,
        checkpoint=checkpoint,
        seed=seed,
        num_steps=steps,
        guidance_scale=scale,
        force_procedural=force_procedural,
    )
    if str(gen_meta.get("generation_mode") or "") == "failed" or (
        gen_meta.get("error") and not out_path.exists()
    ):
        raise RuntimeError(str(gen_meta.get("error") or "Image generation failed"))
    if not out_path.exists():
        raise RuntimeError(str(gen_meta.get("error") or "Image generation produced no file"))

    used_procedural = bool(gen_meta.get("procedural"))
    generation_mode = str(gen_meta.get("generation_mode") or "unknown")
    reference_info = None

    apply_guide = bool(infer_cfg.get("apply_reference_guide", False)) and not bool(
        infer_cfg.get("custom_trained_only", False)
    )
    if apply_guide and generation_mode in ("diffusion", "diffusers") and not used_procedural:
        try:
            from navine.reference.config import ensure_reference_dirs
            from navine.reference.media import apply_reference_to_output

            ensure_reference_dirs()
            ckpt_path = Path(checkpoint) if checkpoint else resolve_checkpoint_dir("image", config) / "latest.pt"
            reference_info = apply_reference_to_output(
                out_path,
                raw_prompt,
                size=int(infer_cfg.get("output_size", 512)),
                has_checkpoint=ckpt_path.exists(),
            )
            if reference_info:
                generation_mode = str(reference_info.get("generation_mode") or "diffusion_guide")
        except Exception:
            reference_info = None
    elif gen_meta.get("reference_used"):
        reference_info = gen_meta

    info = parse_prompt(raw_prompt)
    nsfw_info = {}
    try:
        from navine.nsfw.prompts import analyze_prompt

        nsfw_info = analyze_prompt(raw_prompt)
    except Exception:
        nsfw_info = {}

    meta = {
        "prompt": raw_prompt,
        "enhanced_prompt": effective_prompt,
        "num_steps": steps,
        "guidance_scale": scale,
        "seed": seed,
        "procedural": used_procedural,
        "nsfw": bool(nsfw_info.get("nsfw") or info.get("nsfw")),
        "category": nsfw_info.get("category"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": str(out_path),
        "generation_mode": generation_mode,
    }

    if reference_info:
        meta["reference_used"] = True
        meta["reference_path"] = reference_info.get("reference_path")
        meta["keyword_matched"] = reference_info.get("keyword_matched")
        meta["reference_category"] = reference_info.get("category")
        meta["reference_strength"] = reference_info.get("reference_strength")
    else:
        meta["reference_used"] = bool(gen_meta.get("reference_used"))

    ckpt_path = Path(checkpoint) if checkpoint else resolve_checkpoint_dir("image", config) / "latest.pt"
    if generation_mode in ("diffusion", "diffusers") and ckpt_path.exists():
        meta["checkpoint"] = str(ckpt_path)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if infer_cfg.get("postprocess"):
        try:
            from navine.image.postprocess import postprocess_image
            from PIL import Image

            img = Image.open(out_path).convert("RGB")
            img = postprocess_image(img, upscale=bool(infer_cfg.get("postprocess_upscale", False)))
            img.save(out_path, format="PNG")
        except Exception:
            pass

    return out_path


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "a colorful pattern"
    path = generate(prompt)
    print(f"Saved to {path}")
