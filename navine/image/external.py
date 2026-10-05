import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from navine.image.backends import generate_from_diffusers
from navine.image.procedural import prompt_seed
from navine.image.prompts import enhance_prompt
from navine.nsfw.prompts import enhance_nsfw_prompt
from navine.utils.config import load_config
from navine.utils.paths import get_project_root

EXTERNAL_CONFIG_NAME = "image_external"
DIFFUSERS_INSTALL_HINT = (
    "Install HuggingFace diffusers for Navine AI - Python secondary generation:\n"
    "  pip install diffusers transformers accelerate safetensors"
)
NSFW_NEGATIVE_EXTRA = (
    "censored, mosaic, bar censor, pixelated, low resolution, extra limbs, "
    "malformed hands, duplicate, cropped"
)
DEFAULT_HENTAI_PROMPTS = [
    "anime hentai girl, detailed illustration, soft shading, explicit adult scene",
    "anime ecchi waifu, vibrant colors, detailed linework, lewd pose",
    "doujin style hentai character art, explicit adult, high detail",
    "anime girl blue hair, hentai illustration, sensual pose, detailed eyes",
]
DEFAULT_PORN_PROMPTS = [
    "photorealistic explicit adult woman, natural lighting, detailed skin texture",
    "amateur adult photo, bedroom lighting, sharp focus, realistic body",
    "explicit adult couple, photorealistic, natural pose, high detail",
    "adult model portrait, studio lighting, photorealistic explicit content",
]
DEFAULT_REAL_PROMPTS = [
    "photorealistic adult portrait, natural lighting, realistic skin texture",
    "realistic adult photo, soft window light, detailed face and body",
    "live action adult scene, cinematic lighting, photorealistic",
    "adult human subject, natural pose, sharp focus, realistic photography",
]
NSFW_PRESET_DEFAULTS = {
    "hentai": DEFAULT_HENTAI_PROMPTS,
    "porn": DEFAULT_PORN_PROMPTS,
    "real": DEFAULT_REAL_PROMPTS,
    "anime": DEFAULT_HENTAI_PROMPTS,
}


def diffusers_available() -> Tuple[bool, Optional[str]]:
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        return True, None
    except ImportError as exc:
        return False, str(exc)


def require_diffusers() -> None:
    ok, detail = diffusers_available()
    if ok:
        return
    message = DIFFUSERS_INSTALL_HINT
    if detail:
        message = f"{message}\nImport error: {detail}"
    raise RuntimeError(message)


def load_external_config() -> Dict[str, Any]:
    return load_config(EXTERNAL_CONFIG_NAME)


def external_output_dir(subdir: Optional[str] = None) -> Path:
    base = get_project_root() / "outputs" / "external"
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base


def nsfw_negative_prompt(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config or load_external_config()
    diff_cfg = (cfg.get("inference") or {}).get("diffusers") or {}
    base = str(
        diff_cfg.get("negative_prompt")
        or "blurry, low quality, deformed, ugly, bad anatomy, watermark, text"
    )
    return f"{base}, {NSFW_NEGATIVE_EXTRA}"


def _prompt_from_metadata(data: Dict[str, Any], category: str) -> str:
    tags = [str(t) for t in (data.get("tags") or []) if str(t).strip()]
    skip = {"infini", "atomic", "waifu"}
    tag_text = ", ".join(t for t in tags if t.lower() not in skip)
    media = str(data.get("media_category") or data.get("infini_tag") or category)
    subject = tag_text or media.replace("_", " ")
    if category in ("hentai", "anime"):
        return f"anime {media} illustration, {subject}, detailed linework, explicit adult"
    if category == "porn":
        return f"photorealistic explicit adult, {subject}, natural lighting, sharp focus"
    return f"photorealistic adult content, {subject}, natural lighting, detailed"


def load_local_nsfw_prompts(category: str, limit: int = 64) -> List[str]:
    folder = get_project_root() / "data" / "nsfw" / "local" / category
    if not folder.is_dir():
        return []
    prompts: List[str] = []
    for json_path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            text = _prompt_from_metadata(data, category).strip()
            if text and text not in prompts:
                prompts.append(text)
        except Exception:
            continue
        if len(prompts) >= limit:
            break
    if len(prompts) < limit:
        for image_path in sorted(folder.glob("*.*")):
            if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            stem = image_path.stem.replace("_", " ").replace("-", " ")
            if category in ("hentai", "anime"):
                text = f"anime hentai illustration, {stem}, detailed linework, explicit adult"
            elif category == "porn":
                text = f"photorealistic explicit adult photo, {stem}, natural lighting"
            else:
                text = f"photorealistic adult content, {stem}, natural lighting"
            if text not in prompts:
                prompts.append(text)
            if len(prompts) >= limit:
                break
    return prompts


def pick_nsfw_prompt(category: str, index: int = 0) -> str:
    key = "hentai" if category == "anime" else category
    local = load_local_nsfw_prompts(key)
    defaults = NSFW_PRESET_DEFAULTS.get(key) or NSFW_PRESET_DEFAULTS["hentai"]
    pool = local or defaults
    return pool[index % len(pool)]


def batch_nsfw_prompts(category: str, count: int) -> List[str]:
    key = "hentai" if category == "anime" else category
    local = load_local_nsfw_prompts(key)
    defaults = NSFW_PRESET_DEFAULTS.get(key) or NSFW_PRESET_DEFAULTS["hentai"]
    pool = local or defaults
    if count <= 0:
        return []
    if count <= len(pool):
        return list(pool[:count])
    items: List[str] = []
    idx = 0
    while len(items) < count:
        items.append(pool[idx % len(pool)])
        idx += 1
    return items


def _timestamp_name(prefix: str, suffix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{suffix}"


def list_teacher_model_profiles() -> Dict[str, Dict[str, Any]]:
    config = load_external_config()
    section = config.get("teacher_models") or {}
    profiles = section.get("profiles") or {}
    if profiles:
        return dict(profiles)
    diff_cfg = (config.get("inference") or {}).get("diffusers") or {}
    return {
        "sd15": {
            "label": "Stable Diffusion 1.5",
            "model_id": diff_cfg.get("model_id", "runwayml/stable-diffusion-v1-5"),
            "pipeline": diff_cfg.get("pipeline", "stable_diffusion"),
            "width": diff_cfg.get("width", 512),
            "height": diff_cfg.get("height", 512),
            "num_inference_steps": diff_cfg.get("num_inference_steps", 28),
            "guidance_scale": diff_cfg.get("guidance_scale", 7.5),
            "cpu_ok": True,
        }
    }


def resolve_teacher_model(
    profile_name: Optional[str] = None,
    cycle_index: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    config = load_external_config()
    section = config.get("teacher_models") or {}
    profiles = list_teacher_model_profiles()
    if profile_name:
        key = profile_name.strip()
        if key not in profiles:
            raise ValueError(
                f"Unknown teacher model profile '{key}'. "
                f"Available: {', '.join(sorted(profiles.keys()))}"
            )
        return key, dict(profiles[key])
    default_key = str(section.get("default") or "sd15")
    rotation = list(section.get("rotation") or [default_key])
    if not rotation:
        rotation = [default_key]
    key = rotation[cycle_index % len(rotation)]
    if key not in profiles:
        key = default_key if default_key in profiles else next(iter(profiles))
    return key, dict(profiles[key])


def apply_teacher_model(
    config: Dict[str, Any],
    profile_name: Optional[str] = None,
    cycle_index: int = 0,
) -> Dict[str, Any]:
    key, profile = resolve_teacher_model(profile_name, cycle_index)
    merged = copy.deepcopy(config)
    infer_cfg = merged.setdefault("inference", {})
    diff_cfg = dict(infer_cfg.get("diffusers") or {})
    for field in (
        "model_id",
        "pipeline",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "negative_prompt",
    ):
        if field in profile and profile[field] is not None:
            diff_cfg[field] = profile[field]
    diff_cfg["teacher_profile"] = key
    diff_cfg["teacher_label"] = profile.get("label", key)
    infer_cfg["diffusers"] = diff_cfg
    return merged


def external_learn_dir() -> Path:
    config = load_external_config()
    rel = (config.get("learn") or {}).get("output_dir") or "data/learn/external/images"
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_external(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    enhance: bool = True,
    config_name: Optional[str] = None,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
    nsfw_enhance: bool = False,
    negative_prompt: Optional[str] = None,
) -> Path:
    from navine.utils.generation import require_external_teacher

    require_external_teacher()
    require_diffusers()
    if config_name:
        config = load_config(config_name)
    else:
        config = apply_teacher_model(load_external_config(), model_profile, cycle_index)
    infer_cfg = config.get("inference") or {}
    if bool(infer_cfg.get("custom_trained_only", False)):
        raise RuntimeError(
            "External diffusion config has custom_trained_only enabled. "
            f"Use configs/{EXTERNAL_CONFIG_NAME}.yaml for the secondary path."
        )
    raw_prompt = prompt.strip()
    if not raw_prompt:
        raise ValueError("Prompt is required for external image generation.")
    if nsfw_enhance:
        effective = enhance_nsfw_prompt(raw_prompt)
    else:
        effective = enhance_prompt(raw_prompt) if enhance else raw_prompt
    if seed is None:
        seed = prompt_seed(raw_prompt)
    out_dir = external_output_dir()
    out_path = Path(output_path) if output_path else out_dir / _timestamp_name("image", ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = copy.deepcopy(config)
    if negative_prompt:
        patched_infer = dict(infer_cfg)
        patched_diff = dict(infer_cfg.get("diffusers") or {})
        patched_diff["negative_prompt"] = negative_prompt
        patched_infer["diffusers"] = patched_diff
        run_config["inference"] = patched_infer
    backends = infer_cfg.get("backend_order") or ["diffusers"]
    generated = False
    mode = "diffusers"
    for backend in backends:
        if backend == "diffusers":
            if generate_from_diffusers(effective, out_path, run_config, seed=seed):
                generated = True
                mode = "diffusers"
                break
        elif backend == "checkpoint":
            from navine.image.backends import generate_from_checkpoint

            ok, _bad = generate_from_checkpoint(
                effective,
                out_path,
                run_config,
                seed=seed,
            )
            if ok:
                generated = True
                mode = "checkpoint_fallback"
                break
    if not generated:
        raise RuntimeError(
            "Navine AI - Python external diffusion could not generate an image. "
            f"{DIFFUSERS_INSTALL_HINT}"
        )
    diff_cfg = infer_cfg.get("diffusers") or {}
    meta = {
        "prompt": raw_prompt,
        "enhanced_prompt": effective,
        "seed": seed,
        "generation_mode": mode,
        "secondary_path": True,
        "nsfw_enhance": nsfw_enhance,
        "negative_prompt": negative_prompt or nsfw_negative_prompt(config),
        "teacher_source": (config.get("learn") or {}).get("teacher_source", "huggingface_diffusers"),
        "model_id": diff_cfg.get("model_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": str(out_path),
    }
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


def generate_nsfw_external(
    category: str,
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    raw = (prompt or pick_nsfw_prompt(category, index)).strip()
    neg = nsfw_negative_prompt()
    return generate_external(
        raw,
        output_path=output_path,
        seed=seed,
        enhance=False,
        nsfw_enhance=True,
        negative_prompt=neg,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def generate_hentai_external(
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    return generate_nsfw_external(
        "hentai",
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        index=index,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def generate_porn_external(
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    return generate_nsfw_external(
        "porn",
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        index=index,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def generate_real_external(
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    return generate_nsfw_external(
        "real",
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        index=index,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def batch_generate_nsfw_images(
    category: str,
    count: int = 3,
    save_for_learning: bool = True,
    seed: Optional[int] = None,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> List[Path]:
    require_diffusers()
    prompts = batch_nsfw_prompts(category, count)
    saved: List[Path] = []
    base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    for index, prompt in enumerate(prompts):
        item_seed = base_seed + index * 31
        if save_for_learning:
            learn_dir = external_learn_dir()
            out_path = learn_dir / f"teacher_{category}_{index + 1:03d}.png"
            image_path = generate_nsfw_external(
                category,
                prompt=prompt,
                output_path=str(out_path),
                seed=item_seed,
                index=index,
                model_profile=model_profile,
                cycle_index=cycle_index + index,
            )
            meta_path = image_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved.append(save_teaching_pair(prompt, image_path, caption=prompt.strip(), metadata=meta))
        else:
            saved.append(
                generate_nsfw_external(
                    category,
                    prompt=prompt,
                    seed=item_seed,
                    index=index,
                    model_profile=model_profile,
                    cycle_index=cycle_index + index,
                )
            )
    return saved


def save_teaching_pair(
    prompt: str,
    image_path: Path,
    caption: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    learn_dir = external_learn_dir()
    stem = image_path.stem
    dest_image = learn_dir / image_path.name
    if image_path.resolve() != dest_image.resolve():
        dest_image.write_bytes(image_path.read_bytes())
    sidecar = learn_dir / f"{stem}.json"
    entry: Dict[str, Any] = {
        "prompt": prompt.strip(),
        "caption": caption or prompt.strip(),
        "filename": dest_image.name,
        "teacher_source": "external_diffusion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)
    sidecar.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    caption_path = learn_dir / f"{stem}.txt"
    caption_path.write_text(entry["caption"], encoding="utf-8")
    return dest_image
