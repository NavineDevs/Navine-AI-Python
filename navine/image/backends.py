import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.utils import save_image

from navine.image.procedural import (
    can_procedural,
    generate_procedural,
    is_bland_like,
    is_noise_like,
    is_simple_scene_prompt,
    is_valid_generated_image,
    parse_prompt,
    prompt_seed,
)
from navine.utils.config import load_config
from navine.utils.tier import load_modality_config, resolve_checkpoint_dir

logger = logging.getLogger(__name__)


def _schedule_lora_merge_if_ready() -> None:
    import threading

    def _run() -> None:
        try:
            from navine.image.lora_custom import adapter_exists
            from navine.lora.cycle import merge_image_lora

            if not adapter_exists():
                return
            result = merge_image_lora(force_merge=False)
            if result.get("merged") and result.get("adapter_removed"):
                logger.info("Image LoRA merged into base and adapter removed (score=%s)", result.get("post_score"))
        except Exception as exc:
            logger.debug("LoRA auto-merge skipped: %s", exc)

    threading.Thread(target=_run, daemon=True).start()


def generate_from_lora(prompt: str, out_path: Path, seed: Optional[int] = None) -> bool:
    try:
        from navine.image.lora_infer import generate_with_lora, lora_adapter_exists

        if not lora_adapter_exists():
            return False
        generated = generate_with_lora(prompt, output_path=str(out_path), seed=seed)
        return generated.exists() and is_valid_generated_image(out_path)
    except Exception:
        return False


def generate_from_custom_lora(
    prompt: str,
    out_path: Path,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Tuple[bool, bool]:
    try:
        from navine.image.lora_custom import adapter_exists, generate_custom_lora

        if not adapter_exists():
            return False, False
        generate_custom_lora(
            prompt,
            output_path=str(out_path),
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        if not out_path.exists() or out_path.stat().st_size < 1024:
            return False, False
        if not is_valid_generated_image(out_path):
            return False, True
        return True, False
    except Exception as exc:
        logger.warning("custom LoRA generation failed: %s", exc)
        return False, False


def _output_size(config: dict) -> int:
    return int(config.get("inference", {}).get("output_size", 512))


def _center_crop_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def _pick_local_reference(prompt: str, size: int, seed: Optional[int] = None) -> Optional[Image.Image]:
    try:
        from navine.reference.config import ensure_reference_dirs
        from navine.reference.media import load_reference_image, resolve_reference_for_prompt

        ensure_reference_dirs()
        info = resolve_reference_for_prompt(prompt, prefer_video=False, has_checkpoint=False)
        if not info:
            return None
        ref_path = Path(str(info["reference_path"]))
        ref = load_reference_image(ref_path, size=None).convert("RGB")
        if ref.size[0] < 64 or ref.size[1] < 64:
            return None
        return _center_crop_square(ref, size)
    except Exception:
        return None


def generate_from_reference(
    prompt: str,
    out_path: Path,
    size: int = 512,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    try:
        from navine.reference.config import ensure_reference_dirs
        from navine.reference.media import load_reference_image, resolve_reference_for_prompt
        from navine.utils.paths import get_project_root

        ensure_reference_dirs()
        info = None
        ref_path = None
        root = get_project_root()
        lower = (prompt or "").lower()
        pool_dirs = []
        if any(k in lower for k in ("sky", "cloud", "sunset", "sunrise", "horizon")):
            pool_dirs.append(root / "data" / "image" / "sfw" / "skies")
        if any(k in lower for k in ("hill", "mountain", "landscape", "field", "forest", "tree", "nature", "grass")):
            pool_dirs.append(root / "data" / "image" / "sfw" / "landscapes")
        if any(k in lower for k in ("person", "people", "portrait", "woman", "man", "face", "girl", "boy")):
            pool_dirs.append(root / "data" / "image" / "sfw" / "people")
        if pool_dirs:
            files: List[Path] = []
            for folder in pool_dirs:
                if not folder.is_dir():
                    continue
                files.extend(
                    [
                        p
                        for p in folder.rglob("*")
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                    ]
                )
            if files:
                rng = random.Random(seed if seed is not None else prompt_seed(prompt))
                ref_path = files[rng.randrange(0, len(files))]
                info = {
                    "category": "local_sfw",
                    "keyword_matched": "local_pool",
                    "reference_path": str(ref_path),
                    "reference_used": True,
                    "reference_strength": 0.0,
                }
        if ref_path is None:
            info = resolve_reference_for_prompt(prompt, prefer_video=False, has_checkpoint=False)
            if info:
                ref_path = Path(str(info["reference_path"]))
        if ref_path is None or not ref_path.exists():
            root = get_project_root()
            files = []
            for folder in (
                root / "data" / "image" / "sfw" / "skies",
                root / "data" / "image" / "sfw" / "landscapes",
                root / "data" / "image" / "sfw" / "objects",
            ):
                if folder.is_dir():
                    files.extend(
                        [
                            p
                            for p in folder.rglob("*")
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                        ]
                    )
            if files:
                rng = random.Random(seed if seed is not None else prompt_seed(prompt))
                ref_path = files[rng.randrange(0, len(files))]
                info = {
                    "category": "local_sfw",
                    "keyword_matched": "local_pool",
                    "reference_path": str(ref_path),
                    "reference_used": True,
                    "reference_strength": 0.0,
                }
        if ref_path is None or not ref_path.exists():
            return None
        ref = load_reference_image(ref_path, size=None).convert("RGB")
        if ref.size[0] < 64 or ref.size[1] < 64:
            return None
        rng = random.Random(seed if seed is not None else prompt_seed(prompt))
        img = _center_crop_square(ref, size)
        if rng.random() > 0.55:
            img = ImageOps.mirror(img)
        fill = img.resize((1, 1)).getpixel((0, 0))
        angle = rng.uniform(-1.2, 1.2)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
        img = ImageEnhance.Color(img).enhance(0.94 + rng.random() * 0.14)
        img = ImageEnhance.Brightness(img).enhance(0.90 + rng.random() * 0.18)
        img = ImageEnhance.Contrast(img).enhance(0.92 + rng.random() * 0.14)
        img = ImageEnhance.Sharpness(img).enhance(1.05 + rng.random() * 0.25)
        inset = int(size * rng.uniform(0.0, 0.03))
        if inset > 0:
            img = img.crop((inset, inset, size - inset, size - inset)).resize(
                (size, size), Image.Resampling.LANCZOS
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG")
        if not is_valid_generated_image(out_path) or is_noise_like(out_path):
            return None
        meta = dict(info or {})
        meta["generation_mode"] = "reference_real"
        meta["reference_used"] = True
        meta["reference_strength"] = 0.0
        return meta
    except Exception:
        return None


def generate_from_checkpoint(
    prompt: str,
    out_path: Path,
    config: dict,
    checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Tuple[bool, bool]:
    from navine.device_manager import get_device
    from navine.image.model import NavineDiffusionModel

    ckpt_dir = resolve_checkpoint_dir("image", config)
    infer_cfg = config["inference"]
    candidates: List[Path] = []
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            candidates.append(path)
    else:
        preferred = [str(x).strip() for x in (infer_cfg.get("prefer_checkpoints") or []) if str(x).strip()]
        ordered_names = preferred + [
            "latest_good.pt",
            "latest_post_mix.pt",
            "latest_mix3500.pt",
            "latest_pre_mix.pt",
            "latest.pt",
        ]
        init_path = ckpt_dir / "latest_800m_init.pt"
        init_size = init_path.stat().st_size if init_path.exists() else -1
        seen = set()
        for name in ordered_names:
            path = Path(name) if (":" in name or name.startswith("/") or name.startswith("\\")) else ckpt_dir / name
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen or not path.exists():
                continue
            if path.name in ("latest_800m_init.pt",):
                continue
            if init_size > 0 and path.name == "latest.pt" and path.stat().st_size == init_size:
                logger.warning("skipping untrained image init checkpoint %s", path.name)
                continue
            seen.add(key)
            candidates.append(path)
    if not candidates:
        return False, False

    steps = int(num_steps if num_steps is not None else infer_cfg["num_steps"])
    scale = float(guidance_scale if guidance_scale is not None else infer_cfg["guidance_scale"])
    if bool(infer_cfg.get("custom_trained_only", False)) or bool(infer_cfg.get("prefer_trained_model", False)):
        scale = float(min(max(scale, 1.0), 2.2))
        steps = int(max(8, min(steps, 48)))
    device = get_device()
    if seed is not None:
        torch.manual_seed(seed)
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = float(free_mem) / (1024**3)
            if free_gb < 1.5:
                steps = min(steps, 12)
                logger.warning("low VRAM free=%.2fGB; using %s image steps", free_gb, steps)
            if free_gb < 0.6:
                return False, True
    except Exception:
        pass

    last_noise = False
    for path in candidates:
        model = None
        try:
            model = NavineDiffusionModel.load_checkpoint(
                path,
                config,
                device,
                prefer_saved_architecture=True,
                allow_fresh_fallback=False,
            )
        except Exception as exc:
            logger.warning("image checkpoint load failed (%s): %s", path, exc)
            continue
        if not getattr(model, "checkpoint_compatible", True):
            continue
        model = model.to(device)
        model_size = int(model.image_size)
        scheduler = str(infer_cfg.get("scheduler") or "ddim").lower()
        use_steps = steps
        try:
            samples = model.sample(
                batch_size=1,
                text=prompt,
                device=device,
                num_steps=use_steps,
                guidance_scale=scale,
                scheduler=scheduler,
                init_image=None,
                init_noise_strength=float(infer_cfg.get("init_noise_strength", 0.55)),
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image((samples + 1) / 2, out_path)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise RuntimeError(
                    "Image generation ran out of GPU memory. Stop training, then retry."
                ) from exc
            logger.warning("image sample failed (%s): %s", path.name, exc)
            last_noise = True
            continue
        finally:
            try:
                del model
            except Exception:
                pass
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            size = _output_size(config)
            img = Image.open(out_path).convert("RGB")
            if img.size[0] != size or img.size[1] != size:
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            try:
                from navine.image.postprocess import postprocess_image

                img = postprocess_image(img, upscale=False, sharpen=True, denoise=True)
            except Exception:
                pass
            img.save(out_path, format="PNG")
        except Exception:
            pass
        if is_noise_like(out_path) or is_bland_like(out_path):
            last_noise = True
            logger.warning("image checkpoint %s produced noise/bland; trying next", path.name)
            continue
        if not is_valid_generated_image(out_path):
            last_noise = True
            continue
        try:
            from navine.image.quality import score_image_path

            scored = score_image_path(out_path)
            if scored and not bool(scored.get("passes", False)):
                score = float(scored.get("score") or 0.0)
                structure = float(scored.get("structure") or 0.0)
                # Prefer any non-noise diffusion sample over stock/reference fakes.
                if score >= 0.18 and structure >= 0.12:
                    logger.info(
                        "image accepting checkpoint %s with soft quality score=%s structure=%s",
                        path.name,
                        score,
                        structure,
                    )
                else:
                    last_noise = True
                    logger.warning(
                        "image checkpoint %s failed quality score=%s structure=%s; trying next",
                        path.name,
                        scored.get("score"),
                        scored.get("structure"),
                    )
                    continue
        except Exception:
            pass
        logger.info("image generated from %s", path.name)
        return True, False
    return False, last_noise


def generate_from_diffusers(
    prompt: str,
    out_path: Path,
    config: dict,
    seed: Optional[int] = None,
) -> bool:
    return False


def generate_procedural_art(prompt: str, out_path: Path, size: int = 512) -> bool:
    info = parse_prompt(prompt)
    if not (info.get("anime") or info.get("nsfw") or info.get("character")):
        return False
    render_size = max(512, int(size))
    img = generate_procedural(prompt, size=render_size)
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path.exists() and not is_noise_like(out_path)


def generate_simple_procedural(prompt: str, out_path: Path, size: int = 512) -> bool:
    info = parse_prompt(prompt)
    allowed = any(
        info[k]
        for k in ("moon", "sun", "sunset", "sky", "cat", "dog", "ocean", "heart", "gradient")
    )
    if not allowed and not info.get("color") and not is_simple_scene_prompt(prompt):
        return False
    if info.get("character") or info.get("nsfw") or info.get("anime"):
        return False
    render_size = max(512, int(size))
    img = generate_procedural(prompt, size=render_size)
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    if is_noise_like(out_path):
        return False
    return True


def generate_image(
    prompt: str,
    out_path: Path,
    effective_prompt: str,
    config: Optional[dict] = None,
    checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    force_procedural: bool = False,
) -> Dict[str, Any]:
    config = config or load_modality_config("image")
    size = _output_size(config)
    infer_cfg = config.get("inference") or {}
    custom_only = True
    ordered = [str(b).strip().lower() for b in (infer_cfg.get("backend_order") or []) if str(b).strip()]
    backends = ordered or ["checkpoint", "reference", "procedural"]
    backends = [b for b in backends if b not in ("diffusers", "lora", "custom_lora")]
    meta: Dict[str, Any] = {
        "generation_mode": "unknown",
        "reference_used": False,
        "procedural": False,
    }
    prompt_info = parse_prompt(prompt)

    allow_reference = bool(infer_cfg.get("allow_reference_fallback", False)) or bool(
        infer_cfg.get("allow_reference_on_noise", False)
    )
    allow_procedural = bool(infer_cfg.get("allow_procedural_on_noise", False))
    ckpt_dir = resolve_checkpoint_dir("image", config)
    ckpt_path = Path(checkpoint) if checkpoint else ckpt_dir / "latest_good.pt"
    has_ckpt = ckpt_path.exists() or (ckpt_dir / "latest_good.pt").exists() or (ckpt_dir / "latest.pt").exists()

    if force_procedural and generate_simple_procedural(prompt, out_path, size):
        meta["generation_mode"] = "procedural"
        meta["procedural"] = True
        return meta

    prefer_simple = bool(infer_cfg.get("prefer_procedural_simple_scenes", False))
    if (
        prefer_simple
        and allow_procedural
        and is_simple_scene_prompt(prompt)
        and generate_simple_procedural(prompt, out_path, size)
    ):
        meta["generation_mode"] = "procedural"
        meta["procedural"] = True
        return meta

    if custom_only:
        backends = [b for b in backends if b not in ("diffusers", "lora", "custom_lora")]

    checkpoint_was_noisy = False
    for backend in backends:
        if backend == "reference" and has_ckpt and not allow_reference and not checkpoint_was_noisy:
            continue
        if backend in ("custom_lora", "diffusers", "lora"):
            continue
        if backend == "checkpoint":
            ok, bad = generate_from_checkpoint(
                effective_prompt, out_path, config, checkpoint, seed, num_steps, guidance_scale
            )
            if ok and not bad:
                meta["generation_mode"] = "diffusion"
                return meta
            if bad:
                checkpoint_was_noisy = True
                meta["checkpoint_rejected_noise"] = True
                if allow_reference and bool(infer_cfg.get("allow_reference_on_noise", False)):
                    ref_meta = generate_from_reference(prompt, out_path, size=size, seed=seed)
                    if ref_meta:
                        meta.update(ref_meta)
                        meta["recovered_from_noise"] = True
                        return meta
                if allow_procedural and generate_simple_procedural(prompt, out_path, size):
                    meta["generation_mode"] = "procedural"
                    meta["procedural"] = True
                    meta["recovered_from_noise"] = True
                    return meta
        elif backend == "procedural_art":
            if allow_procedural and generate_procedural_art(prompt, out_path, size):
                meta["generation_mode"] = "procedural_art"
                meta["procedural"] = True
                return meta
        elif backend == "diffusers":
            if custom_only:
                continue
            if generate_from_diffusers(effective_prompt, out_path, config, seed):
                meta["generation_mode"] = "diffusers"
                return meta
        elif backend == "lora":
            if custom_only:
                continue
            if generate_from_lora(effective_prompt, out_path, seed=seed):
                meta["generation_mode"] = "lora_sd"
                return meta
        elif backend == "reference":
            if not allow_reference:
                continue
            ref_meta = generate_from_reference(prompt, out_path, size=size, seed=seed)
            if ref_meta:
                meta.update(ref_meta)
                return meta
        elif backend == "procedural":
            if not allow_procedural:
                continue
            if generate_simple_procedural(prompt, out_path, size):
                meta["generation_mode"] = "procedural"
                meta["procedural"] = True
                return meta

    if allow_procedural and can_procedural(prompt) and generate_simple_procedural(prompt, out_path, size):
        meta["generation_mode"] = "procedural"
        meta["procedural"] = True
        meta["recovered_from_noise"] = True
        return meta

    meta["generation_mode"] = "failed"
    meta["error"] = (
        "Image model could not produce a real diffusion sample. "
        "Trained checkpoint may still be learning — keep limit-train running, then retry."
    )
    return meta
