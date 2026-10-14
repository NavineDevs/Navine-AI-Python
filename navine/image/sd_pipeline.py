from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PIPE_CACHE: Dict[str, Any] = {}


def _cache_key(model_id: str, pipeline_kind: str, device: str) -> str:
    return f"{pipeline_kind}|{model_id}|{device}"


def clear_pipeline_cache() -> None:
    with _LOCK:
        _PIPE_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_sd_pipeline(
    model_id: str,
    pipeline_kind: str = "stable_diffusion",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    local_files_only: bool = False,
) -> Tuple[Any, str]:
    from navine.device_manager import get_device, get_dtype

    device = device or get_device()
    dtype = dtype or get_dtype(device)
    kind = (pipeline_kind or "stable_diffusion").lower()
    key = _cache_key(model_id, kind, str(device))
    with _LOCK:
        cached = _PIPE_CACHE.get(key)
        if cached is not None:
            return cached, key

        logger.info("Loading image pipeline %s (%s) on %s", model_id, kind, device)
        kwargs = {
            "torch_dtype": dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if local_files_only:
            kwargs["local_files_only"] = True

        if kind in ("turbo", "sd_turbo", "auto"):
            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
        elif kind == "sdxl":
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)
        else:
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

        try:
            pipe.safety_checker = None
        except Exception:
            pass

        if device.type == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            pipe = pipe.to(device)
        else:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        _PIPE_CACHE[key] = pipe
        return pipe, key


def run_sd_txt2img(
    prompt: str,
    *,
    model_id: str,
    pipeline_kind: str = "stable_diffusion",
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    out_path: Optional[Path] = None,
    local_files_only: bool = False,
) -> Path:
    from navine.device_manager import get_device
    from navine.utils.paths import get_output_dir

    device = get_device()
    pipe, _ = load_sd_pipeline(
        model_id=model_id,
        pipeline_kind=pipeline_kind,
        device=device,
        local_files_only=local_files_only,
    )
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=str(gen_device) if gen_device.type == "cuda" else "cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    kind = (pipeline_kind or "stable_diffusion").lower()
    call_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "width": int(width),
        "height": int(height),
        "generator": generator,
    }
    if kind not in ("turbo", "sd_turbo") and negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    with torch.inference_mode():
        result = pipe(**call_kwargs)
    image = result.images[0]
    path = Path(out_path) if out_path else get_output_dir("image") / "generated.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path
