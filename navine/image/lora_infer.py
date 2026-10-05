from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.config import load_config
from navine.utils.paths import get_project_root

CONFIG_NAME = "lora_sd"
INSTALL_HINT = (
    "Navine AI - Python LoRA generation needs extra packages:\n"
    "  pip install -r requirements-external.txt\n"
    "  pip install peft"
)


def _output_dir() -> Path:
    path = get_project_root() / "outputs" / "lora"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adapter_dir(config: Dict[str, Any]) -> Path:
    out_cfg = config.get("output") or {}
    return get_project_root() / str(out_cfg.get("dir") or "checkpoints/lora_sd")


def lora_adapter_exists(config: Optional[Dict[str, Any]] = None) -> bool:
    cfg = config or load_config(CONFIG_NAME)
    adapter = _adapter_dir(cfg)
    return (adapter / "adapter_config.json").exists()


def _build_scheduler(pipe, name: str):
    try:
        if name == "dpmpp":
            from diffusers import DPMSolverMultistepScheduler

            return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if name == "euler_a":
            from diffusers import EulerAncestralDiscreteScheduler

            return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        return pipe.scheduler
    return pipe.scheduler


def generate_with_lora(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    lora_scale: Optional[float] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Path:
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(f"{INSTALL_HINT}\nImport error: {exc}") from None

    config = load_config(CONFIG_NAME)
    base_cfg = config.get("base_model") or {}
    infer_cfg = config.get("inference") or {}
    adapter = _adapter_dir(config)
    if not lora_adapter_exists(config):
        raise RuntimeError(
            "No trained LoRA adapter found. Train first: python -m navine.cli learn lora train"
        )

    from navine.device_manager import get_device, get_dtype

    device = get_device()
    dtype = get_dtype(device)
    model_id = str(base_cfg.get("model_id") or "runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = _build_scheduler(pipe, str(infer_cfg.get("scheduler") or "dpmpp"))
    pipe.load_lora_weights(str(adapter))
    scale = float(lora_scale if lora_scale is not None else infer_cfg.get("lora_scale", 0.85))
    try:
        pipe.fuse_lora(lora_scale=scale)
    except Exception:
        pass
    if device.type == "cuda":
        pipe = pipe.to(device)
    else:
        pipe.enable_attention_slicing()

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or str(infer_cfg.get("negative_prompt") or ""),
        num_inference_steps=int(steps or infer_cfg.get("num_inference_steps", 28)),
        guidance_scale=float(guidance_scale or infer_cfg.get("guidance_scale", 7.5)),
        width=int(infer_cfg.get("width", 512)),
        height=int(infer_cfg.get("height", 512)),
        generator=gen,
    )
    image = result.images[0]
    if output_path:
        out_path = Path(output_path)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = _output_dir() / f"lora_{stamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    meta = {
        "prompt": prompt,
        "model": model_id,
        "lora_adapter": str(adapter),
        "lora_scale": scale,
        "seed": seed,
        "generation_mode": "lora_sd",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.with_suffix(".json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    return out_path
