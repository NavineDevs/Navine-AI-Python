from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.paths import get_output_dir, get_project_root


INSTALL_HINT = (
    "Navine AI - Python video diffusion needs extra packages:\n"
    "  pip install -r requirements-external.txt\n"
    "And for some pipelines you may need:\n"
    "  pip install diffusers transformers accelerate imageio[ffmpeg] av\n"
)


def _stamp(prefix: str, suffix: str) -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%d_%H%M%S{suffix}")


def generate_diffusers_video(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: int = 24,
    fps: int = 12,
    width: int = 512,
    height: int = 512,
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    image_seed: Optional[str] = None,
) -> Optional[Path]:
    try:
        import torch
    except Exception:
        return None

    try:
        from diffusers import StableVideoDiffusionPipeline
    except Exception:
        return None

    try:
        from navine.device_manager import get_device, get_dtype
        from navine.video.interpolate_rife import interpolate_rife
    except Exception:
        return None

    device = get_device()
    dtype = get_dtype(device)

    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if device.type == "cuda":
            pipe = pipe.to(device)
        else:
            pipe.enable_attention_slicing()
    except Exception:
        return None

    init_image = None
    if image_seed:
        try:
            from PIL import Image

            init_image = Image.open(image_seed).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        except Exception:
            init_image = None

    if init_image is None:
        try:
            from navine.image.external import generate_external

            seed_path = get_output_dir("video") / "_svd_seed.png"
            generated = generate_external(prompt, output_path=str(seed_path), seed=seed)
            from PIL import Image

            init_image = Image.open(generated).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        except Exception:
            return None

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    try:
        result = pipe(
            image=init_image,
            num_frames=int(num_frames),
            generator=gen,
        )
        frames = [frame.convert("RGB") for frame in result.frames[0]]
    except Exception:
        return None

    if len(frames) >= 2 and fps >= 20 and len(frames) < 40:
        try:
            frames = interpolate_rife(frames, multiplier=2)
        except Exception:
            pass

    out_path = Path(output_path) if output_path else (get_output_dir("video") / _stamp("diffvid", ".mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v3 as iio

        iio.imwrite(out_path, frames, fps=int(fps))
    except Exception:
        return None

    meta = {
        "prompt": prompt,
        "model_id": model_id,
        "seed": seed,
        "num_frames": num_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "generation_mode": "diffusers_video",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.with_suffix(".json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    return out_path

