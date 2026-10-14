import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from navine.image.external import (
    EXTERNAL_CONFIG_NAME,
    _timestamp_name,
    batch_nsfw_prompts,
    external_output_dir,
    nsfw_negative_prompt,
    pick_nsfw_prompt,
    require_diffusers,
)
from navine.image.procedural import prompt_seed
from navine.nsfw.prompts import enhance_nsfw_prompt
from navine.utils.config import load_config
from navine.utils.paths import get_output_dir, get_project_root

EXTERNAL_CONFIG_NAME_VIDEO = "video_external"


def load_external_config():
    return load_config(EXTERNAL_CONFIG_NAME_VIDEO)


def external_video_learn_dir() -> Path:
    config = load_external_config()
    rel = (config.get("learn") or {}).get("output_dir") or "data/learn/external/video"
    path = get_project_root() / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_video_output_path() -> Path:
    return external_output_dir("video") / _timestamp_name("video", ".mp4")


def generate_external(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    config_name: Optional[str] = None,
    nsfw_enhance: bool = False,
    category: Optional[str] = None,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    from navine.utils.generation import require_external_teacher

    require_external_teacher()
    require_diffusers()
    config = load_config(config_name) if config_name else load_external_config()
    infer_cfg = config.get("inference") or {}
    raw_prompt = prompt.strip() or "motion sequence"
    if seed is None:
        seed = prompt_seed(raw_prompt)
    n_frames = num_frames if num_frames is not None else infer_cfg.get("num_frames", 32)
    frame_fps = fps if fps is not None else infer_cfg.get("fps", 24)
    if output_path:
        out_path = Path(output_path)
        if out_path.suffix.lower() not in (".mp4", ".webm", ".gif"):
            out_path = out_path.with_suffix(".mp4")
    else:
        out_path = _default_video_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_config_name = str(config.get("image_config") or EXTERNAL_CONFIG_NAME)
    fallback_cfg = dict(config.get("fallback") or {})
    fallback_cfg["use_external_image"] = True
    fallback_cfg["image_config"] = image_config_name
    patched = dict(config)
    patched["fallback"] = fallback_cfg
    effective = enhance_nsfw_prompt(raw_prompt) if nsfw_enhance else raw_prompt
    result_path = _generate_hybrid_external(
        effective,
        out_path,
        patched,
        n_frames,
        frame_fps,
        seed,
        image_config_name=image_config_name,
        nsfw_enhance=nsfw_enhance,
        category=category,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )
    diff_cfg = load_config(image_config_name).get("inference", {}).get("diffusers") or {}
    meta = {
        "prompt": raw_prompt,
        "enhanced_prompt": effective,
        "num_frames": n_frames,
        "fps": frame_fps,
        "seed": seed,
        "fallback": True,
        "secondary_path": True,
        "nsfw_enhance": nsfw_enhance,
        "category": category,
        "generation_mode": "external_diffusion_hybrid",
        "teacher_source": (config.get("learn") or {}).get("teacher_source", "huggingface_diffusers"),
        "image_model_id": diff_cfg.get("model_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": str(result_path),
    }
    meta_path = result_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_path


def generate_nsfw_external(
    category: str,
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    raw = (prompt or pick_nsfw_prompt(category, index)).strip()
    return generate_external(
        raw,
        output_path=output_path,
        seed=seed,
        num_frames=num_frames,
        fps=fps,
        nsfw_enhance=True,
        category=category,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def generate_hentai_external(
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    return generate_nsfw_external(
        "hentai",
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        num_frames=num_frames,
        fps=fps,
        index=index,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def generate_porn_external(
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    return generate_nsfw_external(
        "porn",
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        num_frames=num_frames,
        fps=fps,
        index=index,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def generate_real_external(
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    index: int = 0,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    return generate_nsfw_external(
        "real",
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        num_frames=num_frames,
        fps=fps,
        index=index,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )


def batch_generate_nsfw_videos(
    category: str,
    count: int = 3,
    save_for_learning: bool = True,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
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
            learn_dir = external_video_learn_dir()
            out_path = learn_dir / f"teacher_{category}_{index + 1:03d}.mp4"
            video_path = generate_nsfw_external(
                category,
                prompt=prompt,
                output_path=str(out_path),
                seed=item_seed,
                num_frames=num_frames,
                fps=fps,
                index=index,
                model_profile=model_profile,
                cycle_index=cycle_index + index,
            )
            meta_path = video_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved.append(save_teaching_video(prompt, video_path, metadata=meta))
        else:
            saved.append(
                generate_nsfw_external(
                    category,
                    prompt=prompt,
                    seed=item_seed,
                    num_frames=num_frames,
                    fps=fps,
                    index=index,
                    model_profile=model_profile,
                    cycle_index=cycle_index + index,
                )
            )
    return saved


def _generate_hybrid_external(
    prompt: str,
    out_path: Path,
    config: dict,
    num_frames: int,
    fps: int,
    seed: Optional[int],
    image_config_name: str,
    nsfw_enhance: bool = False,
    category: Optional[str] = None,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    from navine.video import infer as video_infer

    fallback_cfg = config.get("fallback") or {}
    infer_cfg = config.get("inference") or {}
    frame_size = int(infer_cfg.get("frame_output_size") or config["model"]["frame_size"])
    image_steps = int(fallback_cfg.get("image_steps", 28))
    motion = float(fallback_cfg.get("motion", 0.02))
    variation = float(fallback_cfg.get("keyframe_variation", 0.02))
    profile_name, profile_scale = video_infer._motion_profile(prompt, config)
    keyframe_count = 2
    keyframes = _render_external_keyframes(
        prompt,
        frame_size,
        image_steps,
        keyframe_count,
        seed,
        variation,
        image_config_name=image_config_name,
        nsfw_enhance=nsfw_enhance,
        category=category,
        model_profile=model_profile,
        cycle_index=cycle_index,
    )
    base = keyframes[0]
    frames = video_infer._animate_advanced(
        base,
        max(4, min(num_frames, int(fallback_cfg.get("max_frames", num_frames)))),
        motion,
        profile_name,
        profile_scale * 0.65,
        keyframes=keyframes,
    )
    return video_infer._save_video(frames, out_path, fps, config)


def _render_external_keyframes(
    prompt: str,
    frame_size: int,
    image_steps: int,
    count: int,
    seed: Optional[int],
    variation: float,
    image_config_name: str,
    nsfw_enhance: bool = False,
    category: Optional[str] = None,
    model_profile: Optional[str] = None,
    cycle_index: int = 0,
) -> list:
    import math

    from navine.image.external import generate_external, generate_nsfw_external
    from PIL import Image

    frames = []
    base_seed = seed if seed is not None else prompt_seed(prompt)
    tmp_dir = get_output_dir("video") / "external_keyframes"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    neg = nsfw_negative_prompt() if nsfw_enhance else None
    for idx in range(max(1, count)):
        frame_seed = base_seed + idx * 17
        tmp_path = tmp_dir / f"keyframe_{idx}.png"
        if nsfw_enhance and category:
            generate_nsfw_external(
                category,
                prompt=prompt,
                output_path=str(tmp_path),
                seed=frame_seed,
                model_profile=model_profile,
                cycle_index=cycle_index + idx,
            )
        else:
            generate_external(
                prompt,
                output_path=str(tmp_path),
                seed=frame_seed,
                enhance=not nsfw_enhance,
                config_name=image_config_name,
                nsfw_enhance=nsfw_enhance,
                negative_prompt=neg,
                model_profile=model_profile,
                cycle_index=cycle_index + idx,
            )
        img = Image.open(tmp_path).convert("RGB")
        if img.size[0] != frame_size or img.size[1] != frame_size:
            img = img.resize((frame_size, frame_size), Image.Resampling.LANCZOS)
        if idx > 0 and variation > 0:
            angle = math.sin(idx * 1.7) * variation * 12
            shift = int(variation * frame_size * 0.08 * idx)
            img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
            img = img.transform(
                (frame_size, frame_size),
                Image.AFFINE,
                (1, 0, -shift, 0, 1, 0),
                resample=Image.Resampling.BICUBIC,
            )
        frames.append(img.convert("RGB"))
    return frames


def save_teaching_video(
    prompt: str,
    video_path: Path,
    metadata: Optional[dict] = None,
) -> Path:
    learn_dir = external_video_learn_dir()
    stem = video_path.stem
    dest = learn_dir / video_path.name
    if video_path.resolve() != dest.resolve():
        dest.write_bytes(video_path.read_bytes())
    sidecar = learn_dir / f"{stem}.json"
    entry = {
        "prompt": prompt.strip(),
        "caption": prompt.strip(),
        "filename": dest.name,
        "teacher_source": "external_diffusion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)
    sidecar.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    caption_path = learn_dir / f"{stem}.txt"
    caption_path.write_text(entry["caption"], encoding="utf-8")
    return dest
