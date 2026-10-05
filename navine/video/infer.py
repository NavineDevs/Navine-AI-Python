import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter

from navine.video.model import NavineVideoModel
from navine.utils.config import load_config
from navine.utils.paths import get_output_dir
from navine.utils.tier import load_modality_config, resolve_checkpoint_dir


def _image_tensor_from_prompt(
    prompt: str,
    frame_size: int,
    seed: Optional[int] = None,
) -> Optional[torch.Tensor]:
    try:
        from navine.image.backends import generate_image
        from navine.image.procedural import is_bland_like
        from navine.image.prompts import enhance_prompt

        config = load_modality_config("image")
        tmp_path = get_output_dir("video") / "_seed_frame.png"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        generate_image(
            prompt,
            tmp_path,
            enhance_prompt(prompt),
            config=config,
            seed=seed,
        )
        if not tmp_path.exists() or is_bland_like(tmp_path):
            return None
        img = Image.open(tmp_path).convert("RGB").resize((frame_size, frame_size), Image.Resampling.LANCZOS)
        import numpy as np

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0
    except Exception:
        return None


def _render_base_image(prompt: str, frame_size: int, image_steps: int, seed: Optional[int] = None) -> Image.Image:
    from navine.image.backends import generate_image
    from navine.image.prompts import enhance_prompt

    config = load_modality_config("image")
    tmp_path = get_output_dir("video") / "_keyframe.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        generate_image(
            prompt,
            tmp_path,
            enhance_prompt(prompt),
            config=config,
            seed=seed,
            num_steps=image_steps,
        )
        img = Image.open(tmp_path).convert("RGB")
        if frame_size and (img.size[0] != frame_size or img.size[1] != frame_size):
            img = img.resize((frame_size, frame_size), Image.Resampling.LANCZOS)
        return img
    except Exception:
        raise RuntimeError("Could not render video keyframe from the image model.")


def _ken_burns_frame(base: Image.Image, progress: float, zoom_start: float = 1.0, zoom_end: float = 1.08) -> Image.Image:
    w, h = base.size
    zoom = zoom_start + (zoom_end - zoom_start) * progress
    nw, nh = max(w + 2, int(w * zoom)), max(h + 2, int(h * zoom))
    enlarged = base.resize((nw, nh), Image.Resampling.LANCZOS)
    pan_x = int((nw - w) * progress * 0.35)
    pan_y = int((nh - h) * (0.15 + 0.25 * progress))
    left = max(0, min(nw - w, pan_x))
    top = max(0, min(nh - h, pan_y))
    return enlarged.crop((left, top, left + w, top + h))


def _animate_reference_frames(
    reference_frames: List[Image.Image],
    num_frames: int,
    frame_size: int,
) -> List[Image.Image]:
    if not reference_frames:
        return []
    frames: List[Image.Image] = []
    sources = [
        frame.convert("RGB").resize((frame_size, frame_size), Image.Resampling.LANCZOS)
        for frame in reference_frames
    ]
    if len(sources) == 1:
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            frames.append(_ken_burns_frame(sources[0], t))
        return frames
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        seg = t * (len(sources) - 1)
        left = int(seg)
        right = min(len(sources) - 1, left + 1)
        blend = seg - left
        if blend < 0.01:
            frame = sources[left]
        else:
            frame = Image.blend(sources[left], sources[right], blend)
        frames.append(_ken_burns_frame(frame, t * 0.6, 1.0, 1.05))
    return frames


def _render_keyframes(
    prompt: str,
    frame_size: int,
    image_steps: int,
    count: int,
    seed: Optional[int],
    variation: float,
    reference_frames: Optional[List[Image.Image]] = None,
    reference_category: str = "",
) -> List[Image.Image]:
    from navine.image.procedural import prompt_seed

    if reference_frames:
        from navine.reference.config import get_reference_settings
        from navine.reference.media import apply_reference_blend, reference_strength_for_category

        settings = get_reference_settings()
        ref_mode = str(settings.get("mode") or "guide").lower()
        category = str(reference_category or "")
        strength = reference_strength_for_category(category, has_checkpoint=True) if category else 0.35
        frames = []
        base_seed = seed if seed is not None else prompt_seed(prompt)
        for idx in range(max(1, count)):
            frame_seed = base_seed + idx * 17
            img = _render_base_image(prompt, frame_size, max(40, image_steps - idx * 8), frame_seed)
            ref = reference_frames[min(idx, len(reference_frames) - 1)]
            img = apply_reference_blend(
                img,
                ref,
                strength=strength,
                seed=base_seed + idx,
                category=category,
                mode=ref_mode,
            )
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
    frames = []
    base_seed = seed if seed is not None else prompt_seed(prompt)
    for idx in range(max(1, count)):
        frame_seed = base_seed + idx * 17
        img = _render_base_image(prompt, frame_size, max(40, image_steps - idx * 8), frame_seed)
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


def _motion_profile(prompt: str, config: dict) -> Tuple[str, float]:
    lower = prompt.lower()
    profiles = (config.get("fallback") or {}).get("motion_profiles") or {}
    dancing = False
    walking = False
    bouncing = False
    try:
        from navine.nsfw.prompts import analyze_prompt

        info = analyze_prompt(prompt)
        dancing = bool(info.get("motion"))
    except Exception:
        dancing = False
    if not dancing:
        dancing = any(w in lower for w in ("danc", "dance", "groov", "twirl", "spin", "hip hop", "choreograph"))
    walking = any(w in lower for w in ("walk", "run", "step", "stride", "jog"))
    bouncing = any(w in lower for w in ("bounce", "jump", "hop", "bob"))
    if dancing:
        return "dance", float(profiles.get("dance", 1.8))
    if walking:
        return "walk", float(profiles.get("walk", 1.2))
    if bouncing:
        return "bounce", float(profiles.get("bounce", 1.4))
    return "sway", float(profiles.get("sway", 1.0))


def _regional_warp(base: Image.Image, upper_dx: int, upper_dy: int, lower_dx: int, lower_dy: int) -> Image.Image:
    w, h = base.size
    split = int(h * 0.55)
    upper = base.crop((0, 0, w, split))
    lower = base.crop((0, split, w, h))
    upper = upper.transform(
        (w, split),
        Image.AFFINE,
        (1, 0, -upper_dx, 0, 1, -upper_dy),
        resample=Image.Resampling.BICUBIC,
    )
    lower = lower.transform(
        (w, h - split),
        Image.AFFINE,
        (1, 0, -lower_dx, 0, 1, -lower_dy),
        resample=Image.Resampling.BICUBIC,
    )
    out = Image.new("RGB", (w, h))
    out.paste(upper, (0, 0))
    out.paste(lower, (0, split))
    return out


def _blend_frames(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(a, b, max(0.0, min(1.0, alpha)))


def _animate_advanced(
    base: Image.Image,
    num_frames: int,
    motion: float,
    profile_name: str,
    profile_scale: float,
    keyframes: Optional[List[Image.Image]] = None,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    w, h = base.size
    motion_scale = profile_scale
    max_shift = max(2, int(w * motion * motion_scale))
    keyframes = keyframes or [base]
    for i in range(num_frames):
        phase = (i / max(1, num_frames)) * 2 * math.pi
        t = i / max(1, num_frames - 1)
        if len(keyframes) > 1:
            seg = t * (len(keyframes) - 1)
            left = int(seg)
            right = min(len(keyframes) - 1, left + 1)
            blend = seg - left
            frame_base = _blend_frames(keyframes[left], keyframes[right], blend)
        else:
            frame_base = base.copy()
        dx = int(math.sin(phase) * max_shift)
        dy = int(math.cos(phase * 2) * (max_shift // 2))
        angle = math.sin(phase) * (motion * 35 * motion_scale)
        if profile_name == "dance":
            upper_dx = int(math.sin(phase * 2) * max_shift)
            upper_dy = int(math.sin(phase) * (max_shift // 3))
            lower_dx = int(math.sin(phase * 3 + 1.2) * (max_shift + 2))
            lower_dy = int(math.cos(phase * 2) * (max_shift // 2))
            frame = _regional_warp(frame_base, upper_dx, upper_dy, lower_dx, lower_dy)
            angle += math.sin(phase * 2) * motion * 18
            dy += int(math.sin(phase * 3) * (max_shift // 2))
        elif profile_name == "walk":
            leg_shift = int(math.sin(phase * 2) * (max_shift + 1))
            frame = _regional_warp(frame_base, dx // 3, dy // 4, leg_shift, 0)
            dy += int(abs(math.sin(phase * 2)) * (max_shift // 4))
        elif profile_name == "bounce":
            dy = -int(abs(math.sin(phase * 2)) * max_shift)
            frame = frame_base.transform(
                (w, h),
                Image.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.BICUBIC,
            )
        else:
            frame = frame_base.transform(
                (w, h),
                Image.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.BICUBIC,
            )
        if profile_name != "bounce":
            frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        zoom = 1.0 + math.sin(phase) * motion * 0.06 * motion_scale
        nw, nh = int(w * zoom), int(h * zoom)
        if nw > 4 and nh > 4:
            zoomed = frame.resize((nw, nh), Image.Resampling.BICUBIC)
            left = (nw - w) // 2
            top = (nh - h) // 2
            frame = zoomed.crop((left, top, left + w, top + h))
        brightness = 1.0 + math.sin(phase) * 0.06
        frame = ImageEnhance.Brightness(frame).enhance(brightness)
        if i % 5 == 0:
            frame = frame.filter(ImageFilter.UnsharpMask(radius=0.6, percent=80, threshold=2))
        frames.append(frame.convert("RGB"))
    return frames


def _save_gif(frames: List[Image.Image], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _save_mp4(frames: List[Image.Image], out_path: Path, fps: int) -> bool:
    try:
        import imageio.v3 as iio
        import numpy as np

        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
        iio.imwrite(
            out_path,
            arrays,
            fps=max(1, fps),
            codec="libx264",
            extension=".mp4",
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        try:
            import cv2
            import numpy as np

            out_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frames[0].size[1], frames[0].size[0]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1, fps),
                (width, height),
            )
            if not writer.isOpened():
                return False
            for frame in frames:
                rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            return out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            return False


def _save_webm(frames: List[Image.Image], out_path: Path, fps: int) -> bool:
    try:
        import imageio.v3 as iio
        import numpy as np

        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
        iio.imwrite(
            out_path,
            arrays,
            fps=max(1, fps),
            codec="libvpx-vp9",
            extension=".webm",
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def _resolve_output_path(config: dict, output_path: Optional[Path] = None) -> Path:
    infer_cfg = config.get("inference") or {}
    fmt = str(infer_cfg.get("output_format", "mp4")).lower().strip()
    if fmt not in ("mp4", "webm", "gif"):
        fmt = "mp4"
    out_dir = get_output_dir("video")
    if output_path is None:
        return out_dir / f"generated.{fmt}"
    path = Path(output_path)
    if path.suffix.lower() in (".mp4", ".webm", ".gif"):
        return path
    return path.with_suffix(f".{fmt}")


def _save_video(frames: List[Image.Image], out_path: Path, fps: int, config: dict) -> Path:
    infer_cfg = config.get("inference") or {}
    fmt = str(infer_cfg.get("output_format", "mp4")).lower().strip()
    target = _resolve_output_path(config, out_path)
    if fmt == "webm":
        if _save_webm(frames, target, fps):
            return target
    elif fmt == "gif":
        _save_gif(frames, target, fps)
        return target
    else:
        if _save_mp4(frames, target, fps):
            return target
    gif_fallback = target.with_suffix(".gif")
    _save_gif(frames, gif_fallback, fps)
    return gif_fallback


def _frame_variance(frames: List[Image.Image]) -> float:
    if len(frames) < 2:
        return 0.0
    import numpy as np

    arrays = [np.asarray(frame.convert("L"), dtype=np.float32) for frame in frames[:8]]
    diffs = []
    for idx in range(len(arrays) - 1):
        diffs.append(float(np.mean(np.abs(arrays[idx + 1] - arrays[idx])) / 255.0))
    return sum(diffs) / max(1, len(diffs))


def _frames_are_bland(frames: List[Image.Image]) -> bool:
    if not frames:
        return True
    try:
        from navine.image.procedural import is_bland_like

        tmp = get_output_dir("video") / "_bland_check.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(tmp, format="PNG")
        return is_bland_like(tmp)
    except Exception:
        return False


def _resolve_reference_frames(
    prompt: str,
    frame_size: int,
    max_frames: int = 8,
    has_checkpoint: bool = False,
    use_references: bool = True,
) -> Tuple[Optional[Dict[str, object]], Optional[List[Image.Image]]]:
    if not use_references:
        return None, None
    try:
        from navine.reference.config import ensure_reference_dirs, get_reference_settings
        from navine.reference.media import extract_video_frames, load_reference_image, record_reference_use, resolve_reference_for_prompt

        ensure_reference_dirs()
        settings = get_reference_settings()
        if str(settings.get("mode") or "guide").lower() == "off":
            return None, None
        motion_frames = int(settings.get("video_motion_frames", max_frames))
        info = resolve_reference_for_prompt(prompt, prefer_video=True, has_checkpoint=has_checkpoint)
        if not info:
            return None, None
        path = Path(str(info["reference_path"]))
        category = str(info.get("category") or "")
        if info.get("is_video"):
            raw_frames = extract_video_frames(path, max_frames=max(motion_frames, max_frames))
            if not raw_frames:
                return info, None
            frames = [
                frame.convert("RGB").resize((frame_size, frame_size), Image.Resampling.LANCZOS)
                for frame in raw_frames
            ]
            record_reference_use(category)
            return info, frames
        frame = load_reference_image(path, size=frame_size)
        record_reference_use(category)
        return info, [frame]
    except Exception:
        return None, None


def _generate_hybrid(
    prompt: str,
    out_path: Path,
    config: dict,
    num_frames: int,
    fps: int,
    seed: Optional[int],
    reference_info: Optional[Dict[str, object]] = None,
    reference_frames: Optional[List[Image.Image]] = None,
) -> Path:
    fallback_cfg = config.get("fallback") or {}
    infer_cfg = config.get("inference") or {}
    frame_size = int(infer_cfg.get("frame_output_size") or config["model"]["frame_size"])
    max_frames = int(fallback_cfg.get("max_frames", num_frames))
    frame_count = max(4, min(num_frames, max_frames))

    image_steps = int(fallback_cfg.get("image_steps", 16))
    image_steps = max(6, min(image_steps, 24))
    motion = float(fallback_cfg.get("motion", 0.04))
    variation = float(fallback_cfg.get("keyframe_variation", 0.02))
    profile_name, profile_scale = _motion_profile(prompt, config)
    keyframe_count = 2
    keyframes = _render_keyframes(
        prompt,
        frame_size,
        image_steps,
        keyframe_count,
        seed,
        variation,
        reference_frames=reference_frames,
        reference_category=str((reference_info or {}).get("category") or ""),
    )
    base = keyframes[0]
    frames = _animate_advanced(
        base,
        frame_count,
        motion,
        profile_name,
        profile_scale * 0.65,
        keyframes=keyframes,
    )
    return _save_video(frames, out_path, fps, config)


def _generate_fallback(
    prompt: str,
    out_path: Path,
    config: dict,
    num_frames: int,
    fps: int,
    seed: Optional[int] = None,
    reference_info: Optional[Dict[str, object]] = None,
    reference_frames: Optional[List[Image.Image]] = None,
) -> Path:
    return _generate_hybrid(
        prompt,
        out_path,
        config,
        num_frames,
        fps,
        seed,
        reference_info=reference_info,
        reference_frames=reference_frames,
    )


def _collect_video_data_dirs(config: dict) -> List[Path]:
    from navine.utils.paths import get_project_root

    root = get_project_root()
    dirs: List[Path] = []
    primary = config.get("data", {}).get("train_dir")
    if primary:
        dirs.append(root / primary)
    for extra in config.get("data", {}).get("extra_dirs") or []:
        dirs.append(root / extra)
    return dirs


def generate(
    prompt: str,
    output_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    auto_ingest: bool = True,
    with_audio: bool = False,
    narration: Optional[str] = None,
    voice: Optional[str] = None,
    model_profile: Optional[str] = None,
) -> Path:
    if auto_ingest:
        try:
            from navine.nsfw.ingest import auto_ingest_before_generate

            auto_ingest_before_generate()
        except Exception:
            pass
    from navine.modes.profiles import apply_video_profile

    prompt = apply_video_profile((prompt or "").strip(), model_profile) or "motion sequence"
    config = load_config(config_name) if config_name else load_modality_config("video")
    from navine.image.prompts import enhance_prompt

    ckpt_dir = resolve_checkpoint_dir("video", config)
    primary = Path(checkpoint) if checkpoint else ckpt_dir / "latest.pt"
    good = ckpt_dir / "latest_good.pt"
    if checkpoint:
        ckpt_path = Path(checkpoint)
    elif good.exists():
        ckpt_path = good
    else:
        ckpt_path = primary
    infer_cfg = config["inference"]
    custom_only = True
    try_diffvid = False
    if seed is None:
        from navine.image.procedural import prompt_seed

        seed = prompt_seed(prompt.strip() or "motion sequence")
    n_frames = num_frames if num_frames is not None else infer_cfg["num_frames"]
    frame_fps = fps if fps is not None else infer_cfg.get("fps", 12)
    out_dir = get_output_dir("video")
    out_path = _resolve_output_path(config, Path(output_path) if output_path else None)
    used_fallback = False
    initial_frame = None
    hybrid_enabled = bool(infer_cfg.get("hybrid", False)) or bool((config.get("fallback") or {}).get("enabled", False))
    use_references = bool(infer_cfg.get("use_references", False))
    min_variance = float(infer_cfg.get("min_frame_variance", 0.002))
    reject_bland = bool(infer_cfg.get("reject_bland_frames", True))
    reference_info, reference_frames = _resolve_reference_frames(
        prompt.strip() or "motion sequence",
        int(infer_cfg.get("frame_output_size") or config["model"]["frame_size"]),
        has_checkpoint=ckpt_path.exists() or primary.exists() or good.exists(),
        use_references=use_references,
    )
    use_learned = bool(infer_cfg.get("use_learned_model", True))

    if try_diffvid:
        try:
            from navine.video.diffusers_video import generate_diffusers_video

            dv_cfg = infer_cfg.get("diffusers_video") or {}
            model_id = str(dv_cfg.get("model_id") or "stabilityai/stable-video-diffusion-img2vid-xt")
            width = int(dv_cfg.get("width", int(infer_cfg.get("frame_output_size") or 512)))
            height = int(dv_cfg.get("height", int(infer_cfg.get("frame_output_size") or 512)))
            produced = generate_diffusers_video(
                prompt=prompt.strip() or "motion sequence",
                output_path=str(out_path),
                seed=seed,
                num_frames=int(n_frames),
                fps=int(frame_fps),
                width=width,
                height=height,
                model_id=model_id,
            )
            if produced and produced.exists():
                return produced
        except Exception:
            pass

    if not use_learned or not ckpt_path.exists():
        out_path = _generate_hybrid(
            prompt.strip() or "motion sequence",
            out_path,
            config,
            n_frames,
            frame_fps,
            seed,
            reference_info=reference_info,
            reference_frames=reference_frames,
        )
        used_fallback = True
    else:
        from navine.device_manager import get_device

        device = get_device()
        model = None
        compatible = False
        try:
            model = NavineVideoModel.load_checkpoint(ckpt_path, config, device)
            compatible = getattr(model, "checkpoint_compatible", True)
            if not compatible:
                alt = good if good.exists() and good != ckpt_path else primary if primary.exists() and primary != ckpt_path else None
                if alt is not None:
                    model = NavineVideoModel.load_checkpoint(alt, config, device)
                    compatible = getattr(model, "checkpoint_compatible", True)
        except Exception:
            model = None
            compatible = False
        if not compatible or model is None:
            out_path = _generate_hybrid(
                prompt.strip() or "motion sequence",
                out_path,
                config,
                n_frames,
                frame_fps,
                seed,
                reference_info=reference_info,
                reference_frames=reference_frames,
            )
            used_fallback = True
        else:
            model = model.to(device)
            effective_prompt = enhance_prompt(prompt.strip() or "motion sequence")
            frame_size = int(config["model"]["frame_size"])
            try:
                initial_frame = _image_tensor_from_prompt(
                    prompt.strip() or "motion sequence",
                    frame_size,
                    seed=seed,
                )
                if seed is not None:
                    torch.manual_seed(seed)
                frames_tensor = model.generate(
                    text=effective_prompt,
                    device=device,
                    num_frames=n_frames,
                    seed=seed,
                    initial_frame=initial_frame,
                )
                imgs = []
                for i in range(frames_tensor.size(1)):
                    frame = frames_tensor[0, i].cpu()
                    frame = ((frame + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    imgs.append(Image.fromarray(frame))
            except Exception:
                imgs = []
            if reference_frames and use_references and imgs:
                from navine.reference.config import get_reference_settings
                from navine.reference.media import apply_reference_blend, reference_strength_for_category
                from navine.image.procedural import prompt_seed

                settings = get_reference_settings()
                ref_mode = str(settings.get("mode") or "guide").lower()
                category = str((reference_info or {}).get("category") or "")
                strength = reference_strength_for_category(category, has_checkpoint=True) if category else 0.35
                blended = []
                for idx, frame in enumerate(imgs):
                    ref = reference_frames[min(idx, len(reference_frames) - 1)]
                    blended.append(
                        apply_reference_blend(
                            frame,
                            ref,
                            strength=strength,
                            seed=prompt_seed(prompt.strip()) + idx,
                            category=category,
                            mode=ref_mode,
                        )
                    )
                imgs = blended
            out_size = int(infer_cfg.get("frame_output_size") or config["model"]["frame_size"])
            if imgs and imgs[0].size[0] != out_size:
                imgs = [frame.resize((out_size, out_size), Image.Resampling.LANCZOS) for frame in imgs]
            variance = _frame_variance(imgs) if imgs else 0.0
            bland = (not imgs) or (reject_bland and _frames_are_bland(imgs))
            if bland or variance < min_variance or not imgs:
                out_path = _generate_hybrid(
                    prompt.strip() or "motion sequence",
                    out_path,
                    config,
                    n_frames,
                    frame_fps,
                    seed,
                    reference_info=reference_info,
                    reference_frames=reference_frames,
                )
                used_fallback = True
            else:
                out_path = _save_video(imgs, out_path, frame_fps, config)
    meta = {
        "prompt": prompt.strip(),
        "enhanced_prompt": enhance_prompt(prompt.strip() or "motion sequence"),
        "num_frames": n_frames,
        "fps": frame_fps,
        "seed": seed,
        "fallback": used_fallback,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": str(out_path),
    }
    try:
        from navine.nsfw.prompts import analyze_prompt

        analyzed = analyze_prompt(prompt.strip())
        meta["nsfw"] = bool(analyzed.get("nsfw"))
        meta["category"] = analyzed.get("category")
    except Exception:
        pass
    if reference_info:
        meta["reference_used"] = True
        meta["reference_path"] = reference_info.get("reference_path")
        meta["keyword_matched"] = reference_info.get("keyword_matched")
        meta["reference_category"] = reference_info.get("category")
        meta["reference_strength"] = reference_info.get("reference_strength")
        meta["generation_mode"] = reference_info.get("generation_mode")
    else:
        meta["reference_used"] = False
        if used_fallback and not reference_info:
            meta["generation_mode"] = "diffusion"
        elif used_fallback:
            meta["generation_mode"] = "hybrid"
        else:
            meta["generation_mode"] = "learned"
    if not used_fallback and initial_frame is not None:
        meta["seeded_from_image"] = True
    if not used_fallback:
        meta["checkpoint"] = str(ckpt_path)
    meta_path = out_path.with_suffix(".json")
    if with_audio or narration:
        try:
            from navine.video.audio_mux import narrate_video

            text = (narration or prompt or "Navine AI - Python video narration.").strip()
            out_path = narrate_video(out_path, text, voice=voice)
            meta["audio"] = True
            meta["narration"] = text
            meta["output"] = str(out_path)
        except Exception as exc:
            meta["audio"] = False
            meta["audio_error"] = str(exc)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "waves on a beach"
    path = generate(prompt)
    print(f"Saved to {path}")
