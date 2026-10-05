import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from navine.utils.paths import get_output_dir

INSTALL_HINT = (
    "Optional deepfake backends (install when available):\n"
    "  pip install insightface onnxruntime opencv-python-headless\n"
    "Navine AI - Python includes a built-in face-region transfer fallback that works without those packages."
)


def _face_region(img: Image.Image) -> Tuple[int, int, int, int]:
    w, h = img.size
    left = int(w * 0.22)
    top = int(h * 0.12)
    right = int(w * 0.78)
    bottom = int(h * 0.62)
    return left, top, right, bottom


def _load_image(path: Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def _soft_mask(size: Tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = size
    draw.ellipse((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(4, min(w, h) // 18)))


def _match_face_color(face: Image.Image, target_region: Image.Image) -> Image.Image:
    import numpy as np

    src = np.asarray(face.convert("RGB"), dtype=np.float32)
    tgt = np.asarray(target_region.convert("RGB"), dtype=np.float32)
    if src.size == 0 or tgt.size == 0:
        return face
    out = src.copy()
    for c in range(3):
        s_mean = float(src[:, :, c].mean()) + 1e-5
        t_mean = float(tgt[:, :, c].mean())
        s_std = float(src[:, :, c].std()) + 1e-5
        t_std = float(tgt[:, :, c].std()) + 1e-5
        out[:, :, c] = (src[:, :, c] - s_mean) * (t_std / s_std) + t_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def swap_face_pil(source: Image.Image, target: Image.Image, strength: float = 0.85) -> Image.Image:
    strength = max(0.05, min(1.0, float(strength)))
    neural = _try_neural_swap(source, target, strength=strength)
    if neural is not None:
        return neural
    target = target.convert("RGB")
    source = source.convert("RGB").resize(target.size, Image.Resampling.LANCZOS)
    box = _face_region(target)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    face = source.crop(box).resize((tw, th), Image.Resampling.LANCZOS)
    target_face = target.crop(box)
    try:
        face = _match_face_color(face, target_face)
    except Exception:
        pass
    face = ImageEnhance.Color(face).enhance(1.04)
    face = ImageEnhance.Contrast(face).enhance(1.06)
    face = face.filter(ImageFilter.SMOOTH)
    mask = _soft_mask((tw, th))
    if strength < 1.0:
        mask = mask.point(lambda p: int(p * strength))
    out = target.copy()
    out.paste(face, (box[0], box[1]), mask)
    feather = out.filter(ImageFilter.GaussianBlur(radius=0.8))
    return Image.blend(out, feather, 0.12)


def _try_neural_swap(source: Image.Image, target: Image.Image, strength: float = 0.85):
    try:
        import torch
        from torchvision import transforms

        from navine.deepfake.model import NavineDeepfakeModel
        from navine.utils.paths import get_checkpoint_dir

        ckpt = get_checkpoint_dir("deepfake") / "latest.pt"
        if not ckpt.is_file():
            return None
        model = NavineDeepfakeModel.load(ckpt, map_location="cpu")
        model.eval()
        size = int(getattr(model, "image_size", 128) or 128)
        tfm = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        src = tfm(source.convert("RGB")).unsqueeze(0)
        tgt = tfm(target.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            pred = model(src, tgt)
        pred = ((pred.clamp(-1, 1) + 1) * 0.5).squeeze(0).cpu()
        neural = transforms.ToPILImage()(pred)
        neural = neural.resize(target.size, Image.Resampling.LANCZOS)
        if strength >= 0.99:
            return neural.convert("RGB")
        return Image.blend(target.convert("RGB"), neural.convert("RGB"), strength)
    except Exception:
        return None


def swap_face_files(
    source_path: str,
    target_path: str,
    output_path: Optional[str] = None,
    strength: float = 0.85,
) -> Path:
    source = _load_image(Path(source_path))
    target = _load_image(Path(target_path))
    result = swap_face_pil(source, target, strength=strength)
    if output_path:
        out = Path(output_path)
    else:
        stamp = hashlib.sha1(f"{source_path}:{target_path}".encode("utf-8")).hexdigest()[:10]
        out = get_output_dir("deepfake") / f"face_{stamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out, format="PNG")
    return out


def _extract_frames(video_path: Path, max_frames: int = 24) -> List[Image.Image]:
    frames: List[Image.Image] = []
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(video_path))
        for frame in reader:
            frames.append(Image.fromarray(frame).convert("RGB"))
            if len(frames) >= max_frames:
                break
        reader.close()
        if frames:
            return frames
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // max_frames) if total > 0 else 1
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1
        cap.release()
    except Exception:
        pass
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return frames


def _write_frames(frames: List[Image.Image], out_path: Path, fps: int = 12) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        import numpy as np

        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
        writer.close()
        if out_path.exists():
            return out_path
    except Exception:
        pass
    try:
        import cv2
        import numpy as np

        w, h = frames[0].size
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        for frame in frames:
            bgr = cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        return out_path
    except Exception as exc:
        raise RuntimeError(f"Could not write deepfake video: {exc}") from None


def deepfake_video(
    source_face: str,
    target_video: Optional[str] = None,
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    num_frames: int = 16,
    fps: int = 12,
    strength: float = 0.85,
    seed: Optional[int] = None,
    with_audio: bool = False,
    narration: Optional[str] = None,
) -> Path:
    from navine.video.infer import generate as generate_video

    if target_video and Path(target_video).exists():
        frames = _extract_frames(Path(target_video), max_frames=max(8, num_frames))
        source = _load_image(Path(source_face))
        swapped = [swap_face_pil(source, frame, strength=strength) for frame in frames]
        out = Path(output_path) if output_path else get_output_dir("deepfake") / "deepfake_video.mp4"
        out = _write_frames(swapped, out, fps=fps)
    else:
        video_prompt = prompt or "adult person talking to camera, natural lighting, detailed face"
        base = generate_video(
            video_prompt,
            output_path=str(get_output_dir("deepfake") / "_base_deepfake.mp4"),
            num_frames=num_frames,
            fps=fps,
            seed=seed,
        )
        frames = _extract_frames(Path(base), max_frames=max(8, num_frames))
        source = _load_image(Path(source_face))
        swapped = [swap_face_pil(source, frame, strength=strength) for frame in frames]
        out = Path(output_path) if output_path else get_output_dir("deepfake") / "deepfake_video.mp4"
        out = _write_frames(swapped, out, fps=fps)

    if with_audio or narration:
        try:
            from navine.video.audio_mux import narrate_video

            text = narration or prompt or "Navine AI - Python deepfake narration."
            out = narrate_video(out, text)
        except Exception:
            pass
    return out


def realtime_deepfake_frame(
    source_face: str,
    camera_frame: Image.Image,
    strength: float = 0.8,
) -> Image.Image:
    source = _load_image(Path(source_face), size=camera_frame.size)
    return swap_face_pil(source, camera_frame.convert("RGB"), strength=strength)


def realtime_deepfake_loop(
    source_face: str,
    seconds: int = 8,
    fps: int = 8,
    output_path: Optional[str] = None,
    strength: float = 0.8,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"OpenCV is required for realtime deepfake.\n{INSTALL_HINT}\n{exc}") from None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for realtime deepfake")

    frames: List[Image.Image] = []
    interval = 1.0 / max(1, fps)
    end = time.time() + max(1, seconds)
    try:
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            frames.append(realtime_deepfake_frame(source_face, img, strength=strength))
            time.sleep(interval)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No webcam frames captured")
    out = Path(output_path) if output_path else get_output_dir("deepfake") / "realtime_deepfake.mp4"
    return _write_frames(frames, out, fps=fps)
