from pathlib import Path
from typing import List, Optional

from PIL import Image


def crossfade_frames(a: Image.Image, b: Image.Image, steps: int = 6) -> List[Image.Image]:
    a = a.convert("RGB")
    b = b.convert("RGB").resize(a.size, Image.Resampling.LANCZOS)
    frames: List[Image.Image] = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        blended = Image.blend(a, b, alpha)
        frames.append(blended)
    return frames


def chain_clips(clips: List[List[Image.Image]], crossfade_steps: int = 6) -> List[Image.Image]:
    if not clips:
        return []
    output: List[Image.Image] = list(clips[0])
    for clip in clips[1:]:
        if not clip:
            continue
        if output:
            output.extend(crossfade_frames(output[-1], clip[0], steps=crossfade_steps))
        output.extend(clip)
    return output


def save_clip_sequence(frames: List[Image.Image], out_path: Path, fps: int = 12) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        arrays = [frame.convert("RGB") for frame in frames]
        iio.imwrite(out_path, arrays, fps=fps, codec="libx264")
        return out_path
    except Exception:
        first = frames[0].convert("RGB")
        first.save(out_path.with_suffix(".gif"), save_all=True, append_images=[f.convert("RGB") for f in frames[1:]], duration=int(1000 / fps), loop=0)
        return out_path.with_suffix(".gif")
