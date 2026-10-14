from pathlib import Path
from typing import List, Optional

from PIL import Image


def interpolate_rife(frames: List[Image.Image], multiplier: int = 2) -> List[Image.Image]:
    if multiplier <= 1 or len(frames) < 2:
        return frames
    try:
        import torch
        from rife import RIFEModel
    except ImportError:
        return _blend_interpolate(frames, multiplier)
    try:
        model = RIFEModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        output: List[Image.Image] = [frames[0]]
        for idx in range(len(frames) - 1):
            mid_frames = model.interpolate(frames[idx], frames[idx + 1], times=multiplier - 1)
            output.extend(mid_frames)
            output.append(frames[idx + 1])
        return output
    except Exception:
        return _blend_interpolate(frames, multiplier)


def _blend_interpolate(frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
    if multiplier <= 1:
        return frames
    output: List[Image.Image] = []
    for idx in range(len(frames) - 1):
        a = frames[idx].convert("RGB")
        b = frames[idx + 1].convert("RGB").resize(a.size, Image.Resampling.LANCZOS)
        output.append(a)
        for step in range(1, multiplier):
            alpha = step / multiplier
            output.append(Image.blend(a, b, alpha))
    if frames:
        output.append(frames[-1])
    return output
