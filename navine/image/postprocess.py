from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter


def upscale_image(image: Image.Image, scale: int = 2) -> Image.Image:
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)


def denoise_image(image: Image.Image, strength: float = 0.65) -> Image.Image:
    out = image.convert("RGB")
    if strength <= 0:
        return out
    radius = 1 if strength < 0.5 else 2
    smoothed = out.filter(ImageFilter.MedianFilter(size=3))
    if radius >= 2:
        smoothed = smoothed.filter(ImageFilter.GaussianBlur(radius=0.6))
    return Image.blend(out, smoothed, min(max(strength, 0.0), 0.85))


def sharpen_image(image: Image.Image, amount: float = 0.28) -> Image.Image:
    amount = max(0.05, min(amount, 0.6))
    return image.filter(ImageFilter.UnsharpMask(radius=0.9, percent=int(100 * amount), threshold=4))


def postprocess_image(
    image: Image.Image,
    upscale: bool = False,
    sharpen: bool = True,
    denoise: bool = True,
) -> Image.Image:
    out = image.convert("RGB")
    if denoise:
        out = denoise_image(out, strength=0.55)
    weights_dir = Path(__file__).resolve().parents[2] / "weights"
    if upscale and (weights_dir / "RealESRGAN").exists():
        try:
            from realesrgan import RealESRGANer
            import numpy as np

            upsampler = RealESRGANer(scale=2, model_path=str(weights_dir / "RealESRGAN" / "model.pth"))
            arr = np.array(out)[:, :, ::-1]
            output, _ = upsampler.enhance(arr, outscale=2)
            out = Image.fromarray(output[:, :, ::-1])
        except Exception:
            out = upscale_image(out, 2)
    elif upscale:
        out = upscale_image(out, 2)
        out = denoise_image(out, strength=0.35)
    if sharpen:
        out = sharpen_image(out, amount=0.22)
    out = ImageEnhance.Contrast(out).enhance(1.04)
    out = ImageEnhance.Color(out).enhance(1.03)
    return out
