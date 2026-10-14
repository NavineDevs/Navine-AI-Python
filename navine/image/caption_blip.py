from pathlib import Path
from typing import Optional

from PIL import Image


def caption_with_blip(image_path: Path, device: str = "cpu") -> Optional[str]:
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError:
        return None
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = model.to(device)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        return None


def auto_caption_folder(folder: Path, limit: int = 50) -> int:
    count = 0
    for path in sorted(folder.rglob("*")):
        if count >= limit:
            break
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.exists():
            continue
        cap = caption_with_blip(path)
        if not cap:
            continue
        import json

        sidecar.write_text(json.dumps({"caption": cap, "source": "blip"}, indent=2), encoding="utf-8")
        count += 1
    return count
