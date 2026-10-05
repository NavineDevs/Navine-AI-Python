import json
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def write_sidecar(dst_img: Path, caption: str, category: str, is_nsfw: bool) -> None:
    meta = {
        "caption": caption,
        "prompt": caption,
        "media_category": category,
        "is_nsfw": bool(is_nsfw),
        "sfw": not bool(is_nsfw),
        "rating": "explicit" if is_nsfw else "sfw",
    }
    (dst_img.parent / f"{dst_img.stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def classify(caption: str, name: str) -> str:
    text = f"{caption} {name}".lower()
    if any(k in text for k in ("sky", "cloud", "moon", "sunset", "sunrise", "horizon", "landscape", "mountain", "ocean", "beach", "field")):
        if any(k in text for k in ("sky", "cloud", "moon", "sunset", "sunrise")):
            return "sky"
        return "landscape"
    if any(k in text for k in ("hentai", "anime", "waifu", "manga", "illustration", "linework", "cel shading")):
        return "hentai"
    if any(
        k in text
        for k in (
            "pussy",
            "boob",
            "breast",
            "ass",
            "nude",
            "porn",
            "explicit",
            "genital",
            "nipple",
            "vagina",
            "penis",
            "sex",
            "oppai",
        )
    ):
        return "porn"
    if any(k in text for k in ("photoreal", "photo", "portrait", "realistic", "woman", "man", "person", "human", "candid")):
        return "human"
    return "general"


def caption_for(category: str, base: str) -> str:
    base = (base or "").strip() or category
    if category == "hentai":
        return f"{base}, nude hentai anime girl, detailed face and body, sharp lineart, vibrant colors"
    if category == "porn":
        return f"{base}, photorealistic explicit adult photo, detailed anatomy, natural lighting, sharp focus"
    if category == "human":
        return f"{base}, photorealistic adult photo, natural lighting, detailed skin texture"
    if category == "sky":
        return f"{base}, blue sky clouds atmosphere, wide landscape sky, natural light"
    if category == "landscape":
        return f"{base}, outdoor landscape scenery, natural lighting, sharp focus"
    return base


def collect_sources():
    sources = []
    for folder in (
        ROOT / "data" / "learn" / "image",
        ROOT / "data" / "learn" / "external" / "images",
        ROOT / "data" / "train" / "image_human",
        ROOT / "data" / "train" / "human",
        ROOT / "data" / "image" / "ingested",
        ROOT / "data" / "image" / "learned",
        ROOT / "data" / "image" / "samples",
    ):
        if not folder.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            sources.extend(folder.glob(ext))
    return sources


def load_caption(path: Path) -> str:
    for side in (path.with_suffix(".json"), path.parent / f"{path.stem}.json"):
        if not side.exists():
            continue
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return str(data.get("caption") or data.get("prompt") or path.stem)
        except Exception:
            pass
    return path.stem.replace("_", " ")


def make_sky_images(out_dir: Path, count: int = 48) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    made = 0
    presets = [
        ("clear blue sky with soft white clouds", (70, 140, 220), (180, 210, 245), True, False),
        ("deep blue daytime sky with wispy clouds", (40, 90, 180), (140, 190, 240), True, False),
        ("night sky with a bright blue moon", (5, 10, 35), (20, 40, 90), False, True),
        ("cloudy overcast sky gray blue", (90, 110, 140), (160, 175, 195), True, False),
        ("sunset sky orange pink clouds", (255, 120, 60), (80, 40, 120), True, False),
        ("sunrise sky soft gold and blue", (255, 190, 120), (90, 150, 220), True, False),
        ("wide open blue sky no clouds", (60, 130, 220), (140, 190, 250), False, False),
        ("mountain landscape under blue sky", (70, 140, 210), (120, 170, 230), True, False),
    ]
    for i in range(count):
        label, top, bottom, clouds, moon = presets[i % len(presets)]
        img = Image.new("RGB", (256, 256))
        px = img.load()
        for y in range(256):
            t = y / 255.0
            r = int(top[0] * (1 - t) + bottom[0] * t)
            g = int(top[1] * (1 - t) + bottom[1] * t)
            b = int(top[2] * (1 - t) + bottom[2] * t)
            for x in range(256):
                n = rng.randint(-6, 6)
                px[x, y] = (
                    max(0, min(255, r + n)),
                    max(0, min(255, g + n)),
                    max(0, min(255, b + n)),
                )
        draw = ImageDraw.Draw(img)
        if clouds:
            for _ in range(rng.randint(3, 8)):
                cx = rng.randint(10, 230)
                cy = rng.randint(20, 140)
                w = rng.randint(30, 90)
                h = rng.randint(12, 35)
                col = (240, 245, 250) if "night" not in label else (180, 190, 210)
                draw.ellipse([cx, cy, cx + w, cy + h], fill=col)
                draw.ellipse([cx + 10, cy - 8, cx + w - 5, cy + h - 5], fill=col)
        if moon:
            mx, my = rng.randint(40, 180), rng.randint(30, 100)
            draw.ellipse([mx, my, mx + 36, my + 36], fill=(210, 230, 255))
            draw.ellipse([mx + 8, my + 4, mx + 40, my + 36], fill=(10, 20, 50))
        if "mountain" in label or "landscape" in label:
            base_y = 180
            pts = [(0, 256), (0, base_y)]
            x = 0
            while x < 256:
                peak = base_y - rng.randint(20, 70)
                pts.append((x, peak))
                x += rng.randint(20, 50)
            pts.extend([(256, base_y), (256, 256)])
            draw.polygon(pts, fill=(50, 90, 60))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        category = "landscape" if ("landscape" in label or "mountain" in label) else "sky"
        dest_root = ROOT / "data" / "image" / "sfw" / ("landscapes" if category == "landscape" else "skies")
        dest_root.mkdir(parents=True, exist_ok=True)
        out = dest_root / f"synth_{category}_{i:03d}.png"
        if not out.exists():
            img.save(out)
            write_sidecar(out, caption_for(category, label), category, False)
            made += 1
    return made


def organize() -> dict:
    counts = {"hentai": 0, "porn": 0, "human": 0, "sky": 0, "landscape": 0, "general": 0}
    dest_map = {
        "hentai": ROOT / "data" / "nsfw" / "local" / "hentai",
        "porn": ROOT / "data" / "nsfw" / "local" / "porn",
        "human": ROOT / "data" / "nsfw" / "local" / "human",
        "sky": ROOT / "data" / "image" / "sfw" / "skies",
        "landscape": ROOT / "data" / "image" / "sfw" / "landscapes",
        "general": ROOT / "data" / "nsfw" / "local" / "mixed",
    }
    seen = set()
    for src in collect_sources():
        key = str(src.resolve())
        if key in seen:
            continue
        seen.add(key)
        base_cap = load_caption(src)
        category = classify(base_cap, src.name)
        dest_dir = dest_map[category]
        dst = dest_dir / src.name
        hardlink_or_copy(src, dst)
        write_sidecar(dst, caption_for(category, base_cap), category, category in ("hentai", "porn"))
        counts[category] += 1
    sky_made = make_sky_images(ROOT / "data" / "image" / "sfw" / "skies", count=56)
    counts["sky_synth"] = sky_made
    return counts


if __name__ == "__main__":
    result = organize()
    print(json.dumps(result, indent=2))
