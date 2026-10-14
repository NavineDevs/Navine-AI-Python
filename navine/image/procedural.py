import hashlib
import math
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

COLOR_MAP = {
    "red": (220, 60, 60),
    "blue": (60, 120, 220),
    "green": (60, 180, 90),
    "yellow": (240, 210, 60),
    "orange": (240, 140, 50),
    "purple": (140, 80, 200),
    "pink": (240, 120, 180),
    "white": (240, 240, 240),
    "black": (20, 20, 30),
    "cyan": (60, 200, 220),
    "gold": (220, 180, 60),
    "silver": (180, 190, 200),
}

SIZE = 256

HAIR_PALETTES = {
    "hentai": [(240, 120, 200), (180, 90, 220), (255, 150, 210), (140, 80, 200)],
    "anime": [(80, 160, 240), (60, 200, 220), (240, 140, 180), (100, 180, 255)],
    "real": [(90, 60, 40), (120, 85, 55), (60, 45, 35), (150, 110, 70)],
    "porn": [(95, 65, 50), (125, 90, 60), (70, 50, 40), (160, 115, 75)],
    "default": [(180, 100, 140), (120, 140, 180), (200, 120, 90)],
}

SKIN_PALETTES = {
    "hentai": (252, 220, 205),
    "anime": (245, 215, 190),
    "real": (210, 175, 155),
    "porn": (205, 168, 148),
    "default": (230, 200, 175),
}

BG_PALETTES = {
    "hentai": ((70, 30, 90), (35, 15, 50)),
    "anime": ((55, 35, 75), (25, 18, 40)),
    "real": ((45, 38, 42), (22, 20, 24)),
    "porn": ((38, 32, 36), (18, 16, 20)),
    "default": ((45, 35, 55), (20, 18, 30)),
}


def prompt_seed(prompt: str) -> int:
    digest = hashlib.sha256(prompt.strip().lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def prompt_rng(prompt: str) -> random.Random:
    return random.Random(prompt_seed(prompt))


def resolve_visual_style(prompt: str, info: dict) -> Dict[str, object]:
    category = "default"
    anime = bool(info.get("anime"))
    nsfw = bool(info.get("nsfw"))
    try:
        from navine.nsfw.prompts import analyze_prompt

        nsfw_info = analyze_prompt(prompt)
        category = str(nsfw_info.get("category") or "default")
        anime = bool(nsfw_info.get("anime")) or anime
        nsfw = bool(nsfw_info.get("nsfw")) or nsfw
    except Exception:
        pass
    lower = prompt.lower()
    if "hentai" in lower or category == "hentai":
        style = "hentai"
    elif category == "porn" or ("porn" in lower and not anime):
        style = "porn"
    elif category == "real" or ("real" in lower and not anime):
        style = "real"
    elif anime or category in ("anime", "animated"):
        style = "anime"
    elif nsfw:
        style = "real"
    else:
        style = "default"
    rng = prompt_rng(prompt)
    hair_choices = HAIR_PALETTES.get(style, HAIR_PALETTES["default"])
    accent = info.get("color") or hair_choices[rng.randint(0, len(hair_choices) - 1)]
    skin = SKIN_PALETTES.get(style, SKIN_PALETTES["default"])
    if style == "real" or style == "porn":
        skin = (
            max(0, min(255, skin[0] + rng.randint(-12, 12))),
            max(0, min(255, skin[1] + rng.randint(-10, 10))),
            max(0, min(255, skin[2] + rng.randint(-8, 8))),
        )
    pose = rng.choice(("stand", "lean", "wave"))
    if style == "hentai":
        eye_scale = 1.85
        head_ratio = 0.26
    elif style == "anime":
        eye_scale = 1.65
        head_ratio = 0.28
    else:
        eye_scale = 1.0
        head_ratio = 0.34
    bg = BG_PALETTES.get(style, BG_PALETTES["default"])
    return {
        "style": style,
        "category": category,
        "anime": style in ("hentai", "anime"),
        "nsfw": nsfw or style in ("hentai", "porn"),
        "photoreal": style in ("real", "porn"),
        "accent": accent,
        "skin": skin,
        "bg_top": bg[0],
        "bg_bottom": bg[1],
        "pose": pose,
        "eye_scale": eye_scale,
        "head_ratio": head_ratio,
        "rng": rng,
        "seed": prompt_seed(prompt),
    }


def detect_color(prompt: str) -> Optional[Tuple[int, int, int]]:
    lower = prompt.lower()
    for name, rgb in COLOR_MAP.items():
        if name in lower:
            return rgb
    return None


CHARACTER_WORDS = (
    "girl",
    "woman",
    "boy",
    "man",
    "person",
    "people",
    "character",
    "anime",
    "waifu",
    "human",
    "lady",
    "guy",
    "portrait",
    "face",
    "hentai",
    "ecchi",
    "nude",
    "naked",
    "nsfw",
    "model",
    "figure",
    "body",
)

NSFW_WORDS = (
    "nsfw",
    "hentai",
    "ecchi",
    "lewd",
    "nude",
    "naked",
    "explicit",
    "porn",
    "erotic",
    "adult",
    "rule34",
)


def parse_prompt(prompt: str) -> dict:
    lower = prompt.lower()
    return {
        "moon": "moon" in lower,
        "sun": "sun" in lower and "sunset" not in lower and "sunrise" not in lower,
        "sunset": "sunset" in lower or "sunrise" in lower,
        "sky": any(w in lower for w in ("sky", "night", "cloud", "star")),
        "cat": "cat" in lower or "kitten" in lower,
        "dog": "dog" in lower or "puppy" in lower,
        "tree": "tree" in lower or "forest" in lower,
        "ocean": any(w in lower for w in ("ocean", "sea", "water", "wave")),
        "fire": "fire" in lower or "flame" in lower,
        "heart": "heart" in lower,
        "circle": "circle" in lower or "ball" in lower or "sphere" in lower,
        "gradient": "gradient" in lower,
        "character": any(w in lower for w in CHARACTER_WORDS),
        "anime": any(w in lower for w in ("anime", "waifu", "manga", "hentai", "ecchi", "2d")),
        "nsfw": any(w in lower for w in NSFW_WORDS),
        "color": detect_color(prompt),
    }


def draw_character(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    style: Dict[str, object],
) -> None:
    w, h = img.size
    cx = w // 2
    rng: random.Random = style["rng"]
    accent: Tuple[int, int, int] = style["accent"]
    skin: Tuple[int, int, int] = style["skin"]
    anime = bool(style["anime"])
    nsfw = bool(style["nsfw"])
    photoreal = bool(style["photoreal"])
    pose = str(style["pose"])
    eye_scale = float(style["eye_scale"])
    hair = accent
    hair_dark = tuple(max(0, c - 50) for c in accent)
    tilt = rng.randint(-8, 8)
    head_ratio = float(style.get("head_ratio", 0.34))
    head_r = int(w * head_ratio)
    if pose == "lean":
        cx += rng.randint(-14, 14)
    head_cy = int(h * (0.26 if nsfw and anime else 0.30 if nsfw else 0.34))
    body_top = head_cy + head_r - 4
    shoulder = int(w * (0.26 if style.get("style") == "hentai" else 0.24 if nsfw else 0.22 if anime else 0.20))
    waist = int(w * (0.13 if style.get("style") == "hentai" else 0.14 if nsfw else 0.15))
    hip = int(w * (0.19 if style.get("style") == "hentai" else 0.17 if nsfw else 0.13))
    leg_top = int(h * 0.72)
    outfit = tuple(min(255, c + 30) for c in accent) if not nsfw else skin
    if photoreal:
        outfit = tuple(max(0, c - 25) for c in skin)
    draw.polygon(
        [
            (cx - shoulder, body_top),
            (cx + shoulder, body_top),
            (cx + waist, int(h * 0.58)),
            (cx - waist, int(h * 0.58)),
        ],
        fill=outfit,
    )
    draw.polygon(
        [
            (cx - waist, int(h * 0.58)),
            (cx + waist, int(h * 0.58)),
            (cx + hip, leg_top),
            (cx - hip, leg_top),
        ],
        fill=outfit,
    )
    thigh_w = int(w * (0.10 if photoreal else 0.09))
    for sign in (-1, 1):
        leg_x = cx + sign * int(w * 0.07)
        draw.polygon(
            [
                (leg_x - thigh_w, leg_top),
                (leg_x + thigh_w, leg_top),
                (leg_x + thigh_w // 2, int(h * 0.94)),
                (leg_x - thigh_w // 2, int(h * 0.94)),
            ],
            fill=skin,
        )
    if pose == "wave":
        arm_y = body_top + int(h * 0.08)
        draw.polygon(
            [
                (cx + shoulder, arm_y),
                (cx + shoulder + int(w * 0.14), arm_y - int(h * 0.06)),
                (cx + shoulder + int(w * 0.10), arm_y + int(h * 0.04)),
            ],
            fill=skin,
        )
    neck_w = head_r // 2
    draw.rectangle([cx - neck_w, head_cy + head_r - 8, cx + neck_w, body_top + 6], fill=skin)
    head_box = [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r]
    draw.ellipse(head_box, fill=skin)
    draw.chord([cx - head_r - 4, head_cy - head_r - 10, cx + head_r + 4, head_cy + 6], 180, 360, fill=hair)
    draw.polygon(
        [(cx - head_r - 4, head_cy - 2), (cx - head_r + 6, head_cy + head_r), (cx - head_r // 2, head_cy)],
        fill=hair_dark,
    )
    draw.polygon(
        [(cx + head_r + 4, head_cy - 2), (cx + head_r - 6, head_cy + head_r), (cx + head_r // 2, head_cy)],
        fill=hair_dark,
    )
    eye_dy = head_cy + head_r // 6
    eye_dx = head_r // 2
    eye_w = int(head_r // 3 * eye_scale) if anime else head_r // 4
    eye_h = int(eye_w * (1.6 if anime else 1.0))
    eye_white = (250, 250, 255) if anime else (245, 245, 248)
    iris = accent if anime else (70, 90, 120)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse([ex - eye_w, eye_dy - eye_h, ex + eye_w, eye_dy + eye_h], fill=eye_white)
        draw.ellipse([ex - eye_w // 2, eye_dy - eye_h // 2, ex + eye_w // 2, eye_dy + eye_h], fill=iris)
        draw.ellipse([ex - eye_w // 4, eye_dy, ex + eye_w // 4, eye_dy + eye_h // 2], fill=(20, 20, 30))
        if anime:
            draw.ellipse([ex - eye_w // 5, eye_dy - eye_h // 3, ex, eye_dy], fill=(255, 255, 255))
    mouth_y = head_cy + int(head_r * 0.6)
    if nsfw and anime:
        draw.arc([cx - head_r // 5, mouth_y - 4, cx + head_r // 5, mouth_y + 8], 0, 180, fill=(200, 100, 110), width=2)
    else:
        draw.line([cx - head_r // 6, mouth_y, cx + head_r // 6, mouth_y], fill=(190, 110, 110), width=2)
    if anime:
        blush = (250, 180, 180)
        for sign in (-1, 1):
            bx = cx + sign * (head_r - 6)
            draw.ellipse([bx - 6, mouth_y - 6, bx + 6, mouth_y + 2], fill=blush)
    if anime:
        outline = tuple(max(0, c - 80) for c in accent)
        draw.ellipse([cx - head_r - 1, head_cy - head_r - 1, cx + head_r + 1, head_cy + head_r + 1], outline=outline, width=2)
        highlight = tuple(min(255, c + 40) for c in hair)
        draw.chord([cx - head_r + 8, head_cy - head_r - 6, cx + head_r - 8, head_cy - 4], 200, 340, fill=highlight)
    if tilt and anime:
        highlight_dot = tuple(min(255, c + 20) for c in hair)
        draw.ellipse([cx - head_r // 3 + tilt, head_cy - head_r // 2, cx - head_r // 3 + tilt + 6, head_cy - head_r // 2 + 6], fill=highlight_dot)


def draw_moon(img: Image.Image, draw: ImageDraw.ImageDraw, color: Tuple[int, int, int]):
    w, h = img.size
    cx, cy = int(w * 0.62), int(h * 0.32)
    radius = min(w, h) // 5
    glow = tuple(min(255, c + 35) for c in color)
    for i in range(6, 0, -1):
        alpha_pad = int(radius * (1.0 + i * 0.12))
        shade = tuple(max(0, min(255, int(c * (0.25 + i * 0.08)))) for c in glow)
        draw.ellipse([cx - alpha_pad, cy - alpha_pad, cx + alpha_pad, cy + alpha_pad], fill=shade)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    highlight = tuple(min(255, c + 55) for c in color)
    draw.ellipse(
        [cx - radius // 2, cy - radius // 2, cx - radius // 8, cy - radius // 8],
        fill=highlight,
    )
    crater = tuple(max(0, c - 40) for c in color)
    draw.ellipse(
        [cx + radius // 6, cy + radius // 8, cx + radius // 2, cy + radius // 2],
        fill=crater,
    )
    draw.ellipse(
        [cx - radius // 5, cy + radius // 4, cx + radius // 12, cy + radius // 2],
        fill=tuple(max(0, c - 25) for c in color),
    )


def draw_stars(draw: ImageDraw.ImageDraw, w: int, h: int, seed: int, count: int = 40):
    rng = random.Random(seed)
    for _ in range(count):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, int(h * 0.75))
        size = rng.randint(1, 3)
        bright = rng.randint(200, 255)
        draw.ellipse([x, y, x + size, y + size], fill=(bright, bright, min(255, bright + 10)))


def draw_clouds(draw: ImageDraw.ImageDraw, w: int, h: int, seed: int, count: int = 5) -> None:
    rng = random.Random(seed + 17)
    for _ in range(count):
        cx = rng.randint(int(w * 0.08), int(w * 0.92))
        cy = rng.randint(int(h * 0.08), int(h * 0.42))
        rw = rng.randint(int(w * 0.14), int(w * 0.26))
        rh = rng.randint(int(h * 0.05), int(h * 0.10))
        for dx, dy, s, shade in (
            (0, 0, 1.0, (250, 252, 255)),
            (-0.65, 0.18, 0.8, (236, 242, 252)),
            (0.7, 0.12, 0.75, (242, 246, 255)),
            (0.1, -0.4, 0.55, (255, 255, 255)),
            (-0.2, 0.35, 0.5, (228, 236, 248)),
        ):
            x0 = int(cx + dx * rw - rw * s)
            y0 = int(cy + dy * rh - rh * s)
            x1 = int(cx + dx * rw + rw * s)
            y1 = int(cy + dy * rh + rh * s)
            draw.ellipse([x0, y0, x1, y1], fill=shade)


def draw_day_sky(img: Image.Image, draw: ImageDraw.ImageDraw, color: Optional[Tuple[int, int, int]], seed: int) -> None:
    w, h = img.size
    top = color or (70, 150, 235)
    bottom = (
        min(255, int(top[0] * 0.55 + 160)),
        min(255, int(top[1] * 0.55 + 170)),
        min(255, int(top[2] * 0.35 + 200)),
    )
    for y in range(h):
        t = y / max(h - 1, 1)
        t = t * t * (3 - 2 * t)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    draw_clouds(draw, w, h, seed, count=7)


def draw_night_sky(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    moon_color: Optional[Tuple[int, int, int]],
    seed: int,
    with_moon: bool = True,
) -> None:
    w, h = img.size
    top = (8, 10, 28)
    bottom = (18, 22, 48)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    draw_stars(draw, w, h, seed, count=70)
    if with_moon:
        draw_moon(img, draw, moon_color or (120, 160, 235))


def is_simple_scene_prompt(prompt: str) -> bool:
    info = parse_prompt(prompt)
    if info.get("character") or info.get("nsfw") or info.get("anime"):
        return False
    if any(info.get(k) for k in ("moon", "sun", "sunset", "sky", "ocean", "heart", "cat", "dog", "gradient")):
        return True
    lower = (prompt or "").lower()
    if re.search(r"\b(?:cloud|clouds|star|stars|landscape|horizon|dawn|dusk)\b", lower):
        return True
    if info.get("color") and len(lower.split()) <= 5:
        return True
    return False


def draw_sun(img: Image.Image, draw: ImageDraw.ImageDraw, color: Tuple[int, int, int]):
    w, h = img.size
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 5
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x1 = cx + int(math.cos(rad) * (radius + 8))
        y1 = cy + int(math.sin(rad) * (radius + 8))
        x2 = cx + int(math.cos(rad) * (radius + 28))
        y2 = cy + int(math.sin(rad) * (radius + 28))
        draw.line([x1, y1, x2, y2], fill=color, width=4)


def draw_cat(img: Image.Image, draw: ImageDraw.ImageDraw, color: Tuple[int, int, int]):
    w, h = img.size
    cx, cy = w // 2, h // 2 + 10
    body_w, body_h = 80, 60
    draw.ellipse(
        [cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2],
        fill=color,
    )
    head_r = 35
    hx, hy = cx, cy - body_h // 2 - head_r // 2
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=color)
    ear = tuple(max(0, c - 40) for c in color)
    draw.polygon([(hx - head_r + 5, hy - head_r + 5), (hx - head_r + 20, hy - head_r - 25), (hx - 5, hy - head_r + 5)], fill=ear)
    draw.polygon([(hx + 5, hy - head_r + 5), (hx + head_r - 20, hy - head_r - 25), (hx + head_r - 5, hy - head_r + 5)], fill=ear)
    draw.ellipse([hx - 12, hy - 5, hx - 4, hy + 3], fill=(30, 30, 40))
    draw.ellipse([hx + 4, hy - 5, hx + 12, hy + 3], fill=(30, 30, 40))


def draw_sunset(img: Image.Image, draw: ImageDraw.ImageDraw):
    w, h = img.size
    for y in range(h):
        t = y / h
        r = int(20 + t * 80 + (1 - t) * 180)
        g = int(10 + t * 40 + (1 - t) * 60)
        b = int(40 + t * 100)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    sun_color = (255, 180, 60)
    cx, cy = w // 2, int(h * 0.65)
    radius = w // 6
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=sun_color)


def draw_ocean(img: Image.Image, draw: ImageDraw.ImageDraw):
    w, h = img.size
    for y in range(h):
        t = y / h
        if t < 0.45:
            r, g, b = 135, 190, 240
        else:
            depth = (t - 0.45) / 0.55
            r = int(20 + depth * 10)
            g = int(80 + depth * 40)
            b = int(160 - depth * 60)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    wave_color = (30, 100, 180)
    for i in range(5):
        y = int(h * 0.45) + i * 12
        draw.arc([0, y - 8, w, y + 8], 0, 180, fill=wave_color, width=2)


def draw_heart(img: Image.Image, draw: ImageDraw.ImageDraw, color: Tuple[int, int, int]):
    w, h = img.size
    cx, cy = w // 2, h // 2
    s = min(w, h) // 5
    draw.ellipse([cx - s, cy - s, cx, cy + s // 3], fill=color)
    draw.ellipse([cx, cy - s, cx + s, cy + s // 3], fill=color)
    draw.polygon([(cx - s, cy), (cx + s, cy), (cx, cy + s * 2)], fill=color)


def generate_procedural(prompt: str, size: int = SIZE) -> Image.Image:
    info = parse_prompt(prompt)
    style = resolve_visual_style(prompt, info)
    color = info["color"] or style["accent"]
    img = Image.new("RGB", (size, size), (15, 15, 30))
    draw = ImageDraw.Draw(img)
    lower = prompt.lower()
    night = any(w in lower for w in ("night", "midnight", "dusk", "evening", "star"))

    if info["sunset"]:
        draw_sunset(img, draw)
        return img

    if info["ocean"]:
        draw_ocean(img, draw)
        return img

    if info["moon"] or (night and info["sky"]):
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)
        moon_color = color if (info["color"] or "moon" in lower) else (120, 160, 235)
        if "blue" in lower and "moon" in lower:
            moon_color = info["color"] or (90, 140, 230)
        draw_night_sky(img, draw, moon_color, style["seed"], with_moon=("moon" in lower or info["moon"]))
        return img

    if info["sky"] or (info["color"] and any(w in lower for w in ("sky", "cloud", "clouds"))):
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)
        if night:
            draw_night_sky(img, draw, color if info["color"] else (120, 160, 235), style["seed"], with_moon=False)
        else:
            sky_color = color if info["color"] else (70, 150, 235)
            draw_day_sky(img, draw, sky_color, style["seed"])
        return img

    if info["sun"]:
        img = Image.new("RGB", (size, size), (135, 200, 250))
        draw = ImageDraw.Draw(img)
        draw_sun(img, draw, color if info["color"] else (255, 210, 60))
        return img

    if info["cat"]:
        bg = (30, 30, 45)
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)
        cat_color = color if info["color"] else (180, 130, 80)
        draw_cat(img, draw, cat_color)
        return img

    if info["heart"]:
        bg = (25, 20, 35)
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)
        draw_heart(img, draw, color if info["color"] else (220, 60, 90))
        return img

    if info["character"] or info["nsfw"]:
        top = style["bg_top"]
        bottom = style["bg_bottom"]
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)
        for y in range(size):
            t = y / size
            r = int(top[0] + (bottom[0] - top[0]) * t)
            g = int(top[1] + (bottom[1] - top[1]) * t)
            b = int(top[2] + (bottom[2] - top[2]) * t)
            draw.line([(0, y), (size, y)], fill=(r, g, b))
        draw_character(img, draw, style)
        blur = 0.15 if style["photoreal"] else 0.35 if style["style"] == "hentai" else 0.4
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        if style["photoreal"]:
            img = ImageEnhance.Contrast(img).enhance(1.08)
            img = ImageEnhance.Color(img).enhance(0.95)
        elif style["style"] == "hentai":
            img = ImageEnhance.Color(img).enhance(1.12)
        return img

    if info["gradient"] or info["color"]:
        w, h = size, size
        base = color
        dark = tuple(max(0, c - 80) for c in base)
        for y in range(h):
            t = y / h
            r = int(dark[0] + (base[0] - dark[0]) * t)
            g = int(dark[1] + (base[1] - dark[1]) * t)
            b = int(dark[2] + (base[2] - dark[2]) * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        if info["circle"]:
            cx, cy = w // 2, h // 2
            radius = min(w, h) // 4
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=base)
        return img

    cx, cy = size // 2, size // 2
    radius = size // 4
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    return img


def is_bland_like(image_path: Path) -> bool:
    try:
        import numpy as np
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        if arr.size == 0:
            return True
        mean = arr.mean(axis=(0, 1))
        std = float(arr.std())
        skyish = float(mean[2]) > float(mean[0]) + 8 and float(mean[2]) > float(mean[1]) - 5
        if skyish and 8.0 <= std < 48.0:
            return False
        if std < 28:
            return True
        flat = arr.reshape(-1, 3)
        center = arr[arr.shape[0] // 4: 3 * arr.shape[0] // 4, arr.shape[1] // 4: 3 * arr.shape[1] // 4]
        if center.size > 0:
            center_std = float(center.std())
            if center_std < 22 and not skyish:
                return True
        hist, _ = np.histogram(flat.mean(axis=1), bins=32)
        dominant = float(hist.max()) / max(flat.shape[0], 1)
        if dominant > 0.55 and std < 40 and not skyish:
            return True
        return False
    except Exception:
        return True


def needs_procedural_fallback(image_path: Path, prompt: str) -> bool:
    if not can_procedural(prompt):
        return False
    return is_noise_like(image_path) or is_bland_like(image_path)


def is_noise_like(image_path: Path) -> bool:
    try:
        import numpy as np
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        if arr.size == 0:
            return True
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        std_mean = float(std.mean())
        if std_mean > 70:
            return True
        channel_corr = []
        for i in range(3):
            for j in range(i + 1, 3):
                flat_i = arr[:, :, i].flatten()
                flat_j = arr[:, :, j].flatten()
                corr = np.corrcoef(flat_i, flat_j)[0, 1]
                if not np.isnan(corr):
                    channel_corr.append(float(corr))
        avg_corr = sum(channel_corr) / len(channel_corr) if channel_corr else 0.0
        if channel_corr:
            if avg_corr < 0.55:
                return True
            if avg_corr < 0.72 and std_mean > 52:
                return True
        edges = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
        edges += np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
        if edges > 45 and std_mean > 50:
            return True
        unique_ratio = len(np.unique(arr // 32)) / max(arr.size // 3, 1)
        if unique_ratio > 0.95 and std_mean > 55:
            return True
        if is_unstructured_like(arr):
            return True
        return False
    except Exception:
        return True


def is_unstructured_like(arr) -> bool:
    """Catch muddy 'TV static' outputs that pass classic noise thresholds."""
    try:
        import numpy as np

        gray = arr.mean(axis=2)
        # High-frequency energy without large-scale structure
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        lap_std = float(np.std(lap))
        # Local patch variance should vary a lot in real photos; noise is uniform
        h, w = gray.shape
        ys = max(8, h // 8)
        xs = max(8, w // 8)
        patch_stds = []
        for y in range(0, h - ys, ys):
            for x in range(0, w - xs, xs):
                patch_stds.append(float(gray[y : y + ys, x : x + xs].std()))
        if not patch_stds:
            return True
        patch_stds_arr = np.asarray(patch_stds, dtype=np.float32)
        patch_mean = float(patch_stds_arr.mean())
        patch_spread = float(patch_stds_arr.std())
        global_std = float(gray.std())
        # Uniform grain across tiles + moderate global std = sludge/noise
        if global_std > 24 and patch_spread < max(4.5, 0.18 * patch_mean) and lap_std > 12:
            return True
        # Autocorrelation drop: noise has weak long-range structure
        small = gray[::4, ::4]
        if small.size > 64:
            flat = small.astype(np.float32)
            flat = flat - flat.mean()
            shifted = np.roll(np.roll(flat, 3, axis=0), 3, axis=1)
            denom = float(np.sqrt((flat ** 2).sum() * (shifted ** 2).sum()) + 1e-6)
            ac = float((flat * shifted).sum() / denom)
            if ac < 0.22 and global_std > 14:
                return True
            if ac < 0.30 and patch_spread < 9.0 and global_std > 14:
                return True
        # Color sludge: warm muddy field with soft blobs and little semantic structure
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        if (
            float(mean_rgb[0]) > float(mean_rgb[2]) + 8
            and global_std < 48
            and lap_std < 18
            and patch_spread < 10
        ):
            return True
        return False
    except Exception:
        return True


def is_valid_generated_image(image_path: Path) -> bool:
    if not Path(image_path).exists():
        return False
    return not (is_noise_like(image_path) or is_bland_like(image_path))



def can_procedural(prompt: str) -> bool:
    info = parse_prompt(prompt)
    lower = prompt.lower()
    if is_simple_scene_prompt(prompt):
        return True
    if any(info[k] for k in ("moon", "sun", "sunset", "sky", "cat", "dog", "ocean", "heart", "gradient", "character", "nsfw", "anime")):
        return True
    if info["color"]:
        return True
    if re.search(r"\b(create|draw|make|generate|show)\b", lower):
        return True
    if len(lower.split()) <= 6:
        return True
    return False


def save_procedural(prompt: str, output_path: Path, size: int = SIZE) -> Path:
    img = generate_procedural(prompt, size=size)
    img_large = img.resize((512, 512), Image.Resampling.NEAREST)
    img_large = img_large.filter(ImageFilter.GaussianBlur(radius=0.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_large.save(output_path, format="PNG")
    return output_path
