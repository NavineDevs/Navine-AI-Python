import re
from typing import Optional


COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "cyan", "gold", "silver", "violet", "indigo",
}

SCENE_ENHANCEMENTS = {
    "moon": "photorealistic full moon in a deep night sky, detailed lunar craters, soft atmospheric glow, stars",
    "sun": "bright sun in a clear blue sky, natural daylight",
    "sunset": "dramatic sunset sky with orange and purple clouds",
    "sunrise": "sunrise with warm golden sky, natural lighting",
    "cat": "cute cat, natural lighting, sharp focus",
    "dog": "friendly dog, natural lighting, sharp focus",
    "ocean": "ocean with waves, natural lighting, sharp focus",
    "forest": "green forest landscape, natural lighting",
    "mountain": "mountains under a clear sky, outdoor landscape",
    "flower": "colorful flowers, sharp focus, natural lighting",
    "heart": "heart shape, digital art",
    "star": "stars in a night sky",
    "sky": "clear blue sky with soft white clouds, atmospheric depth, natural daylight",
    "night": "night sky with stars, deep blue atmosphere",
    "clouds": "soft white clouds in a clear blue sky, natural daylight",
    "girl": "adult woman portrait, natural face, soft lighting",
    "woman": "adult woman portrait, natural face, soft lighting",
    "anime": "anime illustration, stylized character art, vibrant colors, detailed linework",
    "hentai": "nude hentai anime girl, detailed face and body, clean lineart, vibrant colors",
    "nude": "nude adult woman, detailed anatomy, natural pose",
    "porn": "photorealistic explicit adult nude, detailed anatomy, natural lighting, sharp focus",
}


def strip_command_prefix(prompt: str) -> str:
    lower = prompt.lower().strip()
    prefixes = [
        "create me a ", "create me ", "create a ", "create an ",
        "make me a ", "make me ", "make a ", "make an ",
        "draw me a ", "draw me ", "draw a ", "draw an ",
        "generate a ", "generate an ", "generate me a ",
        "show me a ", "show me ", "give me a ", "give me ",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            return prompt.strip()[len(prefix):].strip()
    return prompt.strip()


def enhance_prompt(prompt: str, profile: Optional[str] = None) -> str:
    from navine.modes.profiles import apply_image_profile

    text = strip_command_prefix(prompt)
    text = apply_image_profile(text, profile)
    if not text:
        return "a photo of an abstract pattern, digital art, high quality"
    try:
        from navine.nsfw.prompts import analyze_prompt, enhance_nsfw_prompt

        info = analyze_prompt(text)
        lower = text.lower()
        if info.get("real") or str(info.get("category")) in ("real", "human"):
            if any(w in lower for w in ("portrait", "face", "headshot", "photo")):
                subject = text if lower.startswith(("a ", "an ", "the ")) else f"a {text}"
            elif any(w in lower for w in ("girl", "woman", "lady", "model")):
                subject = text if lower.startswith(("a ", "an ", "the ")) else f"a portrait of a {text}"
            elif not lower.startswith(("a ", "an ", "the ")):
                subject = f"a photo of {text}"
            else:
                subject = text
            text = subject
            if "photoreal" not in lower and "photo" not in lower and "realistic" not in lower:
                text = f"{text}, photorealistic, natural lighting"
            if not any(w in lower for w in ("quality", "detailed", "sharp", "4k", "skin")):
                text = f"{text}, high quality, detailed skin texture, sharp focus"
            return text
        if info.get("nsfw") or info.get("anime"):
            enhanced = enhance_nsfw_prompt(text)
            lower_enh = enhanced.lower()
            if str(info.get("category")) in ("hentai", "anime"):
                if not any(w in lower_enh for w in ("anime", "illustration", "linework", "2d")):
                    enhanced = f"{enhanced}, anime illustration, detailed linework"
            elif "photoreal" not in lower_enh and "realistic" not in lower_enh:
                enhanced = f"{enhanced}, photorealistic, natural lighting"
            return enhanced
    except Exception:
        pass
    lower = text.lower()
    if "moon" in lower and "sky" in lower:
        moon_color = None
        sky_color = None
        color_pattern = "|".join(sorted(COLOR_WORDS, key=len, reverse=True))
        moon_match = re.search(rf"\b({color_pattern})\s+moon\b", lower)
        if moon_match:
            moon_color = moon_match.group(1)
        sky_match = re.search(rf"\b({color_pattern})\s+sky\b", lower)
        if sky_match:
            sky_color = sky_match.group(1)
        moon_part = f"{moon_color} moon" if moon_color else "moon"
        sky_part = f"{sky_color} sky" if sky_color else "night sky"
        text = (
            f"a photorealistic {moon_part} in a vivid {sky_part}, stars, "
            f"atmospheric depth, natural lighting, high quality, sharp focus"
        )
        return text
    is_realish = any(w in lower for w in ("real", "photo", "photoreal", "human", "amateur"))
    detected_color = None
    for color in COLOR_WORDS:
        if color in lower.split() or f" {color} " in f" {lower} ":
            detected_color = color
            break
    scene_suffix = None
    for keyword, enhancement in SCENE_ENHANCEMENTS.items():
        if keyword in lower:
            if is_realish and keyword in ("girl", "woman", "anime", "hentai"):
                continue
            scene_suffix = enhancement
            break
    has_quality = any(w in lower for w in ("quality", "detailed", "sharp", "photo", "picture", "photograph", "photoreal"))
    has_article = lower.startswith(("a ", "an ", "the "))
    if scene_suffix:
        if detected_color and detected_color not in scene_suffix:
            text = f"a {detected_color} {scene_suffix}"
        else:
            text = scene_suffix if scene_suffix.startswith(("a ", "an ")) else f"a {scene_suffix}"
    elif detected_color and len(text.split()) <= 4:
        text = f"a photorealistic photo of a {detected_color} {text}"
    elif not has_article and len(text.split()) <= 5:
        text = f"a photorealistic photo of {text}"
    if not has_quality and not any(w in text.lower() for w in ("anime", "hentai", "manga", "sky", "cloud", "moon", "landscape")):
        text = text + ", natural lighting, high quality, sharp focus"
    elif not has_quality and any(w in text.lower() for w in ("sky", "cloud", "moon", "landscape")):
        text = text + ", atmospheric depth, natural lighting, high quality"
    elif not has_quality:
        text = text + ", high quality, detailed, sharp focus, clean composition"
    return text
