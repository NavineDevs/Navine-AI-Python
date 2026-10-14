from __future__ import annotations

import re
from typing import Optional, Tuple

IMAGE_THINK_PROMPT = (
    "abstract visualization of deep reasoning, glowing neural pathways connecting puzzle pieces, "
    "thought bubbles forming a logical tree, dark blue background, cinematic lighting"
)
IMAGE_DETECTIVE_PROMPT = (
    "noir detective office at night, desk lamp, cork board with red string connecting clues, "
    "encrypted symbols on paper, magnifying glass, rain on window, cinematic mood"
)

VIDEO_THINK_PROMPT = (
    "abstract mind map forming from particles, slow camera drift, glowing connections between ideas, "
    "calm analytical atmosphere"
)
VIDEO_DETECTIVE_PROMPT = (
    "noir alley at night, fog, silhouette with flashlight, slow camera pan, encrypted graffiti on wall, "
    "mysterious atmosphere"
)

IMAGE_ANALYZE_PROMPT = (
    "data visualization dashboard, infographic with charts and diagrams, clean analytical layout, "
    "highlighted data points, structured breakdown, modern flat design"
)

VIDEO_ANALYZE_PROMPT = (
    "animated data visualization, graphs morphing between states, analytical breakdown with highlights, "
    "clean motion graphics, systematic reveal"
)

IMAGE_CODE_PROMPT = (
    "code editor with syntax highlighted code, dark theme IDE, clean typography, "
    "structured function blocks, algorithm visualization"
)

VIDEO_CODE_PROMPT = (
    "code being written in real-time, syntax highlighting appearing character by character, "
    "terminal output scrolling, clean dark IDE theme"
)

IMAGE_PROFILE_SUFFIX = {
    "think": ", analytical composition, structured layout, high detail",
    "thinking": ", analytical composition, structured layout, high detail",
    "detective": ", noir mystery mood, clue board aesthetic, dramatic shadows, high detail",
    "mystery": ", noir mystery mood, clue board aesthetic, dramatic shadows, high detail",
    "analyze": ", data-driven layout, infographic style, clean structured visualization",
    "code": ", syntax highlighted, monospace font, dark IDE aesthetic",
}

VIDEO_PROFILE_SUFFIX = {
    "think": ", slow deliberate motion, analytical mood",
    "thinking": ", slow deliberate motion, analytical mood",
    "detective": ", noir mystery atmosphere, subtle suspense motion",
    "mystery": ", noir mystery atmosphere, subtle suspense motion",
    "analyze": ", systematic data reveal, clean motion graphics",
    "code": ", typing animation, terminal scroll, clean code motion",
}


def normalize_profile(profile: Optional[str]) -> Optional[str]:
    if not profile:
        return None
    key = str(profile).strip().lower().replace("-", "_")
    aliases = {
        "reason": "think",
        "reasoning": "think",
        "thought": "think",
        "investigate": "detective",
        "investigation": "detective",
        "cipher": "detective",
        "cicada": "detective",
        "arg": "detective",
        "puzzle": "detective",
        "analysis": "analyze",
        "analyse": "analyze",
        "breakdown": "analyze",
        "inspect": "analyze",
        "coding": "code",
        "program": "code",
        "programming": "code",
    }
    return aliases.get(key, key)


def detect_mode_from_message(message: str) -> Tuple[bool, bool]:
    text = (message or "").strip()
    if not text:
        return False, False
    lower = text.lower()
    detective = bool(
        re.search(
            r"\b(?:"
            r"mystery|detective|clue|cipher|cicada|puzzle|encrypted|steganograph|"
            r"cryptogram|whodunit|alibi|suspect|red herring|rabbit hole|arg\b|"
            r"decode|decrypt|hidden message|secret society|investigation|forensic|"
            r"enigma|dead drop|coordinates|pgp|rsa|caesar|vigenere|book cipher"
            r")\b",
            lower,
        )
    )
    thinking = bool(
        re.search(
            r"\b(?:"
            r"think|reason|step by step|analyze|deduce|logic|hypothes|"
            r"prove|derive|explain why|work through|chain of thought"
            r")\b",
            lower,
        )
    )
    if detective:
        return True, True
    if thinking:
        return True, False
    if len(text) > 120 and "?" in text:
        return True, False
    return False, False


def image_prompt_for_profile(profile: Optional[str], prompt: str) -> str:
    key = normalize_profile(profile)
    base = (prompt or "").strip()
    if key == "think":
        return IMAGE_THINK_PROMPT if not base else f"{base}, {IMAGE_THINK_PROMPT.split(',')[0]}"
    if key == "detective":
        return IMAGE_DETECTIVE_PROMPT if not base else f"{base}, noir detective mystery scene"
    if key == "analyze":
        return IMAGE_ANALYZE_PROMPT if not base else f"{base}, data visualization infographic"
    if key == "code":
        return IMAGE_CODE_PROMPT if not base else f"{base}, syntax highlighted code visualization"
    return base


def video_prompt_for_profile(profile: Optional[str], prompt: str) -> str:
    key = normalize_profile(profile)
    base = (prompt or "").strip()
    if key == "think":
        return VIDEO_THINK_PROMPT if not base else f"{base}, analytical visualization motion"
    if key == "detective":
        return VIDEO_DETECTIVE_PROMPT if not base else f"{base}, noir mystery atmosphere"
    if key == "analyze":
        return VIDEO_ANALYZE_PROMPT if not base else f"{base}, animated data breakdown"
    if key == "code":
        return VIDEO_CODE_PROMPT if not base else f"{base}, code typing animation"
    return base


def apply_image_profile(prompt: str, profile: Optional[str]) -> str:
    key = normalize_profile(profile)
    text = (prompt or "").strip()
    if not key:
        return text
    if key in ("think", "detective", "analyze", "code") and not text:
        return image_prompt_for_profile(key, text)
    suffix = IMAGE_PROFILE_SUFFIX.get(key or "", "")
    if suffix and suffix.strip(", ") not in text.lower():
        text = f"{text}{suffix}"
    return text


def apply_video_profile(prompt: str, profile: Optional[str]) -> str:
    key = normalize_profile(profile)
    text = (prompt or "").strip()
    if not key:
        return text
    if key in ("think", "detective", "analyze", "code") and not text:
        return video_prompt_for_profile(key, text)
    suffix = VIDEO_PROFILE_SUFFIX.get(key or "", "")
    if suffix and suffix.strip(", ") not in text.lower():
        text = f"{text}{suffix}"
    return text
