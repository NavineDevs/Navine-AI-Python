import re
from typing import Dict, List, Optional, Tuple


CURATED_TOPICS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bverity\b.*\bminecraft\b|\bminecraft\b.*\bverity\b", re.I),
        (
            "Verity is a Minecraft horror concept from YouTuber ThatMob's viral ARG series called \"Something.\" "
            "In the story, Verity is a small yellow smiling AI companion that joins your world, talks like a helper, "
            "and slowly becomes creepy and dangerous. The original videos were scripted horror, not a leaked real mod, "
            "but fan-made Verity mods and add-ons later appeared on sites like CurseForge for Java and Bedrock. "
            "If you mean the videos: watch ThatMob's \"Something\" series. If you mean gameplay: search CurseForge for Verity mods and read the page carefully before installing."
        ),
    ),
]


def match_curated_topic(message: str) -> Optional[str]:
    stripped = message.strip()
    if not stripped:
        return None
    for pattern, answer in CURATED_TOPICS:
        if pattern.search(stripped):
            return answer
    return None
