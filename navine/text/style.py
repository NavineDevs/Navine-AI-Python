import re
from typing import Dict, List, Optional

STRAIGHT_ANSWER_RE = re.compile(
    r"\b(?:straight|direct|blunt|short|concise)\s+answers?\b"
    r"|\b(?:answer|reply|respond)\s+(?:straight|directly|bluntly|concisely)\b"
    r"|\b(?:only|just)\s+(?:straight|direct|short)\s+answers?\b"
    r"|\bno\s+(?:fluff|filler|rambling|long\s+answers?)\b"
    r"|\bgive\s+(?:me\s+)?straight\s+answers?\b",
    re.I,
)


def wants_straight_answers(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    if STRAIGHT_ANSWER_RE.search(message or ""):
        return True
    for turn in reversed(history or []):
        user = str(turn.get("user") or turn.get("content") or "").strip()
        if user and STRAIGHT_ANSWER_RE.search(user):
            return True
    return False


def straight_answer_instruction() -> str:
    return (
        "The user wants straight answers only. Reply in one or two short sentences. "
        "No filler, no feature lists, no disclaimers unless required."
    )
