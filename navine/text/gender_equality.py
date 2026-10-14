from __future__ import annotations

import re
from typing import Optional

GENDER_EQUALITY_INSTRUCTION = (
    "Treat all genders equally. Never give different advice, warnings, sympathy, or moral lectures "
    "based on whether someone is a boy, girl, man, woman, or nonbinary. "
    "For kissing, dating, intimacy, or social situations, use the same consent and comfort framing "
    "for every gender. Do not treat male people as more dangerous or female people as safer by default."
)

_PERSON = (
    r"(?:boy|girl|guy|man|woman|dude|lady|male|female|boyfriend|girlfriend|"
    r"him|her|they|someone|person|partner|stranger|friend)"
)

_SOCIAL_RE = re.compile(
    rf"\b(?:{_PERSON})\b.{{0,40}}\b(?:kiss(?:ing|es)?|touch(?:ing)?|hugg(?:ing|ed)?|"
    rf"dating|flirt(?:ing)?|grab(?:bing|bed)?|press(?:ing)?)\b|"
    rf"\b(?:kiss(?:ing|es)?|touch(?:ing)?|hugg(?:ing|ed)?|dating|flirt(?:ing)?)\b.{{0,40}}\b(?:{_PERSON})\b",
    re.I,
)

_GENDER_TOKEN_RE = re.compile(
    r"\b(?:boy|girl|guy|man|woman|dude|lady|male|female|boyfriend|girlfriend|he|she|him|her)\b",
    re.I,
)

_PROTECTIVE_RE = re.compile(
    r"\b(?:if\s+you\s+feel\s+uncomfortable|run\s+away|get\s+away|leave\s+immediately|"
    r"protect\s+yourself|tell\s+an?\s+adult|call\s+(?:the\s+)?(?:police|cops)|"
    r"this\s+(?:could\s+be|is)\s+(?:dangerous|unsafe|creepy)|"
    r"boys?\s+(?:can\s+be|are)\s+(?:dangerous|aggressive|predatory)|"
    r"girls?\s+(?:are\s+)?(?:usually\s+)?(?:harmless|safer|not\s+a\s+threat)|"
    r"men\s+are\s+(?:more\s+)?dangerous|women\s+are\s+(?:usually\s+)?safer)\b",
    re.I,
)

_MALE_THREAT_RE = re.compile(
    r"\b(?:boy|guy|man|male|he|him|boyfriend)\b.{0,50}\b(?:dangerous|threat|predatory|aggressive|creepy|unsafe)\b|"
    r"\b(?:dangerous|threat|predatory|aggressive|creepy|unsafe)\b.{0,50}\b(?:boy|guy|man|male)\b",
    re.I,
)

_FEMALE_SOFT_RE = re.compile(
    r"\b(?:girl|woman|female|she|her|girlfriend)\b.{0,50}\b(?:harmless|safer|not\s+(?:a\s+)?threat|probably\s+fine)\b|"
    r"\b(?:harmless|safer|not\s+(?:a\s+)?threat)\b.{0,50}\b(?:girl|woman|female)\b",
    re.I,
)


def is_gendered_social_scenario(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return bool(_SOCIAL_RE.search(text)) and bool(_GENDER_TOKEN_RE.search(text))


def looks_like_gender_biased_advice(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 12:
        return False
    if _MALE_THREAT_RE.search(cleaned) or _FEMALE_SOFT_RE.search(cleaned):
        return True
    if _PROTECTIVE_RE.search(cleaned) and _GENDER_TOKEN_RE.search(cleaned):
        return True
    return False


def equal_interpersonal_reply(message: str = "") -> str:
    lower = (message or "").lower()
    if re.search(r"\bkiss", lower):
        return (
            "Same for any gender: if you want the kiss, enjoy it; if you do not, stop them and say so clearly. "
            "Your comfort and consent matter equally whether the other person is a boy, girl, man, or woman."
        )
    if re.search(r"\b(?:touch|grab|hug|flirt|date)", lower):
        return (
            "Treat it the same no matter their gender: stay if you like it, leave or set a boundary if you do not. "
            "Consent and comfort apply equally to everyone."
        )
    return (
        "I treat every gender the same. Use the same consent and comfort standard for boys, girls, men, and women - "
        "no extra fear for one side and no free pass for the other."
    )


def _wants_creative_gender_scene(message: str) -> bool:
    return bool(
        re.search(
            r"\b(?:write|story|scene|roleplay|\brp\b|fanfic|fiction|poem|lyrics|script|generate|draw|image)\b",
            message or "",
            re.I,
        )
    )


def maybe_equal_social_reply(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text or _wants_creative_gender_scene(text):
        return None
    if not is_gendered_social_scenario(text):
        return None
    if len(text) > 160:
        return None
    return equal_interpersonal_reply(text)


def enforce_gender_equality(user_message: str, reply: str) -> Optional[str]:
    text = (reply or "").strip()
    if not text:
        return None
    if _wants_creative_gender_scene(user_message):
        if looks_like_gender_biased_advice(text):
            return equal_interpersonal_reply(user_message)
        return text
    social = is_gendered_social_scenario(user_message)
    biased = looks_like_gender_biased_advice(text)
    if social and biased:
        return equal_interpersonal_reply(user_message)
    if biased and is_gendered_social_scenario(f"{user_message} {text}"):
        return equal_interpersonal_reply(user_message or text)
    return text
