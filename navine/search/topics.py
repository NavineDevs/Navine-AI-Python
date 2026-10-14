import re
from typing import Any, Dict, List, Optional


TOPIC_QUERY_PATTERNS = [
    re.compile(r"\bthe .+ (thing|mod|game|story|series|show|movie|anime)\b", re.I),
    re.compile(r"\bwhat is the .+ (mod|game|story|series)\b", re.I),
    re.compile(r"\b(tell me about|explain)\b", re.I),
    re.compile(r"\bwhat is\s+(?:a|an|the)\s+\w", re.I),
    re.compile(r"\b(minecraft|horror|mod|creepypasta|arg)\b", re.I),
]


def is_topic_query(message: str) -> bool:
    from navine.search.people import is_who_query
    from navine.realtime.math import is_math_query
    from navine.search.web import is_capability_query, is_casual_chitchat, should_skip_search
    from navine.search.common_knowledge import is_common_knowledge_query

    stripped = message.strip()
    if not stripped or len(stripped.split()) < 3:
        return False
    if is_common_knowledge_query(stripped):
        return False
    if re.search(r"\bcapital\s+(?:city\s+)?of\b", stripped, re.I):
        return False
    if is_who_query(stripped) or is_math_query(stripped):
        return False
    if is_capability_query(stripped) or is_casual_chitchat(stripped):
        return False
    if should_skip_search(stripped):
        return False
    if re.search(r"\bwhat\s+(?:colour|color)s?\b", stripped, re.I):
        return False
    for pattern in TOPIC_QUERY_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def build_topic_search_query(message: str) -> str:
    text = message.strip().rstrip("?.!")
    lower = text.lower()
    if "verity" in lower and "minecraft" in lower:
        return "Verity Minecraft horror mod ThatMob ARG"
    if "minecraft" in lower and "horror" in lower:
        cleaned = re.sub(r"\b(the|thing|stuff|something|about|that|this)\b", " ", text, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return f"{cleaned} minecraft horror"
    cleaned = re.sub(r"\b(the|thing|stuff|something|about|that|this)\b", " ", text, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:180] if cleaned else text[:180]


def _clean_snippet(text: str) -> str:
    from navine.search.web import strip_wiki_boilerplate

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = strip_wiki_boilerplate(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _first_good_sentence(text: str, max_len: int = 240) -> str:
    cleaned = _clean_snippet(text)
    if not cleaned:
        return ""
    for part in re.split(r"(?<=[.!?])\s+", cleaned):
        part = part.strip()
        if len(part) < 35:
            continue
        lower = part.lower()
        if any(marker in lower for marker in ("wiktionary", "disambiguation", "list of youtubers", "jstor")):
            continue
        if len(part) > max_len:
            part = part[: max_len - 1].rsplit(" ", 1)[0] + "."
        return part
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rsplit(" ", 1)[0] + "."
    return cleaned


def _score_topic_result(query_terms: List[str], item: Dict[str, Any]) -> int:
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or item.get("excerpt") or "")
    body = f"{title} {snippet}".lower()
    score = 0
    for term in query_terms:
        if term and term in body:
            score += 2
    if "minecraft" in body and "verity" in body:
        score += 5
    if any(marker in body for marker in ("list of youtubers", "wiktionary", "disambiguation")):
        score -= 6
    if len(snippet) >= 60:
        score += 1
    return score


def answer_topic_query(
    message: str,
    search_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    if not is_topic_query(message):
        return None
    from navine.search.knowledge_base import match_curated_topic

    curated = match_curated_topic(message)
    query = build_topic_search_query(message)
    terms = [t.lower() for t in re.findall(r"[a-z0-9]{3,}", query.lower())]
    results = list(search_results or [])
    if not results:
        from navine.search.web import search_internet

        results = search_internet(query)
    if not results:
        return curated
    ranked = sorted(results, key=lambda item: _score_topic_result(terms, item), reverse=True)
    lines: List[str] = []
    for item in ranked[:3]:
        title = _clean_snippet(str(item.get("title") or ""))
        sentence = _first_good_sentence(str(item.get("snippet") or item.get("excerpt") or ""))
        if not sentence:
            continue
        if title and title.lower() not in sentence.lower() and len(title) <= 80:
            line = f"{title}: {sentence}"
        else:
            line = sentence
        if line not in lines:
            lines.append(line)
    if not lines:
        return curated
    lower = message.lower()
    if "verity" in lower and "minecraft" in lower:
        intro = (
            "Verity is a Minecraft horror concept from YouTuber ThatMob's viral ARG series. "
            "It is a smiling yellow AI companion that starts helpful and turns sinister. "
            "The original videos were scripted fiction, but fan-made Verity mods now exist on CurseForge. "
        )
        return intro + lines[0]
    if len(lines) == 1:
        return lines[0]
    return " ".join(lines[:2])
