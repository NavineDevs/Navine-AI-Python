import re
from typing import Any, Dict, List, Optional

WHO_QUERY_PATTERNS = [
    re.compile(r"^do you know who (.+?) is\??$", re.I),
    re.compile(r"^who is (.+?)\??$", re.I),
    re.compile(r"^who was (.+?)\??$", re.I),
    re.compile(r"^who are (.+?)\??$", re.I),
    re.compile(r"^tell me who (.+?) is\??$", re.I),
    re.compile(r"^do you know who (.+?) was\??$", re.I),
    re.compile(r"\bwho is (.+?)\??$", re.I),
    re.compile(r"\bdo you know who (.+?) is\b", re.I),
]


def is_who_query(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    from navine.search.web import is_capability_query

    if is_capability_query(stripped):
        return False
    lower = stripped.lower().rstrip("?.! ")
    if lower in {"who are you", "who is you", "who am i talking to", "who is this"}:
        return False
    if re.search(r"\bwho\s+(?:are|is)\s+you\b", stripped, re.I):
        return False
    for pattern in WHO_QUERY_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def extract_who_subject(message: str) -> Optional[str]:
    stripped = message.strip().rstrip("?.!")
    for pattern in WHO_QUERY_PATTERNS:
        match = pattern.search(stripped)
        if match:
            subject = match.group(1).strip().rstrip("?.!")
            if subject and len(subject) >= 2:
                return subject
    return None


def _clean_snippet(text: str) -> str:
    from navine.search.web import strip_wiki_boilerplate

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = strip_wiki_boilerplate(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _first_sentence(text: str, max_len: int = 220) -> str:
    cleaned = _clean_snippet(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for part in parts:
        part = part.strip()
        if len(part) >= 30:
            if len(part) > max_len:
                part = part[: max_len - 1].rsplit(" ", 1)[0] + "."
            return part
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rsplit(" ", 1)[0] + "."
    return cleaned


def _subject_in_text(subject: str, text: str) -> bool:
    subject_compact = re.sub(r"[^a-z0-9]+", "", subject.lower())
    text_compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not subject_compact:
        return False
    return subject_compact in text_compact


def _score_result(subject: str, item: Dict[str, Any]) -> int:
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or item.get("excerpt") or "")
    body = f"{title} {snippet}"
    score = 0
    if _subject_in_text(subject, title):
        score += 4
    if _subject_in_text(subject, snippet):
        score += 3
    if re.search(r"\b(is|was|are|were)\b", snippet, re.I):
        score += 1
    if len(snippet) >= 50:
        score += 1
    lower = body.lower()
    if any(marker in lower for marker in ("disambiguation", "click here", "cookie", "sign in")):
        score -= 3
    if "wiktionary" in lower or "may refer to" in lower:
        score -= 4
    return score


def _format_match(subject: str, item: Dict[str, Any]) -> Optional[str]:
    title = _clean_snippet(str(item.get("title") or ""))
    snippet = _first_sentence(str(item.get("snippet") or item.get("excerpt") or ""))
    if not snippet and not title:
        return None
    if title and _subject_in_text(subject, title) and len(title.split()) <= 6:
        if snippet:
            return f"{title}: {snippet}"
        return title + "."
    if snippet:
        return snippet
    return None


def answer_who_query(
    message: str,
    search_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    if not is_who_query(message):
        return None
    subject = extract_who_subject(message)
    if not subject:
        return None
    results = list(search_results or [])
    if not results:
        from navine.search.web import search_internet

        results = search_internet(f"who is {subject}")
    if not results:
        return None
    ranked = sorted(results, key=lambda item: _score_result(subject, item), reverse=True)
    lines: List[str] = []
    for item in ranked[:3]:
        line = _format_match(subject, item)
        if not line:
            continue
        if line not in lines:
            lines.append(line)
    if not lines:
        return None
    if len(lines) == 1:
        return f"{lines[0]}"
    intro = f"\"{subject}\" can refer to a few different people or characters. Here is what I found:"
    return intro + " " + " ".join(lines[:3])


def build_search_query(message: str) -> str:
    subject = extract_who_subject(message)
    if subject:
        return f"who is {subject}"
    return message.strip()
