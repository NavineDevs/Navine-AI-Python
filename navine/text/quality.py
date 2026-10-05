import re
from typing import List


TAG_JUNK = re.compile(
    r"(?:tags:\s*|\|\s*\[?(?:image|human|hentai|character|video)\b|\[(?:human|image))",
    re.I,
)
STEP_MARK = re.compile(r"^(?:answer|final(?: answer)?)\s*:\s*(.+)$", re.I | re.M)


def extract_final_answer(text: str) -> str:
    if not text:
        return text
    matches = STEP_MARK.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def candidate_score(text: str) -> float:
    if not text or not text.strip():
        return -1000.0
    t = text.strip()
    score = 0.0
    length = len(t)
    if 24 <= length <= 420:
        score += 8.0
    elif length < 8:
        score -= 12.0
    elif length > 900:
        score -= 6.0
    alpha = sum(1 for c in t if c.isalpha())
    if length:
        ratio = alpha / length
        if ratio > 0.55:
            score += 6.0
        elif ratio < 0.28:
            score -= 18.0
    if TAG_JUNK.search(t):
        score -= 40.0
    if t.count("|") >= 3:
        score -= 12.0
    if re.search(r"[=;{}]|-to-go|style=", t):
        score -= 10.0
    if re.search(r"\bthe\s+the\b", t, re.I):
        score -= 8.0
    if re.search(r"(?:<a\s+href=|<!--\s*sc_off|<div\s+class=|&#x2f;)", t, re.I):
        score -= 50.0
    if re.search(r"(?:from wikipedia|for other uses, see|disambiguation)", t, re.I):
        score -= 35.0
    if re.search(r"defy_\w+|I'm notice|,\s*dring\s*,", t, re.I):
        score -= 40.0
    if re.search(r"^\s*Step\s*1:\s*\d+\s*,\s*\d+\s*$", t, re.I):
        score -= 30.0
    words = re.findall(r"\b\w+\b", t.lower())
    if words:
        unique = len(set(words))
        score += min(10.0, unique * 0.35)
        if len(words) > 12 and unique / len(words) < 0.35:
            score -= 15.0
    if re.search(r"[.!?]$", t):
        score += 2.0
    if re.search(r"\b(step\s*\d+|because|therefore|so the answer)\b", t, re.I):
        score += 3.0
    if re.search(r"\b(I am|I'm|Navine AI - Python|here is|the answer)\b", t, re.I):
        score += 4.0
    return score


def pick_best_candidate(texts: List[str]) -> str:
    if not texts:
        return ""
    ranked = []
    for t in texts:
        cleaned = extract_final_answer(t) if STEP_MARK.search(t or "") else t
        ranked.append((candidate_score(cleaned), t, cleaned))
    ranked.sort(key=lambda row: row[0], reverse=True)
    best_raw, best_clean = ranked[0][1], ranked[0][2]
    if STEP_MARK.search(best_raw or "") and best_clean and len(best_clean) >= 2:
        return best_clean
    return best_raw


def needs_thinking(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    lower = text.lower()
    if needs_detective(text):
        return True
    if re.search(
        r"\bhow\s+to\s+(?:make|build|craft|assemble|manufacture|cook|bake)\b",
        lower,
    ) and not re.search(
        r"\b(?:code|program|script|function|software|app|algorithm)\b",
        lower,
    ):
        return False
    if len(text) < 12 and not re.search(r"\d", text):
        return False
    if re.search(
        r"\b(?:why|how|explain|reason|compare|analyze|prove|derive|step by step|think)\b",
        lower,
    ):
        return True
    if re.search(r"\b(?:plan|design|architecture|debug|fix|implement|algorithm)\b", lower):
        return True
    if re.search(r"\b(?:versus|vs\.?|difference between|pros and cons)\b", lower):
        return True
    if len(text) > 90:
        return True
    if text.count("?") >= 2:
        return True
    return False


def needs_analyze(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:"
            r"analyze|analyse|breakdown|inspect|evaluate|assess|audit|review|"
            r"compare|contrast|dissect|examine|profile|diagnose|benchmark"
            r")\b",
            lower,
        )
    )


def needs_osint(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    lower = text.lower()
    if needs_detective(text):
        return False
    try:
        from navine.realtime.math import is_math_query

        if is_math_query(text):
            return False
    except Exception:
        pass
    return bool(
        re.search(
            r"\b(?:"
            r"osint|open[- ]source intelligence|username lookup|domain recon|whois|dns lookup|"
            r"subdomain enumeration|breach check|data breach|email footprint|ip lookup|"
            r"reverse dns|asn lookup|social media recon|people search|company footprint|"
            r"google dork|dorking|shodan|censys|passive dns|metadata extract|exif data|"
            r"public records|intel report|reconnaissance|footprint|trace (?:this )?ip|"
            r"find (?:this )?(?:person|username|profile|email|domain)|"
            r"look(?:ing)?\s+(?:up|for)\s+(?:a\s+)?username|search(?:ing)?\s+for\s+(?:a\s+)?username|"
            r"username\s+(?:search|lookup|recon|trace)|trace\s+(?:a\s+)?username|"
            r"can\s+(?:you|it)\s+help.{0,40}username"
            r")\b",
            lower,
        )
    )


def needs_detective(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    lower = text.lower()
    if re.search(
        r"\b(?:are you able|can you|could you|do you know|what is|who (?:made|created)|tell me about)\b",
        lower,
    ):
        if re.search(r"\bcicada(?:\s*3301)?\b", lower) and not re.search(
            r"\b(?:decode|decrypt|solve this|here is|cipher text|pgp)\b",
            lower,
        ):
            return False
    return bool(
        re.search(
            r"\b(?:"
            r"mystery|detective|clue|cipher|cicada|puzzle|encrypted|steganograph|"
            r"cryptogram|whodunit|alibi|suspect|red herring|rabbit hole|\barg\b|"
            r"decode|decrypt|hidden message|secret society|investigation|forensic|"
            r"enigma|dead drop|coordinates|pgp|rsa|caesar|vigenere|book cipher|"
            r"liber primus|noir"
            r")\b",
            lower,
        )
    )
