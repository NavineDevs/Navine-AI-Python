import re
from typing import Any, Dict, List, Optional

CODE_PATTERNS = (
    r"\bdef\s+\w+\s*\(",
    r"\bfunction\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\bfn\s+\w+\s*\(",
    r"\bpackage\s+\w+",
    r"\bfunc\s+\w+\s*\(",
    r"\bpublic\s+class\s+\w+",
    r"\binterface\s+\w+",
    r"\bstruct\s+\w+",
    r"\benum\s+\w+",
    r"#include\s+<",
    r"\busing\s+namespace\b",
    r"\bmodule\s+\w+",
    r"\bend\s*$",
    r"^\s*#!/",
    r"\blocal\s+\w+\s*=",
)

QA_PATTERNS = (
    r"\?$",
    r"^(what|how|why|when|where|who|which|can|could|should|is|are|does|do)\b",
    r"\bquestion:\s*",
    r"\banswer:\s*",
)

STORY_PATTERNS = (
    r"\b(story|poem|fiction|chapter|once upon|novel|tale)\b",
    r"^###\s*Title:",
)

SCIENCE_PATTERNS = (
    r"\b(physics|chemistry|biology|quantum|molecule|experiment|hypothesis|theorem)\b",
    r"\b(research|scientific|laboratory|particle|genome)\b",
)

HISTORY_PATTERNS = (
    r"\b(century|historical|ancient|medieval|war|empire|revolution|dynasty)\b",
    r"\b(born|died|founded|invented)\s+\d{3,4}\b",
)

NEWS_PATTERNS = (
    r"\b(breaking|reported|announced|today|yesterday|officials|government)\b",
    r"\b(news|headline|press release)\b",
)

TECHNICAL_PATTERNS = (
    r"\b(algorithm|architecture|framework|api|database|compiler|protocol)\b",
    r"\b(documentation|implementation|specification|RFC)\b",
)

MATH_PATTERNS = (
    r"\b(equation|integral|derivative|matrix|probability|geometry|algebra)\b",
    r"\b=\s*\d",
    r"\b\d+\s*[\+\-\*/]\s*\d+",
)

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n([\s\S]*?)```", re.MULTILINE)
FENCED_LANG_RE = re.compile(r"```(\w+)\n([\s\S]*?)```", re.MULTILINE)
INDENT_CODE_RE = re.compile(r"(?:^|\n)((?:    |\t).+(?:\n(?:    |\t).+)*)", re.MULTILINE)

LANGUAGE_HINTS = {
    "python": (r"\bdef\s+\w+", r"\bimport\s+\w+", r":\s*$"),
    "javascript": (r"\bfunction\s+\w+", r"\bconst\s+\w+\s*=", r"=>"),
    "typescript": (r"\binterface\s+\w+", r":\s*\w+\s*[;,]", r"\btype\s+\w+"),
    "rust": (r"\bfn\s+\w+", r"\bimpl\s+", r"\bpub\s+"),
    "go": (r"\bfunc\s+\w+", r"\bpackage\s+\w+", r":="),
    "java": (r"\bpublic\s+class", r"\bSystem\.out", r"\bimport\s+java\."),
    "cpp": (r"#include\s+", r"\bstd::", r"\bnamespace\s+"),
    "csharp": (r"\busing\s+System", r"\bnamespace\s+\w+", r"\bpublic\s+class"),
    "ruby": (r"\bdef\s+\w+", r"\bend\s*$", r"\brequire\s+"),
    "php": (r"<\?php", r"\$\w+\s*=", r"\bfunction\s+\w+"),
    "swift": (r"\bfunc\s+\w+", r"\bvar\s+\w+", r"\bimport\s+Foundation"),
    "kotlin": (r"\bfun\s+\w+", r"\bval\s+\w+", r"\bclass\s+\w+"),
    "lua": (r"\bfunction\s+\w+", r"\blocal\s+\w+", r"\bend\s*$"),
    "shell": (r"^\s*#!/", r"\becho\s+", r"\bif\s+\["),
}


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    seen = set()
    for match in FENCED_LANG_RE.finditer(text):
        lang = match.group(1).lower() or "text"
        code = match.group(2).strip()
        if code and code not in seen:
            seen.add(code)
            blocks.append({"language": lang, "code": code})
    if not blocks:
        for match in CODE_BLOCK_RE.finditer(text):
            code = match.group(1).strip()
            if code and code not in seen:
                seen.add(code)
                blocks.append({"language": "text", "code": code})
    for match in INDENT_CODE_RE.finditer(text):
        code = match.group(1).strip()
        if len(code) > 40 and code not in seen:
            seen.add(code)
            blocks.append({"language": "text", "code": code})
    return blocks


def looks_like_code(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    hits = sum(1 for pattern in CODE_PATTERNS if re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    code_line_ratio = sum(1 for ln in lines if re.match(r"^\s*(def |class |import |func |fn |#include|public |package )", ln)) / max(1, len(lines))
    return hits >= 2 or code_line_ratio > 0.3


def looks_like_qa(title: str, text: str) -> bool:
    combined = f"{title} {text}".strip()
    if not combined:
        return False
    if "?" in combined:
        return True
    lower = combined.lower()
    return any(re.search(pattern, lower, re.IGNORECASE) for pattern in QA_PATTERNS)


def looks_like_story(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in STORY_PATTERNS)


def looks_like_science(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in SCIENCE_PATTERNS)


def looks_like_history(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in HISTORY_PATTERNS)


def looks_like_news(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in NEWS_PATTERNS)


def looks_like_technical(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in TECHNICAL_PATTERNS)


def looks_like_math(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in MATH_PATTERNS)


def detect_language(code: str, hint: Optional[str] = None) -> str:
    if hint and hint not in ("text", "txt", ""):
        return hint
    scores: Dict[str, int] = {}
    for lang, patterns in LANGUAGE_HINTS.items():
        score = sum(1 for pattern in patterns if re.search(pattern, code, re.MULTILINE))
        if score:
            scores[lang] = score
    if scores:
        return max(scores, key=scores.get)
    return hint or "python"


def classify_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = item.get("title", "")
    text = item.get("text", "")
    url = item.get("url", "")
    source = item.get("source", "unknown")
    source_id = item.get("source_id", url or title)
    language = item.get("language")
    forced = item.get("category")
    kind = item.get("kind")
    results: List[Dict[str, Any]] = []

    if forced:
        results.append({"category": forced, "item": item})
        return results

    if kind == "qa":
        results.append({"category": "qa", "item": item})
        return results

    if kind == "technical":
        results.append({"category": "technical", "item": item})
        return results

    code_blocks = extract_code_blocks(text)
    for block in code_blocks:
        lang = detect_language(block["code"], block.get("language"))
        results.append(
            {
                "category": "coding",
                "item": {
                    "source": source,
                    "source_id": f"{source_id}:code:{hash(block['code']) & 0xFFFFFFFF}",
                    "title": title or f"Code from {source}",
                    "text": block["code"],
                    "url": url,
                    "language": lang,
                    "prompt": title or f"Example code from {source}",
                },
            }
        )

    if item.get("kind") == "code" or (language and looks_like_code(text)):
        lang = detect_language(text, language)
        results.append(
            {
                "category": "coding",
                "item": {
                    "source": source,
                    "source_id": source_id,
                    "title": title,
                    "text": text,
                    "url": url,
                    "language": lang,
                    "prompt": title or f"Code sample from {source}",
                },
            }
        )
        if results:
            return results

    if code_blocks:
        return results

    if source in ("wikipedia",):
        if looks_like_history(title, text):
            results.append({"category": "history", "item": item})
        elif looks_like_science(title, text):
            results.append({"category": "science", "item": item})
        else:
            results.append({"category": "general", "item": item})
            if len(text) > 80:
                results.append({"category": "chat", "item": item})
        return results

    if source in ("arxiv", "docs"):
        results.append({"category": "technical", "item": item})
        if looks_like_science(title, text):
            results.append({"category": "science", "item": item})
        return results

    if source == "gutenberg":
        results.append({"category": "creative", "item": item})
        return results

    if source == "news":
        results.append({"category": "news", "item": item})
        if looks_like_qa(title, text):
            results.append({"category": "qa", "item": item})
        elif len(text) > 80:
            results.append({"category": "chat", "item": item})
        return results

    if source == "humans":
        if looks_like_qa(title, text):
            results.append({"category": "qa", "item": item})
        else:
            results.append({"category": "chat", "item": item})
        if looks_like_news(title, text):
            results.append({"category": "news", "item": item})
        return results

    if source in ("feeds", "hackernews", "reddit"):
        if looks_like_news(title, text):
            results.append({"category": "news", "item": item})
        if looks_like_technical(title, text):
            results.append({"category": "technical", "item": item})

    if kind == "qa" or looks_like_qa(title, text):
        if looks_like_math(title, text):
            results.append({"category": "math", "item": item})
        else:
            results.append({"category": "qa", "item": item})
        return results

    if looks_like_story(title, text):
        results.append({"category": "creative", "item": item})
        return results

    if looks_like_math(title, text):
        results.append({"category": "math", "item": item})
        return results

    if looks_like_science(title, text):
        results.append({"category": "science", "item": item})
        return results

    if looks_like_history(title, text):
        results.append({"category": "history", "item": item})
        return results

    if looks_like_technical(title, text):
        results.append({"category": "technical", "item": item})
        return results

    if len(text) > 80:
        results.append({"category": "chat", "item": item})
    else:
        results.append({"category": "general", "item": item})
    return results
