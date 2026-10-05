import re
from typing import List, Optional

from navine.selfcode.io import read_file
from navine.selfcode.modules import list_modules
from navine.selfcode.search import search_code

MAX_CONTEXT_CHARS = 4200
MAX_FILE_EXCERPT = 2200
MAX_SEARCH_TERMS = 4

SELFCODE_INSPECT_PATTERNS = [
    re.compile(
        r"\b(look at|read|inspect|examine|browse|show me|open)\s+(your\s+)?(own\s+)?(code|source|codebase|project)\b",
        re.I,
    ),
    re.compile(r"\b(your|navine'?s?)\s+(source|codebase|architecture|implementation)\b", re.I),
    re.compile(r"\bhow\s+does\s+(your|the)\s+\w+", re.I),
    re.compile(
        r"\bhow\s+does\s+(chat|image|video|server|api|search|memory|training|inference|generation)\b",
        re.I,
    ),
    re.compile(r"\bwhat\s+does\s+(your|the)\s+(server|api|chat|image|video)\b", re.I),
    re.compile(r"\bwhere\s+is\s+.+\s+(in|inside)\s+(your|the)\s+(code|project|codebase)\b", re.I),
    re.compile(r"\b(navine/[\w./-]+\.(py|js|ts|json|yaml|yml|html|css))\b", re.I),
    re.compile(r"\bexplain\s+(your|the)\s+\w+\s+(module|code|system|pipeline)\b", re.I),
    re.compile(r"\bhow\s+(chat|image|video|server|api)\s+works?\b", re.I),
]

TOPIC_SEARCH_TERMS = {
    "image": ["generate", "NavineDiffusionModel", "image infer", "diffusion"],
    "video": ["video infer", "generate", "NavineVideo"],
    "chat": ["chat_with_meta", "build_chat_prompt", "chat"],
    "server": ["FastAPI", "uvicorn", "server"],
    "api": ["router", "ChatRequest", "routes"],
    "search": ["search_internet", "web search"],
    "memory": ["conversation", "retrieve_conversation"],
    "train": ["train", "checkpoint"],
    "code": ["generate", "format_code"],
    "learn": ["rag", "retrieve_context", "ingest"],
    "auth": ["api_key", "middleware"],
    "cli": ["argparse", "subparsers"],
}

TOPIC_FILES = {
    "image": ["navine/image/infer.py", "navine/image/model.py"],
    "video": ["navine/video/infer.py", "navine/video/model.py"],
    "chat": ["navine/text/chat.py", "navine/text/infer.py"],
    "server": ["navine/server.py", "navine/api/routes.py"],
    "api": ["navine/api/routes.py", "navine/api/schemas.py"],
    "search": ["navine/search/web.py"],
    "memory": ["navine/memory/conversations.py"],
    "code": ["navine/code/generate.py"],
    "learn": ["navine/learn/rag.py"],
    "cli": ["navine/cli.py"],
    "selfcode": ["navine/selfcode/io.py", "navine/selfcode/intent.py"],
}

FILE_PATH_PATTERN = re.compile(
    r"\b((?:navine|web|configs|scripts|app)/[\w./-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|html|css|md|sh))\b",
    re.I,
)


def is_selfcode_inspect_query(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if "your own code" in lower or "your code" in lower:
        if any(
            token in lower
            for token in ("look at", "read", "inspect", "show", "examine", "how does")
        ):
            return True
    for pattern in SELFCODE_INSPECT_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def extract_file_paths(message: str) -> List[str]:
    found = []
    for match in FILE_PATH_PATTERN.finditer(message):
        path = match.group(1).replace("\\", "/")
        if path not in found:
            found.append(path)
    return found


def detect_topics(message: str) -> List[str]:
    lower = message.lower()
    topics: List[str] = []
    for topic in TOPIC_SEARCH_TERMS:
        if topic in lower:
            topics.append(topic)
    if "image generation" in lower or "generate images" in lower or "image model" in lower:
        if "image" not in topics:
            topics.append("image")
    if "video generation" in lower or "generate videos" in lower:
        if "video" not in topics:
            topics.append("video")
    if "chat" in lower and ("work" in lower or "how" in lower):
        if "chat" not in topics:
            topics.append("chat")
    return topics


def _search_terms_for_message(message: str) -> List[str]:
    terms: List[str] = []
    for topic in detect_topics(message):
        for term in TOPIC_SEARCH_TERMS.get(topic, []):
            if term not in terms:
                terms.append(term)
        for path in TOPIC_FILES.get(topic, []):
            func_guess = path.rsplit("/", 1)[-1].replace(".py", "")
            if func_guess not in terms:
                terms.append(func_guess)
    lower = message.lower()
    quoted = re.findall(r'"([^"]{2,40})"|\'([^\']{2,40})\'', message)
    for pair in quoted:
        for item in pair:
            if item and item not in terms:
                terms.append(item)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", lower)
    for token in tokens:
        if token in {"navine", "your", "code", "does", "work", "what", "where", "show", "read", "look"}:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= MAX_SEARCH_TERMS:
            break
    return terms[:MAX_SEARCH_TERMS]


def _format_search_hits(hits: List[dict]) -> str:
    if not hits:
        return ""
    parts = ["Search matches:"]
    seen_paths = set()
    for hit in hits:
        path = hit["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        parts.append(f"- {path}:{hit['line']}")
        parts.append(hit["snippet"])
    return "\n".join(parts)


def _format_file_excerpt(result: dict) -> str:
    if not result.get("ok"):
        return f"Could not read {result.get('path')}: {result.get('error')}"
    path = result["path"]
    content = result["content"]
    note = " (truncated)" if result.get("truncated") else ""
    return f"File: {path}{note}\n```\n{content}\n```"


def gather_selfcode_context(message: str) -> Optional[str]:
    sections: List[str] = []
    char_budget = MAX_CONTEXT_CHARS

    explicit_paths = extract_file_paths(message)
    topics = detect_topics(message)
    for topic in topics:
        for path in TOPIC_FILES.get(topic, []):
            if path not in explicit_paths:
                explicit_paths.append(path)

    if not explicit_paths and is_selfcode_inspect_query(message):
        overview = list_modules()
        if overview:
            preview = ", ".join(overview[:24])
            sections.append(f"Project modules: {preview}")

    for path in explicit_paths[:3]:
        result = read_file(path, max_chars=min(MAX_FILE_EXCERPT, char_budget // 2))
        block = _format_file_excerpt(result)
        if len(block) > char_budget:
            block = block[: char_budget - 3] + "..."
        sections.append(block)
        char_budget -= len(block)
        if char_budget <= 400:
            break

    if char_budget > 500:
        for term in _search_terms_for_message(message):
            hits = search_code(term, max_results=4)
            block = _format_search_hits(hits)
            if not block:
                continue
            if len(block) > char_budget:
                block = block[: char_budget - 3] + "..."
            sections.append(block)
            char_budget -= len(block)
            if char_budget <= 400:
                break

    if not sections:
        return None
    header = (
        "Navine AI - Python local project source (read from disk). "
        "Answer using these excerpts. Cite file paths naturally."
    )
    body = "\n\n".join(sections)
    combined = f"{header}\n\n{body}"
    if len(combined) > MAX_CONTEXT_CHARS:
        combined = combined[: MAX_CONTEXT_CHARS - 3] + "..."
    return combined
