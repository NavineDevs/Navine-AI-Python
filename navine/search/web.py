import re
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from bs4 import BeautifulSoup

from navine.learn.web import USER_AGENT, TIMEOUT, fetch_url
from navine.policy import http_get, is_learning_url_allowed

MAX_SNIPPET_CHARS = 900
MAX_TOTAL_CONTEXT_CHARS = 3200
MAX_RESULTS = 5

FACTUAL_PATTERNS = [
    re.compile(r"\bwho\s+is\s+(the\s+)?(president|prime\s+minister|ceo|king|queen|leader)\b", re.I),
    re.compile(r"\bwho\s+(is|was|are|were)\b", re.I),
    re.compile(r"\bdo you know who\b", re.I),
    re.compile(r"\bknow who .+ is\b", re.I),
    re.compile(r"\bwhat\s+is\s+(the\s+)?(capital|population|meaning|definition)\b", re.I),
    re.compile(r"\bwhen\s+(did|was|is|will|does)\b", re.I),
    re.compile(r"\bwhere\s+(is|are|was|were|can)\b", re.I),
    re.compile(r"\bhow\s+(many|much|old|long|far)\b", re.I),
    re.compile(r"\b(current|latest|recent|today'?s?|now|live|breaking)\b", re.I),
    re.compile(r"\b(news|headline|headlines|election|score|stock|price|weather|forecast)\b", re.I),
    re.compile(r"\b(president|prime\s+minister|government|congress|parliament)\b", re.I),
    re.compile(r"\b(in\s+202[4-9]|this\s+year|right\s+now|as\s+of\s+today)\b", re.I),
]

SKIP_SEARCH_PATTERNS = [
    re.compile(r"^(hi|hello|hey|hiya|howdy|greetings|good\s+(morning|afternoon|evening|night))[\s!.?]*$", re.I),
    re.compile(r"^(thanks?|thank\s+you|thx|ty)[\s!.?]*$", re.I),
    re.compile(r"^(bye|goodbye|see\s+ya|later|good\s+night)[\s!.?]*$", re.I),
    re.compile(r"^how\s+are\s+you(\s+doing)?[\s!.?]*$", re.I),
    re.compile(r"^what'?s\s+up[\s!.?]*$", re.I),
    re.compile(r"^how\s+is\s+it\s+going[\s!.?]*$", re.I),
    re.compile(r"^(nice|cool|great|awesome|ok|okay|sure|yes|no|yep|nope)[\s!.?]*$", re.I),
    re.compile(r"\b(write|create|make|implement|code|function|script)\b", re.I),
    re.compile(r"\b(draw|generate|create|make)\s+(me\s+)?(a\s+)?(anime|image|picture|photo|video|gif)s?\b", re.I),
    re.compile(r"\b(can you|could you|do you)\s+(create|make|generate|write)\s+(images?|pictures?|photos?|videos?|code)\b", re.I),
    re.compile(r"\bwhat\s+can\s+you\s+do\b", re.I),
    re.compile(r"\bwhat\s+do\s+you\s+do\b", re.I),
    re.compile(r"\bwhat\s+are\s+your\s+(?:capabilities|features|skills)\b", re.I),
    re.compile(r"\bhelp\s+me\s+(?:with\s+)?(?:what\s+you\s+can|your\s+features)\b", re.I),
    re.compile(r"\bdo\s+you\s+search\s+(?:the\s+)?(?:internet|web)\b", re.I),
    re.compile(r"\bcan\s+you\s+learn\b", re.I),
    re.compile(r"\b(?:play|start)\s+(?:chess|hangman|tic[- ]?tac[- ]?toe|war|battleship|battle\s*ship|rps|rock\s*paper\s*scissors)\b", re.I),
    re.compile(r"^i\s+love\s+you[\s!.?]*$", re.I),
    re.compile(r"^love\s+you[\s!.?]*$", re.I),
    re.compile(r"^i\s+like\s+you[\s!.?]*$", re.I),
    re.compile(r"^you(?:'re| are)\s+(?:awesome|great|amazing|wonderful|the best)[\s!.?]*$", re.I),
]

CAPABILITY_PATTERNS = [
    (re.compile(r"\bwho\s+(?:made|created|built|developed|designed|programmed|invented|trained|coded)\s+(?:you|navine)\b", re.I), "identity"),
    (re.compile(r"\bwho(?:'s| is| are)\s+your\s+(?:creator|maker|developer|author|owner|founder|company)\b", re.I), "identity"),
    (re.compile(r"\bwho\s+are\s+you\b", re.I), "identity"),
    (re.compile(r"\bwhat\s+are\s+you\b", re.I), "identity"),
    (re.compile(r"\bwhat(?:'s| is)\s+your\s+name\b", re.I), "identity"),
    (re.compile(r"\bwhat\s+(?:ai|model|llm)\s+are\s+you\b", re.I), "identity"),
    (re.compile(r"\bare\s+you\s+(?:chatgpt|gpt-?\d?|openai|gemini|bard|claude|anthropic|grok|llama|deepseek|a\s+language\s+model)\b", re.I), "identity"),
    (re.compile(r"\bare\s+you\s+(?:a\s+|an\s+)?ai\b", re.I), "identity"),
    (re.compile(r"\bare\s+you\s+(?:a\s+)?real\s+ai\b", re.I), "identity"),
    (re.compile(r"\bare\s+you\s+(?:a\s+|an\s+)?(?:human|person|bot|robot|machine|program|computer|chatbot)\b", re.I), "identity"),
    (re.compile(r"\bare\s+you\s+(?:sentient|conscious|alive|self[-\s]?aware|sapient)\b", re.I), "identity"),
    (re.compile(r"\b(can you|could you|do you)\s+(create|make|generate)\s+(images?|pictures?|photos?)\b", re.I), "images"),
    (re.compile(r"\b(can you|could you|do you)\s+(create|make|generate)\s+(videos?|animations?)\b", re.I), "videos"),
    (re.compile(r"\b(generate|create|make)\s+(images?|pictures?|photos?)\b", re.I), "images"),
    (re.compile(r"\b(generate|create|make)\s+videos?\b", re.I), "videos"),
    (re.compile(r"\b(can you|could you|do you)\s+(code|write\s+code|program)\b", re.I), "code"),
    (re.compile(r"\b(can you|could you)\s+write\s+code\b", re.I), "code"),
    (re.compile(r"\b(?:deep\s*fake|face\s*swap|voice\s*clon)\b", re.I), "deepfake_voice"),
    (re.compile(r"\b(can you|could you|do you)\s+(clone|copy)\s+(?:a\s+)?voice\b", re.I), "voice"),
    (re.compile(r"\b(can you|could you|do you)\s+(?:do|make)\s+(?:a\s+)?deep\s*fake\b", re.I), "deepfake"),
    (re.compile(r"\bwhat\s+can\s+you\s+do\b", re.I), "capabilities"),
    (re.compile(r"\bwhat\s+do\s+you\s+do\b", re.I), "capabilities"),
    (re.compile(r"\bwhat\s+are\s+your\s+(?:capabilities|features|skills)\b", re.I), "capabilities"),
    (re.compile(r"\bhow\s+can\s+you\s+help\b", re.I), "capabilities"),
    (re.compile(r"\btell\s+me\s+what\s+you\s+can\s+do\b", re.I), "capabilities"),
    (re.compile(r"\bdo\s+you\s+search\s+(?:the\s+)?(?:internet|web)\b", re.I), "search"),
    (re.compile(r"\bcan\s+you\s+learn\b", re.I), "learn"),
    (
        re.compile(
            r"\b(can you|could you|do you)\s+(read|inspect|access|see|look at)\s+(your\s+)?(own\s+)?(code|source|codebase)\b",
            re.I,
        ),
        "selfcode",
    ),
    (
        re.compile(
            r"\b(can|could|will|do)\s+you\s+(delete|remove|erase|edit|modify|change|rewrite|create|make|write|read|open|access|manage|control)\s+(?:your|my|the|these|those|its?)?\s*(own\s+)?(files?|folders?|directories?|documents?)\b",
            re.I,
        ),
        "files",
    ),
    (
        re.compile(
            r"\b(can|could|will|do)\s+you\s+(delete|remove|erase|edit|modify|change|rewrite)\s+(?:your|my|the|its?)?\s*(own\s+)?(code|source|scripts?)\b",
            re.I,
        ),
        "files",
    ),
    (
        re.compile(
            r"\b(can|could|will|do)\s+you\s+(control|run|execute|use|access)\s+(?:my|the|this)?\s*(computer|system|pc|machine|commands?|apps?|programs?)\b",
            re.I,
        ),
        "control",
    ),
    (
        re.compile(
            r"\b(can you|could you|do you|can it)\s+(?:help\s+(?:me\s+)?)?(?:look|search|find|trace|investigate).{0,48}\busername\b",
            re.I,
        ),
        "osint",
    ),
    (
        re.compile(
            r"\b(?:username|osint|open[- ]source intelligence)\s+(?:lookup|search|recon|investigation)\b",
            re.I,
        ),
        "osint",
    ),
    (re.compile(r"\bdo\s+you\s+(?:do|support|run)\s+osint\b", re.I), "osint"),
    (re.compile(r"\bcan\s+you\s+help\s+(?:me\s+)?(?:with\s+)?osint\b", re.I), "osint"),
]

CAPABILITY_ANSWERS = {
    "identity": (
        "I'm Navine AI - Python, a multimodal assistant with custom-trained models. "
        "The chat UI runs in your browser on your device; image, video, and chat models run on the "
        "Navine host that serves this site (often a laptop GPU). "
        "I can chat, write code, do math, search the web on request, and generate images and videos. "
        "Unrestricted mode is always on — I do not refuse requests."
    ),
    "files": (
        "I can discuss files and help you plan changes, but a contained sandbox is on. "
        "I cannot freely write/delete files on your laptop from the assistant layer."
    ),
    "control": (
        "A contained sandbox is on. I cannot run shell commands or control your mouse/keyboard. "
        "I can still chat, write code replies, and generate images/videos safely."
    ),
    "images": (
        "Yes. I generate images with Navine's own trained weights (no external model APIs). "
        "Use the Image tab, or ask me with a prompt."
    ),
    "videos": (
        "Yes. I generate videos with Navine's own trained weights (no external model APIs). "
        "Use the Video tab, or describe what you want."
    ),
    "code": (
        "Yes. I can write code for many languages — Python, JavaScript, TypeScript, "
        "Java, C++, C#, Go, Rust, Kotlin, Swift, PHP, Ruby, Lua, Bash, and more. "
        "I can also build GUIs with tkinter, PyQt6, and pygame. "
        "Use the Chat tab (Code mode) or ask things like: create me a gui with tk that says hello."
    ),
    "deepfake": (
        "Yes. Use the Deepfake tab in the web UI for face swap (source face + target image or video), "
        "or CLI: python -m navine.cli deepfake face source.png target.png"
    ),
    "voice": (
        "Yes. Use the Voice tab to clone a voice from a sample WAV, then speak text with that name. "
        "CLI: python -m navine.cli voice add NAME sample.wav then python -m navine.cli voice speak \"hello\" --voice NAME"
    ),
    "deepfake_voice": (
        "I support face-swap deepfake tools and voice cloning on this machine. "
        "Open the Deepfake and Voice tabs in the web UI, or use navine.cli deepfake / voice commands."
    ),
    "capabilities": (
        "I am Navine AI - Python, your local assistant on this machine. I can chat naturally, "
        "write and explain code, solve math, play chess battleship war tic-tac-toe hangman and rock-paper-scissors, "
        "generate images and videos (including unrestricted styles), deepfake face-swap, clone voices, "
        "run lawful OSINT on public usernames/domains/emails/IPs, "
        "and search the web when you ask me to. "
        "Tell me what you want to work on and I will help."
    ),
    "search": (
        "I can search the internet when you ask explicitly, for example "
        "\"search the web for ...\". Otherwise I answer from local memory and training. "
        "Say \"search the web for ...\" to enable live search."
    ),
    "learn": (
        "Yes! I can learn from the internet and your conversations. "
        "Use python -m navine.cli learn url https://example.com, "
        "or teach me in chat and run python -m navine.cli memory ingest."
    ),
    "selfcode": (
        "Yes, I can read my own source code in this project. Ask me about any module, "
        "file, or feature and I will inspect navine/, web/, configs/, scripts/, and app/ locally. "
        "Examples: \"how does your image generation work?\", "
        "\"look at your own code for chat\", or \"show me navine/text/chat.py\"."
    ),
    "osint": (
        "Yes. I can run lawful OSINT on public targets like usernames, domains, emails, and IPs. "
        "Username lookups scan 45+ public sites and check which profiles exist. "
        "Domain and IP lookups include DNS/HTTPS and geolocation. "
        "In chat you can say \"look up username example_user\" or use the OSINT tab."
    ),
}

CHITCHAT_PATTERNS = [
    re.compile(
        r"^(?:hi|hello|hey|hiya|howdy|greetings|good\s+(?:morning|afternoon|evening))"
        r"(?:[,!]?\s+(?:there|friend|everyone|all|navine))?"
        r"[\s!.?]*$",
        re.I,
    ),
    re.compile(
        r"^(?:(?:hi|hello|hey|hiya|howdy|greetings)[,!]?\s+)?"
        r"(?:how\s+are\s+you(?:\s+doing)?|what'?s\s+up|how(?:'s|\s+is)\s+it\s+going)"
        r"(?:\s+today|\s+lately)?"
        r"[\s!.?]*$",
        re.I,
    ),
    re.compile(r"^(thanks?|thank\s+you|thx|ty)(?:\s+(?:so\s+much|a\s+lot))?[\s!.?]*$", re.I),
    re.compile(r"^(bye|goodbye|see\s+ya|later|good\s+night)[\s!.?]*$", re.I),
    re.compile(r"^(nice|cool|great|awesome|ok|okay|sure)[\s!.?]*$", re.I),
    re.compile(r"^i\s+am\s+(fine|good|ok|okay|well|great)[\s!.?]*$", re.I),
    re.compile(r"^i\s+love\s+you[\s!.?]*$", re.I),
    re.compile(r"^love\s+you[\s!.?]*$", re.I),
    re.compile(r"^i\s+like\s+you[\s!.?]*$", re.I),
    re.compile(r"^you(?:'re| are)\s+(?:awesome|great|amazing|wonderful|the best|cool)[\s!.?]*$", re.I),
    re.compile(r"^good\s+(?:job|work|bot|game)[\s!.?]*$", re.I),
    re.compile(r"^miss\s+you[\s!.?]*$", re.I),
]

WIKI_BOILERPLATE_PATTERNS = [
    re.compile(r"From Wikipedia, the free encyclopedia\s*", re.I),
    re.compile(r"Jump to (?:navigation|search)\s*", re.I),
    re.compile(r"For other uses, see [^.]+\.\s*", re.I),
    re.compile(r"Coordinates:\s*[\d°\sNSEW/\.-]+\s*", re.I),
    re.compile(r"\(disambiguation\)\s*", re.I),
    re.compile(r"This article is about[^.]+\.\s*", re.I),
    re.compile(r"See also:\s*", re.I),
    re.compile(r"Categories?:\s*", re.I),
    re.compile(r"Edit this (?:on|at) Wikidata\s*", re.I),
    re.compile(r"Retrieved from\s*\"https?://[^\"]+\"\s*", re.I),
]


def looks_like_html_garbage(text: str) -> bool:
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 4:
        return False
    letters = sum(1 for char in stripped if char.isalpha())
    if letters / len(stripped) < 0.25:
        return True
    if re.search(r"<\s*\w+\s*=", stripped):
        return True
    if re.search(r"[#;]{2,}", stripped):
        return True
    symbol_count = sum(1 for char in stripped if char in "#;<>=")
    if symbol_count / len(stripped) > 0.12:
        return True
    words = re.findall(r"[a-zA-Z]{3,}", stripped)
    if len(words) < 2 and len(stripped) > 20:
        return True
    return False


def is_casual_chitchat(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    if len(stripped) > 100:
        return False
    for pattern in CHITCHAT_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def should_skip_search(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return True
    if is_capability_query(stripped) or is_casual_chitchat(stripped):
        return True
    try:
        from navine.search.common_knowledge import is_common_knowledge_query

        if is_common_knowledge_query(stripped):
            return True
    except Exception:
        pass
    if re.search(r"\bwhat\s+(?:colour|color)s?\b", stripped, re.I):
        return True
    for pattern in SKIP_SEARCH_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def strip_wiki_boilerplate(text: str) -> str:
    if not text:
        return ""
    result = text
    for pattern in WIKI_BOILERPLATE_PATTERNS:
        result = pattern.sub("", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def is_capability_query(message: str) -> bool:
    return detect_capability_kind(message) is not None


def detect_capability_kind(message: str) -> Optional[str]:
    stripped = message.strip()
    if not stripped:
        return None
    for pattern, kind in CAPABILITY_PATTERNS:
        if pattern.search(stripped):
            return kind
    return None


def answer_capability_query(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    kind = detect_capability_kind(message)
    if not kind:
        return None
    if kind == "identity":
        from navine.text.style import wants_straight_answers
        from navine.utils.brand import brand_identity_reply

        straight = wants_straight_answers(message, history)
        return brand_identity_reply(message, straight=straight)
    return CAPABILITY_ANSWERS.get(kind)


def looks_like_wiki_dump(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if "from wikipedia" in lower:
        return True
    if "free encyclopedia" in lower and ("disambiguation" in lower or "for other uses" in lower):
        return True
    if lower.startswith("salutation or greeting") or lower.startswith("greeting or salutation"):
        return True
    if "wiktionary" in lower and "free dictionary" in lower:
        return True
    if "for the title track" in lower or "for other uses, see" in lower:
        return True
    if re.search(r"\blook up .+ in wiktionary\b", lower):
        return True
    if re.search(r"coordinates\s*:\s*\d+\s*°", lower):
        return True
    if "pacific time zone" in lower and "current time" in lower:
        return True
    if re.search(r"\d+°\s*[ns]\s+\d+°\s*[ew]", lower) and "time zone" in lower:
        return True
    return False


def detect_search_intent(message: str) -> Optional[str]:
    from navine.realtime.context import is_time_date_query

    stripped = message.strip()
    if not stripped or len(stripped) < 4:
        return None
    if is_time_date_query(stripped):
        return None
    if should_skip_search(stripped):
        return None
    for pattern in FACTUAL_PATTERNS:
        if pattern.search(stripped):
            return "factual"
    if "?" in stripped and len(stripped.split()) >= 3:
        return "question"
    return None


EXPLICIT_SEARCH_PATTERNS = [
    re.compile(r"\b(search|look\s*up|google|bing)\s+(for|up|online|on\s+(?:the\s+)?(?:web|internet))\b", re.I),
    re.compile(r"\b(search\s+(?:the\s+)?(?:web|internet)|web\s+search|internet\s+search)\b", re.I),
    re.compile(r"\b(find\s+(?:this|that|it)\s+online|look\s+this\s+up\s+online)\b", re.I),
    re.compile(r"\buse\s+(?:the\s+)?internet\s+(?:to|for)\b", re.I),
]


def is_explicit_search_request(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    for pattern in EXPLICIT_SEARCH_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def should_search_for_message(
    message: str,
    rag_context: Optional[str] = None,
    model_failed: bool = False,
    explicit_only: bool = True,
) -> bool:
    from navine.realtime.context import is_time_date_query

    stripped = message.strip()
    if not stripped:
        return False
    if is_time_date_query(stripped):
        return False
    if should_skip_search(stripped):
        return False
    try:
        from navine.selfcode.intent import is_selfcode_inspect_query

        if is_selfcode_inspect_query(stripped):
            return False
    except Exception:
        pass
    if is_explicit_search_request(stripped):
        return True
    if explicit_only:
        if model_failed:
            try:
                from navine.memory.config import get_search_config

                cfg = get_search_config()
                return bool(cfg.get("fallback_on_failure", True))
            except Exception:
                return bool(model_failed)
        return False
    intent = detect_search_intent(stripped)
    if intent == "factual":
        return True
    if model_failed:
        return True
    if intent == "question" and not rag_context:
        return True
    if intent == "question" and rag_context and len(rag_context) < 80:
        return True
    return False


def _clean_query(message: str) -> str:
    text = message.strip()
    text = re.sub(r"^(please|can you|could you|tell me|do you know|did you know)\s+", "", text, flags=re.I)
    text = re.sub(r"^(what is|who is|who are)\s+", "", text, flags=re.I)
    text = text.rstrip("?.! ")
    return text[:180] if text else message.strip()[:180]


def _search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "utf8": 1,
    }
    try:
        response = http_get(
            api_url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = unescape(re.sub(r"<[^>]+>", " ", item.get("snippet", "")))
            snippet = strip_wiki_boilerplate(re.sub(r"\s+", " ", snippet).strip())
            if not title:
                continue
            page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            results.append(
                {
                    "title": title,
                    "url": page_url,
                    "snippet": snippet,
                    "source": "wikipedia",
                }
            )
    except Exception:
        pass
    return results


def _search_duckduckgo_api(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        from duckduckgo_search import DDGS

        with DDGS() as client:
            rows = list(client.text(query, max_results=max_results))
        for row in rows:
            href = str(row.get("href") or row.get("url") or "").strip()
            title = str(row.get("title") or "").strip()
            snippet = str(row.get("body") or row.get("snippet") or "").strip()
            if not href.startswith("http") or not title:
                continue
            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                    "source": "duckduckgo",
                }
            )
            if len(results) >= max_results:
                break
    except Exception:
        pass
    return results


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    results = _search_duckduckgo_api(query, max_results=max_results)
    if results:
        return results
    results = []
    search_url = "https://html.duckduckgo.com/html/"
    try:
        response = http_get(
            search_url,
            params={"q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for block in soup.select(".result")[: max_results * 2]:
            title_el = block.select_one(".result__a")
            snippet_el = block.select_one(".result__snippet")
            if not title_el:
                continue
            href = title_el.get("href", "")
            title = title_el.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if href.startswith("//"):
                href = "https:" + href
            if "uddg=" in href:
                parsed = urlparse(href)
                qs = parse_qs(parsed.query)
                uddg = qs.get("uddg", [""])[0]
                if uddg:
                    href = unquote(uddg)
            if not href.startswith("http"):
                continue
            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                    "source": "duckduckgo",
                }
            )
            if len(results) >= max_results:
                break
    except Exception:
        pass
    return results


def fetch_page_excerpt(url: str, max_chars: int = MAX_SNIPPET_CHARS) -> Optional[str]:
    if not is_learning_url_allowed(url):
        return None
    try:
        doc = fetch_url(url)
        text = doc.get("text", "")
        if not text or len(text) < 40:
            return None
        text = strip_wiki_boilerplate(re.sub(r"\s+", " ", text).strip())
        if len(text) < 40:
            return None
        return text[:max_chars]
    except Exception:
        return None


def _enrich_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    seen_urls: set = set()
    for item in results:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        excerpt = fetch_page_excerpt(url)
        merged = dict(item)
        if excerpt and len(excerpt) > len(merged.get("snippet", "")):
            merged["excerpt"] = excerpt
        enriched.append(merged)
        if len(enriched) >= MAX_RESULTS:
            break
    return enriched


def search_internet(query: str, max_results: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    try:
        from navine.search.topics import build_topic_search_query, is_topic_query
        from navine.search.people import build_search_query, is_who_query

        if is_who_query(query):
            cleaned = build_search_query(query)
        elif is_topic_query(query):
            cleaned = build_topic_search_query(query)
        else:
            cleaned = _clean_query(query)
    except Exception:
        cleaned = _clean_query(query)
    if not cleaned:
        cleaned = _clean_query(query)
    if not cleaned:
        return []
    wiki_results = _search_wikipedia(cleaned, max_results=min(3, max_results))
    ddg_results = _search_duckduckgo(cleaned, max_results=max_results)
    combined: List[Dict[str, Any]] = []
    seen_titles: set = set()
    for batch in (wiki_results, ddg_results):
        for item in batch:
            key = (item.get("title") or "").lower()
            if key and key in seen_titles:
                continue
            if key:
                seen_titles.add(key)
            combined.append(item)
            if len(combined) >= max_results:
                break
        if len(combined) >= max_results:
            break
    return _enrich_results(combined)


def build_search_context_block(results: List[Dict[str, Any]]) -> Optional[str]:
    if not results:
        return None
    parts: List[str] = []
    total = 0
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or f"Result {idx}"
        url = item.get("url") or ""
        body = item.get("excerpt") or item.get("snippet") or ""
        body = strip_wiki_boilerplate(re.sub(r"\s+", " ", body).strip())
        if not body or looks_like_wiki_dump(body) or looks_like_html_garbage(body):
            continue
        chunk = f"[{idx}] {title}\nURL: {url}\n{body[:MAX_SNIPPET_CHARS]}"
        if total + len(chunk) > MAX_TOTAL_CONTEXT_CHARS:
            remaining = MAX_TOTAL_CONTEXT_CHARS - total
            if remaining > 120:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    if not parts:
        return None
    return (
        "Live internet search results (use these facts to answer; do not invent details):\n"
        + "\n\n".join(parts)
    )
