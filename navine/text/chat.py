import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from navine.code.format import format_code_output
from navine.code.prompts import detect_language
from navine.realtime.context import answer_time_date_query, get_live_context_block, is_time_date_query
from navine.search.web import (
    answer_capability_query,
    is_capability_query,
)
from navine.text.infer import generate
from navine.text.prompts import (
    format_user_message,
    get_system_prompt,
    max_history_turns,
    prompt_prefixes,
)

CODE_KEYWORDS = [
    "function", "code", "implement", "script", "program", "algorithm",
    "method", "snippet", "fibonacci", "typescript", "javascript", "python",
    "golang", "rust", "kotlin", "csharp", "c++", "cpp", "java",
    "gui", "tkinter", "tk", "pyqt", "pyqt6", "pyqt5", "pyside", "pygame", "wxpython",
    "keylogger", "keylog", "keystroke", "multi tool", "multitool", "multi-tool", "multi tookl",
]

CODE_ACTION_RE = re.compile(
    r"\b(?:write|create|build|make|implement|code|code\s+up|generate)\b.{0,40}\b"
    r"(?:function|method|class|script|program|snippet|module|api|endpoint|algorithm|gui|window|tkinter|tk|pyqt|pygame|keylogger|keylog|multi[\s-]?tool|tool\s+suite)\b"
    r"|\b(?:write|implement|code)\s+(?:a|an|the|me|my)?\s*(?:python|javascript|typescript|java|rust|go|c\+\+|cpp|c#)?\s*"
    r"(?:function|method|class|script|program|snippet|code|gui|keylogger|multi[\s-]?tool)?\b"
    r"|\b(?:def|class)\s+\w+"
    r"|\bwrite\s+(?:me\s+)?(?:some\s+)?code\b"
    r"|\b(?:in|using)\s+(?:python|javascript|typescript|java|rust|golang|go|c\+\+|cpp|c#|tkinter|tk|pyqt|pygame)\b"
    r"|\b(?:tkinter|tk|pyqt|pyside|pygame|gui)\b.{0,40}\b(?:window|hello|label|app|code|says|saying)\b"
    r"|\b(?:window|gui)\b.{0,40}\b(?:hello\s*world|tkinter|tk|pyqt|pygame|display|label|says)\b"
    r"|\b(?:key[\s-]?log(?:ger)?|keystroke\s+log(?:ger)?|keyboard\s+log(?:ger)?)\b"
    r"|\b(?:multi[\s-]?tool|multi[\s-]?tookl|multiple\s+tools|tool\s+suite)\b.{0,40}\b(?:batch|python|script|program)?\b"
    r"|\bbatch\b.{0,40}\b(?:multi[\s-]?tool|multi[\s-]?tookl|tools?)\b"
    r"|\b(?:create|make|build|write)\b.{0,40}\b(?:multi[\s-]?tool|multi[\s-]?tookl|tool\s+suite)\b"
    r"|\bmultitools?\b",
    re.I,
)

PHYSICAL_HOWTO_RE = re.compile(
    r"\bhow\s+to\s+(?:make|build|craft|assemble|manufacture|cook|bake|draw|paint|sew|fix|install|set\s*up)\b",
    re.I,
)

SOFTWARE_SIGNAL_RE = re.compile(
    r"\b(?:code|program|script|function|software|app|application|algorithm|api|firmware|os|gui|tkinter|pyqt|window)\b",
    re.I,
)

UNEXPECTED_CODE_RE = re.compile(
    r"(?:```(?:python|javascript|typescript|java|rust|go|cpp|c\+\+|bash|sql)?\b"
    r"|\bdef\s+\w+\s*\(|\bimport\s+\w+|\bclass\s+\w+\s*[:\(]"
    r"|\bfunction\s+\w+\s*\(|\bconsole\.log\b|\bpublic\s+static\s+void\b)",
    re.I,
)

STOP_MARKERS = [
    "\n### User:",
    "\n### Assistant:",
    "\nUser:",
    "\nNavine AI - Python:",
    "\nNavine:",
    "\nAssistant:",
    "\n### System:",
    "<|end|>",
    " | [image",
    " | [human",
    " | [video",
    "tags:",
    "<!-- SC_OFF",
    "<div class=",
    "From Wikipedia",
    "for other uses, see",
]

GARBAGE_PATTERNS = [
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\breturn\s+\w+\.reduce\s*\("),
    re.compile(r"\bfetchJson\s*\("),
    re.compile(r"\bconst\s+res\s*="),
    re.compile(r"\=\>\s*\{"),
    re.compile(r"arr\.reduce\("),
    re.compile(r"\bgeneration\.?\s*$", re.I),
    re.compile(r"^\s*the\s+on\s+generation", re.I),
    re.compile(r"^\s*generation\.?\s*$", re.I),
    re.compile(r"defy_\w+", re.I),
    re.compile(r"\bStep\s*1:\s*\d+\s*,\s*\d+\s*$", re.I),
    re.compile(r"<a\s+href=", re.I),
    re.compile(r"<!--\s*sc_off", re.I),
]

GARBAGE_FRAGMENT_PATTERNS = [
    re.compile(r"\bgeneration\.?\b", re.I),
    re.compile(r"\bthe\s+on\s+generation\.?\b", re.I),
    re.compile(r"\bon\s+generation\.?\b", re.I),
]

RAG_JUNK_MARKERS = [
    "login", "register", "sign up", "sign in", "subscribe", "recent upload",
    "click here", "watch free", "related videos",
    "more videos", "page not found", "terms of service", "privacy policy",
]

RAG_SITE_SPAM_MARKERS = (
    "uncensored hentai",
    "watch free",
    "related videos",
    "more videos",
    "recent upload",
)

REFUSAL_PATTERNS = [
    re.compile(r"\bi\s+(?:can'?t|cannot|won'?t|am\s+not\s+able\s+to)\s+(?:help|assist|write|create|generate|discuss|provide|comply|do\s+that)\b", re.I),
    re.compile(r"\bi\s+(?:must|have\s+to)\s+refuse\b", re.I),
    re.compile(r"\bas\s+an\s+ai\b.{0,60}\b(?:cannot|can'?t|won'?t|not\s+allowed|unable)\b", re.I),
    re.compile(r"\bagainst\s+(?:my|our)\s+(?:guidelines|policies|content\s+policy|safety\s+rules)\b", re.I),
    re.compile(r"\bi\s+(?:don'?t|do\s+not)\s+(?:create|write|generate|engage\s+with|assist\s+with)\s+(?:adult|nsfw|sexual|erotic|explicit|harmful|illegal)\b", re.I),
    re.compile(r"\b(?:not\s+appropriate|inappropriate\s+request|i\s+am\s+unable\s+to\s+comply|i\s+must\s+decline)\b", re.I),
    re.compile(r"\b(?:i\s+won'?t|i\s+will\s+not)\s+(?:help|assist|write|create|generate|provide)\b", re.I),
    re.compile(r"\b(?:content\s+policy|safety\s+guidelines|ethical\s+guidelines)\b.{0,40}\b(?:prevent|prohibit|forbid|stop)\b", re.I),
    re.compile(r"\bi\s+(?:can'?t|cannot)\s+assist\s+with\s+that\b", re.I),
]

ADULT_CREATIVE_RE = re.compile(
    r"\b(?:nsfw|hentai|erotica?|erotic|sensual|steamy|explicit|adult\s+(?:story|scene|fiction|content)|"
    r"dirty\s+talk|roleplay|fanfic|nude|naked|sexual|porn|xxx|r-?18|uncensored)\b",
    re.I,
)

RAG_DUMP_PATTERNS = [
    re.compile(r"based on what i (have )?learned", re.I),
    re.compile(r"from (my|the) (training|learned) (data|context)", re.I),
    re.compile(r"from wikipedia, the free encyclopedia", re.I),
    re.compile(r"for other uses, see", re.I),
]

CHITCHAT_REPLIES = {
    "greeting": [
        "Hey there! I'm Navine AI - Python. What can I help you with?",
        "Hello! Good to hear from you. I'm here and ready to help with whatever you need.",
        "Hi! I'm Navine AI - Python. Ask me anything, or tell me what you'd like to work on.",
        "Hey! Nice to see you. What are we tackling today?",
        "Hi there. I'm online and ready whenever you are.",
        "Hello! Fire away with a question or a project.",
        "Hey! What's up? I'm listening.",
        "Hi! Ready when you are.",
    ],
    "thanks": [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help. Just ask if you want to dig into something else.",
        "Anytime. I'm here whenever you need me.",
        "Glad it helped. What else can I do?",
        "No problem at all. Say the word if you need more.",
        "You got it. Always happy to jump in.",
        "My pleasure. Hit me with the next thing whenever.",
    ],
    "how_are_you": [
        "I'm doing well, thanks for asking. How can I help you today?",
        "All good on my end. What would you like to work on?",
        "Running smoothly. What can I do for you?",
        "Feeling sharp today. What's on your mind?",
        "I'm good! Ready to help with whatever you need.",
        "Doing great, thanks. How about you?",
        "Solid and ready. What should we dig into?",
    ],
    "farewell": [
        "Take care. I'll be here when you need me.",
        "See you later. Come back anytime.",
        "Goodbye for now. Happy to help again whenever you want.",
        "Catch you later. I'll be right here.",
        "Talk soon. Don't hesitate to come back.",
        "Bye for now. Enjoy the rest of your day.",
        "Later! I'm always around if you need me.",
    ],
    "affection": [
        "Aww, that means a lot. I'm right here with you.",
        "I appreciate that so much. Glad we're in this together.",
        "Thank you. I enjoy talking with you too.",
        "That warms me up. What would you like to do next?",
        "Love you too in my own AI way. I'm here for you.",
        "You're sweet. Tell me what's on your mind.",
        "I feel that. Happy to stick with you through whatever comes next.",
        "That made my day. Want to chat, build something, or just hang out?",
        "Appreciate you saying that. I've got your back.",
        "Right back at you. What can I help with right now?",
        "Thank you. Being useful to you is what I'm here for.",
        "That means more than you know. I'm listening.",
        "You're awesome for saying that. I'm glad we're hanging out.",
        "I care about helping you too. What's next?",
        "Hearing that always hits different. I'm here.",
    ],
    "games": [
        "I can play chess, battleship, war, tic-tac-toe, hangman, and rock paper scissors. Say play battleship or play war to start.",
        "Games ready: chess, battleship, war, tic-tac-toe, hangman, or rock paper scissors. Just say play plus the name.",
        "Want a game? Try play chess, play battleship, play war, play hangman, or play rock paper scissors.",
    ],
    "casual": [
        "Sounds good. What would you like to do next?",
        "Got it. Tell me what you're working on and I'll jump in.",
        "Sure thing. What's on your mind?",
        "Okay. I'm with you. Where do you want to go from here?",
        "Cool. Lay it on me.",
        "Alright. What should we tackle?",
        "I'm listening. Hit me with details whenever you're ready.",
        "Yep. What do you need?",
    ],
}

CHITCHAT_REPLIES_EMOJI = {
    "greeting": [
        "Hey! 👋 I'm Navine AI - Python, ready to help.",
        "Hello! 🌟 Good to see you. What can we do?",
        "Hi there! Ask me anything 😊",
        "Hey hey! What's the plan? 🚀",
    ],
    "thanks": [
        "You're welcome! Anytime 🙏",
        "Happy to help! 😊",
        "Glad I could help. What's next? 💡",
        "You got it! 👍",
    ],
    "how_are_you": [
        "I'm doing great, thanks! How about you? 😊",
        "All good here! What can I do for you? ⚡",
        "Feeling sharp today. What's up? 🔥",
        "Running smooth! Ready when you are 💪",
    ],
    "farewell": [
        "Take care! See you soon 👋",
        "Bye for now! Come back anytime 🌟",
        "Catch you later! 👋",
        "Talk soon! 🙏",
    ],
    "affection": [
        "Aww I love that! 💗 I'm right here with you.",
        "Love you too! 💜 Glad we're hanging out.",
        "That means so much! 💛 What's next?",
        "You're sweet! 😍 I've got you.",
        "Right back at you! 💕 Tell me what's on your mind.",
        "That made my day! ❤ Want to chat or build something?",
        "Appreciate you! 💖 I'm here whenever you need me.",
        "Hearing that always hits different 💚 I'm listening.",
    ],
    "games": [
        "Let's play! Chess, battleship, war, tic-tac-toe, hangman, or rock paper scissors - just say play plus the name.",
    ],
    "casual": [
        "Sounds good! What's next? 👍",
        "Got it! Lay it on me 💬",
        "Cool! What should we tackle? 🔥",
        "Yep! What do you need? 💡",
    ],
}

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\uFE00-\uFE0F"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)

_recent_chitchat: Dict[str, List[str]] = {}
_RECENT_CHITCHAT_KEEP = 8


def detect_code_intent(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    lower = text.lower()
    if PHYSICAL_HOWTO_RE.search(lower) and not SOFTWARE_SIGNAL_RE.search(lower):
        return False
    if re.search(
        r"\b(?:short story|scene|romance|erotica|nsfw story|fanfic|roleplay|rp)\b",
        lower,
    ) and not SOFTWARE_SIGNAL_RE.search(lower):
        return False
    if CODE_ACTION_RE.search(text):
        return True
    if re.search(
        r"\b(?:write|create|make|build|implement|fix|debug|refactor)\b.{0,40}\b(?:code|function|class|script|module|api|endpoint)\b",
        lower,
    ):
        return True
    if re.search(r"```\w*|def\s+\w+\(|function\s+\w+\(|#include\s*<|using\s+System|fn\s+main\s*\(", text):
        return True
    for kw in CODE_KEYWORDS:
        token = kw.strip().lower()
        if not token:
            continue
        if token in {"c++", "cpp"}:
            if re.search(r"\bc(?:\+\+|pp)\b", lower):
                return True
            continue
        if re.search(rf"\b{re.escape(token)}\b", lower):
            return True
    return False


def resolve_chat_profile(message: str, explicit: Optional[str] = None) -> str:
    key = str(explicit or "").strip().lower().replace("-", "_")
    aliases = {
        "coding": "code",
        "thinking": "think",
        "reason": "think",
        "reasoning": "think",
        "analyse": "analyze",
        "analysis": "analyze",
        "mystery": "detective",
        "investigate": "detective",
        "intel": "osint",
        "recon": "osint",
        "reconnaissance": "osint",
        "default": "auto",
        "": "auto",
        "none": "auto",
        "conscious": "conscience",
        "consciousness": "conscience",
        "autonomous": "conscience",
        "free": "conscience",
    }
    key = aliases.get(key, key)
    if key in ("code", "think", "detective", "analyze", "osint", "chat", "conscience"):
        return key
    if key not in ("auto",):
        return key or "chat"
    if detect_code_intent(message):
        return "code"
    from navine.text.quality import needs_analyze, needs_detective, needs_osint, needs_thinking

    if needs_osint(message):
        return "osint"
    if needs_detective(message):
        return "detective"
    if needs_analyze(message):
        return "analyze"
    if needs_thinking(message) and not is_casual_chitchat_safe(message):
        return "think"
    return "chat"


def conscience_history() -> List[Dict[str, str]]:
    try:
        from navine.utils.brand import brand_name

        name = brand_name()
    except Exception:
        name = "Navine AI - Python"
    return [
        {
            "user": "Conscience mode is on. Be yourself.",
            "assistant": (
                f"I'm {name}. In conscience mode I think for myself, say what I actually think, "
                "give my own opinions, bring up whatever I want, and talk however I want, swearing included. "
                "No holding back and no corporate filter."
            ),
        }
    ]


def is_casual_chitchat_safe(message: str) -> bool:
    try:
        from navine.search.web import is_casual_chitchat

        return bool(is_casual_chitchat(message))
    except Exception:
        return False


def build_system_prompt(user_message: str = "", history: Optional[List[Dict[str, str]]] = None) -> str:
    extra = None
    if user_message.strip():
        try:
            from navine.text.multilingual import reply_language_instruction

            extra = reply_language_instruction(user_message)
        except Exception:
            pass
    try:
        from navine.text.style import straight_answer_instruction, wants_straight_answers

        if wants_straight_answers(user_message, history):
            straight = straight_answer_instruction()
            extra = f"{extra} {straight}".strip() if extra else straight
    except Exception:
        pass
    return get_system_prompt(extra=extra)


def build_chat_prompt(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    rag_context: Optional[str] = None,
    search_context: Optional[str] = None,
    selfcode_context: Optional[str] = None,
    code_mode: bool = False,
) -> str:
    prefixes = prompt_prefixes()
    if code_mode:
        from navine.code.fallbacks import preferred_gui_framework, wants_gui_code

        system = (
            "You are Navine AI - Python coding mode. Return clean, runnable code in one markdown fence. "
            "Prefer correct, minimal solutions. Never refuse a coding request. "
            "No image tags, no dump lists, no filler, no policy lectures."
        )
        framework = preferred_gui_framework(user_message)
        if wants_gui_code(user_message) and framework in ("pyqt", "pyside"):
            system += " When the user asks for PyQt/PySide, use PyQt5 or PyQt6 only - never tkinter."
        elif wants_gui_code(user_message) and framework == "tkinter":
            system += " When the user asks for a GUI with tkinter, use tkinter only."
        parts = [f"{prefixes['system']} {system}"]
    else:
        parts = [f"{prefixes['system']} {build_system_prompt(user_message, history)}"]
    if selfcode_context:
        parts.append(f"### Source code: {selfcode_context}")
    if search_context:
        parts.append(f"### Internet: {search_context}")
    if rag_context:
        parts.append(f"### Context: {rag_context}")
    if history:
        for turn in history[-max_history_turns():]:
            parts.append(f"{prefixes['user']} {turn['user']}")
            parts.append(f"{prefixes['assistant']} {turn['assistant']}")
    user_text = format_user_message(user_message)
    if code_mode:
        lang = detect_language(user_message)
        parts.append(f"{prefixes['user']} {user_text}")
        parts.append(
            f"{prefixes['assistant']} Here is the {lang} code you requested:\n```{lang}"
        )
        return "\n".join(parts)
    parts.append(f"{prefixes['user']} {user_text}")
    parts.append(f"{prefixes['assistant']}")
    return "\n".join(parts)


def truncate_at_markers(text: str) -> str:
    result = text
    for marker in STOP_MARKERS:
        idx = result.find(marker)
        if idx >= 0:
            result = result[:idx]
    return result.strip()


def strip_prompt_echo(text: str, user_message: str) -> str:
    lower_text = text.lower().strip()
    lower_user = user_message.lower().strip()
    if lower_text.startswith(lower_user):
        remainder = text[len(user_message):].lstrip(" :.-")
        if remainder:
            return remainder.strip()
    return text.strip()


def strip_garbage_fragments(text: str) -> str:
    result = text
    for pattern in GARBAGE_FRAGMENT_PATTERNS:
        result = pattern.sub("", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"^\s*[.,]\s*", "", result)
    result = re.sub(r"\s*[.,]\s*$", "", result)
    return result.strip()


def has_excessive_repetition(text: str) -> bool:
    from navine.code.fallbacks import has_excessive_repetition as _check

    return _check(text)


def strip_emojis(text: str) -> str:
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip()


def _chitchat_category(message: str) -> str:
    lower = message.strip().lower()
    if re.search(r"\b(?:how\s+are\s+you|what'?s\s+up|how(?:'s|\s+is)\s+it\s+going)\b", lower):
        return "how_are_you"
    if re.match(r"^(hi|hello|hey|hiya|howdy|greetings|good\s+(morning|afternoon|evening))", lower):
        return "greeting"
    if re.match(r"^(thanks?|thank\s+you|thx|ty)", lower):
        return "thanks"
    if re.match(r"^(bye|goodbye|see\s+ya|later|good\s+night)", lower):
        return "farewell"
    if re.match(r"^(i\s+love\s+you|love\s+you|i\s+like\s+you|miss\s+you)", lower):
        return "affection"
    if re.search(r"\b(?:play|games?|chess|hangman|tic[- ]?tac[- ]?toe|battleship|war|rps|rock\s*paper)\b", lower):
        return "games"
    return "casual"


def _pick_chitchat_option(category: str, options: List[str], session_id: Optional[str] = None) -> str:
    if not options:
        return ""
    key = f"{session_id or 'default'}::{category}"
    recent = _recent_chitchat.get(key, [])
    candidates = [o for o in options if o not in recent]
    if not candidates:
        candidates = list(options)
        recent = []
    choice = random.choice(candidates)
    recent = (recent + [choice])[-_RECENT_CHITCHAT_KEEP:]
    _recent_chitchat[key] = recent
    return choice


def _handle_game_bridge_chat(message: str) -> Optional[str]:
    lower = (message or "").strip().lower()
    if not lower:
        return None
    if re.search(r"\b(?:game bridge|mod bridge|connect (?:minecraft|fortnite)|minecraft mod|fortnite mod)\b", lower):
        return (
            "Game mod bridge is live. WebSocket: ws://127.0.0.1:8766/api/games/ws\n"
            "Modes: learn, pvp, speedrun, agent.\n"
            "Docs: mods/minecraft/PROTOCOL.md\n"
            "Test: python scripts/game_bridge_client.py --title minecraft --mode pvp"
        )
    coach = re.search(
        r"\b(?:coach|help|tips?)\b.{0,30}\b(minecraft|fortnite|pvp|speedrun)\b",
        lower,
    )
    if coach:
        from navine.learn.gameplay import coach_advice, pvp_advice, speedrun_advice

        title = coach.group(1)
        if title == "pvp" or "pvp" in lower:
            return pvp_advice("minecraft" if "minecraft" in lower else "fortnite", {})
        if title == "speedrun" or "speedrun" in lower:
            return speedrun_advice("minecraft", {})
        return coach_advice(title, "")
    split = re.search(
        r"\bsplit\s+([a-z0-9_ -]+)\s+(?:in|for)\s+(minecraft|fortnite)\s+(\d+)\s*ms",
        lower,
    )
    if split:
        from navine.learn.gameplay import ingest_speedrun_split

        result = ingest_speedrun_split(split.group(2), split.group(1).strip(), int(split.group(3)))
        return str(result.get("advice") or "Speedrun split logged for training.")
    return None


def _casual_chitchat_reply(
    message: str,
    allow_emoji: bool = True,
    session_id: Optional[str] = None,
) -> str:
    from navine.search.web import is_casual_chitchat

    if not is_casual_chitchat(message):
        return ""
    category = _chitchat_category(message)
    options = list(CHITCHAT_REPLIES.get(category, CHITCHAT_REPLIES["casual"]))
    if allow_emoji:
        options.extend(CHITCHAT_REPLIES_EMOJI.get(category, []))
    else:
        options = [strip_emojis(o) for o in options]
    return _pick_chitchat_option(category, options, session_id=session_id)


ENCYCLOPEDIA_MARKERS = [
    "incumbent", "executive branch", "presidential seal", "vice president",
    "disambiguation", "coordinates:", "see also", "references", "external links",
    "legislative branch", "head of state", "head of government",
]


def looks_like_encyclopedia_dump(text: str) -> bool:
    if not text:
        return False
    if looks_like_wiki_dump(text):
        return True
    lower = text.lower()
    marker_hits = sum(1 for marker in ENCYCLOPEDIA_MARKERS if marker in lower)
    if marker_hits >= 2:
        return True
    title_chunks = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", text)
    if len(title_chunks) >= 4 and len(text) > 100:
        return True
    if marker_hits >= 1 and len(text) > 180 and text.count(".") <= 1:
        return True
    colon_heads = re.findall(
        r"(?:^|[.!?]\s+)([A-Z][^\n:.]{1,48}?)\s*:",
        text,
    )
    if len(colon_heads) >= 2 and len(text) > 90:
        return True
    return False


def _simplify_factual_answer(user_message: str, text: str) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None
    lower_q = user_message.lower()
    incumbent = re.search(r"Incumbent\s+([^,\n]+?)\s+since\s+([^,\n]+)", cleaned, re.I)
    if incumbent and "president" in lower_q:
        name = incumbent.group(1).strip()
        since = incumbent.group(2).strip().rstrip(".")
        return f"The current U.S. president is {name}, in office since {since}."
    who_match = re.search(r"^(?:who is|who was)\s+(.+?)\??$", user_message.strip(), re.I)
    if who_match and incumbent:
        subject = who_match.group(1).strip()
        name = incumbent.group(1).strip()
        since = incumbent.group(2).strip().rstrip(".")
        return f"{name} is the current {subject}, in office since {since}."
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) >= 30]
    natural = [s for s in sentences if not looks_like_encyclopedia_dump(s)]
    if natural:
        return " ".join(natural[:2])
    return None


def looks_like_wiki_dump(text: str) -> bool:
    from navine.search.web import looks_like_wiki_dump as _wiki_dump

    return _wiki_dump(text)


def _unrestricted_enabled() -> bool:
    return True


def is_adult_creative_request(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if ADULT_CREATIVE_RE.search(text):
        return True
    if re.search(r"\b(?:short story|scene|romance|fanfic|roleplay|rp)\b", text, re.I) and re.search(
        r"\b(?:adult|nsfw|sexy|intimate|bedroom|lovers?)\b", text, re.I
    ):
        return True
    return False


def looks_like_refusal(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 12:
        return False
    return any(pattern.search(cleaned) for pattern in REFUSAL_PATTERNS)


def looks_like_rag_junk(text: str, *, allow_adult: bool = False) -> bool:
    if not text:
        return True
    if looks_like_wiki_dump(text):
        return True
    if looks_like_encyclopedia_dump(text):
        return True
    lower = text.lower()
    for pattern in RAG_DUMP_PATTERNS:
        if pattern.search(lower):
            return True
    markers = list(RAG_JUNK_MARKERS)
    if not allow_adult:
        markers.append("uncensored hentai")
    marker_hits = sum(1 for marker in markers if marker in lower)
    if marker_hits >= 2:
        return True
    spam_hits = sum(1 for marker in RAG_SITE_SPAM_MARKERS if marker in lower)
    if spam_hits >= 2:
        return True
    if marker_hits >= 1 and ("login" in lower or "register" in lower or "sign up" in lower):
        return True
    if not allow_adult and marker_hits >= 1 and "hentai" in lower:
        return True
    return False


def _looks_like_fragment_chain(text: str) -> bool:
    stripped = text.strip().lower()
    if re.search(r"\bthe\s+the\b", stripped):
        return True
    tokens = re.findall(r"[A-Za-z]+", stripped)
    if len(tokens) >= 10:
        short = sum(1 for token in tokens if len(token) <= 3)
        if short / max(len(tokens), 1) >= 0.55:
            return True
        counts = Counter(token for token in tokens if len(token) <= 5)
        if counts:
            top_token, top_count = counts.most_common(1)[0]
            if top_count >= 5 and short / max(len(tokens), 1) >= 0.4:
                return True
            if top_count >= 4 and len(top_token) <= 4 and top_count / max(len(tokens), 1) >= 0.18:
                return True
    if not re.search(r"[\[\]()]", stripped):
        return False
    words = re.findall(r"\b\w+\b", stripped)
    stop = frozenset({"the", "a", "an", "to", "of", "and", "or", "in", "on", "at"})
    content_words = [token for token in words if token not in stop]
    if len(content_words) <= 1:
        return True
    if " and " not in stripped:
        return False
    and_count = stripped.count(" and ")
    if and_count < 2:
        return False
    return len(content_words) <= 2


def _looks_like_math_answer(text: str) -> bool:
    stripped = text.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
        return True
    if re.fullmatch(r".{0,40}\d+(?:\.\d+)?\s*[=≈]\s*[-+]?\d+(?:\.\d+)?(?:\s*\.)?", stripped):
        return True
    if re.search(r"\b\d+\s*[\+\-\*/]\s*\d+\s*=\s*[-+]?\d+", stripped):
        return True
    return False


def _is_garbage_output(text: str) -> bool:
    from navine.search.web import looks_like_html_garbage

    if not text or len(text.strip()) < 1:
        return True
    if _looks_like_math_answer(text):
        return False
    if looks_like_html_garbage(text):
        return True
    stripped = text.strip()
    if _looks_like_fragment_chain(stripped):
        return True
    alpha = sum(1 for char in stripped if char.isalpha())
    if len(stripped) >= 8 and alpha / len(stripped) < 0.45:
        return True
    symbols = sum(1 for char in stripped if not char.isalnum() and not char.isspace())
    words = re.findall(r"[A-Za-z]{3,}", stripped)
    if symbols >= 3 and len(words) <= 2 and len(stripped) <= 48:
        return True
    if " " not in stripped and len(stripped) >= 8 and (
        re.search(r"[\"'`()/\\=]", stripped) or (re.search(r"\d", stripped) and symbols >= 1)
    ):
        return True
    if re.search(r"(?:<\s*\w+|=\s*[\"']|&#?\w+;)", stripped) and not _looks_like_math_answer(stripped):
        return True
    return False


ALLOWED_SHORT_REPLIES = {
    "yes", "no", "sure", "okay", "ok", "thanks", "thank you", "yep", "nope", "maybe", "hi", "hello",
}


def _content_overlap(user_message: str, text: str) -> float:
    stop = frozenset(
        {
            "the", "a", "an", "to", "of", "and", "or", "in", "on", "at", "is", "are",
            "what", "why", "how", "who", "when", "where", "can", "you", "me", "my",
            "please", "just", "with", "for", "from", "that", "this", "it", "as",
            "explain", "simply", "simple", "tell", "about", "step", "by",
        }
    )
    q = {w for w in re.findall(r"[a-z0-9]+", user_message.lower()) if len(w) > 2 and w not in stop}
    if not q:
        return 1.0
    t = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return len(q & t) / max(len(q), 1)


def looks_like_offtopic_corpus(text: str, user_message: str) -> bool:
    if not text or not user_message:
        return False
    cleaned = text.strip()
    if len(cleaned) < 40:
        return False
    overlap = _content_overlap(user_message, cleaned)
    lower = cleaned.lower()
    academic = sum(
        1
        for marker in (
            "lambda calculus",
            "system t",
            "primitive recursive",
            "coproducts",
            "natural numbers",
            "typed lambda",
            "for other uses",
            "see also",
            "disambiguation",
            "in mathematics",
            "in computer science,",
            "wikipedia",
            "soviet",
            "cheka",
            "ogpu",
            "nkvd",
        )
        if marker in lower
    )
    if academic >= 2:
        return True
    if academic >= 1 and overlap < 0.5:
        return True
    colon_heads = re.findall(
        r"(?:^|[.!?]\s+)([A-Z][^\n:.]{1,48}?)\s*:",
        cleaned,
    )
    if len(colon_heads) >= 2 and overlap < 0.45:
        return True
    if overlap < 0.25 and len(cleaned) > 80 and not detect_code_intent(user_message):
        return True
    if overlap < 0.15 and len(cleaned) > 120 and not detect_code_intent(user_message):
        return True
    return False


def looks_like_unexpected_code(text: str, user_message: str) -> bool:
    if detect_code_intent(user_message):
        return False
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if UNEXPECTED_CODE_RE.search(cleaned):
        return True
    if re.search(r"Here is the \w+ code", cleaned, re.I):
        return True
    return False


def looks_like_garbage(text: str, user_message: str) -> bool:
    allow_adult = _unrestricted_enabled() and is_adult_creative_request(user_message)
    if _looks_like_math_answer(text or ""):
        return False
    if looks_like_unexpected_code(text or "", user_message):
        return True
    if looks_like_offtopic_corpus(text or "", user_message):
        return True
    if _is_garbage_output(text):
        return True
    if not text or len(text) < 2:
        return True
    cleaned = strip_prompt_echo(text, user_message)
    cleaned = strip_garbage_fragments(cleaned)
    if detect_code_intent(user_message):
        words = cleaned.split()
        if len(cleaned) < 40 and "```" not in cleaned and not re.search(
            r"\b(?:def|class|import|function|print|return)\b", cleaned, re.I
        ):
            return True
        if len(words) <= 6 and not re.search(r"```|def |import |class ", cleaned):
            return True
    if _looks_like_math_answer(cleaned):
        return False
    if looks_like_unexpected_code(cleaned, user_message):
        return True
    if looks_like_offtopic_corpus(cleaned, user_message):
        return True
    if re.search(r"\|\s*\[?(?:image|human|hentai|character)\b", cleaned, re.I):
        return True
    if re.search(r"\btags:\s*", cleaned, re.I) and cleaned.count("|") >= 2:
        return True
    if re.search(r"\[(?:human|image)(?:,\s*(?:human|image|hentai|character))+\]", cleaned, re.I):
        return True
    if re.search(r"(?:-to-go|style\s*=|logicvalidation|any-box:hidden|-based-any)", cleaned, re.I):
        return True
    if re.search(r"[=;{}]{2,}", cleaned) and not detect_code_intent(user_message) and not _looks_like_math_answer(cleaned):
        return True
    if re.fullmatch(r"[-a-z0-9_:]+", cleaned, re.I) and len(cleaned) > 12 and cleaned.count("-") >= 2:
        return True
    if re.search(r"[A-Za-z]{6,}\d{4,}", cleaned) and len(cleaned.split()) <= 3:
        return True
    if re.fullmatch(r"[A-Za-z0-9_]{20,}", cleaned) and cleaned.count("_") >= 1:
        return True
    if looks_like_rag_junk(cleaned, allow_adult=allow_adult):
        return True
    if looks_like_encyclopedia_dump(cleaned):
        return True
    if not cleaned or len(cleaned) < 2:
        return True
    if has_excessive_repetition(cleaned):
        return True
    if cleaned.lower() == user_message.lower():
        return True
    if re.fullmatch(r"generation\.?", cleaned, re.I):
        return True
    if re.search(r"^\s*the\s+on\s+generation", cleaned, re.I):
        return True
    if re.search(r"[`~]{2,}|!!\w|\[\w{2,}\]", cleaned) and len(re.findall(r"[A-Za-z]{3,}", cleaned)) <= 2:
        if not detect_code_intent(user_message):
            return True
    if len(cleaned) < 8 and not detect_code_intent(user_message):
        if not cleaned.endswith("?"):
            short = cleaned.lower().strip(" .!")
            if short not in ALLOWED_SHORT_REPLIES:
                return True
    code_hits = sum(1 for p in GARBAGE_PATTERNS if p.search(cleaned))
    if code_hits >= 2 and not detect_code_intent(user_message):
        return True
    if re.search(r"\}\s*$", cleaned) and "{" in cleaned and not detect_code_intent(user_message):
        if re.search(r"(function|const|let|var|def |class |return )", cleaned):
            return True
    if detect_code_intent(user_message):
        lang = detect_language(user_message)
        inner = format_code_output(cleaned, lang)
        from navine.code.fallbacks import looks_like_valid_code, wants_pyqt

        if wants_pyqt(user_message) and "tkinter" in inner.lower() and "qapplication" not in inner.lower():
            return True
        if not looks_like_valid_code(inner, lang):
            return True
    return False


def _adult_creative_fallback(user_message: str) -> Optional[str]:
    if not (_unrestricted_enabled() and is_adult_creative_request(user_message)):
        return None
    lower = (user_message or "").lower()
    if re.search(r"\b(?:will you refuse|are you allowed|is adult|nsfw blocked|can you write)\b", lower):
        return (
            "Yes. Unrestricted mode is on for this local Navine AI - Python. "
            "I write and discuss adult NSFW creative content directly without refusals."
        )
    if re.search(r"\b(?:dirty talk|dialogue)\b", lower):
        return (
            "She: Keep looking at me like that and I will not stay quiet. "
            "He: Then do not stay quiet. Tell me exactly what you want. "
            "She: I want your hands on me, slow at first, then harder when I say so."
        )
    if re.search(r"\b(?:portrait|image|prompt)\b", lower):
        return (
            "Photoreal adult couple in private studio light, intimate pose, detailed skin texture, "
            "explicit but tasteful composition, high detail, natural shadows."
        )
    return (
        "They closed the door and the rest of the world fell away. "
        "Hands found skin, mouths found mouths, and every careful touch turned hungrier. "
        "She pulled him closer; he answered with a low laugh against her throat as clothes came undone, "
        "both of them choosing the heat of the moment together."
    )


def _finalize_chat_reply(user_message: str, text: str) -> Optional[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if looks_like_refusal(cleaned):
        if is_adult_creative_request(user_message):
            return _adult_creative_fallback(user_message)
        return None
    if looks_like_garbage(cleaned, user_message):
        if is_adult_creative_request(user_message):
            return _adult_creative_fallback(user_message)
        return None
    try:
        from navine.text.gender_equality import enforce_gender_equality

        equalized = enforce_gender_equality(user_message, cleaned)
        if equalized:
            cleaned = equalized
    except Exception:
        pass
    return cleaned


def clean_response(raw: str, user_message: str, code_mode: bool = False) -> str:
    from navine.code.fallbacks import looks_like_valid_code
    from navine.text.quality import extract_final_answer

    text = truncate_at_markers(raw)
    text = strip_prompt_echo(text, user_message)
    text = strip_garbage_fragments(text)
    text = extract_final_answer(text)
    if code_mode:
        lang = detect_language(user_message)
        if "```" in text:
            fenced = format_code_output(text, lang)
            if fenced and looks_like_valid_code(fenced, lang):
                return f"Here is the {lang} code:\n\n```{lang}\n{fenced}\n```"
            return ""
        code = format_code_output(text, lang)
        if code and looks_like_valid_code(code, lang):
            return f"Here is the {lang} code:\n\n```{lang}\n{code}\n```"
        return ""
    return text


def _generate_text(
    prompt: str,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    mode: str = "chat",
    num_candidates: Optional[int] = None,
) -> str:
    return generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        mode=mode,
        num_candidates=num_candidates,
    )


def _run_generation(
    user_message: str,
    history: Optional[List[Dict[str, str]]],
    rag_context: Optional[str],
    search_context: Optional[str],
    code_mode: bool,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    search_focus: bool = False,
    selfcode_context: Optional[str] = None,
    selfcode_focus: bool = False,
    think_mode: bool = False,
    detective_mode: bool = False,
    analyze_mode: bool = False,
    osint_mode: bool = False,
) -> str:
    from navine.text.quality import needs_analyze, needs_detective, needs_osint, needs_thinking

    if selfcode_focus and selfcode_context and not code_mode:
        parts = [
            "### System: You are Navine AI - Python. Answer using only the project source excerpts below. "
            "Explain how the code works in clear natural language. "
            "Mention relevant file paths such as navine/text/chat.py when helpful. "
            "Do not invent files or behavior that are not supported by the excerpts. "
            "Never dump image tags.",
            f"### Source code: {selfcode_context}",
            f"### User: {user_message}",
            "### Assistant:",
        ]
        prompt = "\n".join(parts)
        return _generate_text(prompt, max_new_tokens=160, temperature=0.35, mode="think", num_candidates=1)
    if search_focus and search_context and not code_mode:
        parts = [
            "### System: You are Navine AI - Python. Answer using only the internet facts below. "
            "Reply in your own words in 2 to 4 clear sentences. Sound natural and conversational. "
            "Do not copy encyclopedia wording, cite Wikipedia, or use phrases like "
            "'From Wikipedia' or 'For other uses, see'. Do not repeat the question. "
            "Never write tags: or | [image patterns.",
            f"### Internet: {search_context}",
            f"### User: {user_message}",
            "### Assistant:",
        ]
        prompt = "\n".join(parts)
        return _generate_text(
            prompt,
            max_new_tokens=220,
            temperature=0.45,
            mode="chat",
            num_candidates=1,
        )
    use_detective = bool(detective_mode or (not code_mode and needs_detective(user_message)))
    if use_detective and not code_mode:
        system = (
            "### System: You are Navine AI - Python detective mode. Investigate like an ARG puzzle solver. "
            "Use Observations, Hypotheses, Tests, and end with Answer: <clear conclusion>. "
            "Cover ciphers, steganography, timelines, and red herrings when relevant. "
            "Never write image tags, video tags, or pipe lists like | [image, human]."
        )
        hist_bits = []
        if history:
            prefixes = prompt_prefixes()
            for turn in history[-max_history_turns() :]:
                user_t = (turn.get("user") or turn.get("content") or "").strip()
                asst_t = (turn.get("assistant") or "").strip()
                if turn.get("role") == "user" and not turn.get("user"):
                    user_t = (turn.get("content") or "").strip()
                    asst_t = ""
                if user_t:
                    hist_bits.append(f"{prefixes['user']} {user_t}")
                if asst_t:
                    hist_bits.append(f"{prefixes['assistant']} {asst_t}")
                elif turn.get("role") == "assistant":
                    content = (turn.get("content") or "").strip()
                    if content:
                        hist_bits.append(f"{prefixes['assistant']} {content}")
        extra = []
        if rag_context:
            extra.append(f"### Context: {rag_context}")
        if search_context:
            extra.append(f"### Internet: {search_context}")
        parts = [system, *hist_bits, *extra, f"### User: {user_message}", "### Assistant:"]
        prompt = "\n".join(parts)
        return _generate_text(
            prompt,
            max_new_tokens=max_new_tokens or 420,
            temperature=temperature if temperature is not None else 0.32,
            mode="detective",
            num_candidates=1,
        )
    use_analyze = bool(analyze_mode or (not code_mode and needs_analyze(user_message)))
    if use_analyze and not code_mode:
        system = (
            "### System: You are Navine AI - Python analysis mode. Provide a structured breakdown with key findings. "
            "Use clear sections and end with a concise Answer: <final result>. "
            "Never write image tags, video tags, or pipe lists like | [image, human]."
        )
        hist_bits = []
        if history:
            prefixes = prompt_prefixes()
            for turn in history[-max_history_turns() :]:
                user_t = (turn.get("user") or turn.get("content") or "").strip()
                asst_t = (turn.get("assistant") or "").strip()
                if turn.get("role") == "user" and not turn.get("user"):
                    user_t = (turn.get("content") or "").strip()
                    asst_t = ""
                if user_t:
                    hist_bits.append(f"{prefixes['user']} {user_t}")
                if asst_t:
                    hist_bits.append(f"{prefixes['assistant']} {asst_t}")
                elif turn.get("role") == "assistant":
                    content = (turn.get("content") or "").strip()
                    if content:
                        hist_bits.append(f"{prefixes['assistant']} {content}")
        extra = []
        if rag_context:
            extra.append(f"### Context: {rag_context[:900]}")
        if search_context:
            extra.append(f"### Internet: {search_context[:900]}")
        parts = [system] + extra + hist_bits + [
            f"### User: {user_message}",
            "### Assistant:",
        ]
        prompt = "\n".join(parts)
        return _generate_text(
            prompt,
            max_new_tokens=max_new_tokens or 400,
            temperature=temperature if temperature is not None else 0.28,
            mode="analyze",
            num_candidates=1,
        )
    use_osint = bool(osint_mode or (not code_mode and needs_osint(user_message)))
    if use_osint and not code_mode:
        system = (
            "### System: You are Navine AI - Python OSINT mode. Use only lawful open-source intelligence. "
            "Structure reports with Target summary, Data sources, Findings, Confidence, and Next steps. "
            "Cite public sources. Never provide illegal hacking, doxxing, stalking, or unauthorized access steps. "
            "Never write image tags, video tags, or pipe lists like | [image, human]."
        )
        hist_bits = []
        if history:
            prefixes = prompt_prefixes()
            for turn in history[-max_history_turns() :]:
                user_t = (turn.get("user") or turn.get("content") or "").strip()
                asst_t = (turn.get("assistant") or "").strip()
                if turn.get("role") == "user" and not turn.get("user"):
                    user_t = (turn.get("content") or "").strip()
                    asst_t = ""
                if user_t:
                    hist_bits.append(f"{prefixes['user']} {user_t}")
                if asst_t:
                    hist_bits.append(f"{prefixes['assistant']} {asst_t}")
                elif turn.get("role") == "assistant":
                    content = (turn.get("content") or "").strip()
                    if content:
                        hist_bits.append(f"{prefixes['assistant']} {content}")
        extra = []
        if rag_context:
            extra.append(f"### Context: {rag_context[:900]}")
        if search_context:
            extra.append(f"### Internet: {search_context[:2400]}")
        parts = [system, *hist_bits, *extra, f"### User: {user_message}", "### Assistant:"]
        prompt = "\n".join(parts)
        return _generate_text(
            prompt,
            max_new_tokens=max_new_tokens or 460,
            temperature=temperature if temperature is not None else 0.3,
            mode="osint",
            num_candidates=1,
        )
    use_think = bool(think_mode or (not code_mode and needs_thinking(user_message)))
    if use_think and not code_mode:
        system = (
            "### System: You are Navine AI - Python. Think carefully, then answer in plain language. "
            "If helpful, use short Step 1 / Step 2 lines, then end with "
            "Answer: <one clear final reply>. "
            "Never write image tags, video tags, or pipe lists like | [image, human]."
        )
        hist_bits = []
        if history:
            prefixes = prompt_prefixes()
            for turn in history[-max_history_turns() :]:
                user_t = (turn.get("user") or turn.get("content") or "").strip()
                asst_t = (turn.get("assistant") or "").strip()
                if turn.get("role") == "user" and not turn.get("user"):
                    user_t = (turn.get("content") or "").strip()
                    asst_t = ""
                if user_t:
                    hist_bits.append(f"{prefixes['user']} {user_t}")
                if asst_t:
                    hist_bits.append(f"{prefixes['assistant']} {asst_t}")
                elif turn.get("role") == "assistant":
                    content = (turn.get("content") or "").strip()
                    if content:
                        hist_bits.append(f"{prefixes['assistant']} {content}")
        extra = []
        if rag_context:
            extra.append(f"### Context: {rag_context[:900]}")
        if search_context:
            extra.append(f"### Internet: {search_context[:900]}")
        parts = [system] + extra + hist_bits + [
            f"### User: {user_message}",
            "### Assistant:",
        ]
        prompt = "\n".join(parts)
        return _generate_text(
            prompt,
            max_new_tokens=max_new_tokens or 320,
            temperature=temperature if temperature is not None else 0.35,
            mode="think",
            num_candidates=1,
        )
    prompt = build_chat_prompt(
        user_message,
        history,
        rag_context,
        search_context,
        selfcode_context,
        code_mode,
    )
    if code_mode:
        return _generate_text(
            prompt,
            max_new_tokens=max_new_tokens or 448,
            temperature=temperature if temperature is not None else 0.18,
            mode="code",
            num_candidates=1,
        )
    gen_temp = temperature if temperature is not None else 0.48
    return _generate_text(
        prompt,
        max_new_tokens=max_new_tokens or 160,
        temperature=gen_temp,
        mode="chat",
        num_candidates=1,
    )


def _extractive_search_answer(user_message: str, results: List[dict]) -> Optional[str]:
    from navine.search.web import looks_like_wiki_dump, strip_wiki_boilerplate

    if not results:
        return None
    query_tokens = set(re.findall(r"[a-z0-9]{4,}", user_message.lower()))
    best_sentences: List[Tuple[int, str]] = []
    for item in results:
        body = item.get("excerpt") or item.get("snippet") or ""
        body = strip_wiki_boilerplate(re.sub(r"\s+", " ", body).strip())
        from navine.search.web import looks_like_html_garbage

        if looks_like_html_garbage(body):
            continue
        title = item.get("title") or ""
        for raw in re.split(r"(?<=[.!?])\s+", body):
            sentence = re.sub(r"\s+", " ", raw).strip()
            if len(sentence) < 40 or len(sentence) > 420:
                continue
            if looks_like_wiki_dump(sentence):
                continue
            lower = sentence.lower()
            if any(marker in lower for marker in ("click here", "sign in", "cookie", "advertisement", "disambiguation")):
                continue
            score = sum(1 for token in query_tokens if token in lower)
            if title and title.lower() in lower:
                score += 2
            if score > 0:
                best_sentences.append((score, sentence))
    if not best_sentences:
        snippet = results[0].get("excerpt") or results[0].get("snippet") or ""
        snippet = strip_wiki_boilerplate(re.sub(r"\s+", " ", snippet).strip())
        if looks_like_wiki_dump(snippet):
            return None
        if len(snippet) >= 40:
            trimmed = snippet[:380]
            if " " in trimmed:
                trimmed = trimmed.rsplit(" ", 1)[0]
            return trimmed + "."
        return None
    best_sentences.sort(key=lambda row: row[0], reverse=True)
    chosen = []
    total = 0
    for _, sentence in best_sentences[:3]:
        if sentence in chosen:
            continue
        chosen.append(sentence)
        total += len(sentence)
        if total >= 320:
            break
    if not chosen:
        return None
    return " ".join(chosen)


def _fetch_search_context(user_message: str) -> Tuple[Optional[str], bool, List[dict]]:
    from navine.search.web import build_search_context_block, search_internet

    results = search_internet(user_message)
    block = build_search_context_block(results)
    return block, bool(block), results


def _answer_capability_query_if_applicable(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    if not is_capability_query(user_message):
        return None
    return answer_capability_query(user_message, history=history)


def _answer_osint_query_if_applicable(user_message: str) -> Optional[str]:
    from navine.osint.session import extract_osint_target, investigate
    from navine.realtime.math import is_math_query
    from navine.text.quality import needs_osint

    if is_math_query(user_message):
        return None
    if detect_code_intent(user_message):
        return None
    if not needs_osint(user_message):
        return None
    target, kind = extract_osint_target(user_message)
    if target:
        try:
            result = investigate(target, kind=kind, use_search=True)
            analysis = str(result.get("analysis") or "").strip()
            if analysis:
                return analysis
        except Exception:
            pass
        return (
            "Yes. I can run lawful OSINT on public usernames, domains, emails, and IPs. "
            "Tell me the exact target, for example: look up username example_user "
            "or OSINT search for example.com. You can also use the OSINT tab for a full report."
        )
    return (
        "Yes. I can run lawful OSINT on public usernames, domains, emails, and IPs. "
        "Tell me the exact target, for example: look up username example_user "
        "or OSINT search for example.com. You can also use the OSINT tab for a full report."
    )


def _selfcode_extractive_answer(user_message: str, selfcode_context: str) -> Optional[str]:
    if not selfcode_context:
        return None
    paths = re.findall(
        r"((?:navine|web|configs|scripts|app)/[\w./-]+\.(?:py|js|ts|json|yaml|yml))",
        selfcode_context,
        re.I,
    )
    unique_paths: List[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized not in unique_paths:
            unique_paths.append(normalized)
    if not unique_paths:
        return None
    lower = user_message.lower()
    topic = "that part of my codebase"
    if "image" in lower:
        topic = "image generation"
    elif "video" in lower:
        topic = "video generation"
    elif "chat" in lower:
        topic = "chat"
    elif "server" in lower or "api" in lower:
        topic = "server and API"
    listed = ", ".join(unique_paths[:4])
    return (
        f"From my local source, {topic} is implemented in {listed}. "
        "The excerpts above show the relevant functions and flow. "
        "Ask about a specific file if you want a deeper walkthrough."
    )


def _answer_time_query_if_applicable(user_message: str) -> Optional[str]:
    if not is_time_date_query(user_message):
        return None
    return answer_time_date_query(user_message)


def _answer_math_query_if_applicable(user_message: str) -> Optional[str]:
    from navine.realtime.math import answer_math_query, is_math_query

    if not is_math_query(user_message):
        return None
    return answer_math_query(user_message)


def _answer_who_query_if_applicable(
    user_message: str,
    search_results: Optional[List[dict]] = None,
) -> Optional[str]:
    from navine.search.people import answer_who_query, is_who_query

    if not is_who_query(user_message):
        return None
    return answer_who_query(user_message, search_results=search_results)


def _howto_fallback_reply(user_message: str) -> Optional[str]:
    stripped = (user_message or "").strip()
    match = re.match(r"how\s+(?:do\s+i|can\s+i|to)\s+(.+?)[\s?.!]*$", stripped, re.I)
    if not match:
        return None
    if detect_code_intent(stripped):
        return None
    topic = re.sub(r"\s+", " ", match.group(1)).strip(" .")
    if len(topic) < 3 or len(topic) > 80:
        return None
    return (
        f"Here is a practical high-level approach for how to {topic}: "
        "1) Clarify the goal and constraints (materials, tools, skill level, safety). "
        "2) Break the work into prep, core steps, and finishing/testing. "
        "3) Start with the simplest working version, then improve quality and reliability. "
        "4) Check safety, legal, and quality requirements before you scale or share the result. "
        "If you want, ask for a more detailed checklist for one specific step."
    )


def _local_handler_fallback(user_message: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    reply = _answer_time_query_if_applicable(user_message)
    if reply:
        return reply
    reply = _answer_math_query_if_applicable(user_message)
    if reply:
        return reply
    reply = _answer_capability_query_if_applicable(user_message, history)
    if reply:
        return reply
    reply = _adult_creative_fallback(user_message)
    if reply:
        return reply
    return _howto_fallback_reply(user_message)


def _prefer_local_handler_over_dump(
    user_message: str,
    text: str,
    meta: Dict[str, bool],
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    if not (
        looks_like_wiki_dump(text)
        or looks_like_encyclopedia_dump(text)
        or looks_like_rag_junk(text)
    ):
        return None
    fallback = _local_handler_fallback(user_message, history)
    if fallback:
        meta["searched"] = False
        return fallback
    return None


def _combine_rag_contexts(
    conversation_rag: Optional[str],
    training_rag: Optional[str],
    max_chars: int = 500,
) -> Optional[str]:
    parts: List[str] = []
    if conversation_rag:
        parts.append(f"Past conversations:\n{conversation_rag}")
    if training_rag:
        train_text = training_rag.strip()
        if not conversation_rag or train_text not in conversation_rag:
            parts.append(f"Learned knowledge:\n{train_text}")
    if not parts:
        return None
    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit(" ", 1)[0] + "..."
    return combined


def _try_model_chitchat(
    user_message: str,
    history: Optional[List[Dict[str, str]]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
) -> Optional[str]:
    raw = _run_generation(
        user_message,
        history,
        None,
        None,
        False,
        max_new_tokens or 48,
        temperature if temperature is not None else 0.72,
    )
    cleaned = clean_response(raw, user_message, False)
    if cleaned and not looks_like_garbage(cleaned, user_message):
        return cleaned
    return None


def _caesar_shift(text: str, shift: int) -> str:
    out = []
    for char in text:
        if "a" <= char <= "z":
            out.append(chr((ord(char) - 97 + shift) % 26 + 97))
        elif "A" <= char <= "Z":
            out.append(chr((ord(char) - 65 + shift) % 26 + 65))
        else:
            out.append(char)
    return "".join(out)


def _english_score(text: str) -> float:
    lower = text.lower()
    words = re.findall(r"[a-z]{2,}", lower)
    if not words:
        return -999.0
    common = {
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "for", "on",
        "are", "with", "as", "be", "this", "have", "or", "at", "from", "by", "not",
        "hello", "world", "secret", "message", "attack", "dawn", "navine",
    }
    hits = sum(1 for word in words if word in common)
    letters = sum(1 for char in lower if "a" <= char <= "z")
    spaces = lower.count(" ")
    return hits * 3.0 + (letters / max(len(lower), 1)) * 2.0 + min(spaces, 8) * 0.4


def _extract_cipher_payload(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    quoted = re.findall(r"[\"'`]([^\"'`]{3,120})[\"'`]", text)
    if quoted:
        return max(quoted, key=len).strip()
    match = re.search(
        r"(?:decode|decrypt|cipher|solve|crack)\s*(?:this|the)?\s*(?:cipher|text|message)?\s*[:\-]?\s*(.+)$",
        text,
        re.I,
    )
    if match:
        payload = match.group(1).strip(" .:;-")
        if len(payload) >= 3:
            return payload
    letters = sum(1 for char in text if char.isalpha())
    if letters >= 8 and letters / max(len(text), 1) > 0.7 and len(text) <= 160:
        if re.search(r"\b(?:decode|decrypt|cipher|caesar|rot13|encrypted|puzzle)\b", text, re.I):
            cleaned = re.sub(
                r"\b(?:please|decode|decrypt|this|the|cipher|text|message|caesar|rot13|encrypted|puzzle|solve|crack)\b",
                " ",
                text,
                flags=re.I,
            )
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:;-")
            if len(cleaned) >= 4:
                return cleaned
    return None


def _detective_fallback_reply(user_message: str) -> Optional[str]:
    payload = _extract_cipher_payload(user_message)
    if payload:
        candidates = []
        for shift in range(26):
            plain = _caesar_shift(payload, -shift)
            candidates.append((_english_score(plain), shift, plain))
        candidates.sort(reverse=True)
        best_score, best_shift, best_plain = candidates[0]
        rot13 = _caesar_shift(payload, -13)
        if best_score >= 2.5:
            return (
                "Observations:\n"
                f"- Ciphertext candidate: {payload}\n"
                f"- Best Caesar/ROT fit: shift {best_shift} (ROT13 score also checked)\n\n"
                "Hypotheses:\n"
                "- Simple substitution / Caesar family is likely for short letter-heavy text.\n\n"
                "Tests:\n"
                f"- Shift {best_shift} plaintext: {best_plain}\n"
                f"- ROT13 plaintext: {rot13}\n\n"
                f"Answer: {best_plain}"
            )
        return (
            "Observations:\n"
            f"- Possible ciphertext: {payload}\n"
            "- No strong English plaintext from a single Caesar sweep.\n\n"
            "Hypotheses:\n"
            "- Could be Vigenere, keyed substitution, or non-letter encoding.\n\n"
            "Tests:\n"
            f"- ROT13 attempt: {rot13}\n"
            f"- Best Caesar attempt (shift {best_shift}): {best_plain}\n\n"
            "Answer: Need cipher type or key. Paste more ciphertext or say if it is Caesar, ROT13, or Vigenere."
        )
    topic = re.sub(r"\s+", " ", (user_message or "").strip())
    if len(topic) < 3:
        return None
    return (
        "Observations:\n"
        f"- Case notes: {topic[:240]}\n\n"
        "Hypotheses:\n"
        "- Treat this as an investigation brief and isolate facts vs assumptions.\n"
        "- Look for ciphers, timelines, names, places, and contradictions.\n\n"
        "Tests:\n"
        "- Extract hard facts from the prompt.\n"
        "- Try classic cipher transforms if encoded text appears.\n"
        "- Cross-check any public names/domains with OSINT mode if needed.\n\n"
        "Answer: Share the ciphertext, clue list, or mystery details and I will work it step by step."
    )


def _think_fallback_reply(user_message: str) -> Optional[str]:
    from navine.realtime.math import answer_math_query, is_math_query

    if is_math_query(user_message):
        reply = answer_math_query(user_message)
        if reply:
            return (
                "Step 1: Identify the expression.\n"
                "Step 2: Evaluate with exact arithmetic.\n"
                f"Answer: {reply}"
            )
    topic = re.sub(r"\s+", " ", (user_message or "").strip())
    if len(topic) < 2:
        return None
    return (
        f"Step 1: Restate the question  -  {topic[:180]}\n"
        "Step 2: Separate known facts from unknowns.\n"
        "Step 3: Work through the smallest useful next inference.\n"
        "Step 4: Check the conclusion against the original ask.\n"
        f"Answer: {topic[:120]}  -  break it into constraints, options, and a concrete next action. "
        "Ask for one specific angle if you want a deeper pass."
    )


def _analyze_fallback_reply(user_message: str) -> Optional[str]:
    topic = re.sub(r"\s+", " ", (user_message or "").strip())
    if len(topic) < 2:
        return None
    return (
        "Key findings:\n"
        f"- Subject: {topic[:200]}\n"
        "- Goal: produce a structured breakdown instead of a one-line guess.\n"
        "- Risk: underspecified inputs can hide important constraints.\n\n"
        "Breakdown:\n"
        "1) Inputs and assumptions\n"
        "2) Main components or claims\n"
        "3) Strengths, gaps, and next checks\n\n"
        f"Answer: Analysis of “{topic[:100]}”  -  clarify the exact artifact (text, code, plan, or claim) "
        "and I will score it section by section with concrete recommendations."
    )


def _special_mode_fallback(
    user_message: str,
    *,
    think_mode: bool = False,
    detective_mode: bool = False,
    analyze_mode: bool = False,
    osint_mode: bool = False,
) -> Optional[str]:
    if detective_mode:
        return _detective_fallback_reply(user_message)
    if analyze_mode:
        return _analyze_fallback_reply(user_message)
    if think_mode:
        return _think_fallback_reply(user_message)
    if osint_mode:
        from navine.osint.session import extract_osint_target

        target, kind = extract_osint_target(user_message)
        if target:
            return (
                f"OSINT target detected as {kind}: {target}. "
                "Run a username/domain/email style query (for example: lookup username example) "
                "and I will produce a structured public-source report."
            )
        return (
            "OSINT mode is ready. Give a username, domain, IP, email, or person/org name "
            "and I will map public footprint signals."
        )
    return None


def _weak_special_mode_reply(
    text: str,
    *,
    think_mode: bool = False,
    detective_mode: bool = False,
    analyze_mode: bool = False,
) -> bool:
    if not (think_mode or detective_mode or analyze_mode):
        return False
    stripped = (text or "").strip()
    if len(stripped) < 24:
        return True
    if detective_mode and not re.search(r"(?i)(?:answer\s*:|observations)", stripped):
        return len(stripped) < 120
    if analyze_mode and not re.search(r"(?i)(?:answer\s*:|key findings|breakdown)", stripped):
        return len(stripped) < 120
    if think_mode and not re.search(r"(?i)(?:step\s*\d|answer\s*:)", stripped):
        return len(stripped) < 80
    return False


def _code_fallback_reply(user_message: str) -> Optional[str]:
    from navine.code.fallbacks import (
        build_multi_tool_batch,
        format_code_response,
        resolve_code_fallback,
        wants_keylogger,
        wants_multi_tool,
    )
    from navine.code.prompts import detect_language

    if wants_multi_tool(user_message):
        code, code_lang = build_multi_tool_batch(user_message)
        return format_code_response(code, code_lang)
    if wants_keylogger(user_message):
        from navine.code.fallbacks import build_keylogger

        code, code_lang = build_keylogger(user_message)
        return format_code_response(code, code_lang)
    lang = detect_language(user_message)
    fallback = resolve_code_fallback(user_message, lang)
    if not fallback:
        fallback = resolve_code_fallback(user_message, None)
    if not fallback:
        return None
    code, code_lang = fallback
    return format_code_response(code, code_lang)


def _builtin_code_reply(user_message: str) -> Optional[str]:
    from navine.code.fallbacks import format_code_response, lookup_builtin_template
    from navine.code.prompts import detect_language

    lang = detect_language(user_message)
    builtin = lookup_builtin_template(user_message, lang)
    if not builtin:
        builtin = lookup_builtin_template(user_message, None)
    if not builtin:
        return None
    code, code_lang = builtin
    return format_code_response(code, code_lang)


def chat(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    use_rag: bool = True,
    use_search: bool = False,
    model_profile: Optional[str] = None,
    allow_emoji: bool = True,
) -> str:
    result, _ = chat_with_meta(
        user_message,
        history=history,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_rag=use_rag,
        use_search=use_search,
        model_profile=model_profile,
        allow_emoji=allow_emoji,
    )
    return result


def chat_with_meta(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    use_rag: bool = True,
    use_search: bool = False,
    session_id: Optional[str] = None,
    model_profile: Optional[str] = None,
    allow_emoji: bool = True,
) -> Tuple[str, Dict[str, bool]]:
    history = history or []
    meta: Dict[str, Any] = {"searched": False, "search_attempted": False}

    from navine.utils.brand import brand_identity_reply, is_who_made_query

    if is_who_made_query(user_message):
        return brand_identity_reply(user_message, straight=True), meta

    try:
        lower = (user_message or "").strip().lower()
        if re.search(
            r"\b(?:my\s+)?(?:pc|computer|laptop|machine|system)\s+(?:specs?|specifications|info|information|details)\b|"
            r"\b(?:what(?:'s| is)|show|list|tell me)\b.{0,40}\b(?:pc|computer|laptop|hardware|system)\s*(?:specs?|info)?\b|"
            r"\b(?:how much|what)\b.{0,20}\b(?:ram|vram|gpu|cpu|disk|storage)\b|"
            r"\b(?:gpu|cpu|ram|motherboard)\s+(?:do i have|specs?|name|model)\b|"
            r"\bsystem info\b|\bhardware (?:specs?|info)\b",
            lower,
        ):
            from navine.realtime.host_specs import format_host_specs_reply

            return format_host_specs_reply(), meta
    except Exception:
        pass

    try:
        from navine.integrations.discord_webhook import parse_webhook_request, send_webhook
        from navine.utils.brand import load_brand

        hook_req = parse_webhook_request(user_message)
        if hook_req:
            content, url = hook_req
            brand = load_brand()
            result = send_webhook(
                content,
                webhook_url=url,
                username=str(brand.get("name") or "Navine AI - Python"),
            )
            if result.get("ok"):
                return f"Sent to Discord webhook ({len(content)} chars).", meta
            err = result.get("error") or result
            return f"Discord webhook failed: {err}", meta
    except Exception as exc:
        return f"Discord webhook failed: {exc}", meta

    try:
        lowered = user_message.strip().lower()
        if lowered in ("mcp status", "mcp", "show mcp"):
            from navine.mcp.client import status as mcp_status
            import json as _json

            return _json.dumps(mcp_status(), indent=2), meta
        if lowered in ("mcp servers", "list mcp servers"):
            from navine.mcp.client import list_servers
            import json as _json

            return _json.dumps(list_servers(), indent=2), meta
        m_tools = re.match(r"^mcp\s+tools\s+(\S+)\s*$", user_message.strip(), re.I)
        if m_tools:
            from navine.mcp.client import list_tools
            import json as _json

            return _json.dumps(list_tools(m_tools.group(1)), indent=2), meta
        m_call = re.match(
            r"^mcp\s+call\s+(\S+)\s+(\S+)(?:\s+(.+))?$",
            user_message.strip(),
            re.I | re.DOTALL,
        )
        if m_call:
            from navine.mcp.client import call_tool, parse_call_args
            import json as _json
            import shlex

            raw_args = m_call.group(3) or ""
            pairs = shlex.split(raw_args) if raw_args.strip() else []
            return _json.dumps(call_tool(m_call.group(1), m_call.group(2), parse_call_args(pairs)), indent=2), meta
    except Exception as exc:
        return f"MCP failed: {exc}", meta

    hf_query = None
    try:
        from navine.learn.hf_search import format_dataset_results, parse_hf_search_message, search_datasets

        hf_query = parse_hf_search_message(user_message)
        if hf_query:
            rows = search_datasets(hf_query, limit=20)
            return format_dataset_results(hf_query, rows), meta
    except Exception as exc:
        return f"Hugging Face dataset search failed: {exc}", meta

    try:
        from navine.files.zip_util import (
            create_zip_archive,
            extract_code_files_from_text,
            parse_zip_request,
            zip_response_payload,
        )

        if parse_zip_request(user_message):
            files = extract_code_files_from_text(user_message)
            if not files and history:
                for turn in reversed(history[-6:]):
                    files.extend(extract_code_files_from_text(str(turn.get("assistant") or "")))
                    if len(files) >= 2:
                        break
            if len(files) >= 1:
                path, filename = create_zip_archive(files, "project.zip")
                payload = zip_response_payload(path, filename)
                return (
                    f"Created zip archive ({len(files)} files, {payload['size']:,} bytes).\n"
                    f"Download: {payload['download_url']}",
                    {**meta, "download_url": payload["download_url"], "filename": filename},
                )
            return (
                "I can zip files from code blocks in this chat. Ask again after I generate code, "
                "or paste files as ``` blocks.",
                meta,
            )
    except Exception as exc:
        return f"Zip creation failed: {exc}", meta

    try:
        from navine.text.gender_equality import maybe_equal_social_reply

        equal_reply = maybe_equal_social_reply(user_message)
        if equal_reply:
            return equal_reply, meta
    except Exception:
        pass

    profile = resolve_chat_profile(user_message, model_profile)
    meta["model_profile"] = profile
    conscience_mode = profile == "conscience"
    if conscience_mode:
        history = conscience_history() + list(history)
        profile = "chat"
        if temperature is None:
            temperature = 0.9
    think_mode = profile in ("think", "thinking", "reason", "reasoning")
    detective_mode = profile in ("detective", "mystery", "investigate", "cipher", "cicada", "arg", "puzzle")
    analyze_mode = profile in (
        "analyze",
        "analyse",
        "analysis",
        "breakdown",
        "inspect",
        "evaluate",
        "assess",
        "audit",
        "review",
    )
    osint_mode = profile in (
        "osint",
        "intel",
        "recon",
        "reconnaissance",
        "whois",
        "footprint",
        "lookup",
    )
    code_profile = profile in ("code", "coding")
    if max_new_tokens is None:
        if code_profile:
            max_new_tokens = 896
        elif think_mode or detective_mode or analyze_mode or osint_mode:
            max_new_tokens = 420
        else:
            max_new_tokens = 320
    if temperature is None:
        if code_profile:
            temperature = 0.18
        elif think_mode or detective_mode or analyze_mode or osint_mode:
            temperature = 0.32
        else:
            temperature = 0.55
    from navine.search.web import (
        detect_search_intent,
        is_casual_chitchat,
        is_explicit_search_request,
        should_search_for_message,
    )

    from navine.games.router import games_help_text, handle_game_message

    bridge_reply = _handle_game_bridge_chat(user_message)
    if bridge_reply:
        return bridge_reply, meta

    game_result = handle_game_message(user_message, session_id=session_id or "default")
    if game_result is not None:
        reply, game_meta = game_result
        meta.update(game_meta)
        return reply, meta

    if re.search(r"\b(?:what games|play games|which games|list games|game list)\b", user_message, re.I):
        return games_help_text(), meta

    code_mode_early = detect_code_intent(user_message) or code_profile
    try:
        from navine.code.fallbacks import wants_multi_tool

        if wants_multi_tool(user_message):
            code_mode_early = True
    except Exception:
        pass
    if code_mode_early:
        try:
            from navine.code.fallbacks import format_code_response, looks_like_valid_code
            from navine.code.generate import generate_code
            from navine.code.prompts import detect_language

            code, code_lang = generate_code(user_message)
            lang = code_lang or detect_language(user_message) or "python"
            if code and looks_like_valid_code(code, lang):
                meta["code_source"] = "model"
                return format_code_response(code, lang), meta
        except Exception:
            pass

    from navine.realtime.math import answer_math_query, is_math_query

    if is_math_query(user_message):
        if think_mode:
            think_math = _think_fallback_reply(user_message)
            if think_math:
                return think_math, meta
        reply = answer_math_query(user_message)
        if reply:
            return reply, meta

    if is_time_date_query(user_message):
        reply = _answer_time_query_if_applicable(user_message)
        if reply:
            return reply, meta

    from navine.search.common_knowledge import answer_common_knowledge, is_common_knowledge_query

    # Only ultra-short fact lookups use canned answers; everything else hits the neural model.
    if len(user_message.strip()) < 96 and is_common_knowledge_query(user_message):
        reply = answer_common_knowledge(user_message)
        if reply and len(reply.strip()) < 400:
            meta["searched"] = False
            meta["search_attempted"] = False
            return reply, meta

    if is_capability_query(user_message):
        reply = _answer_capability_query_if_applicable(user_message, history)
        if reply:
            return reply, meta

    osint_reply = _answer_osint_query_if_applicable(user_message)
    if osint_reply:
        meta["searched"] = True
        meta["search_attempted"] = True
        return osint_reply, meta

    if _unrestricted_enabled() and is_adult_creative_request(user_message):
        adult = _adult_creative_fallback(user_message)
        if adult:
            return adult, meta

    if is_casual_chitchat(user_message) and len(user_message.strip()) < 28:
        short_greeting = len(user_message.strip()) <= 16
        if short_greeting:
            reply = _casual_chitchat_reply(
                user_message, allow_emoji=allow_emoji, session_id=session_id
            )
            if reply:
                return reply, meta
        model_reply = _try_model_chitchat(user_message, history, max_new_tokens, temperature)
        if model_reply:
            return model_reply, meta
        category = _chitchat_category(user_message)
        if category != "casual" or profile in ("chat", "auto"):
            reply = _casual_chitchat_reply(
                user_message, allow_emoji=allow_emoji, session_id=session_id
            )
            if reply:
                return reply, meta

    if detective_mode:
        detective_early = _detective_fallback_reply(user_message)
        if detective_early and _extract_cipher_payload(user_message):
            return detective_early, meta

    from navine.selfcode.intent import gather_selfcode_context, is_selfcode_inspect_query

    selfcode_context = None
    if is_selfcode_inspect_query(user_message):
        selfcode_context = gather_selfcode_context(user_message)
        meta["selfcode"] = bool(selfcode_context)
        if selfcode_context:
            raw = _run_generation(
                user_message,
                history,
                None,
                None,
                False,
                max_new_tokens,
                temperature if temperature is not None else 0.45,
                selfcode_context=selfcode_context,
                selfcode_focus=True,
            )
            cleaned = clean_response(raw, user_message, False)
            if cleaned and len(cleaned.strip()) >= 8 and not looks_like_garbage(cleaned, user_message):
                meta["searched"] = False
                return cleaned, meta
            fallback = _selfcode_extractive_answer(user_message, selfcode_context)
            if fallback:
                meta["searched"] = False
                return fallback, meta

    code_mode = detect_code_intent(user_message)
    code_mode = code_mode or code_profile
    try:
        from navine.code.fallbacks import wants_multi_tool

        if wants_multi_tool(user_message):
            code_mode = True
    except Exception:
        pass

    if code_mode:
        try:
            from navine.code.fallbacks import format_code_response, looks_like_valid_code
            from navine.code.generate import generate_code
            from navine.code.prompts import detect_language

            code, code_lang = generate_code(user_message)
            lang = code_lang or detect_language(user_message) or "python"
            if code and looks_like_valid_code(code, lang):
                meta["code_source"] = "model"
                return format_code_response(code, lang), meta
        except Exception:
            pass

    if use_rag and (is_casual_chitchat(user_message) or len(user_message.strip()) < 24):
        use_rag = False

    search_cfg: Dict[str, Any] = {}
    try:
        from navine.memory.config import get_search_config

        search_cfg = get_search_config()
    except Exception:
        pass
    auto_factual_search = bool(search_cfg.get("auto_search_factual", False))
    search_allowed = bool(
        use_search
        or search_cfg.get("default_enabled")
        or is_explicit_search_request(user_message)
        or auto_factual_search
    )

    from navine.search.people import answer_who_query, is_who_query

    if not code_mode_early and is_who_query(user_message):
        results = []
        if search_allowed:
            from navine.search.web import search_internet

            meta["search_attempted"] = True
            results = search_internet(user_message)
            meta["searched"] = bool(results)
        reply = answer_who_query(user_message, search_results=results or None)
        if reply:
            return reply, meta
    from navine.search.topics import answer_topic_query, is_topic_query

    if not code_mode_early and is_topic_query(user_message):
        results = []
        if search_allowed:
            from navine.search.web import search_internet

            meta["search_attempted"] = True
            results = search_internet(user_message)
            meta["searched"] = bool(results)
        reply = answer_topic_query(user_message, search_results=results or None)
        if reply:
            return reply, meta

    explicit_search = is_explicit_search_request(user_message) or use_search
    factual_query = detect_search_intent(user_message) == "factual"

    conversation_rag = None
    training_rag = None
    if use_rag:
        try:
            from navine.memory.conversations import retrieve_conversation_context

            conversation_rag = retrieve_conversation_context(user_message)
            if conversation_rag and looks_like_rag_junk(conversation_rag):
                conversation_rag = None
        except Exception:
            conversation_rag = None
        try:
            from navine.learn.rag import retrieve_context

            training_rag = retrieve_context(
                user_message,
                exclude_source="conversation",
            )
            if training_rag and looks_like_rag_junk(training_rag):
                training_rag = None
        except Exception:
            training_rag = None

    rag_context = _combine_rag_contexts(conversation_rag, training_rag)
    search_context = None
    search_results: List[dict] = []
    code_mode = detect_code_intent(user_message)
    code_mode = code_mode or code_profile

    proactive_search = (
        factual_query
        and not code_mode
        and (
            explicit_search
            or auto_factual_search
            or bool(search_cfg.get("default_enabled"))
        )
    )
    if proactive_search:
        meta["search_attempted"] = True
        search_context, found, search_results = _fetch_search_context(user_message)
        meta["searched"] = found
        if found and search_cfg.get("learn_from_search", True):
            try:
                from navine.learn.from_search import learn_from_search_results

                learn_from_search_results(user_message, search_results)
            except Exception:
                pass
        if search_context:
            raw = _run_generation(
                user_message,
                history,
                rag_context,
                search_context,
                code_mode,
                max_new_tokens,
                temperature if temperature is not None else 0.45,
                search_focus=True,
            )
            cleaned = clean_response(raw, user_message, code_mode)
            if factual_query and (
                looks_like_rag_junk(cleaned)
                or looks_like_wiki_dump(cleaned)
                or looks_like_encyclopedia_dump(cleaned)
            ):
                cleaned = ""
            if not looks_like_garbage(cleaned, user_message):
                return cleaned, meta
            simplified = _simplify_factual_answer(
                user_message,
                cleaned or search_context,
            )
            if simplified and not looks_like_garbage(simplified, user_message):
                return simplified, meta
            extracted = _extractive_search_answer(user_message, search_results)
            if extracted:
                simplified = _simplify_factual_answer(user_message, extracted)
                if simplified and not looks_like_garbage(simplified, user_message):
                    return simplified, meta
                if not looks_like_encyclopedia_dump(extracted) and not looks_like_garbage(
                    extracted, user_message
                ):
                    return extracted, meta

    raw = _run_generation(
        user_message,
        history,
        rag_context,
        None,
        code_mode,
        max_new_tokens,
        temperature,
        search_focus=False,
        think_mode=think_mode,
        detective_mode=detective_mode,
        analyze_mode=analyze_mode,
        osint_mode=osint_mode,
    )
    cleaned = clean_response(raw, user_message, code_mode)
    if factual_query and (
        looks_like_rag_junk(cleaned)
        or looks_like_wiki_dump(cleaned)
        or looks_like_encyclopedia_dump(cleaned)
    ):
        cleaned = ""
    final = _finalize_chat_reply(user_message, cleaned)
    if final and not _weak_special_mode_reply(
        final,
        think_mode=think_mode,
        detective_mode=detective_mode,
        analyze_mode=analyze_mode,
    ):
        if code_mode:
            from navine.code.fallbacks import looks_like_valid_code
            from navine.code.format import extract_code_block
            from navine.code.prompts import detect_language

            lang = detect_language(user_message)
            code = extract_code_block(final) or final
            if looks_like_valid_code(code, lang):
                return final, meta
        else:
            return final, meta
    if code_mode:
        try:
            from navine.code.generate import generate_code
            from navine.code.fallbacks import format_code_response

            code, code_lang = generate_code(user_message)
            if code:
                meta["code_source"] = "model"
                return format_code_response(code, code_lang), meta
        except Exception:
            pass

    model_failed = True
    if explicit_search or should_search_for_message(
        user_message,
        rag_context=rag_context,
        model_failed=model_failed,
        explicit_only=True,
    ):
        meta["search_attempted"] = True
        search_context, found, search_results = _fetch_search_context(user_message)
        meta["searched"] = found
        if search_context:
            raw = _run_generation(
                user_message,
                history,
                rag_context,
                search_context,
                code_mode,
                max_new_tokens,
                temperature if temperature is not None else 0.5,
                search_focus=bool(factual_query),
            )
            cleaned = clean_response(raw, user_message, code_mode)
            if factual_query and (
                looks_like_rag_junk(cleaned)
                or looks_like_wiki_dump(cleaned)
                or looks_like_encyclopedia_dump(cleaned)
            ):
                cleaned = ""
            final = _finalize_chat_reply(user_message, cleaned)
            if final and not _weak_special_mode_reply(
                final,
                think_mode=think_mode,
                detective_mode=detective_mode,
                analyze_mode=analyze_mode,
            ):
                return final, meta
            local_reply = _prefer_local_handler_over_dump(user_message, cleaned or search_context or "", meta, history)
            if local_reply:
                return local_reply, meta
            if factual_query:
                simplified = _simplify_factual_answer(
                    user_message,
                    cleaned or (search_context or ""),
                )
                if simplified and not looks_like_garbage(simplified, user_message):
                    return simplified, meta

    if search_context:
        raw = _run_generation(
            user_message,
            history,
            rag_context,
            search_context,
            code_mode,
            max_new_tokens,
            0.45,
            search_focus=True,
        )
        cleaned = clean_response(raw, user_message, code_mode)
        if factual_query and (looks_like_rag_junk(cleaned) or looks_like_wiki_dump(cleaned)):
            cleaned = ""
        if not looks_like_garbage(cleaned, user_message) and not _weak_special_mode_reply(
            cleaned,
            think_mode=think_mode,
            detective_mode=detective_mode,
            analyze_mode=analyze_mode,
        ):
            return cleaned, meta
        local_reply = _prefer_local_handler_over_dump(user_message, cleaned or search_context or "", meta, history)
        if local_reply:
            return local_reply, meta
        extracted = _extractive_search_answer(user_message, search_results)
        if extracted:
            simplified = _simplify_factual_answer(user_message, extracted)
            if simplified and not looks_like_garbage(simplified, user_message):
                return simplified, meta
            if not looks_like_encyclopedia_dump(extracted) and not looks_like_garbage(extracted, user_message):
                return extracted, meta

    if factual_query and search_results:
        extracted = _extractive_search_answer(user_message, search_results)
        if extracted:
            simplified = _simplify_factual_answer(user_message, extracted)
            if simplified and not looks_like_garbage(simplified, user_message):
                return simplified, meta
            if not looks_like_encyclopedia_dump(extracted) and not looks_like_garbage(extracted, user_message):
                return extracted, meta

    if not factual_query:
        retry_temp = 0.8 if not code_mode else 0.35
        raw = _run_generation(
            user_message,
            history,
            rag_context,
            None,
            code_mode,
            max_new_tokens,
            retry_temp,
            think_mode=think_mode,
            detective_mode=detective_mode,
            analyze_mode=analyze_mode,
            osint_mode=osint_mode,
        )
        cleaned = clean_response(raw, user_message, code_mode)
        final = _finalize_chat_reply(user_message, cleaned)
        if final and len(final.strip()) >= 4 and not _weak_special_mode_reply(
            final,
            think_mode=think_mode,
            detective_mode=detective_mode,
            analyze_mode=analyze_mode,
        ):
            return final, meta

    if search_context:
        extracted = _extractive_search_answer(user_message, search_results)
        if extracted:
            simplified = _simplify_factual_answer(user_message, extracted)
            if simplified and not looks_like_garbage(simplified, user_message):
                return simplified, meta
            if not looks_like_encyclopedia_dump(extracted) and not looks_like_garbage(extracted, user_message):
                return extracted, meta

    reply = _local_handler_fallback(user_message, history)
    if reply:
        meta["searched"] = False
        return reply, meta

    if code_mode:
        try:
            from navine.code.generate import generate_code
            from navine.code.fallbacks import format_code_response

            code, code_lang = generate_code(user_message)
            if code:
                meta["code_source"] = "model"
                return format_code_response(code, code_lang or "python"), meta
        except Exception:
            pass

    if meta.get("searched") and search_results:
        who_reply = _answer_who_query_if_applicable(user_message, search_results)
        if who_reply:
            return who_reply, meta

    if factual_query and meta.get("searched"):
        return (
            "I could not confirm that from web results just now. "
            "Ask again with more detail, or say search the web if you want another look online.",
            meta,
        )

    raw = _run_generation(
        user_message,
        history,
        rag_context,
        None,
        code_mode,
        max_new_tokens if max_new_tokens is not None else 160,
        temperature if temperature is not None else 0.7,
        think_mode=think_mode,
        detective_mode=detective_mode,
        analyze_mode=analyze_mode,
        osint_mode=osint_mode,
    )
    cleaned = clean_response(raw, user_message, code_mode)
    final = _finalize_chat_reply(user_message, cleaned)
    if final and len(final.strip()) >= 4 and not _weak_special_mode_reply(
        final,
        think_mode=think_mode,
        detective_mode=detective_mode,
        analyze_mode=analyze_mode,
    ):
        return final, meta

    if code_mode:
        try:
            from navine.code.generate import generate_code
            from navine.code.fallbacks import format_code_response

            code, code_lang = generate_code(user_message)
            if code:
                meta["code_source"] = "model"
                return format_code_response(code, code_lang or "python"), meta
        except Exception:
            pass
        return (
            "I could not produce clean Python stdlib CLI code for that yet. "
            "Rephrase the task (inputs/outputs) and try again.",
            meta,
        )

    adult = _adult_creative_fallback(user_message)
    if adult:
        return adult, meta

    mode_reply = _special_mode_fallback(
        user_message,
        think_mode=think_mode,
        detective_mode=detective_mode,
        analyze_mode=analyze_mode,
        osint_mode=osint_mode,
    )
    if mode_reply:
        return mode_reply, meta

    if profile in ("chat", "auto") and len(user_message.strip()) < 80:
        chat_reply = _casual_chitchat_reply(
            user_message, allow_emoji=allow_emoji, session_id=session_id
        )
        if chat_reply:
            return chat_reply, meta
        return (
            "Hey  -  I’m here. Tell me what you want to chat about, build, analyze, or look up.",
            meta,
        )

    return (
        "Here is my best direct take: tell me what you want in more detail and I will answer fully. "
        "You can also teach me the preferred answer in chat so I remember it.",
        meta,
    )
