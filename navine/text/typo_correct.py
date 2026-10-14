from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")
_PROTECTED_RE = re.compile(
    r"("
    r"```[\s\S]*?```"
    r"|`[^`]+`"
    r"|https?://\S+"
    r"|www\.\S+"
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    r"|@[A-Za-z0-9_]{2,}"
    r"|\$\{[^}]+\}"
    r")"
)

_COMMON_FIXES: Dict[str, str] = {
    "teh": "the",
    "thier": "their",
    "recieve": "receive",
    "recieved": "received",
    "seperate": "separate",
    "occured": "occurred",
    "definately": "definitely",
    "definetly": "definitely",
    "wierd": "weird",
    "becuase": "because",
    "becasue": "because",
    "enviroment": "environment",
    "enviorment": "environment",
    "lenght": "length",
    "heigth": "height",
    "widht": "width",
    "fucntion": "function",
    "funtion": "function",
    "functoin": "function",
    "retrun": "return",
    "reutrn": "return",
    "improt": "import",
    "impoty": "import",
    "pritn": "print",
    "pirnt": "print",
    "calss": "class",
    "clas": "class",
    "strenght": "strength",
    "paramters": "parameters",
    "paramater": "parameter",
    "arguement": "argument",
    "arguemnt": "argument",
    "langauge": "language",
    "lanugage": "language",
    "pytohn": "python",
    "pyhton": "python",
    "javascirpt": "javascript",
    "typescipt": "typescript",
    "typscript": "typescript",
    "respons": "response",
    "requst": "request",
    "requset": "request",
    "generat": "generate",
    "generater": "generator",
    "iamge": "image",
    "imgae": "image",
    "vidoe": "video",
    "vedio": "video",
    "audoi": "audio",
    "passowrd": "password",
    "pasword": "password",
    "usename": "username",
    "userame": "username",
    "emial": "email",
    "adress": "address",
    "addres": "address",
    "tommorow": "tomorrow",
    "tommorrow": "tomorrow",
    "yseterday": "yesterday",
    "yesterady": "yesterday",
    "favouite": "favourite",
    "favorit": "favorite",
    "succesful": "successful",
    "sucessful": "successful",
    "acommodate": "accommodate",
    "acheive": "achieve",
    "beleive": "believe",
    "calender": "calendar",
    "cemetary": "cemetery",
    "commited": "committed",
    "concious": "conscious",
    "dilemna": "dilemma",
    "existance": "existence",
    "experiance": "experience",
    "goverment": "government",
    "happend": "happened",
    "independant": "independent",
    "knowlege": "knowledge",
    "liason": "liaison",
    "maintainance": "maintenance",
    "necesary": "necessary",
    "occassion": "occasion",
    "persistant": "persistent",
    "priviledge": "privilege",
    "probaly": "probably",
    "probly": "probably",
    "publically": "publicly",
    "recomend": "recommend",
    "refered": "referred",
    "relevent": "relevant",
    "remeber": "remember",
    "responce": "response",
    "rythm": "rhythm",
    "sentance": "sentence",
    "sieze": "seize",
    "similiar": "similar",
    "speach": "speech",
    "succes": "success",
    "suprise": "surprise",
    "tatoo": "tattoo",
    "threshhold": "threshold",
    "tomatos": "tomatoes",
    "truely": "truly",
    "untill": "until",
    "vaccuum": "vacuum",
    "wether": "whether",
    "wich": "which",
    "writting": "writing",
    "writen": "written",
    "tookl": "tool",
    "toool": "tool",
    "mutli": "multi",
    "mulit": "multi",
    "osnit": "osint",
    "oinsit": "osint",
    "navien": "navine",
    "naveine": "navine",
    "plese": "please",
    "plase": "please",
    "pls": "please",
    "plz": "please",
    "waht": "what",
    "wahts": "what's",
    "whats": "what's",
    "hwo": "how",
    "hwat": "what",
    "adn": "and",
    "nad": "and",
    "taht": "that",
    "thta": "that",
    "jsut": "just",
    "jstu": "just",
    "abot": "about",
    "abotu": "about",
    "becouse": "because",
    "exapmle": "example",
    "exmaple": "example",
    "explian": "explain",
    "explane": "explain",
    "sumarize": "summarize",
    "summrize": "summarize",
    "traslate": "translate",
    "tranlate": "translate",
}


@lru_cache(maxsize=1)
def _vocab() -> Tuple[str, ...]:
    words = set(_COMMON_FIXES.values())
    words.update(
        {
            "python",
            "javascript",
            "typescript",
            "function",
            "class",
            "return",
            "import",
            "print",
            "image",
            "video",
            "audio",
            "voice",
            "password",
            "username",
            "email",
            "website",
            "generate",
            "generator",
            "parameter",
            "parameters",
            "argument",
            "language",
            "response",
            "request",
            "navine",
            "osint",
            "deepfake",
            "checkpoint",
            "training",
            "model",
            "chat",
            "code",
            "write",
            "create",
            "build",
            "explain",
            "summarize",
            "translate",
            "please",
            "thanks",
            "hello",
            "world",
            "server",
            "client",
            "database",
            "memory",
            "gpu",
            "cuda",
            "windows",
            "linux",
            "mac",
            "file",
            "folder",
            "download",
            "upload",
            "search",
            "report",
            "breach",
            "domain",
            "address",
            "phone",
            "person",
            "organization",
            "sky",
            "cloud",
            "landscape",
            "portrait",
            "anime",
            "photo",
            "picture",
            "message",
            "question",
            "answer",
            "help",
            "fix",
            "error",
            "debug",
            "script",
            "program",
            "algorithm",
            "array",
            "object",
            "string",
            "number",
            "boolean",
            "true",
            "false",
            "null",
            "async",
            "await",
            "promise",
            "json",
            "yaml",
            "html",
            "css",
            "api",
            "http",
            "https",
            "url",
            "path",
            "route",
            "token",
            "session",
            "login",
            "signup",
            "account",
            "admin",
            "user",
            "password",
            "security",
            "public",
            "private",
            "local",
            "remote",
            "online",
            "offline",
        }
    )
    try:
        from navine.utils.paths import get_project_root

        vocab_file = get_project_root() / "data" / "learn" / "vocab_typo.txt"
        if vocab_file.is_file():
            for line in vocab_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                token = line.strip().lower()
                if token.isalpha() and 3 <= len(token) <= 24:
                    words.add(token)
    except Exception:
        pass
    return tuple(sorted(words))


def _best_match(word: str, vocabulary: Iterable[str]) -> Optional[str]:
    low = word.lower()
    if low in _COMMON_FIXES:
        return _COMMON_FIXES[low]
    if low in vocabulary:
        return None
    if len(low) < 4:
        return None
    cutoff = 0.78 if len(low) <= 5 else 0.72
    matches = difflib.get_close_matches(low, list(vocabulary), n=1, cutoff=cutoff)
    if not matches:
        return None
    cand = matches[0]
    if cand == low:
        return None
    if abs(len(cand) - len(low)) > max(2, len(low) // 3):
        return None
    return cand


def _restore_case(original: str, fixed: str) -> str:
    if original.isupper():
        return fixed.upper()
    if original[:1].isupper() and original[1:].islower():
        return fixed[:1].upper() + fixed[1:]
    if original.istitle():
        return fixed.title()
    return fixed


def correct_typos(text: str, *, aggressive: bool = False) -> Tuple[str, List[Tuple[str, str]]]:
    raw = str(text or "")
    if not raw.strip():
        return raw, []
    vocabulary = _vocab()
    changes: List[Tuple[str, str]] = []
    parts: List[str] = []
    cursor = 0
    for match in _PROTECTED_RE.finditer(raw):
        parts.append(_correct_chunk(raw[cursor : match.start()], vocabulary, changes, aggressive=aggressive))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_correct_chunk(raw[cursor:], vocabulary, changes, aggressive=aggressive))
    return "".join(parts), changes


def _correct_chunk(
    chunk: str,
    vocabulary: Tuple[str, ...],
    changes: List[Tuple[str, str]],
    *,
    aggressive: bool,
) -> str:
    if not chunk:
        return chunk

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        if any(ch.isupper() for ch in word[1:]) and sum(1 for ch in word if ch.isupper()) >= 2:
            return word
        fixed = _best_match(word, vocabulary)
        if not fixed:
            if aggressive and len(word) >= 5:
                loose = difflib.get_close_matches(word.lower(), list(vocabulary), n=1, cutoff=0.66)
                fixed = loose[0] if loose and loose[0] != word.lower() else None
            if not fixed:
                return word
        restored = _restore_case(word, fixed)
        if restored != word:
            changes.append((word, restored))
        return restored

    return _WORD_RE.sub(repl, chunk)


def normalize_user_text(text: str) -> str:
    corrected, _changes = correct_typos(text, aggressive=False)
    return corrected
