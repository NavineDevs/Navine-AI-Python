from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from navine.utils.paths import get_project_root

LANG_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}

C4_LANG_CODES = ("en", "es", "fr", "de", "pt", "it", "ja", "ru", "ar", "hi", "zh")

_SCRIPT_RANGES: Tuple[Tuple[str, str, str], ...] = (
    ("zh", "\u4e00", "\u9fff"),
    ("ja", "\u3040", "\u30ff"),
    ("ko", "\uac00", "\ud7af"),
    ("ar", "\u0600", "\u06ff"),
    ("ru", "\u0400", "\u04ff"),
    ("hi", "\u0900", "\u097f"),
    ("th", "\u0e00", "\u0e7f"),
)

_LATIN_HINTS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("es", re.compile(r"\b(hola|gracias|qu[eé]|c[oó]mo|por qu[eé]|buenos)\b", re.I)),
    ("fr", re.compile(r"\b(bonjour|merci|comment|pourquoi|salut|oui|non)\b", re.I)),
    ("de", re.compile(r"\b(hallo|danke|warum|wie|bitte|guten)\b", re.I)),
    ("pt", re.compile(r"\b(ol[aá]|obrigad|como|por que|bom dia)\b", re.I)),
    ("it", re.compile(r"\b(ciao|grazie|come|perch[eé]|buongiorno)\b", re.I)),
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def detect_text_language(text: str, default: str = "en") -> str:
    sample = normalize_unicode(text).strip()
    if not sample:
        return default
    counts: Counter = Counter()
    for lang, pattern in _LATIN_HINTS:
        if pattern.search(sample):
            counts[lang] += 5
    for ch in sample:
        code = ord(ch)
        for lang, start, end in _SCRIPT_RANGES:
            if ord(start) <= code <= ord(end):
                counts[lang] += 3
                break
    if not counts:
        if sample.isascii():
            return "en"
        return default
    return counts.most_common(1)[0][0]


def language_display(code: str) -> str:
    return LANG_NAMES.get(str(code).lower(), str(code))


def reply_language_instruction(user_message: str) -> str:
    code = detect_text_language(user_message)
    name = language_display(code)
    if code == "en" and user_message.strip().isascii():
        return (
            "Respond in the same language the user writes in. "
            "If they use English, reply in English. "
            "You understand and can reply in many languages including "
            "Spanish, French, German, Portuguese, Italian, Japanese, Chinese, Korean, "
            "Russian, Arabic, and Hindi."
        )
    return f"Respond in {name}. Match the user's language unless they ask for another language."


def seed_multilingual_sft_blocks() -> List[str]:
    return [
        "### User: Hello\n### Assistant: Hello! I'm Navine AI - Python. How can I help you today?",
        "### User: Hola\n### Assistant: Hola. Soy Navine AI - Python. ¿En qué puedo ayudarte?",
        "### User: Bonjour\n### Assistant: Bonjour. Je suis Navine AI - Python. Comment puis-je vous aider?",
        "### User: Hallo\n### Assistant: Hallo. Ich bin Navine AI - Python. Wie kann ich Ihnen helfen?",
        "### User: Olá\n### Assistant: Olá. Eu sou a Navine AI - Python. Como posso ajudar?",
        "### User: Ciao\n### Assistant: Ciao. Sono Navine AI - Python. Come posso aiutarti?",
        "### User: 你好\n### Assistant: 你好！我是 Navine AI - Python。有什么可以帮你的？",
        "### User: こんにちは\n### Assistant: こんにちは。Navine AI - Python です。何をお手伝いしましょうか？",
        "### User: 안녕하세요\n### Assistant: 안녕하세요. Navine AI - Python입니다. 무엇을 도와드릴까요?",
        "### User: Привет\n### Assistant: Привет. Я Navine AI - Python. Чем могу помочь?",
        "### User: مرحبا\n### Assistant: مرحبا. أنا Navine AI - Python. كيف يمكنني مساعدتك؟",
        "### User: नमस्ते\n### Assistant: नमस्ते। मैं Navine AI - Python हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
        "### User: What is Python?\n### Assistant: Python is a popular programming language for apps, scripts, and AI.",
        "### User: ¿Qué es Python?\n### Assistant: Python es un lenguaje de programación popular para apps, scripts e IA.",
        "### User: Qu'est-ce que Python?\n### Assistant: Python est un langage de programmation populaire pour les apps, scripts et l'IA.",
        "### User: Python是什么？\n### Assistant: Python 是一种常用的编程语言，适合应用、脚本和人工智能。",
        "### User: 日本語で答えてください。今日の調子はどうですか？\n### Assistant: 調子は良好です。Navine AI - Python としてお手伝いできます。",
        "### User: Réponds en français. Quelle heure est-il?\n### Assistant: Je n'ai pas l'heure exacte sans contexte en direct, mais je peux aider avec d'autres questions.",
        "### User: Antwort auf Deutsch: Was kannst du?\n### Assistant: Ich kann Texte schreiben, Fragen beantworten, Code erklären und lokal auf deinem Rechner helfen.",
    ]


def write_seed_multilingual_file() -> Path:
    out = get_project_root() / "data" / "train" / "multilingual" / "chat.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(seed_multilingual_sft_blocks()) + "\n", encoding="utf-8")
    return out


def load_multilingual_sft_blocks() -> List[str]:
    root = get_project_root() / "data" / "train" / "multilingual"
    blocks: List[str] = []
    if not root.exists():
        return blocks
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".md", ".jsonl"):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path.suffix.lower() == ".jsonl":
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = str(row.get("instruction") or row.get("question") or "").strip()
                response = str(row.get("response") or row.get("answer") or "").strip()
                if instruction and response:
                    blocks.append(f"### User: {instruction}\n### Assistant: {response}")
            continue
        for block in raw.split("\n\n"):
            piece = block.strip()
            if "### User:" in piece and "### Assistant:" in piece:
                blocks.append(piece)
            elif len(piece) >= 20 and not piece.startswith("###"):
                blocks.append(piece)
    return blocks


def fetch_c4_samples(lang: str, max_rows: int = 150) -> List[str]:
    rows: List[str] = []
    try:
        from datasets import load_dataset

        stream = load_dataset("allenai/c4", lang, split="train", streaming=True, trust_remote_code=True)
        for i, row in enumerate(stream):
            if i >= max_rows:
                break
            text = str(row.get("text") or "").strip()
            if len(text) >= 40:
                rows.append(normalize_unicode(text))
    except Exception:
        pass
    return rows


def fetch_multilingual_corpus(
    languages: Optional[Iterable[str]] = None,
    rows_per_lang: int = 150,
) -> Path:
    out = get_project_root() / "data" / "train" / "multilingual" / "corpus.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    langs = [str(x).lower() for x in (languages or C4_LANG_CODES)]
    lines: List[str] = []
    for lang in langs:
        batch = fetch_c4_samples(lang, max_rows=max(20, int(rows_per_lang)))
        lines.extend(batch)
        print(f"Multilingual corpus: {lang} -> {len(batch)} lines")
    if not lines:
        lines = [
            "Navine AI - Python supports many languages including English, Spanish, French, German, Japanese, and Chinese.",
            "La inteligencia artificial local puede responder en varios idiomas.",
            "L'intelligence artificielle locale peut répondre dans plusieurs langues.",
            "Lokale KI kann in mehreren Sprachen antworten.",
            "ローカルAIは複数の言語で回答できます。",
            "本地 AI 可以用多种语言回答。",
        ]
    write_seed_multilingual_file()
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote multilingual corpus ({len(lines)} lines) -> {out}")
    return out


def load_multilingual_pretrain_lines() -> List[str]:
    root = get_project_root() / "data" / "train" / "multilingual"
    lines: List[str] = []
    corpus = root / "corpus.txt"
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8", errors="ignore").splitlines():
            piece = line.strip()
            if len(piece) >= 16:
                lines.append(piece)
    for block in load_multilingual_sft_blocks():
        if "### User:" not in block:
            lines.append(block)
    return lines
