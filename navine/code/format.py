import json
import re
from pathlib import Path
from typing import List, Optional


def extract_code_block(text: str, language: Optional[str] = None) -> str:
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    if "<code>" in text:
        start = text.find("<code>")
        end = text.find("</code>")
        if end > start:
            return text[start + 6 : end].strip()
    return text.strip()


def format_code_output(raw: str, language: Optional[str] = None) -> str:
    code = extract_code_block(raw, language)
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        if line.strip().startswith("language="):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def load_code_dataset(path: Path) -> List[dict]:
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries
