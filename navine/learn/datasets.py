import json
from pathlib import Path
from typing import List

from navine.policy import http_get
from navine.utils.paths import get_project_root

DATASET_SOURCES = [
    {
        "name": "tiny_stories_sample",
        "url": "https://raw.githubusercontent.com/karpathy/llm.c/master/dev/data/tinyshakespeare/tiny_shakespeare.txt",
        "format": "text",
        "max_chars": 50000,
    },
    {
        "name": "python_snippets",
        "url": "https://raw.githubusercontent.com/python/cpython/main/Lib/textwrap.py",
        "format": "code",
        "language": "python",
        "prompt": "Show a Python standard library module example",
    },
    {
        "name": "quotes_facts",
        "url": "https://raw.githubusercontent.com/JamesFT/Database-Quotes-JSON/master/quotes.json",
        "format": "quotes",
    },
    {
        "name": "science_facts",
        "url": "https://raw.githubusercontent.com/owid/energy-data/master/README.md",
        "format": "text",
        "max_chars": 8000,
    },
]


def download_datasets() -> List[str]:
    root = get_project_root()
    out_dir = root / "data" / "learn" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for spec in DATASET_SOURCES:
        try:
            resp = http_get(spec["url"], timeout=60)
            resp.raise_for_status()
            content = resp.text
            if spec.get("max_chars"):
                content = content[: spec["max_chars"]]
            if spec["format"] == "text":
                path = out_dir / f"{spec['name']}.txt"
                path.write_text(content, encoding="utf-8")
            elif spec["format"] == "quotes":
                path = out_dir / f"{spec['name']}.jsonl"
                try:
                    quotes_data = json.loads(content)
                except json.JSONDecodeError:
                    continue
                entries = quotes_data if isinstance(quotes_data, list) else quotes_data.get("quotes", [])
                with open(path, "w", encoding="utf-8") as f:
                    for entry in entries[:200]:
                        if isinstance(entry, dict):
                            quote_text = entry.get("quote") or entry.get("text") or ""
                            author = entry.get("author") or entry.get("source") or "Unknown"
                        else:
                            quote_text = str(entry)
                            author = "Unknown"
                        if not quote_text:
                            continue
                        line = json.dumps({"quote": quote_text, "author": author}, ensure_ascii=False)
                        f.write(line + "\n")
            else:
                path = out_dir / f"{spec['name']}.jsonl"
                entry = {
                    "language": spec.get("language", "python"),
                    "prompt": spec.get("prompt", "Example code"),
                    "code": content[:4000],
                }
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            saved.append(str(path))
        except Exception:
            continue
    return saved


def ingest_learned_data() -> str:
    root = get_project_root()
    learn_dir = root / "data" / "learn"
    out_path = root / "data" / "text" / "learned_corpus.txt"
    lines: List[str] = []
    for sub in ("pages", "crawl", "datasets", "conversations"):
        subdir = learn_dir / sub
        if not subdir.exists():
            continue
        for path in subdir.rglob("*"):
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                text = data.get("text", "")
                if text:
                    lines.append(text[:4000])
            elif path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    if not text and entry.get("user") and entry.get("assistant"):
                        text = f"Q: {entry['user']}\nA: {entry['assistant']}"
                    if text:
                        lines.append(text[:4000])
            elif path.suffix == ".txt":
                lines.append(path.read_text(encoding="utf-8")[:4000])
    conv_train = root / "data" / "train" / "chat" / "conversation_learned.txt"
    if conv_train.exists():
        lines.append(conv_train.read_text(encoding="utf-8")[:8000])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(lines), encoding="utf-8")
    return str(out_path)
