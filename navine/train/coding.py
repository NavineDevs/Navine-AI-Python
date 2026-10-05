from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_jsonl, read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config
from navine.utils.paths import get_project_root


def format_coding_entry(entry: dict) -> str:
    language = entry.get("language", "python")
    prompt = entry.get("prompt", "")
    code = entry.get("code", "")
    return f"### Task: {prompt}\n### Language: {language}\n### Code:\n{code}"


def load_coding_texts(
    language: Optional[str] = None,
    max_texts: int = 12000,
) -> List[str]:
    config = load_train_config("coding")
    root = get_project_root()
    texts: List[str] = []
    data_dir = root / config["data"].get("train_dir", "data/train/coding")
    priority = [
        data_dir / "real_code_sft.jsonl",
        data_dir / "real_code_sft.txt",
        data_dir / "verified.jsonl",
        data_dir / "python.jsonl",
    ]
    if language:
        paths = [data_dir / f"{language}.jsonl"]
    else:
        paths = list(priority) + sorted(
            p for p in data_dir.glob("*.jsonl") if p not in priority
        )
        paths += sorted(p for p in data_dir.glob("*.txt") if p not in priority)
    train_file = resolve_data_path(config)
    if train_file and train_file.exists() and train_file not in paths:
        paths.insert(0, train_file)
    seen_paths = set()
    seen_text = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if path.suffix.lower() == ".txt":
            for block in read_txt_blocks(path):
                cleaned = block.strip()
                if not cleaned or cleaned in seen_text:
                    continue
                seen_text.add(cleaned)
                texts.append(cleaned)
                if len(texts) >= max_texts:
                    return texts
            continue
        for entry in read_jsonl(path):
            if language and entry.get("language") and entry.get("language") != language:
                continue
            formatted = format_coding_entry(entry).strip()
            if not formatted or formatted in seen_text:
                continue
            seen_text.add(formatted)
            texts.append(formatted)
            if len(texts) >= max_texts:
                return texts
    return texts


def train(language: Optional[str] = None, steps: Optional[int] = None) -> None:
    texts = load_coding_texts(language=language)
    if not texts:
        lang_msg = f" for language '{language}'" if language else ""
        raise ValueError(f"No coding training data found{lang_msg}. Run: python -m navine.cli learn coding")
    label = f"Navine AI - Python Coding ({language or 'all'})"
    finetune_texts("coding", texts, desc=label, max_steps=steps)
