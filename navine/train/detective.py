from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_jsonl, read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config


def format_cipher_entry(entry: dict) -> str:
    task = entry.get("task", entry.get("prompt", ""))
    language = entry.get("language", "python")
    code = entry.get("code", entry.get("solution", ""))
    explanation = entry.get("explanation", entry.get("notes", ""))
    block = f"### Task: {task}\n### Language: {language}\n### Code:\n{code}"
    if explanation:
        block += f"\n### Notes:\n{explanation}"
    return block


def load_detective_texts() -> List[str]:
    config = load_train_config("detective")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path:
        texts.extend(read_txt_blocks(train_path))
    extra = config["data"].get("extra_file")
    if extra:
        from navine.utils.paths import get_project_root

        extra_path = get_project_root() / extra
        if extra_path.exists() and extra_path.suffix == ".jsonl":
            for entry in read_jsonl(extra_path):
                formatted = format_cipher_entry(entry)
                if formatted.strip():
                    texts.append(formatted)
    return texts


def load_detective_cipher_texts() -> List[str]:
    texts: List[str] = []
    from navine.utils.paths import get_project_root

    path = get_project_root() / "data" / "train" / "detective" / "ciphers.jsonl"
    if path.exists():
        for entry in read_jsonl(path):
            formatted = format_cipher_entry(entry)
            if formatted.strip():
                texts.append(formatted)
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_detective_texts()
    if not texts:
        raise ValueError("No detective training data found. Add data/train/detective/mysteries.txt")
    finetune_texts("detective", texts, desc="Navine AI - Python Detective", max_steps=steps)
