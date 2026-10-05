from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_jsonl, read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config


def format_creative_entry(entry: dict) -> str:
    title = entry.get("title", "")
    text = entry.get("text", entry.get("content", ""))
    if title:
        return f"### Title: {title}\n### Story:\n{text}"
    return text


def load_creative_texts() -> List[str]:
    config = load_train_config("creative")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path:
        if train_path.suffix == ".jsonl":
            for entry in read_jsonl(train_path):
                formatted = format_creative_entry(entry)
                if formatted.strip():
                    texts.append(formatted)
        else:
            texts.extend(read_txt_blocks(train_path))
    extra = config["data"].get("extra_file")
    if extra:
        from navine.utils.paths import get_project_root
        extra_path = get_project_root() / extra
        if extra_path.exists():
            texts.extend(read_txt_blocks(extra_path))
    from navine.utils.paths import get_project_root
    autolearn = get_project_root() / "data" / "train" / "creative" / "autolearn_stories.txt"
    if autolearn.exists():
        texts.extend(read_txt_blocks(autolearn))
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_creative_texts()
    if not texts:
        raise ValueError("No creative training data found. Add data/train/creative/stories.txt")
    finetune_texts("creative", texts, desc="Navine AI - Python Creative", max_steps=steps)
