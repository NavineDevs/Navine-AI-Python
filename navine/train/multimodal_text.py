from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_jsonl, resolve_data_path
from navine.utils.config import load_train_config
from navine.utils.paths import get_project_root


def format_multimodal_entry(entry: dict) -> str:
    caption = entry.get("caption", entry.get("text", ""))
    media = entry.get("media", entry.get("source", ""))
    media_type = entry.get("type", "image")
    return f"### Media: {media}\n### Type: {media_type}\n### Caption: {caption}"


def load_multimodal_texts() -> List[str]:
    config = load_train_config("multimodal-text")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path:
        for entry in read_jsonl(train_path):
            formatted = format_multimodal_entry(entry)
            if formatted.strip():
                texts.append(formatted)
    root = get_project_root()
    for name in (
        "captions.jsonl",
        "captions_extra.jsonl",
        "game_frames.jsonl",
        "autolearn_captions.jsonl",
        "video_captions.jsonl",
        "alignment_extra.jsonl",
        "gpu_upgrade_captions.jsonl",
    ):
        path = root / "data" / "train" / "multimodal-text" / name
        if not path.exists():
            continue
        if train_path and path.resolve() == train_path.resolve():
            continue
        for entry in read_jsonl(path):
            formatted = format_multimodal_entry(entry)
            if formatted.strip():
                texts.append(formatted)
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_multimodal_texts()
    if not texts:
        raise ValueError("No multimodal-text training data found. Add data/train/multimodal-text/captions.jsonl")
    finetune_texts("multimodal-text", texts, desc="Navine AI - Python Multimodal Text", max_steps=steps)
