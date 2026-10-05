from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config


def load_thinking_texts() -> List[str]:
    config = load_train_config("thinking")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path:
        texts.extend(read_txt_blocks(train_path))
    from navine.utils.paths import get_project_root

    for rel in (
        "data/train/chat/dense_reasoning.md",
        "data/train/chat/gpu_upgrade_reasoning.md",
        "data/train/chat/neural_reason_core.txt",
        "data/train/chat/reason_predict_dialogue.txt",
    ):
        path = get_project_root() / rel
        if path.exists():
            texts.extend(read_txt_blocks(path))
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_thinking_texts()
    if not texts:
        raise ValueError("No thinking training data found. Add data/train/thinking/reasoning.txt")
    finetune_texts("thinking", texts, desc="Navine AI - Python Thinking", max_steps=steps)
