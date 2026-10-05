from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_jsonl, resolve_data_path
from navine.utils.config import load_train_config


def format_math_entry(entry: dict) -> str:
    question = entry.get("question", entry.get("prompt", ""))
    answer = entry.get("answer", entry.get("response", ""))
    steps = entry.get("steps", "")
    block = f"### Question: {question}\n### Answer: {answer}"
    if steps:
        block += f"\n### Steps:\n{steps}"
    return block


def load_math_texts() -> List[str]:
    config = load_train_config("math")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path:
        for entry in read_jsonl(train_path):
            formatted = format_math_entry(entry)
            if formatted.strip():
                texts.append(formatted)
    from navine.utils.paths import get_project_root
    autolearn = get_project_root() / "data" / "train" / "math" / "autolearn_qa.jsonl"
    if autolearn.exists():
        for entry in read_jsonl(autolearn):
            formatted = format_math_entry(entry)
            if formatted.strip():
                texts.append(formatted)
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_math_texts()
    if not texts:
        raise ValueError("No math training data found. Add data/train/math/qa.jsonl")
    finetune_texts("math", texts, desc="Navine AI - Python Math", max_steps=steps)
