from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.coding import load_coding_texts
from navine.utils.paths import get_checkpoint_dir
from navine.train.detective import load_detective_texts
from navine.train.general import load_general_texts
from navine.train.thinking import load_thinking_texts
from navine.train.unrestricted import load_unrestricted_texts


def load_hitboyx23_texts() -> List[str]:
    return (
        load_general_texts()
        + load_unrestricted_texts()
        + load_coding_texts(language=None)
        + load_thinking_texts()
        + load_detective_texts()
    )


def load_hitboyx23_python_texts() -> List[str]:
    return (
        load_general_texts()
        + load_unrestricted_texts()
        + load_coding_texts(language="python")
        + load_thinking_texts()
        + load_detective_texts()
    )


def train(language: Optional[str] = None, steps: Optional[int] = None) -> None:
    texts = load_hitboyx23_texts()
    if not texts:
        raise ValueError("No HitBoyXx23 AI training data found.")
    finetune_texts(
        "hitboyx23",
        texts,
        desc="HitBoyXx23 AI (all languages)",
        max_steps=steps,
        checkpoint_dir=get_checkpoint_dir("hitboyx23_ai"),
    )


def train_python(language: Optional[str] = None, steps: Optional[int] = None) -> None:
    texts = load_hitboyx23_python_texts()
    if not texts:
        raise ValueError("No HitBoyXx23 AI Python training data found.")
    finetune_texts(
        "hitboyx23_python",
        texts,
        desc="HitBoyXx23 AI (Python only)",
        max_steps=steps,
        checkpoint_dir=get_checkpoint_dir("hitboyx23_ai_python"),
    )
