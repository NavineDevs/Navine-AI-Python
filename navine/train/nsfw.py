from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config
from navine.utils.paths import get_project_root


def load_nsfw_texts() -> List[str]:
    config = load_train_config("nsfw")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path and train_path.exists():
        texts.extend(read_txt_blocks(train_path))
    extra = config["data"].get("extra_file")
    if extra:
        extra_path = get_project_root() / extra
        if extra_path.exists():
            texts.extend(read_txt_blocks(extra_path))
    autolearn = get_project_root() / "data" / "train" / "nsfw" / "autolearn_nsfw.txt"
    if autolearn.exists():
        texts.extend(read_txt_blocks(autolearn))
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_nsfw_texts()
    if not texts:
        raise ValueError("No NSFW training data found. Enable configs/nsfw.yaml and run: learn nsfw run")
    finetune_texts("nsfw", texts, desc="Navine AI - Python NSFW", max_steps=steps)
