from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config
from navine.utils.paths import get_project_root


def load_unrestricted_texts() -> List[str]:
    config = load_train_config("unrestricted")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path and train_path.exists():
        texts.extend(read_txt_blocks(train_path))
    extra = config["data"].get("extra_file")
    if extra:
        extra_path = get_project_root() / extra
        if extra_path.exists():
            texts.extend(read_txt_blocks(extra_path))
    for rel in config["data"].get("topic_files") or []:
        topic_path = get_project_root() / rel
        if topic_path.exists():
            texts.extend(read_txt_blocks(topic_path))
    autolearn = get_project_root() / "data" / "train" / "unrestricted" / "autolearn_unrestricted.txt"
    if autolearn.exists():
        texts.extend(read_txt_blocks(autolearn))
    nsfw_extra = get_project_root() / "data" / "train" / "nsfw" / "autolearn_nsfw.txt"
    if nsfw_extra.exists():
        texts.extend(read_txt_blocks(nsfw_extra))
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_unrestricted_texts()
    if not texts:
        raise ValueError("No unrestricted training data found. Enable NSFW autolearn or add data/train/unrestricted/")
    finetune_texts("unrestricted", texts, desc="Navine AI - Python Unrestricted", max_steps=steps)
