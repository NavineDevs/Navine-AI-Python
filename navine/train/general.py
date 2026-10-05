from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config
from navine.utils.paths import get_project_root


def load_general_texts() -> List[str]:
    config = load_train_config("general")
    texts: List[str] = []
    train_path = resolve_data_path(config)
    if train_path:
        texts.extend(read_txt_blocks(train_path))
    for key in ("code_file", "learned_file"):
        rel = config["data"].get(key)
        if not rel:
            continue
        path = get_project_root() / rel
        if path.exists() and path.suffix == ".txt":
            texts.extend(read_txt_blocks(path))
    autolearn_dir = get_project_root() / "data" / "train" / "general"
    for name in (
        "autolearn_corpus.txt",
        "autolearn_science.txt",
        "autolearn_history.txt",
        "autolearn_news.txt",
        "autolearn_technical.txt",
        "nn_pipeline_notes.txt",
        "enterprise_stack_notes.txt",
        "gpu_train_ops.txt",
        "gpu_upgrade_stack.txt",
    ):
        path = autolearn_dir / name
        if path.exists():
            texts.extend(read_txt_blocks(path))
    for path in sorted(autolearn_dir.glob("*.md")):
        texts.extend(read_txt_blocks(path))
    diy_dir = get_project_root() / "data" / "train" / "diy"
    for path in sorted(diy_dir.glob("*.txt")) + sorted(diy_dir.glob("*.md")):
        texts.extend(read_txt_blocks(path))
    unrestricted_dir = get_project_root() / "data" / "train" / "unrestricted"
    for path in sorted(unrestricted_dir.glob("*.txt")):
        texts.extend(read_txt_blocks(path))
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_general_texts()
    if not texts:
        raise ValueError("No general training data found. Add corpus text to data/train/general/")
    finetune_texts("general", texts, desc="Navine AI - Python General Text", max_steps=steps)
