from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_jsonl, read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config


def format_source_entry(entry: dict) -> str:
    task = entry.get("task", entry.get("prompt", ""))
    target = entry.get("target", entry.get("subject", ""))
    kind = entry.get("kind", "general")
    answer = entry.get("answer", entry.get("response", ""))
    sources = entry.get("sources") or []
    block = f"### Task: {task}\n### Target: {target}\n### Kind: {kind}\n### Response:\n{answer}"
    if sources:
        src_lines = []
        for row in sources[:6]:
            if isinstance(row, dict):
                title = str(row.get("title") or "").strip()
                url = str(row.get("url") or "").strip()
                if title or url:
                    src_lines.append(f"- {title} {url}".strip())
            else:
                src_lines.append(f"- {row}")
        if src_lines:
            block += "\n### Sources:\n" + "\n".join(src_lines)
    return block


def load_osint_texts() -> List[str]:
    config = load_train_config("osint")
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
                formatted = format_source_entry(entry)
                if formatted.strip():
                    texts.append(formatted)
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_osint_texts()
    if not texts:
        raise ValueError("No OSINT training data found. Add data/train/osint/intel.txt")
    finetune_texts("osint", texts, desc="Navine AI - Python OSINT", max_steps=steps)
