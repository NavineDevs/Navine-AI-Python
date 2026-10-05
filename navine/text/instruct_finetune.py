from pathlib import Path
from typing import List, Optional

from navine.text.data_pipeline import build_training_corpus
from navine.utils.paths import get_project_root

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"


def format_instruct(system: str, user: str, assistant: str) -> str:
    return f"{SYSTEM}\n{system.strip()}\n{USER}\n{user.strip()}\n{ASSISTANT}\n{assistant.strip()}"


def parse_instruct_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    parts = text.split(SYSTEM)
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if USER in chunk and ASSISTANT in chunk:
            blocks.append(f"{SYSTEM}\n{chunk}")
    return blocks


def export_instruct_dataset(
    output_path: Optional[Path] = None,
    system_prompt: str = "You are Navine AI - Python, a helpful local assistant.",
) -> Path:
    root = get_project_root()
    out = output_path or (root / "data" / "text" / "instruction_finetune.txt")
    lines: List[str] = []
    chat_path = root / "data" / "text" / "instruction_chat.txt"
    if chat_path.exists():
        content = chat_path.read_text(encoding="utf-8")
        if USER in content and ASSISTANT in content:
            lines.extend(parse_instruct_blocks(content))
        else:
            for block in content.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                if "### User:" in block and "### Assistant:" in block:
                    user_part = block.split("### User:", 1)[1].split("### Assistant:", 1)[0].strip()
                    asst_part = block.split("### Assistant:", 1)[1].strip()
                    lines.append(format_instruct(system_prompt, user_part, asst_part))
    corpus = build_training_corpus()
    for idx, text in enumerate(corpus[:200]):
        lines.append(format_instruct(system_prompt, f"Summarize: {text[:120]}", text[:400]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return out
