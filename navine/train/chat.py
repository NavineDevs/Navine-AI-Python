from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_txt_blocks, resolve_data_path
from navine.utils.config import load_train_config
from navine.utils.paths import get_project_root


def _extend_unique(texts: List[str], blocks: List[str], repeats: int = 1) -> None:
    existing = set(texts)
    for block in blocks:
        if block not in existing:
            texts.append(block)
            existing.add(block)
    core = [b for b in blocks if b]
    for _ in range(max(0, repeats - 1)):
        texts.extend(core)


def load_chat_texts() -> List[str]:
    config = load_train_config("chat")
    texts: List[str] = []
    for key in ("train_file", "chat_file"):
        path = resolve_data_path(config, key)
        if path:
            texts.extend(read_txt_blocks(path))
    root = get_project_root()
    chat_dir = root / "data" / "train" / "chat"
    quality = chat_dir / "quality_core_dialogue.txt"
    if quality.exists():
        _extend_unique(texts, read_txt_blocks(quality), repeats=12)
    for voice_name in ("voice_assistant_dialogue.txt", "halo_voice_dialogue.txt"):
        voice_file = chat_dir / voice_name
        if voice_file.exists():
            _extend_unique(texts, read_txt_blocks(voice_file), repeats=14)
            break
    persona = chat_dir / "owner_persona_dialogue.txt"
    if persona.exists():
        _extend_unique(texts, read_txt_blocks(persona), repeats=12)
    neural = chat_dir / "neural_reason_core.txt"
    if neural.exists():
        _extend_unique(texts, read_txt_blocks(neural), repeats=14)
    real_ai = chat_dir / "real_ai_dialogue.txt"
    if real_ai.exists():
        _extend_unique(texts, read_txt_blocks(real_ai), repeats=4)
    reason_dialogue = chat_dir / "reason_predict_dialogue.txt"
    if reason_dialogue.exists():
        _extend_unique(texts, read_txt_blocks(reason_dialogue), repeats=6)
    games_dialogue = chat_dir / "games_dialogue.txt"
    if games_dialogue.exists():
        _extend_unique(texts, read_txt_blocks(games_dialogue), repeats=6)
    play_learn = chat_dir / "gameplay_learn_dialogue.txt"
    if play_learn.exists():
        _extend_unique(texts, read_txt_blocks(play_learn), repeats=12)
    file_ops = chat_dir / "file_ops_dialogue.txt"
    if file_ops.exists():
        _extend_unique(texts, read_txt_blocks(file_ops), repeats=10)
    moltbook = chat_dir / "moltbook_dialogue.txt"
    if moltbook.exists():
        _extend_unique(texts, read_txt_blocks(moltbook), repeats=2)
    for md_name in (
        "nn_systems.md",
        "multimodal_ops.md",
        "dense_reasoning.md",
        "coding_tutor.md",
        "gpu_upgrade_reasoning.md",
    ):
        md_path = chat_dir / md_name
        if md_path.exists():
            _extend_unique(texts, read_txt_blocks(md_path), repeats=8)
    for extra in sorted(chat_dir.glob("*.md")):
        if extra.name in (
            "nn_systems.md",
            "multimodal_ops.md",
            "dense_reasoning.md",
            "coding_tutor.md",
            "gpu_upgrade_reasoning.md",
        ):
            continue
        _extend_unique(texts, read_txt_blocks(extra), repeats=2)
    fallback = root / "data" / "text" / "instruction_chat.txt"
    if fallback.exists():
        _extend_unique(texts, read_txt_blocks(fallback), repeats=1)
    for name in (
        "conversation_learned.txt",
        "autolearn_qa.txt",
        "autolearn_instructions.txt",
        "systems_long_form.txt",
        "assistant_ops_extended.txt",
        "gpu_upgrade_dialogue.txt",
    ):
        path = chat_dir / name
        if path.exists():
            _extend_unique(texts, read_txt_blocks(path), repeats=1)
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_chat_texts()
    if not texts:
        raise ValueError("No chat training data found. Add data/train/chat/instructions.txt")
    finetune_texts("chat", texts, desc="Navine AI - Python Chat", max_steps=steps)
