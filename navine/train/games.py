from typing import List, Optional

from navine.train.base import finetune_texts
from navine.train.data_loaders import read_txt_blocks
from navine.utils.paths import get_project_root


def load_games_texts() -> List[str]:
    root = get_project_root()
    texts: List[str] = []
    paths = [
        root / "data" / "train" / "games" / "strategy_corpus.txt",
        root / "data" / "train" / "games" / "observe_log.txt",
        root / "data" / "train" / "games" / "action_labels.txt",
        root / "data" / "train" / "games" / "platform_fps_moba.txt",
        root / "data" / "train" / "games" / "skill_drills.md",
        root / "data" / "train" / "games" / "coach_extended.md",
        root / "data" / "train" / "games" / "gpu_upgrade_drills.txt",
        root / "data" / "train" / "games" / "gpu_upgrade_coach.md",
        root / "data" / "train" / "chat" / "games_dialogue.txt",
        root / "data" / "train" / "chat" / "gameplay_learn_dialogue.txt",
    ]
    games_dir = root / "data" / "train" / "games"
    if games_dir.exists():
        for path in sorted(games_dir.glob("*.md")):
            if path not in paths:
                paths.append(path)
        for path in sorted(games_dir.glob("*.txt")):
            if path not in paths:
                paths.append(path)
    for path in paths:
        if path.exists():
            blocks = read_txt_blocks(path)
            texts.extend(blocks)
    try:
        from navine.learn.gameplay import load_game_train_texts

        for block in load_game_train_texts():
            if block and block not in texts:
                texts.append(block)
    except Exception:
        pass
    return texts


def train(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    texts = load_games_texts()
    if not texts:
        raise ValueError(
            "No games training data found. Add data/train/games/strategy_corpus.txt "
            "or run assist observe-game captures."
        )
    finetune_texts("games", texts, desc="Navine AI - Python Games", max_steps=steps)
