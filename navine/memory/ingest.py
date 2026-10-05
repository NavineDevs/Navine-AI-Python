import json
from pathlib import Path
from typing import Any, Dict, List

from navine.memory.config import get_conversation_learning_config
from navine.memory.conversations import (
    _format_training_block,
    _is_good_exchange,
    conversations_dir,
    learn_dir,
    train_file_path,
)
from navine.utils.paths import get_project_root


def _load_session_turns(path: Path) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def ingest_conversations(rebuild_rag: bool = True) -> Dict[str, Any]:
    cfg = get_conversation_learning_config()
    session_root = conversations_dir()
    qa_path = learn_dir() / "qa.jsonl"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    train_path = train_file_path()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set = set()
    if qa_path.exists():
        for line in qa_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen_hashes.add(line.strip())
    new_qa = 0
    new_train = 0
    sessions = 0
    for path in sorted(session_root.glob("*.jsonl")):
        sessions += 1
        for turn in _load_session_turns(path):
            user = turn.get("user", "")
            assistant = turn.get("assistant", "")
            if not _is_good_exchange(user, assistant):
                continue
            entry = {
                "title": user.strip()[:200],
                "text": f"Q: {user.strip()}\nA: {assistant.strip()[:4000]}",
                "source": "conversation",
            }
            line = json.dumps(entry, ensure_ascii=False)
            if line not in seen_hashes:
                with open(qa_path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                seen_hashes.add(line)
                new_qa += 1
            block = _format_training_block(user, assistant)
            with open(train_path, "a", encoding="utf-8") as handle:
                handle.write(block + "\n")
            new_train += 1
    rag_count = 0
    if rebuild_rag and cfg.get("enabled", True):
        from navine.learn.rag import rebuild_index

        rag_count = rebuild_index()
    return {
        "sessions_scanned": sessions,
        "new_qa_pairs": new_qa,
        "training_blocks": new_train,
        "rag_documents": rag_count,
    }


def merge_into_learn_corpus() -> str:
    root = get_project_root()
    out_path = root / "data" / "text" / "conversation_corpus.txt"
    chunks: List[str] = []
    qa_path = learn_dir() / "qa.jsonl"
    if qa_path.exists():
        for line in qa_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            text = entry.get("text", "")
            if text:
                chunks.append(text[:4000])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(chunks), encoding="utf-8")
    return str(out_path)
