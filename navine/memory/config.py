from typing import Any, Dict

from navine.policy import load_policy_config

DEFAULT_CONVERSATION_LEARNING: Dict[str, Any] = {
    "enabled": True,
    "auto_ingest": True,
    "auto_index": True,
    "auto_train_chat": True,
    "train_steps": 120,
    "online_train_steps": 50,
    "min_pairs_before_train": 2,
    "train_cooldown_seconds": 90,
    "train_in_background": True,
    "adapt_assist_actions": True,
    "skip_garbled_outputs": True,
    "boost_corrections": True,
    "min_turn_length": 8,
    "min_answer_length": 12,
    "session_dir": "data/conversations",
    "learn_dir": "data/learn/conversations",
    "train_file": "data/train/chat/conversation_learned.txt",
}

DEFAULT_SEARCH: Dict[str, Any] = {
    "default_enabled": False,
    "fallback_on_failure": False,
    "auto_search_factual": False,
    "learn_from_search": True,
}


def get_conversation_learning_config() -> Dict[str, Any]:
    cfg = load_policy_config()
    merged = dict(DEFAULT_CONVERSATION_LEARNING)
    merged.update(cfg.get("conversation_learning") or {})
    return merged


def get_search_config() -> Dict[str, Any]:
    cfg = load_policy_config()
    merged = dict(DEFAULT_SEARCH)
    merged.update(cfg.get("search") or {})
    return merged


def conversation_learning_enabled() -> bool:
    return bool(get_conversation_learning_config().get("enabled", True))
