import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.memory.config import get_conversation_learning_config
from navine.utils.paths import get_project_root

_STATE_LOCK = threading.RLock()
_TRAIN_THREAD: Optional[threading.Thread] = None
_PENDING_SINCE_TRAIN = 0
_LAST_TRAIN_TS = 0.0
_ADAPT_RUNNING = False


def _state_path() -> Path:
    path = get_project_root() / "data" / "learn" / "conversations" / "adapt_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"pending_good": 0, "last_train_ts": 0.0, "total_trains": 0, "last_train_result": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"pending_good": 0, "last_train_ts": 0.0, "total_trains": 0, "last_train_result": None}


def _save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def looks_garbled(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    if len(raw) < 2:
        return True
    if re.search(r"[`~]{2,}.*[\[\]()]|!!\w+\(|\[\w{2,5}\]", raw) and len(raw) < 80:
        words = re.findall(r"[A-Za-z]{3,}", raw)
        if len(words) <= 2:
            return True
    compact = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", raw)
    good = sum(ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"-()[]/" for ch in raw)
    if good / max(len(raw), 1) < 0.62:
        return True
    probe = compact if compact else raw
    if " " not in probe and len(probe) >= 7:
        if re.search(r"[_/\\|]", probe) and "http" not in probe.lower():
            return True
        if re.fullmatch(r"[A-Za-z0-9_/\\.-]{7,}", probe) and not re.search(
            r"(the|and|you|for|with|this|that|from|open|file|train|game|path|time|date|yes|okay)",
            probe.lower(),
        ):
            vowels = sum(ch in "aeiou" for ch in probe.lower() if ch.isalpha())
            letters = sum(ch.isalpha() for ch in probe)
            if letters and vowels / letters < 0.34:
                return True
            if letters >= 8 and re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", probe.lower()):
                return True
            if letters >= 8 and re.fullmatch(r"[A-Za-z\-]+", probe):
                if not re.search(
                    r"(tion|ing|ment|ness|able|ally|over|under|about|with|play|file|code|help|hello|navine)",
                    probe.lower(),
                ):
                    return True
    words = re.findall(r"[A-Za-z]{3,}", raw)
    if not words and re.search(r"[/\\|_\d]{3,}", raw):
        return True
    if words:
        weird = 0
        for word in words:
            letters = word.lower()
            vowels = sum(ch in "aeiou" for ch in letters)
            if len(letters) >= 5 and vowels / len(letters) < 0.2:
                weird += 1
            if re.search(r"(.)\1{3,}", letters):
                weird += 1
            if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", letters):
                weird += 1
        if weird / len(words) >= 0.4:
            return True
    if re.search(r"[/\\]{1,}[a-z0-9_]{6,}", raw.lower()) and "http" not in raw.lower():
        return True
    return False


def is_correction_turn(user: str) -> bool:
    lower = (user or "").lower().strip()
    patterns = (
        r"\bthat(?:'s| is)? (?:wrong|incorrect)\b",
        r"\bno[,.]?\s+(?:actually|the answer|you should|say)\b",
        r"\bactually\b.{0,40}\b(?:is|are|was|were|means)\b",
        r"\bremember (?:that|this)\b",
        r"\bfrom now on\b",
        r"\bcorrect answer is\b",
        r"\bdon'?t say that\b",
        r"\bprefer (?:to )?(?:answer|say|reply)\b",
    )
    return any(re.search(p, lower) for p in patterns)


def _cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return default


def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


def note_good_pair_recorded(count: int = 1) -> None:
    global _PENDING_SINCE_TRAIN
    with _STATE_LOCK:
        state = _load_state()
        pending = int(state.get("pending_good", 0)) + max(1, count)
        state["pending_good"] = pending
        _PENDING_SINCE_TRAIN = pending
        _save_state(state)


def _run_chat_finetune(steps: int) -> Dict[str, Any]:
    from navine.train.chat import load_chat_texts
    from navine.train.base import finetune_texts
    from navine.text.infer import clear_model_cache
    from navine.utils.gpu_session import inference_active, training_session
    from navine.utils.training_lock import training_lock

    if inference_active():
        return {"ok": False, "error": "inference active; adapt deferred"}

    texts = load_chat_texts()
    learned = get_project_root() / "data" / "train" / "chat" / "conversation_learned.txt"
    if learned.exists():
        try:
            from navine.train.data_loaders import read_txt_blocks

            blocks = read_txt_blocks(learned)
            if blocks:
                texts = blocks[- min(len(blocks), 400) :] + texts[: max(40, min(200, len(texts)))]
        except Exception:
            pass
    if not texts:
        return {"ok": False, "error": "No chat training texts available"}
    capped_steps = max(8, min(int(steps), 30))
    try:
        with training_session(wait_for_idle_seconds=90.0):
            with training_lock("text"):
                finetune_texts(
                    "chat",
                    texts,
                    desc="Navine online adapt (chat)",
                    max_steps=capped_steps,
                )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        clear_model_cache()
    except Exception:
        pass
    return {"ok": True, "steps": capped_steps, "samples": len(texts)}


def run_adapt_train(force: bool = False, steps: Optional[int] = None) -> Dict[str, Any]:
    global _ADAPT_RUNNING, _LAST_TRAIN_TS, _PENDING_SINCE_TRAIN, _TRAIN_THREAD
    cfg = get_conversation_learning_config()
    if not cfg.get("enabled", True):
        return {"ok": False, "error": "conversation learning disabled"}
    if not cfg.get("auto_train_chat", False) and not force:
        return {"ok": False, "error": "auto_train_chat disabled"}

    now = time.time()
    min_pairs = _cfg_int(cfg, "min_pairs_before_train", 2)
    cooldown = _cfg_float(cfg, "train_cooldown_seconds", 90.0)
    train_steps = steps if steps is not None else _cfg_int(cfg, "online_train_steps", _cfg_int(cfg, "train_steps", 20))
    if train_steps > 60 and not force:
        train_steps = 30

    with _STATE_LOCK:
        state = _load_state()
        pending = int(state.get("pending_good", 0))
        last_ts = float(state.get("last_train_ts", 0.0) or 0.0)
        if _ADAPT_RUNNING:
            return {"ok": False, "error": "adapt train already running", "pending": pending}
        if not force:
            if pending < min_pairs:
                return {"ok": False, "error": "not enough new pairs", "pending": pending}
            if now - last_ts < cooldown:
                return {
                    "ok": False,
                    "error": "cooldown",
                    "pending": pending,
                    "wait_s": round(cooldown - (now - last_ts), 1),
                }
        _ADAPT_RUNNING = True

    result: Dict[str, Any]
    try:
        result = _run_chat_finetune(train_steps)
        result["at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    with _STATE_LOCK:
        state = _load_state()
        if result.get("ok"):
            state["pending_good"] = 0
            state["last_train_ts"] = time.time()
            state["total_trains"] = int(state.get("total_trains", 0)) + 1
            _PENDING_SINCE_TRAIN = 0
            _LAST_TRAIN_TS = float(state["last_train_ts"])
        state["last_train_result"] = result
        _save_state(state)
        _ADAPT_RUNNING = False
        _TRAIN_THREAD = None
    return result


def maybe_schedule_adapt_train(foreground: bool = False) -> Dict[str, Any]:
    global _TRAIN_THREAD
    cfg = get_conversation_learning_config()
    if not cfg.get("auto_train_chat", False):
        return {"scheduled": False, "reason": "auto_train_chat off"}
    background = bool(cfg.get("train_in_background", True)) and not foreground

    probe = run_adapt_train(force=False, steps=None) if not background else None
    if not background:
        return {"scheduled": False, "result": probe}

    with _STATE_LOCK:
        state = _load_state()
        pending = int(state.get("pending_good", 0))
        min_pairs = _cfg_int(cfg, "min_pairs_before_train", 2)
        cooldown = _cfg_float(cfg, "train_cooldown_seconds", 90.0)
        last_ts = float(state.get("last_train_ts", 0.0) or 0.0)
        if _ADAPT_RUNNING or (_TRAIN_THREAD is not None and _TRAIN_THREAD.is_alive()):
            return {"scheduled": False, "reason": "already running", "pending": pending}
        if pending < min_pairs:
            return {"scheduled": False, "reason": "need more pairs", "pending": pending}
        if time.time() - last_ts < cooldown:
            return {"scheduled": False, "reason": "cooldown", "pending": pending}

        def _worker() -> None:
            try:
                run_adapt_train(force=True)
            except Exception:
                pass

        _TRAIN_THREAD = threading.Thread(target=_worker, name="navine-adapt-train", daemon=True)
        _TRAIN_THREAD.start()
        return {"scheduled": True, "pending": pending}


def record_action_learning(
    user_request: str,
    result_summary: str,
    session_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    from navine.memory.conversations import record_chat_exchange

    cfg = get_conversation_learning_config()
    if not cfg.get("adapt_assist_actions", True):
        return session_id or ""
    user = (user_request or "").strip()
    assistant = (result_summary or "").strip()
    if not user or not assistant:
        return session_id or ""
    if looks_garbled(assistant):
        return session_id or ""
    meta_out = dict(meta or {})
    meta_out["source"] = meta_out.get("source") or "assist_action"
    return record_chat_exchange(user, assistant, session_id=session_id, meta=meta_out)


def adapt_status() -> Dict[str, Any]:
    with _STATE_LOCK:
        state = _load_state()
        return {
            "pending_good": int(state.get("pending_good", 0)),
            "last_train_ts": state.get("last_train_ts"),
            "total_trains": int(state.get("total_trains", 0)),
            "last_train_result": state.get("last_train_result"),
            "running": _ADAPT_RUNNING,
            "config": {
                "auto_train_chat": bool(get_conversation_learning_config().get("auto_train_chat", False)),
                "train_in_background": bool(get_conversation_learning_config().get("train_in_background", True)),
                "min_pairs_before_train": _cfg_int(get_conversation_learning_config(), "min_pairs_before_train", 2),
                "online_train_steps": _cfg_int(
                    get_conversation_learning_config(),
                    "online_train_steps",
                    40,
                ),
                "train_cooldown_seconds": _cfg_float(get_conversation_learning_config(), "train_cooldown_seconds", 90.0),
            },
        }
