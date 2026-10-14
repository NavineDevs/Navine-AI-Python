import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_project_root


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_user(username: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (username or "").strip())[:64]
    return name or "user"


def user_chats_dir(username: str) -> Path:
    path = get_project_root() / "data" / "accounts" / "chats" / _safe_user(username)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(username: str) -> Path:
    return user_chats_dir(username) / "index.json"


def _chat_path(username: str, chat_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "", chat_id)[:40]
    return user_chats_dir(username) / f"{safe_id}.json"


def _load_index(username: str) -> List[Dict[str, Any]]:
    path = _index_path(username)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("chats"), list):
            return data["chats"]
    except Exception:
        pass
    return []


def _save_index(username: str, chats: List[Dict[str, Any]]) -> None:
    _index_path(username).write_text(json.dumps(chats, indent=2), encoding="utf-8")


def _title_from_message(message: str) -> str:
    text = re.sub(r"\s+", " ", (message or "").strip())
    if not text:
        return "New chat"
    return text[:48] + ("…" if len(text) > 48 else "")


def list_chats(username: str) -> List[Dict[str, Any]]:
    chats = _load_index(username)
    chats.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    return chats


def create_chat(username: str, title: Optional[str] = None) -> Dict[str, Any]:
    chat_id = uuid.uuid4().hex[:16]
    now = _now_iso()
    meta = {
        "id": chat_id,
        "title": (title or "New chat").strip()[:80] or "New chat",
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }
    payload = {
        "id": chat_id,
        "title": meta["title"],
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _chat_path(username, chat_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    chats = _load_index(username)
    chats.insert(0, meta)
    _save_index(username, chats)
    return meta


def get_chat(username: str, chat_id: str) -> Optional[Dict[str, Any]]:
    path = _chat_path(username, chat_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def rename_chat(username: str, chat_id: str, title: str) -> Optional[Dict[str, Any]]:
    chat = get_chat(username, chat_id)
    if not chat:
        return None
    clean = (title or "").strip()[:80] or "New chat"
    chat["title"] = clean
    chat["updated_at"] = _now_iso()
    _chat_path(username, chat_id).write_text(json.dumps(chat, indent=2), encoding="utf-8")
    chats = _load_index(username)
    for row in chats:
        if row.get("id") == chat_id:
            row["title"] = clean
            row["updated_at"] = chat["updated_at"]
            break
    _save_index(username, chats)
    return {"id": chat_id, "title": clean, "updated_at": chat["updated_at"], "message_count": len(chat.get("messages") or [])}


def delete_chat(username: str, chat_id: str) -> bool:
    path = _chat_path(username, chat_id)
    existed = path.exists()
    if existed:
        path.unlink()
    chats = [row for row in _load_index(username) if row.get("id") != chat_id]
    _save_index(username, chats)
    return existed


def append_exchange(
    username: str,
    chat_id: str,
    user: str,
    assistant: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    chat = get_chat(username, chat_id)
    if not chat:
        return None
    messages = list(chat.get("messages") or [])
    now = _now_iso()
    messages.append(
        {
            "role": "user",
            "content": (user or "").strip(),
            "timestamp": now,
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": (assistant or "").strip(),
            "timestamp": now,
            "meta": meta or {},
        }
    )
    chat["messages"] = messages
    chat["updated_at"] = now
    if len(messages) <= 2 or chat.get("title") in (None, "", "New chat"):
        chat["title"] = _title_from_message(user)
    _chat_path(username, chat_id).write_text(json.dumps(chat, indent=2), encoding="utf-8")
    chats = _load_index(username)
    found = False
    for row in chats:
        if row.get("id") == chat_id:
            row["title"] = chat["title"]
            row["updated_at"] = now
            row["message_count"] = len(messages)
            found = True
            break
    if not found:
        chats.insert(
            0,
            {
                "id": chat_id,
                "title": chat["title"],
                "created_at": chat.get("created_at") or now,
                "updated_at": now,
                "message_count": len(messages),
            },
        )
    _save_index(username, chats)
    return chat
