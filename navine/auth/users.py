import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from navine.utils.paths import get_project_root


def _users_path() -> Path:
    path = get_project_root() / "data" / "accounts" / "users.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def _session_secret() -> str:
    try:
        from navine.auth.admin import load_admin_config

        return str(load_admin_config().get("secret") or "navine-session-v1")
    except Exception:
        return "navine-session-v1"


def _is_bootstrap_admin(username: str) -> bool:
    try:
        from navine.auth.admin import load_admin_config

        users = load_admin_config().get("users") or {}
        return str(username or "").strip() in users
    except Exception:
        return False


class UserStore:
    def _load(self) -> Dict[str, Any]:
        path = _users_path()
        if not path.exists():
            return {"users": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("users"), list):
                return data
        except Exception:
            pass
        return {"users": []}

    def _save(self, data: Dict[str, Any]) -> None:
        path = _users_path()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def find(self, username: str) -> Optional[Dict[str, Any]]:
        key = str(username or "").strip()
        if not key:
            return None
        for row in self._load()["users"]:
            if str(row.get("username") or "").lower() == key.lower():
                return row
        return None

    def signup(self, username: str, password: str) -> Dict[str, Any]:
        name = str(username or "").strip()
        if len(name) < 2:
            raise ValueError("Username must be at least 2 characters")
        if len(name) > 32:
            raise ValueError("Username too long")
        if not password or len(password) < 4:
            raise ValueError("Password must be at least 4 characters")
        if self.find(name) or _is_bootstrap_admin(name):
            raise ValueError("Username already taken")
        salt = secrets.token_hex(8)
        row = {
            "username": name,
            "salt": salt,
            "password_hash": _hash_password(password, salt),
            "role": "user",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data = self._load()
        data["users"].append(row)
        self._save(data)
        return row

    def verify(self, username: str, password: str) -> bool:
        if _is_bootstrap_admin(username):
            from navine.auth.admin import verify_admin_credentials

            return verify_admin_credentials(username, password)
        row = self.find(username)
        if not row:
            return False
        expected = row.get("password_hash") or ""
        salt = str(row.get("salt") or "")
        return hmac.compare_digest(str(expected), _hash_password(password, salt))

    def is_admin(self, username: str) -> bool:
        if _is_bootstrap_admin(username):
            return True
        row = self.find(username)
        return str((row or {}).get("role") or "").lower() == "admin"


def create_session_token(username: str) -> Tuple[str, int]:
    from navine.auth.admin import create_admin_token

    if _is_bootstrap_admin(username):
        return create_admin_token(username)
    ttl_hours = 72
    expires = int(time.time()) + max(3600, ttl_hours * 3600)
    nonce = secrets.token_hex(8)
    payload = json.dumps(
        {"user": str(username).strip(), "exp": expires, "nonce": nonce, "kind": "user"},
        separators=(",", ":"),
    )
    sig = hmac.new(_session_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}", expires


def verify_session_token(token: Optional[str]) -> Optional[str]:
    from navine.auth.admin import verify_admin_token

    admin_user = verify_admin_token(token)
    if admin_user:
        return admin_user
    raw = str(token or "").strip()
    if not raw or "." not in raw:
        return None
    payload, sig = raw.rsplit(".", 1)
    expected = hmac.new(_session_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if int(data.get("exp") or 0) < int(time.time()):
        return None
    user = str(data.get("user") or "").strip()
    store = UserStore()
    if store.find(user) or _is_bootstrap_admin(user):
        return user
    return None


def session_from_headers(headers: Any) -> Optional[str]:
    for name in ("X-Session-Token", "X-Admin-Token", "X-Navine-Session"):
        value = headers.get(name) if hasattr(headers, "get") else None
        if value:
            return str(value).strip()
    auth = headers.get("Authorization", "") if hasattr(headers, "get") else ""
    if str(auth).lower().startswith("bearer "):
        return str(auth)[7:].strip()
    if str(auth).lower().startswith("session "):
        return str(auth)[8:].strip()
    return None
