import hashlib
import hmac
import json
import secrets
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import yaml

from navine.utils.paths import get_project_root


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_admin_config() -> Dict[str, Any]:
    path = get_project_root() / "configs" / "admin.yaml"
    if not path.exists():
        return {"users": {}, "secret": "navine-admin-fallback", "token_ttl_hours": 12}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"users": {}, "secret": "navine-admin-fallback", "token_ttl_hours": 12}


def verify_admin_username(username: str) -> bool:
    users = load_admin_config().get("users") or {}
    if not isinstance(users, dict):
        return False
    key = str(username or "").strip()
    return bool(key and key in users)


def verify_admin_credentials(username: str, password: str) -> bool:
    users = load_admin_config().get("users") or {}
    if not isinstance(users, dict):
        return False
    key = str(username or "").strip()
    if not key or key not in users:
        return False
    entry = users.get(key) or {}
    expected = str(entry.get("password_hash") or "").strip().lower()
    if not expected:
        return False
    return hmac.compare_digest(_hash_password(password), expected)


def _sign_payload(payload: str) -> str:
    secret = str(load_admin_config().get("secret") or "navine-admin-fallback")
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_admin_token(username: str) -> Tuple[str, int]:
    ttl_hours = int(load_admin_config().get("token_ttl_hours") or 24)
    expires = int(time.time()) + max(3600, ttl_hours * 3600)
    nonce = secrets.token_hex(8)
    payload = json.dumps({"user": username, "exp": expires, "nonce": nonce}, separators=(",", ":"))
    token = f"{payload}.{_sign_payload(payload)}"
    return token, expires


def verify_admin_token(token: Optional[str]) -> Optional[str]:
    raw = str(token or "").strip()
    if not raw or "." not in raw:
        return None
    payload, sig = raw.rsplit(".", 1)
    if not hmac.compare_digest(_sign_payload(payload), sig):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if int(data.get("exp") or 0) < int(time.time()):
        return None
    user = str(data.get("user") or "").strip()
    users = load_admin_config().get("users") or {}
    if user and isinstance(users, dict) and user in users:
        return user
    return None


def admin_token_from_headers(headers: Any) -> Optional[str]:
    for name in ("X-Admin-Token", "X-Navine-Admin-Token"):
        value = headers.get(name) if hasattr(headers, "get") else None
        if value:
            return str(value).strip()
    auth = headers.get("Authorization", "") if hasattr(headers, "get") else ""
    if str(auth).lower().startswith("admin "):
        return str(auth)[6:].strip()
    return None
