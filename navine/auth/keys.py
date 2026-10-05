import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ApiKeyStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _prefix(raw_key: str) -> str:
        return raw_key[:10] if len(raw_key) >= 10 else raw_key

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"keys": []}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "keys" not in data:
            data["keys"] = []
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def create(
        self,
        name: str = "Navine AI - Python",
        scopes: Optional[List[str]] = None,
        expires_days: Optional[int] = None,
        prefix: str = "nvn",
        owner: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        body = secrets.token_urlsafe(32)
        raw_key = f"{prefix}_{body}"
        key_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_days and int(expires_days) > 0:
            expires_at = (now + timedelta(days=int(expires_days))).isoformat()
        entry = {
            "id": key_id,
            "name": name,
            "hash": self.hash_key(raw_key),
            "prefix": self._prefix(raw_key),
            "scopes": scopes or ["chat", "text", "image", "video", "code", "deepfake"],
            "created_at": now.isoformat(),
            "expires_at": expires_at,
            "last_used_at": None,
            "use_count": 0,
            "revoked": False,
            "owner": str(owner or "").strip() or None,
        }
        data = self._load()
        data["keys"].append(entry)
        self._save(data)
        return key_id, raw_key, entry

    def list_keys(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = []
        owner_key = str(owner or "").strip().lower()
        for entry in self._load()["keys"]:
            if owner_key:
                entry_owner = str(entry.get("owner") or "").lower()
                if entry_owner != owner_key:
                    continue
            rows.append(
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "prefix": entry.get("prefix"),
                    "scopes": entry.get("scopes") or [],
                    "created_at": entry.get("created_at"),
                    "expires_at": entry.get("expires_at"),
                    "last_used_at": entry.get("last_used_at"),
                    "use_count": entry.get("use_count", 0),
                    "revoked": entry.get("revoked", False),
                    "owner": entry.get("owner"),
                }
            )
        return rows

    def delete(self, key_id: str, owner: Optional[str] = None, *, admin: bool = False) -> bool:
        wanted = str(key_id or "").strip()
        if not wanted:
            return False
        data = self._load()
        kept: List[Dict[str, Any]] = []
        found = False
        owner_key = str(owner or "").strip().lower()
        for entry in data["keys"]:
            if str(entry.get("id") or "") != wanted:
                kept.append(entry)
                continue
            if not admin:
                entry_owner = str(entry.get("owner") or "").strip().lower()
                if owner_key and entry_owner != owner_key:
                    kept.append(entry)
                    continue
            found = True
        if found:
            data["keys"] = kept
            self._save(data)
        return found

    def revoke(self, key_id: str) -> bool:
        data = self._load()
        found = False
        for entry in data["keys"]:
            if entry["id"] == key_id:
                entry["revoked"] = True
                found = True
                break
        if found:
            self._save(data)
        return found

    def rotate(self, key_id: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        data = self._load()
        old = None
        for entry in data["keys"]:
            if entry["id"] == key_id and not entry.get("revoked"):
                old = entry
                entry["revoked"] = True
                break
        if not old:
            return None
        self._save(data)
        return self.create(
            name=str(old.get("name") or "Navine AI - Python"),
            scopes=list(old.get("scopes") or []),
            prefix="nvn",
        )

    def verify(self, raw_key: str) -> bool:
        if not raw_key:
            return False
        digest = self.hash_key(raw_key)
        now = datetime.now(timezone.utc)
        data = self._load()
        changed = False
        ok = False
        for entry in data["keys"]:
            if entry.get("revoked"):
                continue
            expires = entry.get("expires_at")
            if expires:
                try:
                    exp = datetime.fromisoformat(str(expires))
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    if now > exp:
                        entry["revoked"] = True
                        changed = True
                        continue
                except Exception:
                    pass
            if entry.get("hash") == digest:
                entry["last_used_at"] = now.isoformat()
                entry["use_count"] = int(entry.get("use_count") or 0) + 1
                changed = True
                ok = True
                break
        if changed:
            self._save(data)
        return ok

    def has_active_keys(self) -> bool:
        for entry in self._load()["keys"]:
            if not entry.get("revoked"):
                return True
        return False

    def ensure_default_key(self) -> Optional[str]:
        if self.has_active_keys():
            return None
        _, raw_key, _ = self.create("default")
        return raw_key
