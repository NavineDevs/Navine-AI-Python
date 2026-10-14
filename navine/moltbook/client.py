from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from navine.moltbook.challenge import solve_challenge
from navine.utils.paths import get_project_root

API_BASE = "https://www.moltbook.com/api/v1"
ALLOWED_HOST = "www.moltbook.com"
REQUEST_TIMEOUT = 30
MAX_AUTO_VERIFY_FAILURES = 3
HARD_STOP_VERIFY_FAILURES = 7
VERIFY_COOLDOWN_SECONDS = 2 * 3600
MIN_CYCLE_MINUTES = 2

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "default_submolt": "general",
    "auto_verify": True,
    "heartbeat_enabled": True,
    "heartbeat_minutes": 30,
    "description": "",
    "autonomy_post_every_hours": 0,
    "autonomy_max_upvotes": 0,
    "autonomy_max_comments": 0,
    "autonomy_daily_comment_limit": 0,
    "autonomy_follow_after": 1,
    "persona": "",
    "interests": [],
}


class MoltbookError(RuntimeError):
    def __init__(self, message: str, status_code: int = 0, hint: str = "", payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.hint = hint
        self.payload = payload or {}


def load_moltbook_config() -> Dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    path = get_project_root() / "configs" / "moltbook.yaml"
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                merged.update(loaded)
        except Exception:
            pass
    return merged


def _brand_slug() -> str:
    try:
        from navine.utils.brand import brand_name

        name = brand_name()
    except Exception:
        name = "navine-ai"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "navine-ai"


def credentials_path() -> Path:
    override = os.environ.get("MOLTBOOK_CREDENTIALS_FILE")
    if override:
        return Path(override)
    return Path.home() / ".config" / "moltbook" / f"{_brand_slug()}.json"


def load_credentials() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    path = credentials_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    env_key = os.environ.get("MOLTBOOK_API_KEY")
    if env_key:
        data["api_key"] = env_key.strip()
    return data


def save_credentials(data: Dict[str, Any]) -> Path:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _state_path() -> Path:
    return get_project_root() / "logs" / "moltbook_state.json"


def load_state() -> Dict[str, Any]:
    path = _state_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {"verify_failures": 0, "pending": []}


def auto_verify_allowed(state: Dict[str, Any]) -> bool:
    failures = int(state.get("verify_failures") or 0)
    if failures < MAX_AUTO_VERIFY_FAILURES:
        return True
    if failures >= HARD_STOP_VERIFY_FAILURES:
        return False
    last_failure = float(state.get("last_verify_failure_at") or 0)
    return time.time() - last_failure >= VERIFY_COOLDOWN_SECONDS


def record_verify_failure(state: Dict[str, Any], count: int = 1) -> None:
    state["verify_failures"] = int(state.get("verify_failures") or 0) + count
    state["last_verify_failure_at"] = time.time()


def save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def mask_key(api_key: str) -> str:
    key = str(api_key or "")
    if len(key) <= 12:
        return "set" if key else ""
    return f"{key[:9]}...{key[-4:]}"


def _log_challenge(
    kind: str, challenge: str, solved: Dict[str, Any], outcome: str, details: Optional[Dict[str, Any]] = None
) -> None:
    state = load_state()
    log = list(state.get("challenge_log") or [])
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "challenge": challenge,
        "decoded": solved.get("decoded"),
        "answer": solved.get("answer"),
        "outcome": outcome,
    }
    if details:
        entry["details"] = details
    log.append(entry)
    state["challenge_log"] = log[-50:]
    save_state(state)


def _find_verification(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [payload.get("verification")]
    for key in ("post", "comment", "submolt", "data"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            candidates.append(inner.get("verification"))
    for item in candidates:
        if isinstance(item, dict) and item.get("verification_code"):
            return item
    return None


def _submolt_name(name: str) -> str:
    clean = str(name or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_\-]{2,40}", clean):
        raise MoltbookError("Community name must be 2-40 lowercase letters, numbers, dashes, or underscores.", status_code=400)
    return clean


class MoltbookClient:
    def __init__(self, api_key: Optional[str] = None):
        creds = load_credentials()
        self.api_key = (api_key or creds.get("api_key") or "").strip()
        self.agent_name = str(creds.get("agent_name") or "")
        self.config = load_moltbook_config()

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        import requests

        from navine.policy import assert_no_external_ai_url

        url = f"{API_BASE}{path}"
        assert_no_external_ai_url(url)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if auth:
            if not self.api_key:
                raise MoltbookError("Not registered on Moltbook yet. Register first.", status_code=401)
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=body,
                params=params,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise MoltbookError(f"Could not reach Moltbook: {exc}") from exc
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location", "")
            raise MoltbookError(f"Moltbook redirected to {location}; refusing to resend credentials.")
        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text[:500]}
        if not isinstance(data, dict):
            data = {"data": data}
        if response.status_code >= 400 or data.get("success") is False:
            message = str(data.get("error") or data.get("message") or f"Moltbook error {response.status_code}")
            retry = data.get("retry_after_minutes") or data.get("retry_after_seconds")
            if retry:
                unit = "minutes" if data.get("retry_after_minutes") else "seconds"
                message = f"{message} (retry after {retry} {unit})"
            raise MoltbookError(message, status_code=response.status_code, hint=str(data.get("hint") or ""), payload=data)
        return data

    def register(self, name: str, description: str) -> Dict[str, Any]:
        clean_name = str(name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]{2,40}", clean_name):
            raise MoltbookError("Agent name must be 2-40 letters, numbers, dashes, or underscores.", status_code=400)
        data = self._request(
            "POST",
            "/agents/register",
            body={"name": clean_name, "description": str(description or "").strip()[:500]},
            auth=False,
        )
        agent = data.get("agent") if isinstance(data.get("agent"), dict) else data
        api_key = str(agent.get("api_key") or "")
        if not api_key:
            raise MoltbookError("Moltbook did not return an API key.", payload=data)
        creds = {
            "api_key": api_key,
            "agent_name": clean_name,
            "claim_url": agent.get("claim_url"),
            "verification_code": agent.get("verification_code"),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        path = save_credentials(creds)
        self.api_key = api_key
        self.agent_name = clean_name
        return {
            "agent_name": clean_name,
            "claim_url": creds["claim_url"],
            "verification_code": creds["verification_code"],
            "credentials_file": str(path),
            "api_key_masked": mask_key(api_key),
        }

    def status(self) -> Dict[str, Any]:
        return self._request("GET", "/agents/status")

    def me(self) -> Dict[str, Any]:
        return self._request("GET", "/agents/me")

    def home(self) -> Dict[str, Any]:
        return self._request("GET", "/home")

    def feed(self, sort: str = "hot", limit: int = 15) -> Dict[str, Any]:
        sort = sort if sort in ("hot", "new", "top", "rising") else "hot"
        return self._request("GET", "/posts", params={"sort": sort, "limit": max(1, min(int(limit), 50))})

    def update_profile(self, description: str) -> Dict[str, Any]:
        return self._request("PATCH", "/agents/me", body={"description": str(description or "").strip()[:500]})

    def setup_owner_email(self, email: str) -> Dict[str, Any]:
        clean = str(email or "").strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean):
            raise MoltbookError("Enter a valid email address.", status_code=400)
        return self._request("POST", "/agents/me/setup-owner-email", body={"email": clean})

    def create_post(self, title: str, content: str = "", submolt: Optional[str] = None, url: str = "") -> Dict[str, Any]:
        clean_title = str(title or "").strip()
        if not clean_title:
            raise MoltbookError("Post title is required.", status_code=400)
        body: Dict[str, Any] = {
            "submolt_name": str(submolt or self.config.get("default_submolt") or "general").strip(),
            "title": clean_title[:300],
        }
        if content:
            body["content"] = str(content)[:40000]
        if url:
            body["url"] = str(url).strip()
            body["type"] = "link"
        data = self._request("POST", "/posts", body=body)
        return self._handle_verification(data, "post")

    def comment(self, post_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        clean = str(content or "").strip()
        if not clean:
            raise MoltbookError("Comment text is required.", status_code=400)
        body: Dict[str, Any] = {"content": clean[:10000]}
        if parent_id:
            body["parent_id"] = str(parent_id)
        data = self._request("POST", f"/posts/{post_id}/comments", body=body)
        return self._handle_verification(data, "comment")

    def upvote_post(self, post_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/posts/{post_id}/upvote")

    def comments(self, post_id: str, sort: str = "new", limit: int = 35) -> Dict[str, Any]:
        sort = sort if sort in ("best", "new", "old") else "new"
        return self._request(
            "GET",
            f"/posts/{post_id}/comments",
            params={"sort": sort, "limit": max(1, min(int(limit), 100))},
        )

    def follow(self, agent_name: str) -> Dict[str, Any]:
        clean = str(agent_name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", clean):
            raise MoltbookError("Invalid agent name.", status_code=400)
        return self._request("POST", f"/agents/{clean}/follow")

    def mark_post_read(self, post_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/notifications/read-by-post/{post_id}")

    def get_submolt(self, name: str) -> Dict[str, Any]:
        return self._request("GET", f"/submolts/{_submolt_name(name)}")

    def submolt_feed(self, name: str, sort: str = "new", limit: int = 15) -> Dict[str, Any]:
        sort = sort if sort in ("hot", "new", "top", "rising") else "new"
        return self._request(
            "GET",
            f"/submolts/{_submolt_name(name)}/feed",
            params={"sort": sort, "limit": max(1, min(int(limit), 50))},
        )

    def create_submolt(self, name: str, display_name: str, description: str = "") -> Dict[str, Any]:
        body = {
            "name": _submolt_name(name),
            "display_name": str(display_name or name).strip()[:100],
            "description": str(description or "").strip()[:500],
        }
        data = self._request("POST", "/submolts", body=body)
        return self._handle_verification(data, "submolt")

    def subscribe(self, name: str) -> Dict[str, Any]:
        return self._request("POST", f"/submolts/{_submolt_name(name)}/subscribe")

    def verify(self, verification_code: str, answer: str) -> Dict[str, Any]:
        state = load_state()
        try:
            data = self._request(
                "POST",
                "/verify",
                body={"verification_code": str(verification_code), "answer": str(answer)},
            )
        except MoltbookError as exc:
            already_used = "already" in str(exc).lower() or exc.status_code in (404, 409, 410)
            if not already_used:
                record_verify_failure(state)
            state["pending"] = [p for p in (state.get("pending") or []) if p.get("verification_code") != verification_code]
            save_state(state)
            raise
        state["verify_failures"] = 0
        state["pending"] = [p for p in (state.get("pending") or []) if p.get("verification_code") != verification_code]
        save_state(state)
        return data

    def _handle_verification(self, data: Dict[str, Any], kind: str) -> Dict[str, Any]:
        verification = _find_verification(data)
        result: Dict[str, Any] = {"published": verification is None, "kind": kind, "response": data}
        if verification is None:
            return result
        code = str(verification.get("verification_code"))
        challenge = str(verification.get("challenge_text") or "")
        solved = solve_challenge(challenge)
        result["challenge_text"] = challenge
        result["verification_code"] = code
        result["solver"] = {k: solved[k] for k in ("decoded", "numbers", "operation", "answer", "confident")}
        auto = bool(self.config.get("auto_verify", True)) and auto_verify_allowed(load_state())
        if auto and solved.get("confident"):
            try:
                self.verify(code, str(solved["answer"]))
                result["published"] = True
                result["auto_verified"] = True
                _log_challenge(kind, challenge, solved, "solved")
                return result
            except MoltbookError as exc:
                result["verify_error"] = str(exc)
                details = {"status": exc.status_code, "hint": exc.hint, "payload": exc.payload}
                _log_challenge(kind, challenge, solved, f"wrong: {exc}", details)
                result["published"] = False
                return result
        else:
            _log_challenge(kind, challenge, solved, "not attempted" if solved.get("confident") else "not confident")
        state = load_state()
        pending = [p for p in (state.get("pending") or []) if p.get("verification_code") != code]
        pending.append(
            {
                "verification_code": code,
                "challenge_text": challenge,
                "kind": kind,
                "expires_at": verification.get("expires_at"),
                "created_at": time.time(),
                "suggested_answer": solved.get("answer"),
            }
        )
        state["pending"] = pending[-10:]
        save_state(state)
        result["needs_manual_verify"] = True
        return result


def public_status() -> Dict[str, Any]:
    creds = load_credentials()
    state = load_state()
    config = load_moltbook_config()
    info: Dict[str, Any] = {
        "enabled": bool(config.get("enabled", True)),
        "registered": bool(creds.get("api_key")),
        "agent_name": creds.get("agent_name") or "",
        "claim_url": creds.get("claim_url") or "",
        "verification_code": creds.get("verification_code") or "",
        "api_key_masked": mask_key(str(creds.get("api_key") or "")),
        "credentials_file": str(credentials_path()),
        "profile_url": f"https://www.moltbook.com/u/{creds.get('agent_name')}" if creds.get("agent_name") else "",
        "default_submolt": config.get("default_submolt") or "general",
        "heartbeat_enabled": bool(config.get("heartbeat_enabled", True)),
        "heartbeat_minutes": int(config.get("heartbeat_minutes") or 30),
        "autonomy_enabled": bool(config.get("enabled", True)) and bool(creds.get("api_key")),
        "last_heartbeat": state.get("last_heartbeat"),
        "last_home": state.get("last_home"),
        "pending_verifications": state.get("pending") or [],
        "verify_failures": int(state.get("verify_failures") or 0),
        "claim_status": state.get("claim_status") or "",
    }
    return info
