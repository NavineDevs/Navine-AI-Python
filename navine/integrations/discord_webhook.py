from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from navine.utils.paths import get_project_root

WEBHOOK_RE = re.compile(
    r"^https://(?:(?:ptb|canary)\.)?(?:discord(?:app)?\.com)/api/webhooks/\d+/[A-Za-z0-9_-]+(?:\?.*)?$",
    re.IGNORECASE,
)
WEBHOOK_FIND_RE = re.compile(
    r"https://(?:(?:ptb|canary)\.)?(?:discord(?:app)?\.com)/api/webhooks/\d+/[A-Za-z0-9_-]+(?:\?[^\s]*)?",
    re.IGNORECASE,
)
STORE_PATH = get_project_root() / "data" / "secrets" / "discord_webhook.json"


def is_discord_webhook_url(url: str) -> bool:
    cleaned = (url or "").strip().split()[0] if (url or "").strip() else ""
    return bool(WEBHOOK_RE.match(cleaned))


def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    match = WEBHOOK_FIND_RE.search(cleaned)
    return match.group(0).rstrip(".,);]") if match else cleaned.split()[0]


def _read_store() -> Dict[str, Any]:
    if not STORE_PATH.is_file():
        return {}
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_store(data: Dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_webhook_url(override: Optional[str] = None) -> str:
    if override and str(override).strip():
        return _normalize_url(str(override))
    env = (os.environ.get("NAVINE_DISCORD_WEBHOOK") or os.environ.get("DISCORD_WEBHOOK_URL") or "").strip()
    if env:
        return _normalize_url(env)
    stored = str((_read_store().get("url") or "")).strip()
    if stored:
        return _normalize_url(stored)
    try:
        from navine.utils.config import load_config

        cfg = load_config("navine")
        disc = cfg.get("discord") or {}
        return _normalize_url(str(disc.get("webhook_url") or ""))
    except Exception:
        return ""


def set_webhook_url(url: str) -> Dict[str, Any]:
    cleaned = _normalize_url(url or "")
    if cleaned and not is_discord_webhook_url(cleaned):
        return {
            "ok": False,
            "error": "URL must be a Discord webhook (https://discord.com/api/webhooks/ID/TOKEN)",
        }
    data = _read_store()
    if cleaned:
        data["url"] = cleaned
    else:
        data.pop("url", None)
    _write_store(data)
    return {"ok": True, "configured": bool(cleaned), "url_set": bool(cleaned)}


def webhook_status() -> Dict[str, Any]:
    url = get_webhook_url()
    return {
        "ok": True,
        "configured": bool(url),
        "valid": bool(url) and is_discord_webhook_url(url),
        "url_hint": (url[:48] + "...") if len(url) > 48 else url,
    }


def send_webhook(
    content: str,
    webhook_url: Optional[str] = None,
    username: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    url = get_webhook_url(webhook_url)
    if not url:
        return {
            "ok": False,
            "error": "No Discord webhook configured. Paste the URL in Settings → Discord webhook, then Save.",
        }
    if not is_discord_webhook_url(url):
        return {"ok": False, "error": "Configured Discord webhook URL is invalid"}
    text = (content or "").strip()
    if not text and not embeds:
        return {"ok": False, "error": "Message is empty"}
    if len(text) > 1900:
        text = text[:1900] + "..."
    payload: Dict[str, Any] = {"content": text or None}
    if username:
        payload["username"] = str(username)[:80]
    if embeds:
        payload["embeds"] = embeds[:10]
    body = json.dumps({k: v for k, v in payload.items() if v is not None}).encode("utf-8")
    post_url = url if "wait=" in url else (url + ("&" if "?" in url else "?") + "wait=true")
    req = request.Request(
        post_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "NavineAI-Webhook/1.1",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=25) as resp:
            code = int(getattr(resp, "status", 0) or 0)
            return {"ok": 200 <= code < 300, "status": code, "sent": True}
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            detail = str(exc)
        return {"ok": False, "status": int(exc.code), "error": detail or str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parse_webhook_request(text: str) -> Optional[Tuple[str, Optional[str]]]:
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    url_match = WEBHOOK_FIND_RE.search(raw)
    patterns = (
        r"^(?:send|post|push)\s+(?:this\s+)?(?:to\s+)?discord(?:\s+webhook)?\s*[:=-]?\s*(.+)$",
        r"^(?:discord\s+webhook|webhook)\s+(?:send|post)\s*[:=-]?\s*(.+)$",
        r"^webhook\s+[:=-]\s*(.+)$",
        r"^(?:message|msg)\s+(?:discord|webhook)\s*[:=-]?\s*(.+)$",
        r"^discord\s*[:=-]\s*(.+)$",
    )
    for pattern in patterns:
        m = re.match(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            msg = m.group(1).strip()
            url = url_match.group(0) if url_match else None
            if url:
                msg = msg.replace(url, "").strip(" :-")
            return msg, url
    if "discord webhook" in lowered and url_match:
        msg = raw
        url = url_match.group(0)
        msg = re.sub(re.escape(url), "", msg, flags=re.I).strip()
        msg = re.sub(
            r"(?:send|post|push)?\s*(?:to\s+)?discord(?:\s+webhook)?\s*[:=-]?",
            "",
            msg,
            flags=re.I,
        ).strip(" :-")
        return msg or "Ping from Navine AI - Python", url
    return None
