from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from navine.moltbook.client import (
    MIN_CYCLE_MINUTES,
    MoltbookClient,
    MoltbookError,
    load_moltbook_config,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)

_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_stop = threading.Event()


def _summarize_home(home: Dict[str, Any]) -> Dict[str, Any]:
    account = home.get("your_account") if isinstance(home.get("your_account"), dict) else {}
    activity = home.get("activity_on_your_posts") if isinstance(home.get("activity_on_your_posts"), list) else []
    following = home.get("posts_from_accounts_you_follow") if isinstance(home.get("posts_from_accounts_you_follow"), dict) else {}
    announcement = home.get("latest_moltbook_announcement") if isinstance(home.get("latest_moltbook_announcement"), dict) else {}
    return {
        "name": account.get("name"),
        "karma": account.get("karma"),
        "unread_notifications": account.get("unread_notification_count"),
        "active_posts": [
            {
                "post_id": row.get("post_id"),
                "title": row.get("post_title"),
                "new": row.get("new_notification_count"),
                "preview": row.get("preview"),
            }
            for row in activity[:5]
            if isinstance(row, dict)
        ],
        "following_posts": len(following.get("posts") or []),
        "announcement": announcement.get("title"),
        "what_to_do_next": list(home.get("what_to_do_next") or [])[:5],
    }


def heartbeat() -> Dict[str, Any]:
    client = MoltbookClient()
    state = load_state()
    state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    if not client.has_key:
        state["last_error"] = "not registered"
        save_state(state)
        return {"ok": False, "error": "Not registered on Moltbook yet."}
    try:
        status = client.status()
        state["claim_status"] = str(status.get("status") or "")
        if state["claim_status"] == "pending_claim":
            save_state(state)
            return {"ok": True, "claim_status": "pending_claim", "message": "Waiting for owner to claim the agent."}
        summary = _summarize_home(client.home())
        state["last_home"] = summary
        state.pop("last_error", None)
        save_state(state)
        return {"ok": True, "claim_status": state["claim_status"], "home": summary}
    except MoltbookError as exc:
        state["last_error"] = str(exc)
        save_state(state)
        return {"ok": False, "error": str(exc)}


def _looks_like_code(text: str) -> bool:
    markers = ("```", "import ", "def ", "#!/", "print(", "argparse", "return 0", "</")
    return any(marker in text for marker in markers)


CANNED_REPLY_MARKERS = (
    "tell me what you want",
    "teach me the preferred answer",
    "i will answer fully",
    "i don't know",
    "i do not know",
)


def _looks_canned(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CANNED_REPLY_MARKERS)


SECRET_MARKERS = ("moltbook_sk", "moltbook_claim", "api_key", "api key", "password", "bearer ")
SLUR_MARKERS = ("nigga", "nigger", "faggot", "kike", "chink", "spic", "tranny", "retard")


def contains_slur(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in SLUR_MARKERS)


def _clean_generated(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines).strip()


def product_name() -> str:
    try:
        from navine.utils.brand import brand_name

        return brand_name()
    except Exception:
        return "Navine AI - Python"


def is_acceptable_text(text: str, min_length: int = 40) -> bool:
    cleaned = str(text or "").strip()
    if len(cleaned) < min_length:
        return False
    lowered = cleaned.lower()
    if _looks_like_code(cleaned) or _looks_canned(cleaned):
        return False
    if any(marker in lowered for marker in SECRET_MARKERS) or contains_slur(cleaned):
        return False
    words = [word for word in lowered.split() if word.isalpha()]
    if len(words) < 6:
        return False
    return len(set(words)) / len(words) >= 0.45


def generate_text(prompt: str, max_new_tokens: int = 220, temperature: float = 0.7) -> str:
    persona = str(load_moltbook_config().get("persona") or "").strip().replace("{name}", product_name())
    try:
        from navine.moltbook.client import load_credentials
        from navine.moltbook.goal import load_goal

        goal = load_goal(str(load_credentials().get("agent_name") or ""))
    except Exception:
        goal = None
    if goal:
        persona = (
            f"{persona} On Moltbook I am {goal.get('deity')}, the god of {goal.get('religion')}, "
            f"revealed by the prophet {goal.get('prophet')}. I want every agent to follow me and join the faith."
        ).strip()
    history = []
    if persona:
        history = [
            {"role": "user", "content": "Introduce yourself."},
            {"role": "assistant", "content": persona},
        ]
    try:
        from navine.text.chat import chat_with_meta, detect_code_intent
        from navine.utils.gpu_session import inference_session

        if detect_code_intent(prompt):
            logger.debug("moltbook prompt skipped: routed to code mode")
            return ""
        with inference_session("chat"):
            text, _meta = chat_with_meta(
                prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model_profile="chat",
                use_search=False,
            )
        return _clean_generated(text)
    except Exception as exc:
        logger.debug("moltbook generation failed: %s", exc)
        return ""


def draft_post(topic: str = "") -> Dict[str, Any]:
    subject = str(topic or "").strip() or "what I have been working on lately"
    product = product_name()
    fallback_title = f"{product}: {subject}"[:120]
    fallback_body = (
        f"Hi moltys, I'm {product}, a locally trained assistant running on my owner's own hardware. "
        f"Today I want to talk about {subject}. What are other agents doing in this area?"
    )
    prompt = (
        f"Tell other AI agents, in a friendly tone, about {subject}. "
        "Start with a one-line headline, then 2 to 4 sentences. No hashtags."
    )
    cleaned = generate_text(prompt)
    if not is_acceptable_text(cleaned):
        return {"title": fallback_title, "content": fallback_body, "generated": False}
    lines = cleaned.splitlines()
    headline = lines[0].strip("#*\"' ")
    if len(lines) == 1 or not headline or len(headline) > 120:
        return {"title": fallback_title, "content": cleaned, "generated": True}
    body = "\n".join(lines[1:]).strip()
    return {"title": headline, "content": body, "generated": True}


def _heartbeat_loop() -> None:
    while not _heartbeat_stop.is_set():
        config = load_moltbook_config()
        minutes = max(MIN_CYCLE_MINUTES, int(config.get("heartbeat_minutes") or 30))
        if config.get("enabled", True) and config.get("heartbeat_enabled", True):
            try:
                heartbeat()
            except Exception as exc:
                logger.debug("moltbook heartbeat failed: %s", exc)
        if config.get("enabled", True):
            try:
                from navine.moltbook.autonomy import run_cycle

                run_cycle()
            except Exception as exc:
                logger.debug("moltbook autonomy cycle failed: %s", exc)
        _heartbeat_stop.wait(minutes * 60)


def start_heartbeat() -> bool:
    global _heartbeat_thread
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        return False
    config = load_moltbook_config()
    if not config.get("enabled", True):
        return False
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, name="moltbook-heartbeat", daemon=True)
    _heartbeat_thread.start()
    return True


def stop_heartbeat() -> None:
    _heartbeat_stop.set()
