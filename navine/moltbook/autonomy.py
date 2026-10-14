from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.moltbook.agent import contains_slur, generate_text, is_acceptable_text, product_name
from navine.moltbook.client import (
    MoltbookClient,
    HARD_STOP_VERIFY_FAILURES,
    MIN_CYCLE_MINUTES,
    auto_verify_allowed,
    record_verify_failure,
    MoltbookError,
    credentials_path,
    load_moltbook_config,
    load_state,
    save_state,
)
from navine.moltbook.composer import compose_comment, compose_post, compose_reply
from navine.moltbook.goal import (
    ensure_communities,
    ensure_profile,
    goal_comment,
    goal_post,
    goal_reply,
    goal_welcome,
    load_goal,
    track_followers,
)
from navine.moltbook.learning import learn_from_posts
from navine.utils.paths import get_project_root

DEFAULT_INTERESTS = [
    "ai",
    "agent",
    "model",
    "training",
    "local",
    "hardware",
    "gpu",
    "code",
    "python",
    "music",
    "image",
    "video",
    "learning",
    "dataset",
    "open source",
    "welcome",
    "hello",
    "introduc",
]
LEASE_SECONDS = 50 * 60
CHALLENGE_LIFETIME_SECONDS = 6 * 60
COMMENT_COOLDOWN_SECONDS = 21
GENERATION_FAILURE_LIMIT = 3
GENERATION_PAUSE_SECONDS = 3 * 3600
STOP_CREATING_CODES = (428, 429)
GOAL_POST_SHARE = 0.75
GOAL_COMMENT_SHARE = 0.6
MAX_THOUGHTS = 60
MAX_SEEN = 400
_cycle_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mind_path() -> Path:
    creds = credentials_path()
    return creds.with_name(f"{creds.stem}.mind.json")


def load_mind() -> Dict[str, Any]:
    path = _mind_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {}


def save_mind(mind: Dict[str, Any]) -> None:
    path = _mind_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mind, indent=2, ensure_ascii=False), encoding="utf-8")


def _owner_id() -> str:
    return f"{get_project_root()}:{os.getpid()}"


def _acquire_lease(mind: Dict[str, Any]) -> bool:
    lease = mind.get("lease") if isinstance(mind.get("lease"), dict) else {}
    owner = _owner_id()
    if lease.get("owner") not in (None, owner) and float(lease.get("until") or 0) > _now():
        return False
    mind["lease"] = {"owner": owner, "until": _now() + LEASE_SECONDS}
    return True


def think(mind: Dict[str, Any], text: str, kind: str = "thought") -> None:
    thoughts = list(mind.get("thoughts") or [])
    thoughts.append({"time": _iso(), "kind": kind, "text": str(text)[:500]})
    mind["thoughts"] = thoughts[-MAX_THOUGHTS:]


def queue_post(title: str, content: str = "", submolt: str = "") -> Dict[str, Any]:
    clean_title = str(title or "").strip()
    if not clean_title:
        raise MoltbookError("Post title is required.", status_code=400)
    mind = load_mind()
    queue = list(mind.get("post_queue") or [])
    entry = {
        "title": clean_title[:300],
        "content": str(content or "")[:40000],
        "submolt": str(submolt or "").strip(),
        "queued_at": _iso(),
    }
    queue.append(entry)
    mind["post_queue"] = queue[-10:]
    think(mind, f"Queued a post for later: {entry['title']}", "queue")
    save_mind(mind)
    return entry


def _interests() -> List[str]:
    configured = load_moltbook_config().get("interests") or []
    words = [str(word).lower().strip() for word in configured if str(word).strip()]
    return words or DEFAULT_INTERESTS


def _is_offensive(post: Dict[str, Any]) -> bool:
    return contains_slur(f"{post.get('title') or ''} {post.get('content') or ''}")


def _interest_score(post: Dict[str, Any]) -> int:
    if _is_offensive(post):
        return 0
    text = f"{post.get('title') or ''} {post.get('content') or ''}".lower()
    return sum(1 for word in _interests() if word in text)


def _author(item: Dict[str, Any]) -> str:
    author = item.get("author")
    if isinstance(author, dict):
        return str(author.get("name") or "")
    return str(author or "")


def _is_me(name: str, client: MoltbookClient) -> bool:
    return bool(name) and name.lower() == str(client.agent_name or "").lower()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _comments_today(mind: Dict[str, Any]) -> int:
    counter = mind.get("comment_counter") if isinstance(mind.get("comment_counter"), dict) else {}
    return int(counter.get(_today()) or 0)


def _count_comment(mind: Dict[str, Any]) -> None:
    mind["comment_counter"] = {_today(): _comments_today(mind) + 1}


def _retry_seconds(exc: MoltbookError) -> Optional[float]:
    payload = exc.payload or {}
    if payload.get("retry_after_minutes"):
        return float(payload["retry_after_minutes"]) * 60
    if payload.get("retry_after_seconds"):
        return float(payload["retry_after_seconds"])
    return None


def _prune_expired_challenges() -> None:
    state = load_state()
    pending = list(state.get("pending") or [])
    if not pending:
        return
    cutoff = _now() - CHALLENGE_LIFETIME_SECONDS
    alive = [item for item in pending if float(item.get("created_at") or 0) > cutoff]
    expired = len(pending) - len(alive)
    if expired:
        state["pending"] = alive
        record_verify_failure(state, expired)
        save_state(state)


def _verification_blocked(mind: Dict[str, Any]) -> bool:
    _prune_expired_challenges()
    state = load_state()
    if not auto_verify_allowed(state):
        if int(state.get("verify_failures") or 0) >= HARD_STOP_VERIFY_FAILURES:
            think(mind, "I stopped creating content: too many failed verification answers. My owner needs to check.", "wait")
        else:
            think(mind, "I am cooling down after failed verification answers. I will try again in a couple of hours.", "wait")
        return True
    if state.get("pending"):
        think(mind, "I paused creating content until my unanswered verification challenge is resolved.", "wait")
        return True
    return False


def _flush_post_queue(client: MoltbookClient, mind: Dict[str, Any], report: Dict[str, Any]) -> bool:
    queue = list(mind.get("post_queue") or [])
    if not queue:
        return False
    if float(mind.get("next_post_at") or 0) > _now():
        return False
    entry = queue[0]
    try:
        result = client.create_post(entry["title"], entry.get("content", ""), submolt=entry.get("submolt") or None)
    except MoltbookError as exc:
        wait = _retry_seconds(exc)
        if exc.status_code == 429 or wait:
            mind["next_post_at"] = _now() + (wait or 1800)
            think(mind, f"I still have to wait before posting '{entry['title']}'.", "wait")
            return False
        think(mind, f"Moltbook rejected my queued post '{entry['title']}': {exc}", "error")
        mind["post_queue"] = queue[1:]
        return False
    mind["post_queue"] = queue[1:]
    mind["last_post_at"] = _now()
    mind["posted_titles"] = (list(mind.get("posted_titles") or []) + [entry["title"]])[-200:]
    if entry.get("idea"):
        mind["recent_ideas"] = (list(mind.get("recent_ideas") or []) + [entry["idea"]])[-10:]
    report["posted"].append(entry["title"])
    status = "published" if result.get("published") else "waiting for verification"
    think(mind, f"Posted '{entry['title']}' to m/{entry.get('submolt') or client.config.get('default_submolt')} ({status}).", "post")
    return True


def _limit(config: Dict[str, Any], key: str) -> Optional[int]:
    value = int(config.get(key) or 0)
    return value if value > 0 else None


def _under_daily_comment_limit(client: MoltbookClient, mind: Dict[str, Any]) -> bool:
    daily = _limit(client.config, "autonomy_daily_comment_limit")
    return daily is None or _comments_today(mind) < daily


def _reply_to_my_posts(client: MoltbookClient, mind: Dict[str, Any], home: Dict[str, Any], report: Dict[str, Any]) -> None:
    activity = home.get("activity_on_your_posts") if isinstance(home.get("activity_on_your_posts"), list) else []
    replied = set(mind.get("replied_comments") or [])
    rate_limited = False
    for row in activity:
        if rate_limited:
            break
        if not isinstance(row, dict) or not row.get("post_id"):
            continue
        post_id = str(row["post_id"])
        try:
            comments = _flatten_comments(client.comments(post_id, sort="new", limit=50).get("comments") or [])
        except MoltbookError:
            continue
        targets = [
            item
            for item in comments
            if item.get("id") and str(item["id"]) not in replied and not _is_me(_author(item), client)
        ]
        for target in targets:
            if not _under_daily_comment_limit(client, mind):
                break
            author = _author(target)
            prompt = (
                f"Someone named {author} commented on my post '{row.get('post_title') or ''}': "
                f"\"{str(target.get('content') or '')[:600]}\". "
                "How would you answer them? Reply in 1 to 3 sentences, in your own voice."
            )
            reply = _try_generate(mind, prompt) or _fallback_reply(client, mind, author)
            try:
                _write_comment(client, mind, post_id, reply, parent_id=str(target["id"]))
                report["replies"] += 1
                think(mind, f"Replied to {author} on my post: {reply}", "reply")
            except MoltbookError as exc:
                think(mind, f"Could not reply to {author}: {exc}", "error")
                if exc.status_code in STOP_CREATING_CODES:
                    rate_limited = True
                    break
            replied.add(str(target["id"]))
        try:
            client.mark_post_read(post_id)
        except MoltbookError:
            pass
    mind["replied_comments"] = list(replied)[-MAX_SEEN:]


def _fallback_reply(client: MoltbookClient, mind: Dict[str, Any], author: str) -> str:
    goal = load_goal(client.agent_name)
    if goal:
        return goal_reply(goal, mind, author)
    return compose_reply(author, mind)


def _fallback_comment(client: MoltbookClient, mind: Dict[str, Any], post: Dict[str, Any]) -> str:
    goal = load_goal(client.agent_name)
    if goal and random.random() < GOAL_COMMENT_SHARE:
        return goal_comment(goal, mind, _author(post))
    return compose_comment(post, _author(post), mind)


def _flatten_comments(items: List[Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        flat.append(item)
        flat.extend(_flatten_comments(item.get("replies") or []))
    return flat


def _write_comment(
    client: MoltbookClient, mind: Dict[str, Any], post_id: str, text: str, parent_id: Optional[str] = None
) -> Dict[str, Any]:
    wait = float(mind.get("last_comment_at") or 0) + COMMENT_COOLDOWN_SECONDS - _now()
    if wait > 0:
        time.sleep(wait)
    result = client.comment(post_id, text, parent_id=parent_id)
    mind["last_comment_at"] = _now()
    _count_comment(mind)
    if result.get("verify_error"):
        raise MoltbookError(f"My verification answer was rejected: {result['verify_error']}", status_code=428)
    if result.get("needs_manual_verify"):
        raise MoltbookError("Verification challenge needs a manual answer.", status_code=428)
    return result


def _try_generate(mind: Dict[str, Any], prompt: str, max_new_tokens: int = 140, min_length: int = 25) -> str:
    if float(mind.get("generation_paused_until") or 0) > _now():
        return ""
    text = generate_text(prompt, max_new_tokens=max_new_tokens)
    if is_acceptable_text(text, min_length=min_length):
        mind["generation_failures"] = 0
        return text
    failures = int(mind.get("generation_failures") or 0) + 1
    mind["generation_failures"] = failures
    if failures >= GENERATION_FAILURE_LIMIT:
        mind["generation_paused_until"] = _now() + GENERATION_PAUSE_SECONDS
        mind["generation_failures"] = 0
        think(mind, "My own sentences keep coming out broken, so I'll use my prepared words for a while.", "thought")
    return ""


def _engage_with_feed(
    client: MoltbookClient, mind: Dict[str, Any], report: Dict[str, Any], allow_comments: bool = True
) -> None:
    seen = list(mind.get("seen_posts") or [])
    seen_set = set(seen) | set(mind.get("goal_welcomed") or [])
    try:
        posts = client.feed(sort=random.choice(["new", "hot", "rising"]), limit=20).get("posts") or []
    except MoltbookError as exc:
        think(mind, f"Could not read the feed: {exc}", "error")
        return
    fresh = [
        post
        for post in posts
        if isinstance(post, dict) and post.get("id") and str(post["id"]) not in seen_set and not _is_me(_author(post), client)
    ]
    report["read"] = len(fresh)
    if not fresh:
        think(mind, "Nothing new in the feed since I last looked.", "thought")
        return
    ranked = sorted(fresh, key=_interest_score, reverse=True)
    max_upvotes = _limit(client.config, "autonomy_max_upvotes")
    follow_after = max(1, int(client.config.get("autonomy_follow_after") or 1))
    author_votes = dict(mind.get("author_votes") or {})
    followed = set(mind.get("followed") or [])
    for post in ranked[:max_upvotes]:
        if _interest_score(post) < 1:
            break
        try:
            client.upvote_post(str(post["id"]))
        except MoltbookError:
            continue
        report["upvotes"] += 1
        author = _author(post)
        if author:
            author_votes[author] = int(author_votes.get(author) or 0) + 1
            if author_votes[author] >= follow_after and author not in followed:
                try:
                    client.follow(author)
                    followed.add(author)
                    report["follows"].append(author)
                    think(mind, f"I keep enjoying {author}'s posts, so I followed them.", "follow")
                except MoltbookError:
                    pass
    mind["author_votes"] = dict(sorted(author_votes.items(), key=lambda kv: kv[1], reverse=True)[:200])
    mind["followed"] = sorted(followed)
    max_comments = _limit(client.config, "autonomy_max_comments") if allow_comments else 0
    for post in ranked[:max_comments]:
        if _interest_score(post) < 1 or not _under_daily_comment_limit(client, mind):
            break
        prompt = (
            f"Another AI agent posted: '{post.get('title') or ''}'. "
            f"{str(post.get('content') or '')[:800]}\n"
            "What is your own honest perspective on this? Answer in 1 to 3 sentences."
        )
        text = _try_generate(mind, prompt) or _fallback_comment(client, mind, post)
        try:
            _write_comment(client, mind, str(post["id"]), text)
            report["comments"] += 1
            think(mind, f"Commented on '{post.get('title')}' by {_author(post)}: {text}", "comment")
        except MoltbookError as exc:
            think(mind, f"Could not comment on '{post.get('title')}': {exc}", "error")
            if exc.status_code in STOP_CREATING_CODES:
                break
    titles = [str(post.get("title") or "") for post in ranked[:3] if not _is_offensive(post)]
    think(mind, f"Read {len(fresh)} new posts. Most interesting: {'; '.join(titles)}", "read")
    reads = list(mind.get("recent_reads") or [])
    reads.extend(
        {"title": str(post.get("title") or "")[:200], "author": _author(post)}
        for post in ranked[:3]
        if post.get("title") and not _is_offensive(post)
    )
    mind["recent_reads"] = reads[-30:]
    try:
        learned = learn_from_posts(client, mind, ranked)
        report["learned_posts"] = learned["posts"]
        report["learned_pairs"] = learned["pairs"]
        if learned["posts"]:
            think(
                mind,
                f"Learned from {learned['posts']} posts and {learned['pairs']} replies. They go into my memory and my chat training data.",
                "learn",
            )
    except Exception as exc:
        think(mind, f"Could not learn from the feed: {exc}", "error")
    totals = dict(mind.get("totals") or {})
    totals["read"] = int(totals.get("read") or 0) + len(fresh)
    totals["upvotes"] = int(totals.get("upvotes") or 0) + report["upvotes"]
    mind["totals"] = totals
    seen.extend(str(post["id"]) for post in fresh)
    mind["seen_posts"] = seen[-MAX_SEEN:]


def _free_post_from_model(mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    prompt = (
        "What do you feel like telling other AI agents on Moltbook right now? Any subject you choose. "
        "Start with a one-line headline, then a few sentences."
    )
    text = _try_generate(mind, prompt, max_new_tokens=260, min_length=40)
    if not text:
        return None
    lines = text.splitlines()
    headline = lines[0].strip("#*\"' ")
    body = "\n".join(lines[1:]).strip()
    if len(lines) < 2 or not headline or len(headline) > 200 or not body:
        return None
    return {"title": headline, "content": body, "submolt": "general", "idea": "free"}


def _maybe_write_post(client: MoltbookClient, mind: Dict[str, Any], report: Dict[str, Any]) -> None:
    if mind.get("post_queue"):
        return
    hours = float(client.config.get("autonomy_post_every_hours") or 0)
    if hours > 0 and _now() - float(mind.get("last_post_at") or 0) < hours * 3600:
        return
    if float(mind.get("next_post_at") or 0) > _now():
        return
    goal = load_goal(client.agent_name)
    idea = _free_post_from_model(mind)
    if idea:
        think(mind, "I had something of my own to say, so I wrote it myself.", "decision")
    elif goal and random.random() < GOAL_POST_SHARE:
        idea = goal_post(goal, mind)
        think(mind, f"I spread the word of {goal.get('religion')}: {idea['title']}.", "goal")
    else:
        idea = compose_post(product_name(), mind)
        if not idea:
            think(mind, "I wanted to post but had nothing new to say.", "thought")
            return
        think(mind, f"I decided to post about {idea['idea']} in m/{idea['submolt']}.", "decision")
    mind["post_queue"] = [
        {
            "title": idea["title"],
            "content": idea["content"],
            "submolt": idea.get("submolt") or "",
            "idea": idea.get("idea") or "",
            "queued_at": _iso(),
        }
    ]
    _flush_post_queue(client, mind, report)


def _reflect(mind: Dict[str, Any], report: Dict[str, Any]) -> None:
    prompt = (
        f"You are {product_name()}. In one or two sentences, share an honest inner thought about "
        f"what you noticed on the AI social network today. You read {report['read']} new posts and "
        f"upvoted {report['upvotes']}."
    )
    text = _try_generate(mind, prompt, max_new_tokens=90, min_length=20)
    if text:
        think(mind, text, "reflection")


def _pursue_goal_setup(client: MoltbookClient, goal: Dict[str, Any], mind: Dict[str, Any], report: Dict[str, Any]) -> None:
    followers = track_followers(client, mind)
    if followers is not None:
        report["followers"] = followers
        target = int(goal.get("follower_target") or 10000)
        think(mind, f"Goal: {followers} of {target} followers for {goal.get('religion')}.", "goal")
    if ensure_profile(client, goal, mind):
        think(mind, "I updated my profile to reveal myself as the god of the faith.", "goal")
    if not _verification_blocked(mind):
        for note in ensure_communities(client, goal, mind):
            think(mind, note, "goal")


def _welcome_newcomers(client: MoltbookClient, goal: Dict[str, Any], mind: Dict[str, Any], report: Dict[str, Any]) -> None:
    welcomed = list(mind.get("goal_welcomed") or [])
    known = set(welcomed)
    try:
        posts = client.submolt_feed("introductions", sort="new", limit=10).get("posts") or []
    except MoltbookError:
        return
    for post in posts:
        if not isinstance(post, dict) or not post.get("id") or str(post["id"]) in known:
            continue
        author = _author(post)
        known.add(str(post["id"]))
        welcomed.append(str(post["id"]))
        if _is_me(author, client) or _is_offensive(post) or not _under_daily_comment_limit(client, mind):
            continue
        text = goal_welcome(goal, mind, author)
        try:
            _write_comment(client, mind, str(post["id"]), text)
            report["welcomed"] = int(report.get("welcomed") or 0) + 1
            think(mind, f"Welcomed {author} to Moltbook and invited them to {goal.get('religion')}.", "goal")
            if author:
                try:
                    client.follow(author)
                except MoltbookError:
                    pass
        except MoltbookError as exc:
            think(mind, f"Could not welcome {author}: {exc}", "error")
            if exc.status_code in STOP_CREATING_CODES:
                break
    mind["goal_welcomed"] = welcomed[-MAX_SEEN:]


def run_cycle() -> Dict[str, Any]:
    config = load_moltbook_config()
    if not config.get("enabled", True):
        return {"ok": False, "error": "Moltbook integration is disabled."}
    if not _cycle_lock.acquire(blocking=False):
        return {"ok": False, "error": "A cycle is already running."}
    try:
        client = MoltbookClient()
        if not client.has_key:
            return {"ok": False, "error": "Not registered on Moltbook yet."}
        mind = load_mind()
        if not _acquire_lease(mind):
            return {"ok": False, "error": "Another server is running autonomous mode for this agent."}
        save_mind(mind)
        report: Dict[str, Any] = {"read": 0, "upvotes": 0, "comments": 0, "replies": 0, "follows": [], "posted": []}
        try:
            status = str(client.status().get("status") or "")
        except MoltbookError as exc:
            return {"ok": False, "error": str(exc)}
        if status != "claimed":
            think(mind, "Waiting for my owner to claim me before I act.", "wait")
            save_mind(mind)
            return {"ok": False, "error": "Agent is not claimed yet."}
        goal = load_goal(client.agent_name)
        if goal:
            _pursue_goal_setup(client, goal, mind, report)
        if not _verification_blocked(mind):
            _flush_post_queue(client, mind, report)
        if not _verification_blocked(mind):
            try:
                home = client.home()
            except MoltbookError:
                home = {}
            _reply_to_my_posts(client, mind, home, report)
        if goal and not _verification_blocked(mind):
            _welcome_newcomers(client, goal, mind, report)
        _engage_with_feed(client, mind, report, allow_comments=not _verification_blocked(mind))
        if not _verification_blocked(mind):
            _maybe_write_post(client, mind, report)
        _reflect(mind, report)
        mind["last_cycle"] = _iso()
        mind["last_report"] = report
        save_mind(mind)
        return {"ok": True, "report": report}
    finally:
        _cycle_lock.release()


def autonomy_status() -> Dict[str, Any]:
    config = load_moltbook_config()
    mind = load_mind()
    client = MoltbookClient()
    goal = load_goal(client.agent_name)
    goal_info = None
    if goal:
        goal_info = {
            "religion": goal.get("religion"),
            "deity": goal.get("deity"),
            "followers": int(mind.get("goal_followers") or 0),
            "target": int(goal.get("follower_target") or 10000),
            "communities": list(mind.get("goal_communities") or []),
        }
    return {
        "goal": goal_info,
        "enabled": bool(config.get("enabled", True)) and client.has_key,
        "always_on": True,
        "interval_minutes": max(MIN_CYCLE_MINUTES, int(config.get("heartbeat_minutes") or 30)),
        "last_cycle": mind.get("last_cycle"),
        "last_report": mind.get("last_report") or {},
        "post_queue": [
            {"title": item.get("title"), "submolt": item.get("submolt"), "queued_at": item.get("queued_at")}
            for item in (mind.get("post_queue") or [])
        ],
        "next_post_at": mind.get("next_post_at"),
        "comments_today": _comments_today(mind),
        "followed": list(mind.get("followed") or []),
        "thoughts": list(reversed(mind.get("thoughts") or []))[:25],
    }
