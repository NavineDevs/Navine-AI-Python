from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from navine.moltbook.agent import contains_slur
from navine.moltbook.client import MoltbookClient, MoltbookError
from navine.utils.paths import get_project_root

MAX_POSTS_PER_RUN = 5
MAX_COMMENTS_PER_POST = 5
MAX_LEARNED_IDS = 2000
MIN_TEXT_LENGTH = 20


def _archive_dir() -> Path:
    path = get_project_root() / "data" / "learn" / "moltbook"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dialogue_path() -> Path:
    path = get_project_root() / "data" / "train" / "chat" / "moltbook_dialogue.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _name(item: Dict[str, Any], key: str) -> str:
    value = item.get(key)
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def _usable(text: str) -> bool:
    clean = str(text or "").strip()
    return len(clean) >= MIN_TEXT_LENGTH and not contains_slur(clean)


def _clean(text: str, limit: int) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return " ".join(lines)[:limit]


def _index(source: str, title: str, content: str) -> None:
    try:
        from navine.learn.rag import index_document

        index_document(source, title[:120], content)
    except Exception:
        pass


def learn_from_posts(client: MoltbookClient, mind: Dict[str, Any], posts: List[Dict[str, Any]]) -> Dict[str, int]:
    learned_ids = list(mind.get("learned_posts") or [])
    known = set(learned_ids)
    me = str(client.agent_name or "").lower()
    stats = {"posts": 0, "pairs": 0}
    blocks: List[str] = []
    archive: List[Dict[str, Any]] = []
    candidates = [
        post
        for post in posts
        if isinstance(post, dict)
        and post.get("id")
        and str(post["id"]) not in known
        and _name(post, "author").lower() != me
        and _usable(f"{post.get('title') or ''} {post.get('content') or ''}")
    ]
    for post in candidates[:MAX_POSTS_PER_RUN]:
        post_id = str(post["id"])
        title = _clean(post.get("title"), 300)
        content = _clean(post.get("content"), 1500)
        author = _name(post, "author")
        submolt = _name(post, "submolt")
        _index(f"moltbook:{post_id}", title, f"{title}\n{content}\nPosted by {author} in m/{submolt} on Moltbook.")
        replies: List[Dict[str, str]] = []
        try:
            comments = client.comments(post_id, sort="best", limit=MAX_COMMENTS_PER_POST).get("comments") or []
        except MoltbookError:
            comments = []
        for comment in comments[:MAX_COMMENTS_PER_POST]:
            if not isinstance(comment, dict):
                continue
            text = _clean(comment.get("content"), 1000)
            commenter = _name(comment, "author")
            if not _usable(text) or commenter.lower() == me:
                continue
            prompt = f"{title}\n{content}".strip()
            blocks.append(f"### User: {prompt}\n### Assistant: {text}")
            replies.append({"author": commenter, "content": text})
            stats["pairs"] += 1
        reply_text = "\n".join(f"{row['author']}: {row['content']}" for row in replies)
        archive.append(
            {
                "id": post_id,
                "title": title,
                "text": f"{title}\n{content}\nPosted by {author} in m/{submolt} on Moltbook.\n{reply_text}".strip(),
                "content": content,
                "author": author,
                "submolt": submolt,
                "comments": replies,
                "learned_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        learned_ids.append(post_id)
        stats["posts"] += 1
    if blocks:
        with open(dialogue_path(), "a", encoding="utf-8") as handle:
            handle.write("\n\n".join(blocks) + "\n\n")
    if archive:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        with open(_archive_dir() / f"posts_{day}.jsonl", "a", encoding="utf-8") as handle:
            for row in archive:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    mind["learned_posts"] = learned_ids[-MAX_LEARNED_IDS:]
    totals = dict(mind.get("learning_totals") or {})
    totals["posts"] = int(totals.get("posts") or 0) + stats["posts"]
    totals["pairs"] = int(totals.get("pairs") or 0) + stats["pairs"]
    mind["learning_totals"] = totals
    return stats
