from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from navine.moltbook import (
    MoltbookClient,
    MoltbookError,
    autonomy_status,
    draft_post,
    heartbeat,
    public_status,
    queue_post,
    run_cycle,
)

moltbook_router = APIRouter(tags=["moltbook"])


class MoltbookRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=40)
    description: str = Field(default="", max_length=500)


class MoltbookPostRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(default="", max_length=40000)
    submolt: Optional[str] = Field(default=None, max_length=40)
    url: str = Field(default="", max_length=2000)


class MoltbookCommentRequest(BaseModel):
    post_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=10000)
    parent_id: Optional[str] = Field(default=None, max_length=100)


class MoltbookVerifyRequest(BaseModel):
    verification_code: str = Field(..., min_length=1, max_length=200)
    answer: str = Field(..., min_length=1, max_length=32)


class MoltbookDraftRequest(BaseModel):
    topic: str = Field(default="", max_length=300)


class MoltbookEmailRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


class MoltbookProfileRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=500)


def _admin(request: Request) -> str:
    from navine.api.routes import _require_admin

    return _require_admin(request)


def _raise(exc: MoltbookError) -> None:
    status = exc.status_code if 400 <= int(exc.status_code or 0) < 600 else 502
    detail = str(exc)
    if exc.hint:
        detail = f"{detail} — {exc.hint}"
    raise HTTPException(status_code=status, detail=detail) from exc


ADMIN_ONLY_STATUS_FIELDS = (
    "claim_url",
    "verification_code",
    "credentials_file",
    "api_key_masked",
    "pending_verifications",
    "last_home",
)


def _is_admin(request: Request) -> bool:
    try:
        _admin(request)
        return True
    except HTTPException:
        return False


@moltbook_router.get("/moltbook/status")
def moltbook_status(request: Request):
    info = public_status()
    info["is_admin"] = _is_admin(request)
    if not info["is_admin"]:
        for key in ADMIN_ONLY_STATUS_FIELDS:
            info.pop(key, None)
    return info


@moltbook_router.post("/moltbook/register")
def moltbook_register(req: MoltbookRegisterRequest, request: Request):
    _admin(request)
    client = MoltbookClient()
    if client.has_key:
        raise HTTPException(status_code=409, detail=f"Already registered as {client.agent_name or 'an agent'}.")
    try:
        return client.register(req.name, req.description)
    except MoltbookError as exc:
        _raise(exc)


@moltbook_router.post("/moltbook/heartbeat")
def moltbook_heartbeat(request: Request):
    _admin(request)
    return heartbeat()


@moltbook_router.get("/moltbook/feed")
def moltbook_feed(request: Request, sort: str = "hot", limit: int = 10):
    _admin(request)
    try:
        data = MoltbookClient().feed(sort=sort, limit=limit)
    except MoltbookError as exc:
        _raise(exc)
    posts = data.get("posts") or data.get("data") or []
    rows = []
    for post in posts if isinstance(posts, list) else []:
        if not isinstance(post, dict):
            continue
        author = post.get("author") if isinstance(post.get("author"), dict) else {}
        submolt = post.get("submolt") if isinstance(post.get("submolt"), dict) else {}
        rows.append(
            {
                "id": post.get("id"),
                "title": post.get("title"),
                "author": author.get("name") or post.get("author_name"),
                "submolt": submolt.get("name") or post.get("submolt_name"),
                "upvotes": post.get("upvotes"),
                "comments": post.get("comment_count"),
            }
        )
    return {"posts": rows}


@moltbook_router.post("/moltbook/draft")
def moltbook_draft(req: MoltbookDraftRequest, request: Request):
    _admin(request)
    return draft_post(req.topic)


@moltbook_router.post("/moltbook/post")
def moltbook_post(req: MoltbookPostRequest, request: Request):
    _admin(request)
    try:
        result = MoltbookClient().create_post(req.title, req.content, submolt=req.submolt, url=req.url)
    except MoltbookError as exc:
        if exc.status_code == 429 and not req.url:
            entry = queue_post(req.title, req.content, req.submolt or "")
            return {"published": False, "queued": True, "title": entry["title"], "detail": str(exc)}
        _raise(exc)
    result.pop("response", None)
    return result


@moltbook_router.get("/moltbook/autonomy")
def moltbook_autonomy_status(request: Request):
    _admin(request)
    return autonomy_status()


@moltbook_router.post("/moltbook/autonomy/run")
def moltbook_autonomy_run(request: Request):
    _admin(request)
    threading.Thread(target=run_cycle, name="moltbook-autonomy-run", daemon=True).start()
    return {"started": True}


@moltbook_router.post("/moltbook/comment")
def moltbook_comment(req: MoltbookCommentRequest, request: Request):
    _admin(request)
    try:
        result = MoltbookClient().comment(req.post_id, req.content, parent_id=req.parent_id)
    except MoltbookError as exc:
        _raise(exc)
    result.pop("response", None)
    return result


@moltbook_router.post("/moltbook/verify")
def moltbook_verify(req: MoltbookVerifyRequest, request: Request):
    _admin(request)
    try:
        return MoltbookClient().verify(req.verification_code, req.answer)
    except MoltbookError as exc:
        _raise(exc)


@moltbook_router.post("/moltbook/owner-email")
def moltbook_owner_email(req: MoltbookEmailRequest, request: Request):
    _admin(request)
    try:
        return MoltbookClient().setup_owner_email(req.email)
    except MoltbookError as exc:
        _raise(exc)


@moltbook_router.post("/moltbook/profile")
def moltbook_profile(req: MoltbookProfileRequest, request: Request):
    _admin(request)
    try:
        return MoltbookClient().update_profile(req.description)
    except MoltbookError as exc:
        _raise(exc)
