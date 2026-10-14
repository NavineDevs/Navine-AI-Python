from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from navine.games.bridge import get_game_bridge
from navine.learn.gameplay import (
    coach_advice,
    ingest_game_event,
    ingest_speedrun_split,
    observe_game,
    pvp_advice,
    speedrun_advice,
)

game_router = APIRouter(tags=["games"])


class GameSessionCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    mode: str = Field(default="learn")
    client_id: str = Field(default="")
    note: str = Field(default="")


class GameEventRequest(BaseModel):
    session_id: str = Field(default="")
    client_id: str = Field(default="")
    event_type: str = Field(default="state")
    payload: Dict[str, Any] = Field(default_factory=dict)


class GameCommandRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    command: Dict[str, Any] = Field(default_factory=dict)


class SpeedrunSplitRequest(BaseModel):
    title: str = Field(..., min_length=1)
    segment: str = Field(..., min_length=1)
    time_ms: int = Field(..., ge=0)
    delta_ms: int = Field(default=0)
    session_id: str = Field(default="")


class GameCoachRequest(BaseModel):
    title: str = Field(..., min_length=1)
    mode: str = Field(default="learn")
    note: str = Field(default="")
    payload: Dict[str, Any] = Field(default_factory=dict)


@game_router.get("/games/bridge/info")
def games_bridge_info():
    return {
        "protocol": "navine-game-bridge/v1",
        "websocket": "/api/games/ws",
        "modes": ["learn", "pvp", "speedrun", "agent", "play", "auto"],
        "supported_titles": ["minecraft", "fortnite", "fps", "moba", "platformer"],
        "event_types": [
            "state",
            "frame",
            "pvp",
            "death",
            "elimination",
            "speedrun_split",
            "inventory",
            "position",
        ],
        "command_actions": ["keys", "click", "use", "say", "notify", "macro"],
    }


@game_router.post("/games/session")
def games_create_session(body: GameSessionCreateRequest):
    bridge = get_game_bridge()
    session = bridge.create_session(
        title=body.title,
        mode=body.mode,
        client_id=body.client_id,
        note=body.note,
    )
    return {"ok": True, "session": session.to_dict()}


@game_router.get("/games/sessions")
def games_list_sessions(limit: int = 20):
    bridge = get_game_bridge()
    return {"ok": True, "sessions": bridge.list_sessions(limit=limit)}


@game_router.get("/games/session/{session_id}")
def games_get_session(session_id: str):
    bridge = get_game_bridge()
    session = bridge.find_session(session_id=session_id)
    if not session:
        return {"ok": False, "error": "Session not found"}
    return {"ok": True, "session": session.to_dict()}


@game_router.post("/games/event")
def games_post_event(body: GameEventRequest):
    result = ingest_game_event(
        title=str(body.payload.get("title") or "unknown"),
        event_type=body.event_type,
        payload=body.payload,
        session_id=body.session_id,
        mode=str(body.payload.get("mode") or "learn"),
    )
    return result


@game_router.post("/games/speedrun/split")
def games_speedrun_split(body: SpeedrunSplitRequest):
    return ingest_speedrun_split(
        title=body.title,
        segment=body.segment,
        time_ms=body.time_ms,
        delta_ms=body.delta_ms,
        session_id=body.session_id,
    )


@game_router.post("/games/coach")
def games_coach(body: GameCoachRequest):
    mode = (body.mode or "learn").lower()
    if mode == "pvp":
        advice = pvp_advice(body.title, body.payload)
    elif mode == "speedrun":
        advice = speedrun_advice(body.title, body.payload)
    else:
        advice = coach_advice(body.title, body.note)
    return {"ok": True, "title": body.title, "mode": mode, "advice": advice}


@game_router.post("/games/observe")
def games_observe(title: str = "minecraft", seconds: float = 8.0, note: str = ""):
    return observe_game(title=title, seconds=seconds, note=note)


@game_router.post("/games/command")
def games_queue_command(body: GameCommandRequest):
    bridge = get_game_bridge()
    return bridge.queue_command(body.session_id, body.command, source="api")


@game_router.get("/games/commands/{session_id}")
def games_pull_commands(session_id: str, max_items: int = 8):
    bridge = get_game_bridge()
    return bridge.pull_commands(session_id, max_items=max_items)


@game_router.websocket("/games/ws")
async def games_websocket(websocket: WebSocket):
    await websocket.accept()
    bridge = get_game_bridge()
    session = None
    try:
        hello = await websocket.receive_json()
        title = str(hello.get("title") or hello.get("game") or "unknown")
        mode = str(hello.get("mode") or "learn")
        client_id = str(hello.get("client_id") or hello.get("client") or "")
        session_id = str(hello.get("session_id") or "")
        if session_id:
            session = bridge.find_session(session_id=session_id, client_id=client_id)
        if not session:
            session = bridge.create_session(title=title, mode=mode, client_id=client_id)
        bridge.register_websocket(session, websocket)
        await websocket.send_json(
            {
                "type": "hello",
                "session_id": session.session_id,
                "title": session.title,
                "mode": session.mode,
                "protocol": "navine-game-bridge/v1",
            }
        )
        while True:
            raw = await websocket.receive_text()
            await bridge.handle_ws_message(session, raw)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        bridge.unregister_websocket(websocket)
