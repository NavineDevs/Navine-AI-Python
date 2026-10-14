from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from navine.learn.gameplay import coach_advice, normalize_title, sessions_dir


@dataclass
class GameSession:
    session_id: str
    title: str
    slug: str
    mode: str
    client_id: str
    created: str
    path: Path
    events_path: Path
    commands_path: Path
    meta_path: Path
    pending_commands: List[Dict[str, Any]] = field(default_factory=list)
    event_count: int = 0
    last_event: Optional[str] = None
    websocket: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "slug": self.slug,
            "mode": self.mode,
            "client_id": self.client_id,
            "created": self.created,
            "event_count": self.event_count,
            "last_event": self.last_event,
            "pending_commands": len(self.pending_commands),
            "path": str(self.path),
        }


class GameBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, GameSession] = {}
        self._client_sessions: Dict[str, str] = {}
        self._ws_clients: Set[Any] = set()

    def create_session(
        self,
        title: str,
        mode: str = "learn",
        client_id: str = "",
        note: str = "",
    ) -> GameSession:
        slug = normalize_title(title)
        session_id = uuid.uuid4().hex[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        folder = sessions_dir() / f"bridge_{slug}_{stamp}_{session_id}"
        folder.mkdir(parents=True, exist_ok=True)
        session = GameSession(
            session_id=session_id,
            title=title or slug,
            slug=slug,
            mode=(mode or "learn").strip().lower(),
            client_id=client_id or session_id,
            created=datetime.now(timezone.utc).isoformat(),
            path=folder,
            events_path=folder / "events.jsonl",
            commands_path=folder / "commands.jsonl",
            meta_path=folder / "session.json",
        )
        meta = {
            "session_id": session_id,
            "title": session.title,
            "slug": session.slug,
            "mode": session.mode,
            "client_id": session.client_id,
            "note": note,
            "created": session.created,
            "bridge": "navine/v1",
        }
        session.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with self._lock:
            self._sessions[session_id] = session
            self._client_sessions[session.client_id] = session_id
        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def find_session(self, session_id: str = "", client_id: str = "") -> Optional[GameSession]:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            if client_id and client_id in self._client_sessions:
                sid = self._client_sessions[client_id]
                return self._sessions.get(sid)
        if session_id:
            return self._load_session_from_disk(session_id)
        return None

    def _load_session_from_disk(self, session_id: str) -> Optional[GameSession]:
        for folder in sessions_dir().glob(f"bridge_*_{session_id}"):
            meta_path = folder / "session.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            session = GameSession(
                session_id=session_id,
                title=str(meta.get("title") or "game"),
                slug=str(meta.get("slug") or "unknown"),
                mode=str(meta.get("mode") or "learn"),
                client_id=str(meta.get("client_id") or session_id),
                created=str(meta.get("created") or ""),
                path=folder,
                events_path=folder / "events.jsonl",
                commands_path=folder / "commands.jsonl",
                meta_path=meta_path,
            )
            if session.events_path.exists():
                session.event_count = sum(1 for _ in session.events_path.open("r", encoding="utf-8"))
            with self._lock:
                self._sessions[session_id] = session
                self._client_sessions[session.client_id] = session_id
            return session
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            items = [s.to_dict() for s in self._sessions.values()]
        if len(items) < limit:
            seen = {i["session_id"] for i in items}
            for folder in sorted(sessions_dir().glob("bridge_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                parts = folder.name.split("_")
                if len(parts) < 2:
                    continue
                sid = parts[-1]
                if sid in seen:
                    continue
                loaded = self._load_session_from_disk(sid)
                if loaded:
                    items.append(loaded.to_dict())
                    seen.add(sid)
                if len(items) >= limit:
                    break
        return items[:limit]

    def ingest_event(
        self,
        session_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        client_id: str = "",
    ) -> Dict[str, Any]:
        session = self.find_session(session_id=session_id, client_id=client_id)
        if not session:
            session = self.create_session(title=str((payload or {}).get("title") or "unknown"), client_id=client_id)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": (event_type or "state").strip().lower(),
            "payload": dict(payload or {}),
        }
        with session.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        session.event_count += 1
        session.last_event = row["type"]
        advice = self._coach_from_event(session, row)
        commands = self._plan_commands(session, row, advice)
        for cmd in commands:
            self.queue_command(session.session_id, cmd, source="bridge")
        return {
            "ok": True,
            "session_id": session.session_id,
            "event_type": row["type"],
            "event_count": session.event_count,
            "advice": advice,
            "commands": commands,
        }

    def queue_command(
        self,
        session_id: str,
        command: Dict[str, Any],
        source: str = "api",
    ) -> Dict[str, Any]:
        session = self.find_session(session_id=session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "command": command,
        }
        with session.commands_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        session.pending_commands.append(command)
        return {"ok": True, "queued": command}

    def pull_commands(self, session_id: str, client_id: str = "", max_items: int = 8) -> Dict[str, Any]:
        session = self.find_session(session_id=session_id, client_id=client_id)
        if not session:
            return {"ok": False, "error": "Session not found", "commands": []}
        max_items = max(1, min(int(max_items), 32))
        batch = session.pending_commands[:max_items]
        session.pending_commands = session.pending_commands[max_items:]
        return {"ok": True, "session_id": session.session_id, "commands": batch}

    def register_websocket(self, session: GameSession, websocket: Any) -> None:
        session.websocket = websocket
        self._ws_clients.add(websocket)

    def unregister_websocket(self, websocket: Any) -> None:
        self._ws_clients.discard(websocket)
        with self._lock:
            for session in self._sessions.values():
                if session.websocket is websocket:
                    session.websocket = None

    async def _push_ws(self, websocket: Any, payload: Dict[str, Any]) -> None:
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    def _coach_from_event(self, session: GameSession, row: Dict[str, Any]) -> str:
        from navine.learn.gameplay import pvp_advice, speedrun_advice

        etype = row.get("type") or ""
        payload = row.get("payload") or {}
        note = str(payload.get("note") or payload.get("message") or "")
        if session.mode == "pvp" or etype.startswith("pvp"):
            return pvp_advice(session.title, payload)
        if session.mode == "speedrun" or etype.startswith("speedrun"):
            return speedrun_advice(session.title, payload)
        if etype == "death":
            return f"You died in {session.title}. Reset positioning and review the last 10 seconds."
        if etype == "elimination":
            return f"Elimination logged in {session.title}. Note cover, aim, and third-party timing."
        return coach_advice(session.title, note)

    def _plan_commands(self, session: GameSession, row: Dict[str, Any], advice: str) -> List[Dict[str, Any]]:
        if session.mode not in ("agent", "play", "auto"):
            return []
        payload = row.get("payload") or {}
        etype = row.get("type") or ""
        commands: List[Dict[str, Any]] = []
        if etype == "chat" and advice:
            commands.append({"action": "say", "text": advice[:180]})
        health = payload.get("health")
        if isinstance(health, (int, float)) and health < 6:
            commands.append({"action": "use", "item": "heal"})
        if payload.get("enemy_near") and session.slug in ("fortnite", "fps", "minecraft"):
            commands.append({"action": "keys", "keys": ["sprint", "crouch"]})
        if etype == "speedrun_split" and payload.get("segment"):
            commands.append({"action": "notify", "text": f"Split: {payload.get('segment')}"})
        return commands[:4]

    async def handle_ws_message(self, session: GameSession, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            await self._push_ws(session.websocket, {"type": "error", "error": "invalid_json"})
            return
        mtype = str(msg.get("type") or "event").lower()
        if mtype == "ping":
            await self._push_ws(session.websocket, {"type": "pong", "t": time.time()})
            return
        if mtype == "pull_commands":
            result = self.pull_commands(session.session_id)
            await self._push_ws(session.websocket, {"type": "commands", **result})
            return
        if mtype in ("event", "state", "frame", "pvp", "speedrun", "death", "elimination", "speedrun_split"):
            result = self.ingest_event(session.session_id, mtype, msg.get("payload") or msg, session.client_id)
            await self._push_ws(session.websocket, {"type": "ack", **result})
            for cmd in result.get("commands") or []:
                await self._push_ws(session.websocket, {"type": "command", "command": cmd})
            return
        await self._push_ws(session.websocket, {"type": "error", "error": f"unknown_type:{mtype}"})


_bridge: Optional[GameBridge] = None
_bridge_lock = threading.Lock()


def get_game_bridge() -> GameBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = GameBridge()
        return _bridge
