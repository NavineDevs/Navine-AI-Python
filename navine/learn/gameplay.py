import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_project_root

KNOWN_TITLES = {
    "fortnite": "fortnite",
    "mario": "mario",
    "super mario": "mario",
    "minecraft": "minecraft",
    "minecraft java": "minecraft",
    "minecraft bedrock": "minecraft",
    "valorant": "valorant",
    "roblox": "roblox",
    "gta": "gta",
    "zelda": "zelda",
    "pokemon": "pokemon",
    "rocket league": "rocket_league",
    "csgo": "fps",
    "counter strike": "fps",
    "call of duty": "fps",
    "apex": "fps",
    "league of legends": "moba",
    "platformer": "platformer",
    "racing": "racing",
}


def games_data_dir() -> Path:
    path = get_project_root() / "data" / "train" / "games"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sessions_dir() -> Path:
    path = games_data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_title(title: str) -> str:
    key = (title or "unknown").strip().lower()
    if not key:
        return "unknown"
    for needle, slug in KNOWN_TITLES.items():
        if needle in key:
            return slug
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
    return cleaned.strip("_")[:48] or "unknown"


def _grab_screen(path: Path) -> Dict[str, Any]:
    try:
        from PIL import ImageGrab

        image = ImageGrab.grab()
        image.save(path, format="PNG")
        return {"ok": True, "path": str(path), "size": image.size}
    except Exception:
        pass
    try:
        import pyautogui

        shot = pyautogui.screenshot()
        shot.save(str(path))
        return {"ok": True, "path": str(path), "size": getattr(shot, "size", None)}
    except Exception as exc:
        return {"ok": False, "error": f"{exc}. Install pillow for screenshots."}


def strategy_tips(title: str) -> List[str]:
    slug = normalize_title(title)
    tips = {
        "fortnite": [
            "Land near edge POIs, loot fast, rotate early toward circle.",
            "Build cover before committing to a fight; disengage when low materials.",
            "Listen for footsteps, check roofs and height advantage before peaking.",
            "Keep loadout flexible: close range plus mid range, heals in bag.",
        ],
        "mario": [
            "Watch enemy patrol timing before jumping into gaps.",
            "Hold jump longer for higher arcs on platform gaps.",
            "Collect power-ups before boss rooms when available.",
            "Slow down near pits and moving platforms; short hops over shell hazards.",
        ],
        "minecraft": [
            "Secure food, tools, and a bed before exploring far.",
            "Light caves and mark exits to avoid getting lost.",
            "Fight one mob at a time; back-pedal while attacking.",
            "Keep a shield or high ground when fighting stronger enemies.",
        ],
        "fps": [
            "Pre-aim common angles and clear corners before running past.",
            "Use sound and mini-map; trade peeks with cover.",
            "Reload behind cover; do not reload mid-swing.",
            "Crosshair at head level while moving between cover.",
        ],
        "platformer": [
            "Learn enemy cycles and jump buffers for consistent clears.",
            "Short hop for low ceilings; full jump for distant platforms.",
            "Do not rush spikes or moving platforms until pattern is clear.",
        ],
        "racing": [
            "Brake before turn apex then accelerate out.",
            "Stay near racing line; avoid walls to keep boost or speed.",
        ],
        "moba": [
            "Farm safely early; trade only with vision and wave advantage.",
            "Group for objectives; do not overextend alone without vision.",
        ],
    }
    return tips.get(slug, [
        "Observe patterns for a few seconds before acting.",
        "Prefer safe positioning, then look for openings.",
        "Practice short action sequences and review mistakes.",
        "Record sessions so the model can learn common situations.",
    ])


def coach_advice(title: str, note: str = "") -> str:
    tips = strategy_tips(title)
    focus = (note or "").strip()
    lead = f"For {title or 'this game'}: "
    body = " ".join(tips[:2])
    if focus:
        body = f"{body} Also note: {focus[:160]}"
    return lead + body


def pvp_advice(title: str, payload: Optional[Dict[str, Any]] = None) -> str:
    payload = payload or {}
    slug = normalize_title(title)
    kills = payload.get("kills")
    deaths = payload.get("deaths")
    health = payload.get("health")
    zone = payload.get("zone_phase") or payload.get("storm_phase")
    tips = []
    if slug == "minecraft":
        tips.extend([
            "Strafe while blocking; break line of sight with blocks.",
            "Use knockback timing and height advantage in crystal or sword fights.",
            "Keep pearls, gaps, and totems hotbar-ready before engaging.",
        ])
    elif slug == "fortnite":
        tips.extend([
            "Take high ground before pushing; edit or reset when cracked.",
            "Box for heals when tagged; avoid long open runs in late zones.",
            "Third-party only when both enemies are low or distracted.",
        ])
    else:
        tips.extend(strategy_tips(title)[:2])
    if isinstance(health, (int, float)) and health < 8:
        tips.append("Heal or reset before re-peeking.")
    if isinstance(deaths, int) and isinstance(kills, int) and deaths > kills:
        tips.append("Play safer angles; disengage after one bad trade.")
    if zone:
        tips.append(f"Rotate early for zone {zone}; do not heal in open.")
    return f"PVP ({title}): " + " ".join(tips[:3])


def speedrun_advice(title: str, payload: Optional[Dict[str, Any]] = None) -> str:
    payload = payload or {}
    slug = normalize_title(title)
    segment = str(payload.get("segment") or payload.get("split") or "")
    time_ms = payload.get("time_ms") or payload.get("split_ms")
    delta = payload.get("delta_ms") or payload.get("behind_ms")
    tips = []
    if slug == "minecraft":
        tips.extend([
            "Route by bed reset or portal timing; prep blocks and food before nether.",
            "Practice one segment at a time; reset on major mistake early.",
            "Keep F3 coords visible and mark practice splits in chat.",
        ])
    elif slug == "mario" or slug == "platformer":
        tips.extend([
            "Use consistent jump buffers and enemy cycle skips.",
            "Reset on missed cycle; do not grind a bad spawn.",
        ])
    else:
        tips.extend([
            "Split every major milestone and compare to gold.",
            "Reset early on run-ending mistakes; drill weak segments offline.",
        ])
    if segment:
        tips.insert(0, f"Segment {segment}:")
    if isinstance(delta, (int, float)) and delta > 0:
        tips.append(f"You are {int(delta)}ms behind pace; tighten the next segment.")
    if isinstance(time_ms, (int, float)):
        tips.append(f"Split time {int(time_ms)}ms logged for training.")
    return " ".join(tips[:4])


def ingest_game_event(
    title: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    mode: str = "learn",
) -> Dict[str, Any]:
    from navine.games.bridge import get_game_bridge

    bridge = get_game_bridge()
    if not session_id:
        session = bridge.create_session(title=title, mode=mode)
        session_id = session.session_id
    return bridge.ingest_event(session_id, event_type, payload, client_id=str((payload or {}).get("client_id") or ""))


def ingest_speedrun_split(
    title: str,
    segment: str,
    time_ms: int,
    delta_ms: int = 0,
    session_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "segment": segment,
        "time_ms": int(time_ms),
        "delta_ms": int(delta_ms),
        "title": title,
        **(extra or {}),
    }
    result = ingest_game_event(title, "speedrun_split", payload, session_id=session_id, mode="speedrun")
    splits_path = games_data_dir() / "speedrun_splits.jsonl"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "segment": segment,
        "time_ms": int(time_ms),
        "delta_ms": int(delta_ms),
        "session_id": result.get("session_id"),
    }
    with splits_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    block = (
        f"Human: speedrun split {segment} in {title} at {time_ms}ms\n"
        f"Assistant: {result.get('advice') or speedrun_advice(title, payload)}\n\n"
    )
    corpus = games_data_dir() / "speedrun_corpus.txt"
    with corpus.open("a", encoding="utf-8") as handle:
        handle.write(block)
    return result


def observe_game(
    title: str = "unknown",
    seconds: float = 8.0,
    fps: float = 1.0,
    note: str = "",
) -> Dict[str, Any]:
    seconds = max(1.0, min(float(seconds), 60.0))
    fps = max(0.2, min(float(fps), 4.0))
    interval = 1.0 / fps
    slug = normalize_title(title)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session = sessions_dir() / f"{slug}_{stamp}"
    session.mkdir(parents=True, exist_ok=True)
    frames: List[Dict[str, Any]] = []
    count = max(1, int(seconds * fps))
    started = time.time()
    for i in range(count):
        frame_path = session / f"frame_{i:03d}.png"
        grab = _grab_screen(frame_path)
        entry = {
            "index": i,
            "path": grab.get("path"),
            "ok": bool(grab.get("ok")),
            "error": grab.get("error"),
            "t": round(time.time() - started, 3),
        }
        frames.append(entry)
        if i + 1 < count:
            time.sleep(interval)
    meta = {
        "title": title,
        "slug": slug,
        "note": note,
        "seconds": seconds,
        "fps": fps,
        "frames": frames,
        "created": datetime.now(timezone.utc).isoformat(),
        "advice": coach_advice(title, note),
    }
    meta_path = session / "session.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ok_frames = sum(1 for f in frames if f.get("ok"))
    _append_learning_record(meta, session)
    return {
        "ok": ok_frames > 0,
        "session_dir": str(session),
        "frames_ok": ok_frames,
        "frames_total": len(frames),
        "title": title,
        "slug": slug,
        "text": (
            f"Recorded {ok_frames} frames for {title or 'game'}. "
            f"{meta['advice']}"
        ),
        "error": None if ok_frames else (frames[0].get("error") if frames else "No frames"),
    }


def label_last_action(action: str, title: str = "") -> Dict[str, Any]:
    sessions = sorted(sessions_dir().glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        return {"ok": False, "error": "No gameplay sessions yet. Say observe game first."}
    session = sessions[0]
    meta_path = session / "session.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    labels_path = session / "labels.jsonl"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "title": title or meta.get("title") or session.name,
    }
    with labels_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    dialogue = games_data_dir() / "action_labels.txt"
    block = (
        f"Human: In {row['title']} what did the player do?\n"
        f"Assistant: Player action: {action}\n\n"
    )
    with dialogue.open("a", encoding="utf-8") as handle:
        handle.write(block)
    return {
        "ok": True,
        "session_dir": str(session),
        "action": action,
        "text": f"Logged action for learning: {action}",
    }


def _append_learning_record(meta: Dict[str, Any], session: Path) -> None:
    corpus = games_data_dir() / "observe_log.txt"
    tips = strategy_tips(str(meta.get("title") or ""))
    lines = [
        f"Human: watch me play {meta.get('title')} for {meta.get('seconds')} seconds",
        f"Assistant: {meta.get('advice')}",
        f"Human: what should I practice in {meta.get('title')}?",
        f"Assistant: {tips[0]} {tips[1] if len(tips) > 1 else ''}",
        "",
    ]
    with corpus.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    captions = get_project_root() / "data" / "train" / "multimodal-text" / "game_frames.jsonl"
    captions.parent.mkdir(parents=True, exist_ok=True)
    with captions.open("a", encoding="utf-8") as handle:
        for frame in meta.get("frames") or []:
            if not frame.get("ok"):
                continue
            row = {
                "media": frame.get("path"),
                "type": "image",
                "caption": (
                    f"Gameplay frame from {meta.get('title') or 'a game'}. "
                    f"{meta.get('note') or 'Local session capture for training.'}"
                ),
            }
            handle.write(json.dumps(row) + "\n")
    session_note = session / "README.txt"
    session_note.write_text(
        "Gameplay observation session for local model training.\n"
        "Frames are screenshots labeled by game title and optional notes.\n",
        encoding="utf-8",
    )


def load_game_train_texts() -> List[str]:
    root = games_data_dir()
    texts: List[str] = []
    for name in (
        "strategy_corpus.txt",
        "observe_log.txt",
        "action_labels.txt",
        "platform_fps_moba.txt",
    ):
        path = root / name
        if path.exists():
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            if raw:
                chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
                texts.extend(chunks)
    for session in sessions_dir().glob("*/session.json"):
        try:
            meta = json.loads(session.read_text(encoding="utf-8"))
        except Exception:
            continue
        advice = meta.get("advice") or ""
        title = meta.get("title") or "game"
        if advice:
            texts.append(
                f"Human: how do I play {title} better?\nAssistant: {advice}"
            )
    for events_file in sessions_dir().glob("bridge_*/events.jsonl"):
        try:
            lines = events_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-20:]
        except Exception:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            etype = row.get("type") or "event"
            payload = row.get("payload") or {}
            title = str(payload.get("title") or events_file.parent.name)
            if etype.startswith("pvp") or payload.get("kills") is not None:
                advice = pvp_advice(title, payload)
            elif etype.startswith("speedrun") or payload.get("segment"):
                advice = speedrun_advice(title, payload)
            else:
                advice = coach_advice(title, str(payload.get("note") or ""))
            texts.append(f"Human: {etype} in {title}\nAssistant: {advice}")
    for path in (games_data_dir() / "speedrun_corpus.txt", games_data_dir() / "pvp_corpus.txt"):
        if path.exists():
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            if raw:
                texts.extend([c.strip() for c in raw.split("\n\n") if c.strip()])
    return texts


def send_keys(keys: str) -> Dict[str, Any]:
    sequence = [part.strip() for part in (keys or "").split(",") if part.strip()]
    if not sequence:
        return {"ok": False, "error": "No keys provided"}
    try:
        import pyautogui
    except Exception as exc:
        return {"ok": False, "error": f"{exc}. Install pyautogui for key casting."}
    try:
        for key in sequence[:12]:
            if "+" in key and not key.startswith("+"):
                parts = [p.strip() for p in key.split("+") if p.strip()]
                if len(parts) >= 2:
                    pyautogui.hotkey(*parts)
                    continue
            pyautogui.press(key)
            time.sleep(0.05)
        return {"ok": True, "keys": sequence[:12], "text": f"Sent keys: {', '.join(sequence[:12])}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
