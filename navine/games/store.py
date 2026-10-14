from typing import Any, Dict, Optional

_sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str = "default") -> Dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "active_game": None,
            "game_type": None,
            "freer": False,
            "difficulty": "normal",
        }
    return _sessions[session_id]


def set_active_game(
    session_id: str,
    game_type: str,
    game: Any,
    freer: bool = False,
    difficulty: str = "normal",
) -> None:
    session = get_session(session_id)
    session["active_game"] = game
    session["game_type"] = game_type
    session["freer"] = freer
    session["difficulty"] = difficulty


def clear_active_game(session_id: str) -> None:
    session = get_session(session_id)
    session["active_game"] = None
    session["game_type"] = None
    session["freer"] = False
    session["difficulty"] = "normal"


def get_active_game(session_id: str = "default") -> Optional[Any]:
    return get_session(session_id).get("active_game")


def get_game_modes(session_id: str = "default") -> Dict[str, Any]:
    session = get_session(session_id)
    return {
        "freer": bool(session.get("freer")),
        "difficulty": session.get("difficulty") or "normal",
        "game_type": session.get("game_type"),
    }
