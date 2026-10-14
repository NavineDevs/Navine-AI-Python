import re
from typing import Dict, Optional, Tuple

from navine.games.battleship import BattleshipGame
from navine.games.chess import CHESS_AVAILABLE, ChessGame
from navine.games.hangman import HangmanGame
from navine.games.rps import RPSGame
from navine.games.store import clear_active_game, get_active_game, set_active_game
from navine.games.tictactoe import TicTacToeGame
from navine.games.war import WarGame

START_PATTERNS = {
    "chess": re.compile(r"\b(?:play|start|let'?s\s+play)\s+chess\b|\bchess\s+game\b", re.I),
    "tictactoe": re.compile(
        r"\b(?:play|start|let'?s\s+play)\s+(?:tic[- ]?tac[- ]?toe|ttt|noughts?\s+and\s+crosses)\b",
        re.I,
    ),
    "hangman": re.compile(r"\b(?:play|start|let'?s\s+play)\s+hangman\b", re.I),
    "war": re.compile(r"\b(?:play|start|let'?s\s+play)\s+(?:war|card\s+war|wars)\b|\bwar\s+card\s+game\b", re.I),
    "battleship": re.compile(
        r"\b(?:play|start|let'?s\s+play)\s+(?:battleship|battle\s*ship|sea\s+battle)\b",
        re.I,
    ),
    "rps": re.compile(
        r"\b(?:play|start|let'?s\s+play)\s+(?:rps|rock\s*paper\s*scissors|roshambo)\b",
        re.I,
    ),
    "racing": re.compile(
        r"\b(?:play|start|let'?s\s+play|launch|open)\s+(?:racing|race|ai\s+racing|self[- ]?driving)\b|"
        r"\b(?:ai\s+racing\s+game|2d\s+racing|genetic\s+racing)\b",
        re.I,
    ),
}

STOP_PATTERNS = re.compile(
    r"\b(?:quit\s+game|stop\s+game|end\s+game|exit\s+game|forfeit|resign)\b",
    re.I,
)

HELP_PATTERNS = re.compile(
    r"\b(?:what\s+games|play\s+games|which\s+games|list\s+games|game\s+list|"
    r"games?\s+help|help\s+games?|available\s+games)\b",
    re.I,
)

FREER_PATTERN = re.compile(
    r"\b(?:anything\s+goes|no\s+rules|illegal\s+moves?(?:\s+ok)?|freestyle|"
    r"loose\s+rules|free\s*play|wild\s+mode|allow\s+illegal|no\s+strict\s+rules)\b",
    re.I,
)
EASY_PATTERN = re.compile(r"\b(?:easy|beginner|casual)\b", re.I)
HARD_PATTERN = re.compile(r"\b(?:hard|difficult|expert|tough)\b", re.I)
GO_FREER = re.compile(r"\b(?:go\s+freestyle|switch\s+to\s+freestyle|enable\s+freestyle|loose\s+rules\s+on)\b", re.I)
GO_STRICT = re.compile(r"\b(?:strict\s+rules|go\s+strict|normal\s+rules|disable\s+freestyle)\b", re.I)

GAME_FACTORY = {
    "chess": ChessGame,
    "tictactoe": TicTacToeGame,
    "hangman": HangmanGame,
    "war": WarGame,
    "battleship": BattleshipGame,
    "rps": RPSGame,
}

NAME_ALIASES = {
    "chess": "chess",
    "tic-tac-toe": "tictactoe",
    "tictactoe": "tictactoe",
    "ttt": "tictactoe",
    "hangman": "hangman",
    "war": "war",
    "wars": "war",
    "battleship": "battleship",
    "battle ship": "battleship",
    "sea battle": "battleship",
    "rps": "rps",
    "rock paper scissors": "rps",
    "roshambo": "rps",
    "racing": "racing",
    "race": "racing",
    "ai racing": "racing",
}


def launch_racing_game(mode: str = "evolve") -> Tuple[str, Dict[str, object]]:
    import subprocess
    import sys

    from navine.utils.paths import get_project_root

    mode = (mode or "evolve").strip().lower()
    if mode not in ("evolve", "race", "play"):
        mode = "evolve"
    try:
        import pygame  # noqa: F401
    except Exception:
        return (
            "Racing needs pygame. Run: pip install pygame\n"
            "Then: python -m navine.games.racing_ai evolve",
            {"game": False, "racing": False},
        )
    cmd = [sys.executable, "-m", "navine.games.racing_ai", mode]
    subprocess.Popen(cmd, cwd=str(get_project_root()))
    return (
        f"Launched Navine 2D AI racing ({mode}). "
        "Cars use ray sensors + a tiny neural net; evolve mode trains with a genetic algorithm "
        "(like the self-driving AI vs Claude video). JAX is used for net math if installed. "
        "ESC closes the window. Modes: evolve, race, play.",
        {"game": True, "game_type": "racing", "racing": True, "mode": mode},
    )


def _parse_modes(message: str) -> Tuple[bool, str]:
    freer = bool(FREER_PATTERN.search(message))
    if HARD_PATTERN.search(message):
        difficulty = "hard"
    elif EASY_PATTERN.search(message):
        difficulty = "easy"
    else:
        difficulty = "normal"
    return freer, difficulty


def _start_game(
    game_type: str,
    session_id: str,
    freer: bool = False,
    difficulty: str = "normal",
) -> Tuple[str, Dict[str, object]]:
    if game_type == "racing":
        mode = "play" if freer else "evolve"
        if difficulty == "hard":
            mode = "race"
        return launch_racing_game(mode=mode)
    if game_type == "chess" and not CHESS_AVAILABLE:
        return (
            "Chess needs python-chess. Run navine setup or: pip install python-chess",
            {"game": False},
        )
    factory = GAME_FACTORY.get(game_type)
    if not factory:
        return "Unknown game.", {"game": False}
    game = factory(freer=freer, difficulty=difficulty)
    set_active_game(session_id, game_type, game, freer=freer, difficulty=difficulty)
    meta: Dict[str, object] = {
        "game": True,
        "game_type": game_type,
        "game_started": True,
        "freer": freer,
        "difficulty": difficulty,
    }
    return game.start_message(), meta


def is_game_message(message: str, session_id: str = "default") -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    if HELP_PATTERNS.search(stripped):
        return True
    if STOP_PATTERNS.search(stripped):
        return True
    if GO_FREER.search(stripped) or GO_STRICT.search(stripped):
        return True
    for pattern in START_PATTERNS.values():
        if pattern.search(stripped):
            return True
    key = stripped.lower()
    if key in NAME_ALIASES:
        return True
    if get_active_game(session_id) is not None:
        return True
    return False


def handle_game_message(
    message: str,
    session_id: str = "default",
) -> Optional[Tuple[str, Dict[str, object]]]:
    stripped = message.strip()
    if not stripped:
        return None

    if HELP_PATTERNS.search(stripped):
        return games_help_text(), {"game": False, "games_help": True}

    if STOP_PATTERNS.search(stripped):
        if get_active_game(session_id):
            clear_active_game(session_id)
            return (
                "Game ended. Say play chess, battleship, war, tic-tac-toe, hangman, or rps to start. "
                "Add anything goes for looser rules.",
                {"game": False, "game_over": True},
            )
        return None

    active = get_active_game(session_id)
    if active is not None:
        if GO_FREER.search(stripped):
            if hasattr(active, "set_freer"):
                active.set_freer(True)
            return (
                "Looser rules on. Illegal or freeform moves are allowed where the game supports them.",
                {"game": True, "freer": True},
            )
        if GO_STRICT.search(stripped):
            if hasattr(active, "set_freer"):
                active.set_freer(False)
            return (
                "Strict rules on. Standard move checks apply.",
                {"game": True, "freer": False},
            )

    freer, difficulty = _parse_modes(stripped)
    for game_type, pattern in START_PATTERNS.items():
        if pattern.search(stripped):
            return _start_game(game_type, session_id, freer=freer, difficulty=difficulty)

    key = stripped.lower()
    for alias, game_type in NAME_ALIASES.items():
        if key == alias or key.startswith(alias + " "):
            return _start_game(game_type, session_id, freer=freer, difficulty=difficulty)

    if active is not None:
        reply, extra = active.handle(stripped)
        meta: Dict[str, object] = {"game": True, "searched": False}
        meta.update(extra)
        if extra.get("game_over"):
            clear_active_game(session_id)
        return reply, meta

    return None


def games_help_text() -> str:
    return (
        "I can play local games with you:\n"
        "- chess (moves like e4, Nf3, e2e4)\n"
        "- battleship (fire A1, C4)\n"
        "- war card game (say go each round)\n"
        "- tic-tac-toe (squares 1-9)\n"
        "- hangman (letter guesses)\n"
        "- rock paper scissors (rock, paper, scissors)\n"
        "- AI racing (2D self-driving cars with neural nets + genetic training; opens a window)\n"
        "Say play chess, play battleship, play war, play racing, and so on.\n"
        "Add easy or hard for difficulty.\n"
        "Add anything goes, freestyle, or illegal moves ok for looser rules "
        "(for example illegal chess moves allowed, occupied squares can be overwritten, re-fire allowed).\n"
        "During a game say go freestyle or strict rules to switch.\n"
        "Say quit game to stop."
    )
