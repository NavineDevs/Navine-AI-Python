import re
from typing import Dict, List, Optional, Tuple

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def _winner(board: List[str]) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def _render(board: List[str]) -> str:
    rows = []
    for i in range(0, 9, 3):
        cells = [board[j] if board[j] else str(j + 1) for j in range(i, i + 3)]
        rows.append(" | ".join(cells))
    return "\n---------\n".join(rows)


def _minimax(board: List[str], player: str, ai: str, user: str) -> int:
    w = _winner(board)
    if w == ai:
        return 10
    if w == user:
        return -10
    if all(cell for cell in board):
        return 0
    empties = [i for i, cell in enumerate(board) if not cell]
    if player == ai:
        best = -999
        for i in empties:
            board[i] = ai
            best = max(best, _minimax(board, user, ai, user))
            board[i] = ""
        return best
    best = 999
    for i in empties:
        board[i] = user
        best = min(best, _minimax(board, ai, ai, user))
        board[i] = ""
    return best


def _best_empty(board: List[str], ai: str, user: str, difficulty: str) -> int:
    empties = [i for i, cell in enumerate(board) if not cell]
    if not empties:
        return -1
    if difficulty == "easy":
        import random
        return random.choice(empties)
    best_score = None
    best_idx = empties[0]
    for i in empties:
        board[i] = ai
        score = _minimax(board, user, ai, user)
        board[i] = ""
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _best_freer(board: List[str], ai: str, user: str) -> int:
    for i in range(9):
        trial = board[:]
        trial[i] = ai
        if _winner(trial) == ai:
            return i
    for i in range(9):
        trial = board[:]
        trial[i] = user
        if _winner(trial) == user:
            return i
    if not board[4]:
        return 4
    for i in (0, 2, 6, 8, 1, 3, 5, 7):
        if board[i] != ai:
            return i
    return 0


class TicTacToeGame:
    def __init__(self, freer: bool = False, difficulty: str = "normal") -> None:
        self.board = [""] * 9
        self.user = "X"
        self.ai = "O"
        self.freer = freer
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"

    def set_freer(self, freer: bool) -> None:
        self.freer = freer

    def start_message(self) -> str:
        rules = (
            "Occupied squares can be overwritten."
            if self.freer
            else "Standard rules: empty squares only."
        )
        return (
            f"Tic-tac-toe started. You are X, I am O. {rules} Difficulty: {self.difficulty}.\n\n"
            f"{_render(self.board)}\n\n"
            "Reply with a square number 1-9. Say quit game to stop."
        )

    def handle(self, message: str) -> Tuple[str, Dict[str, object]]:
        lower = message.strip().lower()
        if lower in ("board", "show board", "status"):
            return f"{_render(self.board)}\n\nYour move.", {"game_over": False}

        match = re.search(r"\b([1-9])\b", message)
        if not match:
            return (f"Pick a square from 1 to 9.\n\n{_render(self.board)}"), {"game_over": False}
        idx = int(match.group(1)) - 1
        if self.board[idx] and not self.freer:
            return (f"Square {idx + 1} is taken.\n\n{_render(self.board)}"), {"game_over": False}

        self.board[idx] = self.user
        winner = _winner(self.board)
        if winner == self.user:
            return (
                f"You win.\n\n{_render(self.board)}\n\nSay play tic-tac-toe to play again."
            ), {"game_over": True, "winner": "user"}
        if not self.freer and all(cell for cell in self.board):
            return f"Draw.\n\n{_render(self.board)}", {"game_over": True, "winner": "draw"}

        if self.freer:
            ai_idx = _best_freer(self.board, self.ai, self.user)
        else:
            ai_idx = _best_empty(self.board, self.ai, self.user, self.difficulty)
            if ai_idx < 0:
                return f"Draw.\n\n{_render(self.board)}", {"game_over": True, "winner": "draw"}

        self.board[ai_idx] = self.ai
        winner = _winner(self.board)
        if winner == self.ai:
            return (
                f"I played square {ai_idx + 1}.\n\n{_render(self.board)}\n\n"
                "I win. Say play tic-tac-toe for a rematch."
            ), {"game_over": True, "winner": "ai"}
        if not self.freer and all(cell for cell in self.board):
            return (
                f"I played square {ai_idx + 1}.\n\n{_render(self.board)}\n\nDraw."
            ), {"game_over": True, "winner": "draw"}
        return (
            f"I played square {ai_idx + 1}.\n\n{_render(self.board)}\n\nYour move."
        ), {"game_over": False}
