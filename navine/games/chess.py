import random
import re
from typing import Dict, List, Optional, Tuple

try:
    import chess
except ImportError:
    chess = None

CHESS_AVAILABLE = chess is not None

PIECE_ASCII = {}
if CHESS_AVAILABLE:
    PIECE_ASCII = {
        chess.PAWN: {"w": "P", "b": "p"},
        chess.KNIGHT: {"w": "N", "b": "n"},
        chess.BISHOP: {"w": "B", "b": "b"},
        chess.ROOK: {"w": "R", "b": "r"},
        chess.QUEEN: {"w": "Q", "b": "q"},
        chess.KING: {"w": "K", "b": "k"},
    }

PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
PST_KING = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]


def _require_chess() -> None:
    if chess is None:
        raise RuntimeError(
            "Chess requires the python-chess package. Run setup or: pip install python-chess"
        )


def _piece_char(piece) -> str:
    symbols = PIECE_ASCII.get(piece.piece_type, {})
    return symbols.get("w" if piece.color == chess.WHITE else "b", "?")


def render_board(board) -> str:
    lines = ["```board"]
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            row.append(_piece_char(piece) if piece else ".")
        lines.append(f"{rank + 1}  " + " ".join(row))
    lines.append("   a b c d e f g h")
    lines.append("```")
    return "\n".join(lines)


def _pst_bonus(piece_type, square: int, color: bool) -> int:
    idx = square if color == chess.WHITE else chess.square_mirror(square)
    if piece_type == chess.PAWN:
        return PST_PAWN[idx]
    if piece_type == chess.KNIGHT:
        return PST_KNIGHT[idx]
    if piece_type == chess.KING:
        return PST_KING[idx]
    if piece_type == chess.BISHOP:
        return PST_KNIGHT[idx] // 2
    if piece_type == chess.ROOK:
        return 5 if chess.square_rank(square) in (0, 7) else 0
    return 0


def _evaluate(board) -> int:
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        val = values.get(piece.piece_type, 0) + _pst_bonus(piece.piece_type, square, piece.color)
        score += val if piece.color == chess.WHITE else -val
    if board.is_check():
        score += 40 if board.turn == chess.BLACK else -40
    return score


def _depth_for(difficulty: str) -> int:
    if difficulty == "hard":
        return 4
    if difficulty == "easy":
        return 1
    return 3


def _minimax(board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0 or board.is_game_over():
        return _evaluate(board)
    moves = list(board.legal_moves)
    if board.turn == chess.WHITE:
        best = -999999
        for move in moves:
            board.push(move)
            best = max(best, _minimax(board, depth - 1, alpha, beta))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = 999999
    for move in moves:
        board.push(move)
        best = min(best, _minimax(board, depth - 1, alpha, beta))
        board.pop()
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def _pick_legal_ai(board, difficulty: str):
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("no legal moves")
    if difficulty == "easy":
        return random.choice(moves)
    depth = _depth_for(difficulty)
    scored = []
    for move in moves:
        board.push(move)
        score = -_minimax(board, depth - 1, -999999, 999999)
        board.pop()
        scored.append((score, move))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = 1 if difficulty == "hard" else min(3, len(scored))
    pool = scored[:top_n]
    return random.choice(pool)[1]


def _parse_uci_pair(text: str) -> Optional[Tuple[int, int, Optional[int]]]:
    cleaned = text.strip().lower().replace(" ", "")
    cleaned = re.sub(r"^(play|move|chess)\s+", "", cleaned)
    m = re.fullmatch(r"([a-h][1-8])([a-h][1-8])([qrbn])?", cleaned)
    if not m:
        return None
    fr = chess.parse_square(m.group(1))
    to = chess.parse_square(m.group(2))
    promo = None
    if m.group(3):
        promo = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}[m.group(3)]
    return fr, to, promo


def _parse_legal_move(text: str, board) -> Optional[object]:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(play|move|chess)\s+", "", cleaned)
    cleaned_ns = cleaned.replace(" ", "")
    if not cleaned_ns:
        return None
    try:
        if len(cleaned_ns) in (4, 5) and cleaned_ns[0] in "abcdefgh" and cleaned_ns[2] in "abcdefgh":
            move = chess.Move.from_uci(cleaned_ns)
            if move in board.legal_moves:
                return move
        move = board.parse_san(cleaned_ns)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    try:
        move = board.parse_uci(cleaned_ns)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    for move in board.legal_moves:
        try:
            if board.san(move).lower().replace(" ", "") == cleaned_ns:
                return move
        except Exception:
            continue
    return None


def _apply_forced(board, fr: int, to: int, promo: Optional[int]) -> Tuple[str, Optional[str]]:
    piece = board.piece_at(fr)
    if not piece:
        return "", "No piece on the start square."
    captured = board.piece_at(to)
    captured_was_king = bool(captured and captured.piece_type == chess.KING)
    label = f"{chess.square_name(fr)}{chess.square_name(to)}"
    board.remove_piece_at(fr)
    if promo:
        piece = chess.Piece(promo, piece.color)
        label += {chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b", chess.KNIGHT: "n"}.get(promo, "")
    board.set_piece_at(to, piece)
    board.turn = not board.turn
    if board.piece_at(chess.E1) is None and board.king(chess.WHITE) is None:
        pass
    return label, "king" if captured_was_king else None


def _count_kings(board) -> Tuple[bool, bool]:
    has_w = any(
        board.piece_at(s) and board.piece_at(s).piece_type == chess.KING and board.piece_at(s).color == chess.WHITE
        for s in chess.SQUARES
    )
    has_b = any(
        board.piece_at(s) and board.piece_at(s).piece_type == chess.KING and board.piece_at(s).color == chess.BLACK
        for s in chess.SQUARES
    )
    return has_w, has_b


def _pick_freer_ai(board, difficulty: str):
    if random.random() < 0.7:
        try:
            legal = list(board.legal_moves)
            if legal:
                if difficulty == "easy":
                    return ("legal", random.choice(legal))
                return ("legal", _pick_legal_ai(board, difficulty))
        except Exception:
            pass
    pieces = [s for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == board.turn]
    if not pieces:
        legal = list(board.legal_moves)
        if legal:
            return ("legal", random.choice(legal))
        raise ValueError("no moves")
    fr = random.choice(pieces)
    to = random.choice(list(chess.SQUARES))
    if to == fr:
        to = (fr + 1) % 64
    promo = chess.QUEEN if board.piece_at(fr).piece_type == chess.PAWN and chess.square_rank(to) in (0, 7) else None
    return ("force", fr, to, promo)


class ChessGame:
    def __init__(self, freer: bool = False, difficulty: str = "normal") -> None:
        _require_chess()
        self.board = chess.Board()
        self.freer = freer
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"

    def set_freer(self, freer: bool) -> None:
        self.freer = freer

    def start_message(self) -> str:
        mode = []
        if self.freer:
            mode.append("Illegal moves are allowed. Capture the king to win.")
        else:
            mode.append("Standard rules.")
        mode.append(f"Difficulty: {self.difficulty}.")
        return (
            "Chess started. You play White. I play Black.\n"
            + " ".join(mode)
            + f"\n\n{render_board(self.board)}\n\n"
            "Send a move like e4, Nf3, or e2e4. Say resign, draw, or quit game to stop. "
            "Say go freestyle or strict rules to change rule strictness mid-game."
        )

    def handle(self, message: str) -> Tuple[str, Dict[str, object]]:
        lower = message.strip().lower()
        if lower in ("resign", "i resign", "forfeit"):
            return "You resigned. I win this game. Say play chess to start again.", {
                "game_over": True,
                "winner": "ai",
            }
        if lower in ("draw", "offer draw", "accept draw"):
            return "Game drawn by agreement. Say play chess for a rematch.", {
                "game_over": True,
                "winner": "draw",
            }
        if lower in ("board", "show board", "status"):
            return self._status_message(), {"game_over": False}

        if self.freer:
            return self._handle_freer(message)
        return self._handle_strict(message)

    def _handle_strict(self, message: str) -> Tuple[str, Dict[str, object]]:
        move = _parse_legal_move(message, self.board)
        if move is None:
            return (
                "I could not parse that as a legal move. Try e4, Nf3, Bb5, or e2e4.\n\n"
                f"{render_board(self.board)}"
            ), {"game_over": False}

        user_san = self.board.san(move)
        self.board.push(move)
        if self.board.is_game_over():
            return self._end_message(), {"game_over": True, "winner": self._winner_meta()}

        ai_move = _pick_legal_ai(self.board, self.difficulty)
        ai_san = self.board.san(ai_move)
        self.board.push(ai_move)
        reply = (
            f"You played {user_san}.\n"
            f"I played {ai_san}.\n\n"
            f"{render_board(self.board)}"
        )
        if self.board.is_game_over():
            reply += "\n\n" + self._end_message()
            return reply, {"game_over": True, "winner": self._winner_meta()}
        reply += "\n\nYour move."
        return reply, {"game_over": False}

    def _handle_freer(self, message: str) -> Tuple[str, Dict[str, object]]:
        user_label = None
        king_taken = None
        legal = _parse_legal_move(message, self.board)
        if legal is not None:
            try:
                user_label = self.board.san(legal)
            except Exception:
                user_label = legal.uci()
            self.board.push(legal)
        else:
            pair = _parse_uci_pair(message)
            if pair is None:
                return (
                    "Could not parse move. Use e2e4 (from-to). Illegal moves are allowed.\n\n"
                    f"{render_board(self.board)}"
                ), {"game_over": False}
            fr, to, promo = pair
            user_label, king_taken = _apply_forced(self.board, fr, to, promo)
            if not user_label:
                return f"{king_taken or 'Invalid move.'}\n\n{render_board(self.board)}", {"game_over": False}

        has_w, has_b = _count_kings(self.board)
        if king_taken == "king" or not has_b:
            return (
                f"You played {user_label}.\n\n{render_board(self.board)}\n\n"
                "You captured the king. You win. Say play chess for a rematch."
            ), {"game_over": True, "winner": "user"}
        if not has_w:
            return (
                f"You played {user_label}.\n\n{render_board(self.board)}\n\n"
                "Your king is gone. I win. Say play chess for a rematch."
            ), {"game_over": True, "winner": "ai"}
        try:
            if self.board.is_game_over():
                return self._end_message(), {"game_over": True, "winner": self._winner_meta()}
        except Exception:
            pass

        ai_kind = _pick_freer_ai(self.board, self.difficulty)
        if ai_kind[0] == "legal":
            ai_move = ai_kind[1]
            try:
                ai_label = self.board.san(ai_move)
            except Exception:
                ai_label = ai_move.uci()
            self.board.push(ai_move)
        else:
            _, fr, to, promo = ai_kind
            ai_label, ai_king = _apply_forced(self.board, fr, to, promo)
            if ai_king == "king":
                return (
                    f"You played {user_label}.\nI played {ai_label}.\n\n{render_board(self.board)}\n\n"
                    "I captured your king. I win."
                ), {"game_over": True, "winner": "ai"}

        has_w, has_b = _count_kings(self.board)
        reply = (
            f"You played {user_label}.\n"
            f"I played {ai_label}.\n\n"
            f"{render_board(self.board)}"
        )
        if not has_w:
            reply += "\n\nI captured your king. I win. Say play chess for a rematch."
            return reply, {"game_over": True, "winner": "ai"}
        if not has_b:
            reply += "\n\nYour king-capture succeeded mid-exchange. You win."
            return reply, {"game_over": True, "winner": "user"}
        try:
            if self.board.is_game_over():
                reply += "\n\n" + self._end_message()
                return reply, {"game_over": True, "winner": self._winner_meta()}
        except Exception:
            pass
        reply += "\n\nYour move. Illegal moves are allowed."
        return reply, {"game_over": False}

    def _status_message(self) -> str:
        turn = "Your turn (White)" if self.board.turn == chess.WHITE else "My turn (Black)"
        mode = "freestyle (illegal moves allowed)" if self.freer else "standard"
        return f"{turn} | {mode} | {self.difficulty}\n\n{render_board(self.board)}"

    def _winner_meta(self) -> str:
        try:
            if self.board.is_checkmate():
                return "ai" if self.board.turn == chess.WHITE else "user"
        except Exception:
            pass
        return "draw"

    def _end_message(self) -> str:
        try:
            if self.board.is_checkmate():
                if self.board.turn == chess.WHITE:
                    return "Checkmate. I win. Say play chess to play again."
                return "Checkmate. You win. Say play chess for a rematch."
            if self.board.is_stalemate():
                return "Stalemate. Draw. Say play chess to play again."
            if self.board.is_insufficient_material():
                return "Draw by insufficient material."
            if self.board.can_claim_threefold_repetition():
                return "Draw by repetition."
        except Exception:
            pass
        return "Game over."
