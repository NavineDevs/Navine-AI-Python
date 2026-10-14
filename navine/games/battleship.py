import random
import re
from typing import Dict, List, Optional, Set, Tuple

SIZE = 8
SHIP_SIZES = (4, 3, 3, 2)
COLS = "ABCDEFGH"


def _coord_to_rc(text: str) -> Optional[Tuple[int, int]]:
    clean = re.sub(r"[^a-zA-Z0-9]", "", text.strip())
    m = re.fullmatch(r"([A-Ha-h])([1-8])", clean)
    if not m:
        m = re.fullmatch(r"([1-8])([A-Ha-h])", clean)
        if not m:
            return None
        row = int(m.group(1)) - 1
        col = COLS.index(m.group(2).upper())
        return row, col
    col = COLS.index(m.group(1).upper())
    row = int(m.group(2)) - 1
    return row, col


def _rc_to_label(row: int, col: int) -> str:
    return f"{COLS[col]}{row + 1}"


def _place_ships() -> Tuple[Set[Tuple[int, int]], List[Set[Tuple[int, int]]]]:
    occupied: Set[Tuple[int, int]] = set()
    ships: List[Set[Tuple[int, int]]] = []
    for length in SHIP_SIZES:
        placed = False
        for _ in range(200):
            horizontal = random.choice([True, False])
            if horizontal:
                row = random.randint(0, SIZE - 1)
                col = random.randint(0, SIZE - length)
                cells = {(row, col + i) for i in range(length)}
            else:
                row = random.randint(0, SIZE - length)
                col = random.randint(0, SIZE - 1)
                cells = {(row + i, col) for i in range(length)}
            if cells & occupied:
                continue
            occupied |= cells
            ships.append(cells)
            placed = True
            break
        if not placed:
            return _place_ships()
    return occupied, ships


def _render(
    hits: Set[Tuple[int, int]],
    misses: Set[Tuple[int, int]],
    ships: Optional[Set[Tuple[int, int]]] = None,
    reveal: bool = False,
) -> str:
    lines = ["    " + " ".join(COLS)]
    for r in range(SIZE):
        row_cells = []
        for c in range(SIZE):
            cell = (r, c)
            if cell in hits:
                row_cells.append("X")
            elif cell in misses:
                row_cells.append("o")
            elif reveal and ships and cell in ships:
                row_cells.append("S")
            else:
                row_cells.append(".")
        lines.append(f"{r + 1:>2}  " + " ".join(row_cells))
    return "\n".join(lines)


class BattleshipGame:
    def __init__(self, freer: bool = False, difficulty: str = "normal") -> None:
        self.user_ships_set, self.user_ships = _place_ships()
        self.ai_ships_set, self.ai_ships = _place_ships()
        self.user_hits: Set[Tuple[int, int]] = set()
        self.user_misses: Set[Tuple[int, int]] = set()
        self.ai_hits: Set[Tuple[int, int]] = set()
        self.ai_misses: Set[Tuple[int, int]] = set()
        self.ai_shots: Set[Tuple[int, int]] = set()
        self.hunt: List[Tuple[int, int]] = []
        self.freer = freer
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"
        self.parity_phase = True

    def set_freer(self, freer: bool) -> None:
        self.freer = freer

    def start_message(self) -> str:
        freer_line = (
            "Re-fire is allowed; shots on known cells still resolve."
            if self.freer
            else "No re-firing the same cell."
        )
        return (
            "Battleship started on an 8x8 grid.\n"
            "Fleets auto-placed: ships of length 4, 3, 3, and 2.\n"
            f"{freer_line} Difficulty: {self.difficulty}.\n"
            "Fire with a coordinate like A1, C4, or H8.\n"
            "X = hit, o = miss, . = unknown.\n\n"
            f"Enemy waters:\n{_render(self.user_hits, self.user_misses)}\n\n"
            f"Your waters (S = ship):\n{_render(self.ai_hits, self.ai_misses, self.user_ships_set, reveal=True)}\n\n"
            "Your shot."
        )

    def _ships_left(self, ships: List[Set[Tuple[int, int]]], hits: Set[Tuple[int, int]]) -> int:
        return sum(1 for ship in ships if not ship.issubset(hits))

    def _density_score(self, row: int, col: int) -> int:
        score = 0
        for ship_len in SHIP_SIZES:
            for horizontal in (True, False):
                for offset in range(ship_len):
                    if horizontal:
                        c0 = col - offset
                        if c0 < 0 or c0 + ship_len > SIZE:
                            continue
                        cells = [(row, c0 + i) for i in range(ship_len)]
                    else:
                        r0 = row - offset
                        if r0 < 0 or r0 + ship_len > SIZE:
                            continue
                        cells = [(r0 + i, col) for i in range(ship_len)]
                    if any(c in self.ai_misses for c in cells):
                        continue
                    if any(c in self.ai_hits and c != (row, col) for c in cells):
                        score += 3
                    else:
                        score += 1
        return score

    def _ai_shot(self) -> Tuple[int, int]:
        while self.hunt:
            target = self.hunt.pop(0)
            if self.freer or target not in self.ai_shots:
                return target
        open_cells = [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if self.freer or (r, c) not in self.ai_shots
        ]
        if not open_cells:
            open_cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        if self.difficulty == "easy":
            return random.choice(open_cells)
        if self.parity_phase and self.difficulty != "easy":
            parity = [(r, c) for r, c in open_cells if (r + c) % 2 == 0]
            if parity:
                open_cells = parity
        ranked = sorted(open_cells, key=lambda rc: self._density_score(*rc), reverse=True)
        top = ranked[: max(1, len(ranked) // 8)]
        return random.choice(top)

    def _enqueue_hunt(self, row: int, col: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE:
                if self.freer or (nr, nc) not in self.ai_shots:
                    self.hunt.append((nr, nc))

    def handle(self, message: str) -> Tuple[str, Dict[str, object]]:
        lower = message.strip().lower()
        if lower in ("board", "status", "show", "map"):
            return (
                f"Enemy waters:\n{_render(self.user_hits, self.user_misses)}\n\n"
                f"Your waters:\n{_render(self.ai_hits, self.ai_misses, self.user_ships_set, reveal=True)}\n"
                f"Enemy ships left: {self._ships_left(self.ai_ships, self.user_hits)} | "
                f"Your ships left: {self._ships_left(self.user_ships, self.ai_hits)}"
            ), {"game_over": False}

        parse_text = re.sub(r"^(fire|shoot|attack|at)\s+", "", message.strip(), flags=re.I)
        coord = _coord_to_rc(parse_text)
        if coord is None:
            return (
                "Fire with a coordinate like B5 or H1.\n"
                f"Enemy waters:\n{_render(self.user_hits, self.user_misses)}"
            ), {"game_over": False}

        row, col = coord
        cell = (row, col)
        if not self.freer and (cell in self.user_hits or cell in self.user_misses):
            return f"You already fired at {_rc_to_label(row, col)}. Try another cell.", {"game_over": False}

        lines = [f"You fire at {_rc_to_label(row, col)}."]
        if cell in self.ai_ships_set:
            self.user_hits.add(cell)
            lines.append("Hit!" if cell not in self.user_hits or True else "Hit again!")
            for ship in self.ai_ships:
                if cell in ship and ship.issubset(self.user_hits):
                    lines.append("You sunk a ship!")
                    break
        else:
            self.user_misses.add(cell)
            lines.append("Miss.")

        if self._ships_left(self.ai_ships, self.user_hits) == 0:
            lines.append(
                f"\nEnemy waters:\n{_render(self.user_hits, self.user_misses)}\n\n"
                "You win Battleship. Say play battleship for a rematch."
            )
            return "\n".join(lines), {"game_over": True, "winner": "user"}

        ar, ac = self._ai_shot()
        self.ai_shots.add((ar, ac))
        lines.append(f"I fire at {_rc_to_label(ar, ac)}.")
        if (ar, ac) in self.user_ships_set:
            self.ai_hits.add((ar, ac))
            lines.append("Hit on your fleet!")
            self._enqueue_hunt(ar, ac)
            self.parity_phase = False
            for ship in self.user_ships:
                if (ar, ac) in ship and ship.issubset(self.ai_hits):
                    lines.append("I sunk one of your ships.")
                    break
        else:
            self.ai_misses.add((ar, ac))
            lines.append("I missed.")

        if self._ships_left(self.user_ships, self.ai_hits) == 0:
            lines.append(
                f"\nYour waters:\n{_render(self.ai_hits, self.ai_misses, self.user_ships_set, reveal=True)}\n\n"
                "I win Battleship. Say play battleship for a rematch."
            )
            return "\n".join(lines), {"game_over": True, "winner": "ai"}

        lines.append(
            f"\nEnemy waters:\n{_render(self.user_hits, self.user_misses)}\n\n"
            f"Your waters:\n{_render(self.ai_hits, self.ai_misses, self.user_ships_set, reveal=True)}\n"
            f"Enemy ships left: {self._ships_left(self.ai_ships, self.user_hits)} | "
            f"Your ships left: {self._ships_left(self.user_ships, self.ai_hits)}\n"
            "Your shot."
        )
        return "\n".join(lines), {"game_over": False}
