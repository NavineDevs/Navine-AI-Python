import random
import re
from collections import Counter
from typing import Dict, List, Tuple

CHOICES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
LOSES_TO = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


class RPSGame:
    def __init__(self, freer: bool = False, difficulty: str = "normal") -> None:
        self.user_score = 0
        self.ai_score = 0
        self.rounds = 0
        self.freer = freer
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"
        self.history: List[str] = []

    def set_freer(self, freer: bool) -> None:
        self.freer = freer

    def start_message(self) -> str:
        freer = (
            "Anything goes: invent a wild move and I will resolve it, or say rock/paper/scissors."
            if self.freer
            else "Say rock, paper, or scissors each round."
        )
        return (
            f"Rock Paper Scissors started. {freer}\n"
            f"First to 3 wins. Difficulty: {self.difficulty}.\n"
            f"Score you {self.user_score} me {self.ai_score}."
        )

    def _ai_pick(self) -> str:
        if self.difficulty == "easy" or not self.history:
            return random.choice(CHOICES)
        counts = Counter(self.history[-5:])
        predicted = counts.most_common(1)[0][0]
        counter = LOSES_TO[predicted]
        if self.difficulty == "hard":
            return counter if random.random() < 0.85 else random.choice(CHOICES)
        return counter if random.random() < 0.65 else random.choice(CHOICES)

    def handle(self, message: str) -> Tuple[str, Dict[str, object]]:
        lower = message.strip().lower()
        pick = None
        wild = None
        if lower in ("r", "rock") or re.search(r"\brock\b", lower):
            pick = "rock"
        elif lower in ("p", "paper") or re.search(r"\bpaper\b", lower):
            pick = "paper"
        elif lower in ("s", "scissors", "scissor") or re.search(r"\bscissors?\b", lower):
            pick = "scissors"
        elif self.freer and lower.strip():
            wild = message.strip()
            pick = random.choice(CHOICES)
        if pick is None:
            hint = "Say rock, paper, or scissors" + (" (or invent a move)" if self.freer else "") + "."
            return hint, {"game_over": False}

        if not wild:
            self.history.append(pick)
        ai = self._ai_pick()
        self.rounds += 1
        if wild:
            if random.random() < 0.5:
                self.user_score += 1
                result = f"Your wild move ({wild}) beats my {ai}."
            else:
                self.ai_score += 1
                result = f"My {ai} answers your wild move ({wild})."
        elif pick == ai:
            result = "Tie."
        elif BEATS[pick] == ai:
            self.user_score += 1
            result = "You win the round."
        else:
            self.ai_score += 1
            result = "I win the round."

        shown = wild or pick
        score = f"Score you {self.user_score} me {self.ai_score}."
        line = f"You: {shown} | Me: {ai}. {result}\n{score}"
        if self.user_score >= 3:
            return f"{line}\nYou win the match. Say play rps for a rematch.", {
                "game_over": True,
                "winner": "user",
            }
        if self.ai_score >= 3:
            return f"{line}\nI win the match. Say play rps for a rematch.", {
                "game_over": True,
                "winner": "ai",
            }
        return f"{line}\nNext throw.", {"game_over": False}
