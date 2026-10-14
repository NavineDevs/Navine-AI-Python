import random
import re
from typing import Dict, Tuple

WORDS = [
    "Navine", "python", "chess", "galaxy", "neural", "tensor", "cipher",
    "rocket", "planet", "dragon", "castle", "wizard", "pixel", "vector",
    "quantum", "memory", "server", "binary", "logic", "model",
    "battleship", "warship", "hangman", "circuit", "prompt", "agent",
    "matrix", "compiler", "kernel", "signal", "nebula", "orbit",
    "diffusion", "latent", "embedding", "tokenizer", "gradient", "encoder",
    "decoder", "attention", "transformer", "scaffold", "pipeline", "gateway",
    "reef", "harbor", "comet", "aurora", "phoenix", "lantern", "mirror",
    "silicon", "cobalt", "nimbus", "volt", "flux", "prism",
]

GALLOW = [
    "  +---+\n  |   |\n      |\n      |\n      |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n      |\n      |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n  |   |\n      |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n /|   |\n      |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n /|\\  |\n      |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n /|\\  |\n /    |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n      |\n=========",
    "  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n *    |\n=========",
    "  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n * *  |\n=========",
]


def _mask(word: str, guessed: set) -> str:
    return " ".join(ch if ch.lower() in guessed or not ch.isalpha() else "_" for ch in word)


class HangmanGame:
    def __init__(self, freer: bool = False, difficulty: str = "normal") -> None:
        self.word = random.choice(WORDS)
        self.guessed: set = set()
        self.wrong = 0
        self.freer = freer
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"
        if freer:
            self.max_wrong = 8
        elif difficulty == "easy":
            self.max_wrong = 7
        elif difficulty == "hard":
            self.max_wrong = 5
        else:
            self.max_wrong = 6

    def set_freer(self, freer: bool) -> None:
        self.freer = freer
        if freer and self.max_wrong < 8:
            self.max_wrong = 8

    def _body(self) -> str:
        stage = min(self.wrong, len(GALLOW) - 1)
        return (
            f"{GALLOW[stage]}\n\n"
            f"{_mask(self.word, self.guessed)}\n"
            f"Guessed: {', '.join(sorted(self.guessed)) or '-'}\n"
            f"Wrong: {self.wrong}/{self.max_wrong}"
        )

    def start_message(self) -> str:
        freer = "Numbers and symbols allowed as flavor guesses (they usually miss)." if self.freer else ""
        return (
            f"Hangman started. {freer} Lives: {self.max_wrong}.\n\n"
            f"{self._body()}\n\n"
            "Reply with a single letter, or guess the full word."
        )

    def handle(self, message: str) -> Tuple[str, Dict[str, object]]:
        text = message.strip()
        if not text:
            return "Send a letter or the full word.", {"game_over": False}
        if any(ch.isspace() for ch in text):
            return (
                "Send a single letter, or the full word with no spaces. Say quit game to stop.",
                {"game_over": False},
            )
        if len(text) > 1 and re.fullmatch(r"[a-zA-Z]+", text):
            if text.lower() == self.word.lower():
                return (
                    f"Correct. The word was {self.word}.\nYou win. Say play hangman to play again."
                ), {"game_over": True, "winner": "user"}
            self.wrong += 1
            if self.wrong >= self.max_wrong:
                return (
                    f"{self._body()}\n\nWrong word. It was {self.word}. I win."
                ), {"game_over": True, "winner": "ai"}
            return f"Not the word.\n\n{self._body()}", {"game_over": False}

        if self.freer and not re.search(r"[a-zA-Z]", text):
            token = text[0]
            if token in self.guessed:
                return f"You already tried {token}.\n\n{self._body()}", {"game_over": False}
            self.guessed.add(token)
            self.wrong += 1
            if self.wrong >= self.max_wrong:
                return (
                    f"{self._body()}\n\nOut of guesses. The word was {self.word}. I win."
                ), {"game_over": True, "winner": "ai"}
            return f"No {token} in the word.\n\n{self._body()}", {"game_over": False}

        if not re.fullmatch(r"[a-zA-Z]", text):
            return "Send one letter guess.", {"game_over": False}
        letter = text.lower()
        if letter in self.guessed:
            return f"You already guessed {letter}.\n\n{self._body()}", {"game_over": False}
        self.guessed.add(letter)
        if letter not in self.word.lower():
            self.wrong += 1
        if all(ch.lower() in self.guessed or not ch.isalpha() for ch in self.word):
            return (
                f"Correct. The word was {self.word}.\nYou win. Say play hangman to play again."
            ), {"game_over": True, "winner": "user"}
        if self.wrong >= self.max_wrong:
            return (
                f"{self._body()}\n\nOut of guesses. The word was {self.word}. I win."
            ), {"game_over": True, "winner": "ai"}
        return self._body() + "\n\nYour guess.", {"game_over": False}
