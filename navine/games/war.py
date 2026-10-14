import random
from typing import Dict, List, Tuple

RANKS = "23456789TJQKA"
SUITS = "SHDC"
RANK_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}


def _rank_name(rank: str) -> str:
    names = {
        "T": "10",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }
    return names.get(rank, rank)


def _card_label(card: str) -> str:
    rank, suit = card[0], card[1]
    suit_name = {"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}[suit]
    return f"{_rank_name(rank)} of {suit_name}"


class WarGame:
    def __init__(self, freer: bool = False, difficulty: str = "normal") -> None:
        deck = [r + s for r in RANKS for s in SUITS]
        random.shuffle(deck)
        self.user: List[str] = deck[:26]
        self.ai: List[str] = deck[26:]
        self.round = 0
        self.user_wins = 0
        self.ai_wins = 0
        self.war_streak = 0
        self.freer = freer
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"
        self.peek_used = False

    def set_freer(self, freer: bool) -> None:
        self.freer = freer

    def _scoreline(self) -> str:
        return (
            f"Round {self.round} | Your cards: {len(self.user)} | My cards: {len(self.ai)} | "
            f"Round wins you {self.user_wins} me {self.ai_wins}"
        )

    def start_message(self) -> str:
        freer = (
            "Freestyle: say peek once to see your top card, or force war to force a war pile."
            if self.freer
            else ""
        )
        return (
            "War started. Each of us has 26 cards.\n"
            "Say go, next, fight, or flip to play a round.\n"
            "Higher card wins both cards. Equal ranks start a war pile.\n"
            f"{freer}\n"
            f"{self._scoreline()}\n"
            "Say quit game to stop."
        ).strip()

    def handle(self, message: str) -> Tuple[str, Dict[str, object]]:
        lower = message.strip().lower()
        if lower in ("board", "status", "score", "cards"):
            return self._scoreline() + "\nSay go to play the next round.", {"game_over": False}

        if self.freer and lower in ("peek", "peek top", "look"):
            if self.peek_used:
                return "You already used peek this game.", {"game_over": False}
            if not self.user:
                return self._end_state()
            self.peek_used = True
            return f"Your top card is {_card_label(self.user[0])}.\n{self._scoreline()}", {"game_over": False}

        force_war = self.freer and lower in ("force war", "war now", "force a war")
        if (
            lower not in {
                "go", "next", "fight", "flip", "play", "war", "round", "draw", "hit me",
                "again", "continue", "ok", "okay", "yes",
            }
            and not lower.startswith("go ")
            and not force_war
        ):
            return (
                "In War, reply with go, next, fight, or flip to play a round.\n"
                f"{self._scoreline()}"
            ), {"game_over": False}

        if not self.user or not self.ai:
            return self._end_state()

        self.round += 1
        pile: List[str] = []
        log: List[str] = [f"Round {self.round}"]
        if force_war:
            log.append("You force a war!")

        first = True
        while True:
            if not self.user or not self.ai:
                break
            u = self.user.pop(0)
            a = self.ai.pop(0)
            pile.extend([u, a])
            log.append(f"You: {_card_label(u)}  |  Me: {_card_label(a)}")
            uv = RANK_VALUE[u[0]]
            av = RANK_VALUE[a[0]]
            if force_war and first:
                first = False
                uv = av
                log.append("War forced - ranks treated as equal.")
            if uv > av:
                random.shuffle(pile)
                self.user.extend(pile)
                self.user_wins += 1
                self.war_streak = 0
                log.append("You take the pile.")
                break
            if av > uv:
                random.shuffle(pile)
                self.ai.extend(pile)
                self.ai_wins += 1
                self.war_streak = 0
                log.append("I take the pile.")
                break
            self.war_streak += 1
            log.append("War! Each side puts two more face down if available.")
            if self.war_streak >= 5:
                log.append("Sudden death: next flip alone decides the pile.")
                if not self.user or not self.ai:
                    break
                u2 = self.user.pop(0)
                a2 = self.ai.pop(0)
                pile.extend([u2, a2])
                log.append(f"Sudden death — You: {_card_label(u2)}  |  Me: {_card_label(a2)}")
                if RANK_VALUE[u2[0]] >= RANK_VALUE[a2[0]]:
                    random.shuffle(pile)
                    self.user.extend(pile)
                    self.user_wins += 1
                    log.append("You win the war pile.")
                else:
                    random.shuffle(pile)
                    self.ai.extend(pile)
                    self.ai_wins += 1
                    log.append("I win the war pile.")
                self.war_streak = 0
                break
            for _ in range(2):
                if self.user:
                    pile.append(self.user.pop(0))
                if self.ai:
                    pile.append(self.ai.pop(0))
            if not self.user or not self.ai:
                break

        if not self.user or not self.ai:
            body = "\n".join(log)
            end, meta = self._end_state()
            return f"{body}\n\n{end}", meta

        body = "\n".join(log)
        return f"{body}\n\n{self._scoreline()}\nSay go for another round.", {"game_over": False}

    def _end_state(self) -> Tuple[str, Dict[str, object]]:
        if len(self.user) > len(self.ai):
            return (
                f"You win War. Final cards you {len(self.user)} me {len(self.ai)}.\n"
                "Say play war for a rematch."
            ), {"game_over": True, "winner": "user"}
        if len(self.ai) > len(self.user):
            return (
                f"I win War. Final cards you {len(self.user)} me {len(self.ai)}.\n"
                "Say play war for a rematch."
            ), {"game_over": True, "winner": "ai"}
        return (
            f"War ends in a draw. Cards you {len(self.user)} me {len(self.ai)}."
        ), {"game_over": True, "winner": "draw"}
