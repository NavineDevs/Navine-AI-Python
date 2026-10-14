from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALES = {"hundred": 100, "thousand": 1000}

MISSPELLED_UNITS = {"fife": 5, "sevn": 7, "sevin": 7, "eigt": 8, "nien": 9, "thri": 3, "sixe": 6}

OPERATION_KEYWORDS: List[Tuple[str, str]] = [
    ("multipl", "*"),
    ("times", "*"),
    ("momentum", "*"),
    ("product", "*"),
    ("dividedby", "/"),
    ("divide", "/"),
    ("splitinto", "/"),
    ("splitequally", "/"),
    ("sharedequally", "/"),
    ("ratio", "/"),
    ("slowsby", "-"),
    ("slowsdown", "-"),
    ("slows", "-"),
    ("slowed", "-"),
    ("slower", "-"),
    ("decelerat", "-"),
    ("brakes", "-"),
    ("sheds", "-"),
    ("gaveaway", "-"),
    ("givesaway", "-"),
    ("falls", "-"),
    ("sinks", "-"),
    ("decreasesby", "-"),
    ("decreases", "-"),
    ("reducesby", "-"),
    ("reduces", "-"),
    ("dropsby", "-"),
    ("drops", "-"),
    ("loses", "-"),
    ("lose", "-"),
    ("lost", "-"),
    ("minus", "-"),
    ("subtract", "-"),
    ("fewer", "-"),
    ("less", "-"),
    ("remain", "-"),
    ("left", "-"),
    ("difference", "-"),
    ("plus", "+"),
    ("adds", "+"),
    ("added", "+"),
    ("gains", "+"),
    ("increasesby", "+"),
    ("increases", "+"),
    ("speedsupby", "+"),
    ("speedsup", "+"),
    ("speeds", "+"),
    ("faster", "+"),
    ("boost", "+"),
    ("finds", "+"),
    ("collects", "+"),
    ("grows", "+"),
    ("rises", "+"),
    ("accelerat", "+"),
    ("combined", "+"),
    ("total", "+"),
    ("sum", "+"),
    ("together", "+"),
    ("more", "+"),
]


SYMBOL_OPERATIONS = {"+": "+", "*": "*", "\u00d7": "*", "\u00f7": "/"}

FALLBACK_KEYWORDS: List[Tuple[str, str]] = [
    ("netforce", "-"),
    ("opos", "-"),
    ("against", "-"),
    ("rival", "-"),
]


def _collapse(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", text)


_NUMBER_WORDS: Dict[str, Tuple[str, int]] = {}
for _word, _value in UNITS.items():
    _NUMBER_WORDS[_collapse(_word)] = ("unit", _value)
for _word, _value in TENS.items():
    _NUMBER_WORDS[_collapse(_word)] = ("tens", _value)
for _word, _value in SCALES.items():
    _NUMBER_WORDS[_collapse(_word)] = ("scale", _value)
for _word, _value in MISSPELLED_UNITS.items():
    _NUMBER_WORDS[_word] = ("unit", _value)
_NUMBER_WORDS["point"] = ("point", 0)


def clean_tokens(challenge_text: str) -> List[str]:
    lowered = (challenge_text or "").lower()
    tokens: List[str] = []
    for raw in lowered.split():
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?[^\w]*", raw):
            tokens.append(re.sub(r"[^\d.\-+]", "", raw))
            continue
        letters = re.sub(r"[^a-z]", "", raw)
        if letters:
            tokens.append(_collapse(letters))
    return tokens


def _split_compound(word: str) -> Optional[List[Tuple[str, int]]]:
    if word in _NUMBER_WORDS:
        return [_NUMBER_WORDS[word]]
    for size in range(len(word) - 1, 2, -1):
        head = word[:size]
        if head in _NUMBER_WORDS:
            rest = _split_compound(word[size:])
            if rest is not None:
                return [_NUMBER_WORDS[head]] + rest
    return None


def _number_parts(tokens: List[str], start: int) -> Tuple[Optional[List[Tuple[str, int]]], int]:
    for span in (3, 2, 1):
        if start + span > len(tokens):
            continue
        merged = _collapse("".join(tokens[start : start + span]))
        parts = _split_compound(merged)
        if parts is not None:
            return parts, span
    return None, 1


def _parts_to_value(parts: List[Tuple[str, int]]) -> Optional[float]:
    total = 0
    current = 0
    decimals: List[int] = []
    in_decimal = False
    previous_kind = ""
    for kind, value in parts:
        if kind == "point":
            in_decimal = True
            continue
        if in_decimal:
            if kind != "unit" or value > 9:
                return None
            decimals.append(value)
            continue
        if kind == "unit":
            current += value
        elif kind == "tens" and previous_kind == "tens":
            current += value // 10
        elif kind == "tens":
            current += value
        elif kind == "scale":
            current = max(1, current) * value
            if value >= 1000:
                total += current
                current = 0
        previous_kind = kind
    whole = total + current
    if decimals:
        return float(f"{whole}." + "".join(str(d) for d in decimals))
    return float(whole)


MEASUREMENT_PREFIXES = (
    "newton",
    "meter",
    "metre",
    "centimet",
    "milimet",
    "kilomet",
    "kilogram",
    "gram",
    "pound",
    "degre",
    "second",
    "minute",
    "hour",
    "liter",
    "litre",
    "joule",
    "wat",
    "mile",
    "fet",
    "fot",
    "inch",
    "knot",
    "kph",
    "mph",
    "cm",
    "kg",
    "km",
)

COUNT_PREFIXES = (
    "claw",
    "lobster",
    "crab",
    "leg",
    "arm",
    "pincer",
    "antena",
    "shel",
    "tail",
    "fin",
    "egs",
    "friend",
    "rock",
)

FILLER_WORDS = {"um", "uh", "er", "like", "and", "of", "the", "its", "so"}


def _next_word(tokens: List[str], index: int) -> str:
    for token in tokens[index : index + 3]:
        if token not in FILLER_WORDS:
            return token
    return ""


def _is_measurement(word: str) -> bool:
    return any(word.startswith(prefix) for prefix in MEASUREMENT_PREFIXES)


def _is_count(word: str) -> bool:
    return any(word.startswith(prefix) for prefix in COUNT_PREFIXES)


def extract_numbers(tokens: List[str]) -> List[float]:
    return [value for value, _end in extract_numbers_with_positions(tokens)]


def extract_numbers_with_positions(tokens: List[str]) -> List[Tuple[float, int]]:
    numbers: List[Tuple[float, int]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token):
            numbers.append((float(token), i + 1))
            i += 1
            continue
        parts, span = _number_parts(tokens, i)
        if parts is None or all(kind == "point" for kind, _ in parts):
            i += 1
            continue
        group = list(parts)
        i += span
        while i < len(tokens):
            more, more_span = _number_parts(tokens, i)
            if more is None:
                break
            if group[-1][0] == "unit" and more[0][0] in ("unit", "tens") and group[-1][1] >= 10:
                break
            if group[-1][0] == "tens" and more[0][0] == "tens":
                break
            group.extend(more)
            i += more_span
        value = _parts_to_value(group)
        if value is not None:
            numbers.append((value, i))
    return numbers


def _is_per_item_total(tokens: List[str], positions: List[Tuple[float, int]]) -> bool:
    if len(positions) != 2:
        return False
    first_word = _next_word(tokens, positions[0][1])
    second_word = _next_word(tokens, positions[1][1])
    return _is_measurement(first_word) and _is_count(second_word)


def _keyword_operation(joined: str, keywords: List[Tuple[str, str]]) -> Optional[str]:
    for keyword, op in keywords:
        if _collapse(keyword) in joined:
            return op
    return None


def _symbol_operation(challenge_text: str) -> Optional[str]:
    for raw in (challenge_text or "").split():
        if raw in SYMBOL_OPERATIONS:
            return SYMBOL_OPERATIONS[raw]
    return None


def detect_operation(tokens: List[str], challenge_text: str = "") -> Optional[str]:
    joined = "".join(t for t in tokens if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t))
    return (
        _keyword_operation(joined, OPERATION_KEYWORDS)
        or _symbol_operation(challenge_text)
        or _keyword_operation(joined, FALLBACK_KEYWORDS)
    )


def solve_challenge(challenge_text: str) -> Dict[str, object]:
    tokens = clean_tokens(challenge_text)
    positions = extract_numbers_with_positions(tokens)
    numbers = [value for value, _end in positions]
    op = detect_operation(tokens, challenge_text)
    if op == "+" and _is_per_item_total(tokens, positions):
        op = "*"
    result: Dict[str, object] = {
        "decoded": " ".join(tokens),
        "numbers": numbers,
        "operation": op,
        "answer": None,
        "confident": False,
    }
    if len(numbers) != 2 or op is None:
        return result
    a, b = numbers
    if op == "+":
        value = a + b
    elif op == "-":
        value = a - b
    elif op == "*":
        value = a * b
    else:
        if b == 0:
            return result
        value = a / b
    result["answer"] = f"{value:.2f}"
    result["confident"] = True
    return result
