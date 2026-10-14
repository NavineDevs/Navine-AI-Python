import math
import re
from typing import Any, Dict, Optional, Tuple

MATH_CONSTANTS = {
    "pi": math.pi,
    "π": math.pi,
    "e": math.e,
}

PI_DIGIT_STRING = (
    "314159265358979323846264338327950288419716939937510582097494459"
    "23078164062862089986280348253421170679"
)

E_DIGIT_STRING = (
    "271828182845904523536028747135266249775724709663995378314528862"
    "373095318781608941422716039886"
)

MATH_FUNCTIONS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "abs": abs,
    "pow": pow,
}

MATH_QUERY_PATTERNS = [
    re.compile(r"\b(?:what(?:'s|s| is| are)|whats)\s+(?:the\s+)?(?:square\s+root|sqrt)\b", re.I),
    re.compile(r"\b(?:square\s+root|sqrt)\s+(?:of|to)\b", re.I),
    re.compile(r"\b(?:calculate|compute|evaluate|solve)\s+", re.I),
    re.compile(r"\bwhat(?:'s|s| is)\s+[\dπpi+\-*/^().\s]+", re.I),
    re.compile(r"^[\dπpi+\-*/^().\s]+\?$", re.I),
    re.compile(r"\b\d+\s*[\+\-\*/^%]\s*\d+", re.I),
    re.compile(
        r"\b\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied\s+by|divided\s+by|over)\s+\d+(?:\.\d+)?\b",
        re.I,
    ),
    re.compile(
        r"\bwhat(?:'s|s| is)\s+\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied\s+by|divided\s+by|over)\s+\d+",
        re.I,
    ),
    re.compile(r"\b\d+\s+(?:digits?|decimal places?|decimals?)\s+(?:of\s+)?(?:pi|π|e)\b", re.I),
    re.compile(r"\b(?:first\s+)?\d+\s+digits?\s+(?:of\s+)?(?:pi|π|e)\b", re.I),
    re.compile(r"\b(?:pi|π|e)\s+(?:to|with)\s+\d+\s+(?:digits?|decimal places?|decimals?)\b", re.I),
    re.compile(r"\bwhat(?:'s|s| is| are)\s+(?:the\s+)?(?:value\s+of\s+)?(?:pi|π|e)\b", re.I),
    re.compile(r"\b\d{3,4}\s*(?:-|\+|minus|plus)\s*\d+\s*years?\b", re.I),
    re.compile(r"\b(?:what|if)\b.+\b\d{3,4}\b.+\b\d+\s*years?\b", re.I),
    re.compile(r"\b(?:years?\s+(?:ago|from\s+now|later))\b", re.I),
    re.compile(
        r"\b(?:how\s+many|how\s+much|word\s+problem|math\s+problem|story\s+problem)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:apples?|oranges?|bananas?|cookies?|cand(?:y|ies)|tickets?|dollars?|miles?|hours?|minutes?)\b.+\b\d+\b",
        re.I,
    ),
    re.compile(
        r"\b\d+\b.+\b(?:apples?|oranges?|bananas?|cookies?|cand(?:y|ies)|tickets?|dollars?|miles?)\b",
        re.I,
    ),
    re.compile(
        r"\bif\s+.+\b(?:has|have|buys?|sells?|gives?|takes?|adds?|removes?)\b.+\b\d+\b",
        re.I,
    ),
]

YEAR_MINUS_PATTERN = re.compile(
    r"(?:if\s+(?:it\s+)?(?:was|were)\s+)?(\d{3,4})\s*(?:-|\u2212|minus)\s*(\d+)\s*years?",
    re.I,
)
YEAR_PLUS_PATTERN = re.compile(
    r"(?:if\s+(?:it\s+)?(?:was|were)\s+)?(\d{3,4})\s*(?:\+|plus)\s*(\d+)\s*years?",
    re.I,
)
YEAR_AGO_PATTERN = re.compile(
    r"\b(\d+)\s*years?\s+ago\s+(?:from\s+)?(\d{3,4})\b|\b(\d{3,4})\s+(?:minus|-\s*)(\d+)\s*years?\s+ago\b",
    re.I,
)
CORRECTION_YEAR_PATTERN = re.compile(
    r"^(?:no[,.]?\s+)?(?:it\s+)?(?:would|should|will)\s+be\s+(\d{3,4})\b",
    re.I,
)
CORRECTION_YEAR_LOOSE = re.compile(
    r"\bno\b.+\b(?:would|should)\s+be\s+(\d{3,4})\b",
    re.I,
)

ROOT_OF_PATTERN = re.compile(
    r"(?:square\s+root|sqrt)\s+(?:of|to)\s+(.+?)(?:\?|$)",
    re.I,
)

DIRECT_EXPR_PATTERN = re.compile(
    r"(?:what(?:'s|s| is)|whats|calculate|compute|evaluate|solve)\s+(.+?)(?:\?|$)",
    re.I,
)

DIGITS_OF_PATTERN = re.compile(
    r"(?:what(?:'s|s| is| are)|whats|give me|show(?: me)?|tell me)?\s*(?:the\s+)?(?:first\s+)?(\d+)\s+"
    r"(digits?|decimal places?|decimals?)\s+(?:of\s+)?(pi|π|e)\b",
    re.I,
)

DIGITS_OF_REVERSE_PATTERN = re.compile(
    r"(?:(?:what(?:'s|s| is)|whats)\s+)?(pi|π|e)\s+(?:to|with)\s+(\d+)\s+(digits?|decimal places?|decimals?)\b",
    re.I,
)

CONSTANT_VALUE_PATTERN = re.compile(
    r"(?:what(?:'s|s| is| are)|whats)\s+(?:the\s+)?(?:value\s+of\s+)?(pi|π|e)\b",
    re.I,
)


def is_math_query(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    if CORRECTION_YEAR_PATTERN.search(stripped) or CORRECTION_YEAR_LOOSE.search(stripped):
        return True
    if YEAR_MINUS_PATTERN.search(stripped) or YEAR_PLUS_PATTERN.search(stripped):
        return True
    for pattern in MATH_QUERY_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def _answer_year_arithmetic(message: str) -> Optional[str]:
    stripped = message.strip()
    corr = CORRECTION_YEAR_PATTERN.search(stripped) or CORRECTION_YEAR_LOOSE.search(stripped)
    if corr:
        year = int(corr.group(1))
        return f"Yes - {year}."
    match = YEAR_MINUS_PATTERN.search(stripped)
    if match:
        base = int(match.group(1))
        years = int(match.group(2))
        result = base - years
        return f"{base} minus {years} years is {result}."
    match = YEAR_PLUS_PATTERN.search(stripped)
    if match:
        base = int(match.group(1))
        years = int(match.group(2))
        result = base + years
        return f"{base} plus {years} years is {result}."
    match = YEAR_AGO_PATTERN.search(stripped)
    if match:
        if match.group(1) and match.group(2):
            years = int(match.group(1))
            base = int(match.group(2))
            return f"{years} years before {base} is {base - years}."
        if match.group(3) and match.group(4):
            base = int(match.group(3))
            years = int(match.group(4))
            return f"{years} years before {base} is {base - years}."
    return None


def _display_constant(name: str) -> str:
    lower = name.lower()
    if lower in ("pi", "π"):
        return "π"
    return "e"


def _digit_source(name: str) -> Optional[str]:
    lower = name.lower()
    if lower in ("pi", "π"):
        return PI_DIGIT_STRING
    if lower == "e":
        return E_DIGIT_STRING
    return None


def _format_first_digits(name: str, count: int) -> Optional[str]:
    digits = _digit_source(name)
    if not digits or count < 1 or count > len(digits):
        return None
    if count == 1:
        return digits[0]
    return f"{digits[0]}.{digits[1:count]}"


def _format_decimal_places(name: str, count: int) -> Optional[str]:
    digits = _digit_source(name)
    if not digits or count < 1 or count > len(digits) - 1:
        return None
    return f"{digits[0]}.{digits[1:count + 1]}"


def _uses_decimal_places(unit: str) -> bool:
    return bool(re.search(r"decimal", unit, re.I))


def _parse_digits_query(message: str) -> Optional[Tuple[str, int, bool]]:
    match = DIGITS_OF_PATTERN.search(message)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        constant = match.group(3)
        return constant, count, _uses_decimal_places(unit)
    match = DIGITS_OF_REVERSE_PATTERN.search(message)
    if match:
        constant = match.group(1)
        count = int(match.group(2))
        unit = match.group(3)
        return constant, count, _uses_decimal_places(unit)
    return None


def _answer_constant_value(constant: str) -> str:
    display = _display_constant(constant)
    if constant.lower() in ("pi", "π"):
        return f"The value of {display} is approximately {_format_number(math.pi)}."
    return f"The value of {display} is approximately {_format_number(math.e)}."


def _answer_digits_query(constant: str, count: int, decimal_places: bool) -> Optional[str]:
    display = _display_constant(constant)
    if decimal_places:
        formatted = _format_decimal_places(constant, count)
        if not formatted:
            return None
        return f"{display} to {count} decimal places is {formatted}."
    formatted = _format_first_digits(constant, count)
    if not formatted:
        return None
    return f"The first {count} digits of {display} are {formatted}."


def _normalize_expression(text: str) -> str:
    expr = text.strip().rstrip("?.!")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.replace("π", "pi")
    expr = re.sub(r"\bmultiplied\s+by\b", "*", expr, flags=re.I)
    expr = re.sub(r"\bdivided\s+by\b", "/", expr, flags=re.I)
    expr = re.sub(r"\btimes\b", "*", expr, flags=re.I)
    expr = re.sub(r"\bplus\b", "+", expr, flags=re.I)
    expr = re.sub(r"\bminus\b", "-", expr, flags=re.I)
    expr = re.sub(r"\bover\b", "/", expr, flags=re.I)
    expr = re.sub(r"\bsquare\s+root\b", "sqrt", expr, flags=re.I)
    expr = re.sub(r"\bsqrt\s+(?:of|to)\s+(\w+)\b", r"sqrt(\1)", expr, flags=re.I)
    expr = re.sub(r"\bsqrt\s+(\w+)\b", r"sqrt(\1)", expr, flags=re.I)
    expr = re.sub(r"\s+", " ", expr)
    open_count = expr.count("(")
    close_count = expr.count(")")
    if open_count > close_count:
        expr = expr + (")" * (open_count - close_count))
    return expr


def _extract_expression(message: str) -> Optional[str]:
    match = ROOT_OF_PATTERN.search(message)
    if match:
        operand = match.group(1).strip().rstrip("?.!")
        return _normalize_expression(f"sqrt {operand}")
    match = DIRECT_EXPR_PATTERN.search(message)
    if match:
        candidate = match.group(1).strip().rstrip("?.!")
        if re.search(
            r"[\d+\-*/().]|sqrt|pi|π|plus|minus|times|divided|multiplied|over",
            candidate,
            re.I,
        ):
            return _normalize_expression(candidate)
    stripped = message.strip().rstrip("?.!")
    if re.fullmatch(
        r"[\dπpi+\-*/().\s]+|(?:\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied by|divided by|over)\s+\d+(?:\.\d+)?)",
        stripped,
        re.I,
    ):
        return _normalize_expression(stripped)
    word_math = re.search(
        r"(\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied\s+by|divided\s+by|over)\s+(\d+(?:\.\d+)?)",
        message,
        re.I,
    )
    if word_math:
        return _normalize_expression(word_math.group(0))
    embedded = re.search(
        r"(\d+(?:\.\d+)?)\s*([\+\-\*/^%])\s*(\d+(?:\.\d+)?)",
        message,
    )
    if embedded:
        return _normalize_expression(f"{embedded.group(1)}{embedded.group(2)}{embedded.group(3)}")
    return None


def _safe_eval(expr: str) -> Optional[float]:
    namespace: Dict[str, Any] = {"__builtins__": {}}
    namespace.update(MATH_CONSTANTS)
    namespace.update(MATH_FUNCTIONS)
    try:
        result = eval(expr, namespace, {})
    except Exception:
        return None
    if isinstance(result, bool):
        return None
    if isinstance(result, (int, float)):
        value = float(result)
        if not math.isfinite(value):
            return None
        return value
    return None


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.12g}"


def _display_operand(operand: str) -> str:
    return operand.replace("pi", "π")


def _parse_number_token(token: str) -> float:
    return float(token) if "." in token else float(int(token))


def _answer_word_problem(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    lower = text.lower()
    nums = [_parse_number_token(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) < 2:
        return None
    a, b = nums[0], nums[1]
    if re.search(
        r"\b(?:gives?(?:\s+away)?|sells?|eats?|loses?|removes?|spends?|takes?\s+away|subtracts?)\b",
        lower,
    ):
        result = a - b
        return f"{_format_number(a)} minus {_format_number(b)} = {_format_number(result)}."
    if re.search(
        r"\b(?:buys?|gets?|adds?|receives?|finds?|gains?|more|plus|altogether|in\s+total|total)\b",
        lower,
    ):
        result = a + b
        return f"{_format_number(a)} plus {_format_number(b)} = {_format_number(result)}."
    if re.search(r"\b(?:times|each|per|multipl)\b", lower):
        result = a * b
        return f"{_format_number(a)} times {_format_number(b)} = {_format_number(result)}."
    if re.search(r"\b(?:shared|split|divided|divide|among|between|each\s+gets?)\b", lower):
        if b == 0:
            return None
        result = a / b
        return f"{_format_number(a)} divided by {_format_number(b)} = {_format_number(result)}."
    if re.search(r"\b(?:how\s+many|how\s+much|left|remain(?:ing)?|now)\b", lower):
        if re.search(r"\b(?:left|remain(?:ing)?)\b", lower):
            result = a - b
            return f"{_format_number(a)} minus {_format_number(b)} = {_format_number(result)}."
        result = a + b
        return f"{_format_number(a)} plus {_format_number(b)} = {_format_number(result)}."
    return None


def answer_math_query(message: str) -> Optional[str]:
    if not is_math_query(message):
        return None
    year_answer = _answer_year_arithmetic(message)
    if year_answer:
        return year_answer
    digits_query = _parse_digits_query(message)
    if digits_query:
        constant, count, decimal_places = digits_query
        return _answer_digits_query(constant, count, decimal_places)
    if CONSTANT_VALUE_PATTERN.search(message) and not ROOT_OF_PATTERN.search(message):
        match = CONSTANT_VALUE_PATTERN.search(message)
        if match:
            return _answer_constant_value(match.group(1))
    expr = _extract_expression(message)
    if expr:
        value = _safe_eval(expr)
        if value is not None:
            formatted = _format_number(value)
            root_match = re.fullmatch(r"sqrt\((.+)\)", expr, flags=re.I)
            if root_match:
                operand_display = _display_operand(root_match.group(1))
                return f"The square root of {operand_display} is approximately {formatted}."
            display = expr.replace("pi", "π").replace("**", "^")
            return f"{display} = {formatted}"
    word = _answer_word_problem(message)
    if word:
        return word
    return None
