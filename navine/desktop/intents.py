import re
from typing import Any, Dict, Optional


_SCREEN_SEE = re.compile(
    r"\b("
    r"look at (my |the )?screen|"
    r"see (my |the )?screen|"
    r"what('?s| is) on (my |the )?screen|"
    r"describe (my |the )?screen|"
    r"screen describe|"
    r"capture (my |the )?screen|"
    r"take a screenshot|"
    r"screenshot|"
    r"screen shot|"
    r"what do you see|"
    r"can you see (my |the )?screen"
    r")\b",
    re.I,
)

_TYPE = re.compile(
    r"^(?:please |could you |can you )?(?:desktop\s+)?(?:type|type out|enter text)\s+(.+)$",
    re.I,
)
_CLICK = re.compile(
    r"^(?:please |could you |can you )?(?:desktop\s+)?click(?:\s+(?:at|on))?\s+"
    r"(?:x\s*=\s*)?(-?\d+)\s*(?:,|\s+)\s*(?:y\s*=\s*)?(-?\d+)"
    r"(?:\s+(left|right|middle))?$",
    re.I,
)
_HOTKEY = re.compile(
    r"^(?:please |could you |can you )?(?:desktop\s+)?(?:press|hotkey|send keys?)\s+(.+)$",
    re.I,
)


def parse_desktop_intent(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()

    m_type = _TYPE.match(raw)
    if m_type:
        return {"action": "desktop_type", "args": {"text": m_type.group(1).strip()}}

    m_click = _CLICK.match(raw)
    if m_click:
        return {
            "action": "desktop_click",
            "args": {
                "x": int(m_click.group(1)),
                "y": int(m_click.group(2)),
                "button": (m_click.group(3) or "left").lower(),
            },
        }

    m_hot = _HOTKEY.match(raw)
    if m_hot and "screenshot" not in lowered:
        keys = m_hot.group(1).strip()
        if keys:
            return {"action": "desktop_hotkey", "args": {"keys": keys}}

    if _SCREEN_SEE.search(raw):
        capture_only = any(
            k in lowered
            for k in ("screenshot", "screen shot", "capture screen", "take a picture of the screen")
        ) and not any(
            k in lowered
            for k in (
                "look",
                "see my screen",
                "see the screen",
                "what's on",
                "whats on",
                "what is on",
                "describe",
                "what do you see",
                "can you see",
            )
        )
        if capture_only:
            return {"action": "screen_capture", "args": {}}
        return {"action": "screen_describe", "args": {"question": raw}}

    return None


def handle_desktop_chat_message(text: str) -> Optional[Dict[str, Any]]:
    intent = parse_desktop_intent(text)
    if intent is None:
        return None
    action = intent.get("action")
    args = intent.get("args") or {}
    if action == "screen_describe":
        from navine.desktop.vision import describe_screen

        result = describe_screen(question=args.get("question") or text)
        result["action"] = action
        return result
    if action == "screen_capture":
        from navine.desktop.capture import capture_screen

        result = capture_screen()
        if result.get("ok"):
            result["text"] = f"Screenshot saved to {result.get('path')}"
        result["action"] = action
        return result
    if action == "desktop_type":
        from navine.desktop.input_control import type_text

        result = type_text(args.get("text", ""))
        result["action"] = action
        return result
    if action == "desktop_click":
        from navine.desktop.input_control import click_at

        result = click_at(args.get("x", 0), args.get("y", 0), button=args.get("button", "left"))
        result["action"] = action
        return result
    if action == "desktop_hotkey":
        from navine.desktop.input_control import press_hotkey

        result = press_hotkey(args.get("keys", ""))
        result["action"] = action
        return result
    return None
