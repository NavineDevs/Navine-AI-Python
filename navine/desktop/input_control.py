import platform
import time
from typing import Any, Dict, List, Optional

from navine.desktop.config import (
    audit_desktop,
    input_enabled,
    load_desktop_config,
)


def _require_input(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not input_enabled(cfg):
        return {
            "ok": False,
            "error": "Desktop input is disabled. Enable Allow input in the UI or set allow_input: true in configs/desktop.yaml.",
        }
    return None


def _pyautogui():
    import pyautogui

    pyautogui.FAILSAFE = True
    return pyautogui


def type_text(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_desktop_config()
    blocked = _require_input(cfg)
    if blocked:
        return blocked
    payload = str(text or "")
    limit = int(cfg.get("max_type_chars") or 2000)
    if len(payload) > limit:
        return {"ok": False, "error": f"Text exceeds max_type_chars ({limit})"}
    if not payload:
        return {"ok": False, "error": "No text provided"}
    try:
        gui = _pyautogui()
        gui.write(payload, interval=0.02)
        result = {"ok": True, "chars": len(payload), "text": f"Typed {len(payload)} characters."}
    except Exception as exc:
        result = {"ok": False, "error": f"{exc}. Install: pip install pyautogui"}
    audit_desktop("type_text", {"ok": result.get("ok"), "chars": len(payload)}, cfg)
    return result


def click_at(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_desktop_config()
    blocked = _require_input(cfg)
    if blocked:
        return blocked
    try:
        xi = int(x)
        yi = int(y)
    except Exception:
        return {"ok": False, "error": "x and y must be integers"}
    if xi < 0 or yi < 0:
        return {"ok": False, "error": "Coordinates must be non-negative"}
    btn = (button or "left").lower()
    if btn not in ("left", "right", "middle"):
        return {"ok": False, "error": f"Unsupported button: {button}"}
    n = max(1, min(int(clicks or 1), 5))
    delay = max(0, int(cfg.get("click_delay_ms") or 50)) / 1000.0
    try:
        gui = _pyautogui()
        time.sleep(delay)
        gui.click(x=xi, y=yi, clicks=n, button=btn)
        result = {
            "ok": True,
            "x": xi,
            "y": yi,
            "button": btn,
            "clicks": n,
            "text": f"Clicked {btn} at ({xi}, {yi}).",
        }
    except Exception as exc:
        result = {"ok": False, "error": f"{exc}. Install: pip install pyautogui"}
    audit_desktop("click_at", {"ok": result.get("ok"), "x": xi, "y": yi, "button": btn}, cfg)
    return result


def press_hotkey(keys: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_desktop_config()
    blocked = _require_input(cfg)
    if blocked:
        return blocked
    raw = str(keys or "").strip()
    if not raw:
        return {"ok": False, "error": "No keys provided"}
    sequence: List[str] = []
    for part in raw.replace("|", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if "+" in token and not token.startswith("+"):
            sequence.append(token)
        else:
            sequence.append(token)
    if not sequence:
        return {"ok": False, "error": "No keys provided"}
    sequence = sequence[:12]
    try:
        gui = _pyautogui()
        for key in sequence:
            if "+" in key and not key.startswith("+"):
                parts = [p.strip() for p in key.split("+") if p.strip()]
                if len(parts) >= 2:
                    gui.hotkey(*parts)
                    continue
            gui.press(key)
            time.sleep(0.05)
        result = {"ok": True, "keys": sequence, "text": f"Sent keys: {', '.join(sequence)}"}
    except Exception as exc:
        result = {"ok": False, "error": f"{exc}. Install: pip install pyautogui"}
    audit_desktop("press_hotkey", {"ok": result.get("ok"), "keys": sequence}, cfg)
    return result


def input_supported() -> bool:
    if platform.system() != "Windows":
        return True
    try:
        import pyautogui  # noqa: F401

        return True
    except Exception:
        return False
