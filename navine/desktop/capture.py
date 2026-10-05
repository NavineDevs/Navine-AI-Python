import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.desktop.config import audit_desktop, load_desktop_config, output_dir, screen_enabled


def _grab_pil(path: Path) -> Dict[str, Any]:
    from PIL import ImageGrab

    image = ImageGrab.grab()
    image.save(path, format="PNG")
    return {"ok": True, "path": str(path), "size": list(image.size), "backend": "pillow"}


def _grab_pyautogui(path: Path) -> Dict[str, Any]:
    import pyautogui

    shot = pyautogui.screenshot()
    shot.save(str(path))
    size = getattr(shot, "size", None)
    return {
        "ok": True,
        "path": str(path),
        "size": list(size) if size else None,
        "backend": "pyautogui",
    }


def capture_screen(
    path: Optional[str] = None,
    force: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_desktop_config()
    if not force and not screen_enabled(cfg):
        return {
            "ok": False,
            "error": "Screen capture disabled. Enable Allow screen in the UI or configs/desktop.yaml.",
        }
    out = Path(path) if path else output_dir(cfg) / f"screen_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any]
    try:
        result = _grab_pil(out)
    except Exception:
        try:
            result = _grab_pyautogui(out)
        except Exception as exc:
            result = {"ok": False, "error": f"{exc}. Install pillow (already required) or pyautogui."}
    if result.get("ok"):
        win = active_window()
        if win.get("ok"):
            result["active_window"] = win.get("title")
            result["active_pid"] = win.get("pid")
    audit_desktop("capture_screen", {"ok": result.get("ok"), "path": result.get("path")}, cfg)
    return result


def active_window() -> Dict[str, Any]:
    if platform.system() != "Windows":
        return {"ok": False, "error": "Active window lookup is Windows-only"}
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return {"ok": False, "error": "No foreground window"}
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return {"ok": True, "title": buf.value, "hwnd": int(hwnd), "pid": int(pid.value)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_windows(limit: int = 40) -> Dict[str, Any]:
    if platform.system() != "Windows":
        return {"ok": False, "error": "Window listing is Windows-only", "windows": []}
    windows: List[Dict[str, Any]] = []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        GetWindowText = user32.GetWindowTextW
        GetWindowTextLength = user32.GetWindowTextLengthW
        IsWindowVisible = user32.IsWindowVisible

        @EnumWindowsProc
        def _callback(hwnd, _lparam):
            if len(windows) >= max(1, min(int(limit), 200)):
                return False
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLength(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buf, length + 1)
            title = (buf.value or "").strip()
            if title:
                windows.append({"title": title, "hwnd": int(hwnd)})
            return True

        EnumWindows(_callback, 0)
        return {"ok": True, "windows": windows, "count": len(windows)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "windows": windows}
