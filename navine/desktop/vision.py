from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.desktop.capture import active_window, capture_screen
from navine.desktop.config import load_desktop_config, screen_enabled


def _dominant_colors(image, count: int = 4) -> List[str]:
    small = image.convert("RGB").resize((48, 48))
    pixels = list(small.getdata())
    buckets: Dict[tuple, int] = {}
    for r, g, b in pixels:
        key = (r // 32 * 32, g // 32 * 32, b // 32 * 32)
        buckets[key] = buckets.get(key, 0) + 1
    ranked = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:count]
    return [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in ranked]


def _brightness(image) -> float:
    gray = image.convert("L").resize((64, 64))
    data = list(gray.getdata())
    if not data:
        return 0.0
    return round(sum(data) / (len(data) * 255.0), 3)


def _optional_ocr(path: Path) -> Optional[str]:
    try:
        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(Image.open(path))
        cleaned = " ".join((text or "").split())
        return cleaned[:1200] if cleaned else None
    except Exception:
        return None


def analyze_image_file(path: str) -> Dict[str, Any]:
    from PIL import Image

    image = Image.open(path)
    width, height = image.size
    colors = _dominant_colors(image)
    bright = _brightness(image)
    ocr = _optional_ocr(Path(path))
    return {
        "ok": True,
        "path": str(path),
        "width": width,
        "height": height,
        "mode": image.mode,
        "dominant_colors": colors,
        "brightness": bright,
        "ocr_text": ocr,
    }


def build_screen_summary(
    capture: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    window: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    if window and window.get("ok") and window.get("title"):
        parts.append(f"Active window: {window.get('title')}")
    if capture.get("path"):
        parts.append(f"Screenshot saved: {capture.get('path')}")
    if analysis and analysis.get("ok"):
        parts.append(
            f"Resolution: {analysis.get('width')}x{analysis.get('height')} ({analysis.get('mode')})"
        )
        parts.append(f"Brightness: {analysis.get('brightness')}")
        colors = analysis.get("dominant_colors") or []
        if colors:
            parts.append("Dominant colors: " + ", ".join(colors))
        ocr = analysis.get("ocr_text")
        if ocr:
            parts.append(f"Visible text (OCR): {ocr}")
    return "\n".join(parts) if parts else "Screenshot captured with no extra analysis."


def describe_image_bytes(
    data: bytes,
    question: Optional[str] = None,
    use_model: bool = True,
    mime: str = "image/jpeg",
    label: str = "camera",
) -> Dict[str, Any]:
    from datetime import datetime, timezone

    from navine.utils.paths import get_project_root

    ext = ".jpg"
    lower = (mime or "").lower()
    if "png" in lower:
        ext = ".png"
    elif "webp" in lower:
        ext = ".webp"
    elif "gif" in lower:
        ext = ".gif"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = get_project_root() / "outputs" / "camera"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}_{stamp}{ext}"
    path.write_bytes(data)
    analysis: Dict[str, Any] = {"ok": False}
    try:
        analysis = analyze_image_file(str(path))
    except Exception as exc:
        analysis = {"ok": False, "error": str(exc)}
    summary_parts: List[str] = [f"{label.title()} frame saved: {path}"]
    if analysis.get("ok"):
        summary_parts.append(
            f"Resolution: {analysis.get('width')}x{analysis.get('height')} ({analysis.get('mode')})"
        )
        summary_parts.append(f"Brightness: {analysis.get('brightness')}")
        colors = analysis.get("dominant_colors") or []
        if colors:
            summary_parts.append("Dominant colors: " + ", ".join(colors))
        ocr = analysis.get("ocr_text")
        if ocr:
            summary_parts.append(f"Visible text (OCR): {ocr}")
    summary = "\n".join(summary_parts)
    result: Dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "summary": summary,
        "analysis": analysis if analysis.get("ok") else None,
        "text": summary,
        "label": label,
    }
    if not use_model:
        return result
    ask = (question or "Describe what you see in this camera frame.").strip()
    prompt = (
        f"{ask}\n\n"
        f"Use only the {label} observation below. Be concrete and practical.\n\n"
        f"### {label.title()} observation\n{summary}"
    )
    try:
        from navine.text.chat import chat_with_meta

        reply, _meta = chat_with_meta(
            prompt,
            history=None,
            use_rag=False,
            use_search=False,
            model_profile="analyze",
        )
        text = (reply or "").strip() or summary
        result["text"] = text
        result["model_used"] = True
    except Exception as exc:
        result["model_used"] = False
        result["model_error"] = str(exc)
        result["text"] = summary
    return result


def describe_screen(
    question: Optional[str] = None,
    use_model: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_desktop_config()
    if not screen_enabled(cfg):
        return {
            "ok": False,
            "error": "Screen capture disabled. Enable Allow screen in the UI or configs/desktop.yaml.",
        }
    capture = capture_screen(force=True, config=cfg)
    if not capture.get("ok"):
        return capture
    window = active_window()
    analysis: Dict[str, Any] = {"ok": False}
    try:
        analysis = analyze_image_file(str(capture["path"]))
    except Exception as exc:
        analysis = {"ok": False, "error": str(exc)}
    summary = build_screen_summary(capture, analysis if analysis.get("ok") else None, window)
    result: Dict[str, Any] = {
        "ok": True,
        "path": capture.get("path"),
        "summary": summary,
        "active_window": window.get("title") if window.get("ok") else None,
        "analysis": analysis if analysis.get("ok") else None,
        "text": summary,
    }
    if not use_model:
        return result
    ask = (question or "Describe what is on my screen and what I appear to be doing.").strip()
    prompt = (
        f"{ask}\n\n"
        "Use only the screen observation below. Be concrete and practical.\n\n"
        f"### Screen observation\n{summary}"
    )
    try:
        from navine.text.chat import chat_with_meta

        reply, _meta = chat_with_meta(
            prompt,
            history=None,
            use_rag=False,
            use_search=False,
            model_profile="analyze",
        )
        text = (reply or "").strip() or summary
        result["text"] = text
        result["model_used"] = True
    except Exception as exc:
        result["model_used"] = False
        result["model_error"] = str(exc)
        result["text"] = summary
    return result
