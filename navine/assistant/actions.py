import json
import os
import platform
import shutil
import subprocess
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.config import load_config
from navine.utils.paths import get_project_root

READ_ONLY_ACTIONS = {
    "system_info",
    "file_read",
    "list_dir",
    "clipboard_read",
    "datetime_now",
    "capabilities",
    "game_coach",
    "screenshot",
    "screen_describe",
    "discord_webhook_status",
}


def load_assistant_config() -> Dict[str, Any]:
    try:
        return load_config("assistant")
    except FileNotFoundError:
        return {"enabled": True, "consent": {"require_confirmation": True}}


def _audit_path(config: Dict[str, Any]) -> Path:
    rel = str((config.get("logging") or {}).get("audit_file") or "logs/assistant/audit.log")
    path = get_project_root() / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def audit(action: str, detail: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
    cfg = config or load_assistant_config()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "detail": detail,
    }
    with _audit_path(cfg).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def is_read_only(action: str) -> bool:
    return action in READ_ONLY_ACTIONS


def is_dangerous(command: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    try:
        from navine.sandbox import guard_shell, load_sandbox_config

        blocked = guard_shell(command, load_sandbox_config())
        if blocked:
            return blocked
    except Exception:
        pass
    cfg = config or load_assistant_config()
    safety = cfg.get("safety") or {}
    if not safety.get("block_dangerous", True):
        return None
    lowered = str(command or "").lower()
    for pattern in safety.get("dangerous_patterns") or []:
        if str(pattern).lower() in lowered:
            return str(pattern)
    return None


def capability_enabled(name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import guard_capability, load_sandbox_config

        blocked = guard_capability(name, load_sandbox_config())
        if blocked:
            return False
    except Exception:
        pass
    caps = cfg.get("capabilities") or {}
    return bool(caps.get(name, False))


def run_command(command: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_shell, load_sandbox_config

        blocked = guard_shell(command, load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    danger = is_dangerous(command, cfg)
    if danger:
        return {"ok": False, "error": f"Blocked dangerous command pattern: {danger}", "sandboxed": True}
    timeout = int((cfg.get("safety") or {}).get("max_command_seconds", 120))
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(get_project_root()),
        )
        result = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        result = {"ok": False, "error": f"Command timed out after {timeout}s"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("run_command", {"command": command, "ok": result.get("ok")}, cfg)
    return result


def open_app(target: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_capability, load_sandbox_config

        blocked = guard_capability("open_app", load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(target)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
        result = {"ok": True, "opened": target}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("open_app", {"target": target, "ok": result.get("ok")}, cfg)
    return result


def open_url(url: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_capability, load_sandbox_config

        blocked = guard_capability("open_url", load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        webbrowser.open(url)
        result = {"ok": True, "url": url}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("open_url", {"url": url, "ok": result.get("ok")}, cfg)
    return result


def system_info(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from navine.realtime.host_specs import collect_host_specs, format_host_specs_reply

        specs = collect_host_specs(force=True)
        info: Dict[str, Any] = {"ok": True, **specs, "text": format_host_specs_reply(specs)}
        return info
    except Exception:
        pass
    info = {
        "ok": True,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        usage = shutil.disk_usage(str(get_project_root()))
        info["disk_total_gb"] = round(usage.total / (1024 ** 3), 1)
        info["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        import psutil

        info["memory_percent"] = psutil.virtual_memory().percent
        info["cpu_percent"] = psutil.cpu_percent(interval=0.3)
    except Exception:
        info["memory_percent"] = None
    return info


def datetime_now(kind: str = "both", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    spoken_time = now.strftime("%I:%M %p").lstrip("0")
    spoken_date = now.strftime("%A, %B %d, %Y")
    kind = (kind or "both").lower()
    if kind == "time":
        text = f"It is {spoken_time}."
    elif kind == "date":
        text = f"Today is {spoken_date}."
    else:
        text = f"It is {spoken_time} on {spoken_date}."
    return {
        "ok": True,
        "kind": kind,
        "iso": now.isoformat(),
        "time": spoken_time,
        "date": spoken_date,
        "text": text,
    }


def capabilities_summary(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    caps = cfg.get("capabilities") or {}
    enabled = [name.replace("_", " ") for name, on in caps.items() if on]
    if not enabled:
        enabled = ["chat answers", "basic system help"]
    text = (
        "I can chat, write code replies, and generate images/videos. "
        "MCP is supported: say mcp status, mcp tools <server>, or mcp call <server> <tool>. "
        "A contained sandbox is on: I cannot run shell commands, delete files, "
        "control your mouse/keyboard, or change your laptop outside the AI workspace."
    )
    try:
        from navine.sandbox import sandbox_status

        status = sandbox_status()
        text = f"{text} Sandbox mode={status.get('mode')} enabled={status.get('enabled')}."
    except Exception:
        pass
    return {"ok": True, "enabled": enabled, "text": text}


def list_dir(path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        blocked = guard_path(path, write=False, config=load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    target = Path(path).expanduser()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {target}"}
    entries = []
    for item in sorted(target.iterdir()):
        entries.append({"name": item.name, "dir": item.is_dir()})
    return {"ok": True, "path": str(target), "entries": entries[:500]}


def read_file(path: str, max_chars: int = 8000, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        blocked = guard_path(path, write=False, config=load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    target = Path(path).expanduser()
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"File not found: {target}"}
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "path": str(target), "content": text[:max_chars], "truncated": len(text) > max_chars}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def write_file(path: str, content: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        blocked = guard_path(path, write=True, config=load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        result = {
            "ok": True,
            "path": str(target),
            "bytes": len(content.encode("utf-8")),
            "text": f"Wrote {target.name}.",
        }
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("write_file", {"path": str(target), "ok": result.get("ok")}, cfg)
    return result


def append_file(path: str, content: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        blocked = guard_path(path, write=True, config=load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content if content.endswith("\n") else content + "\n")
        result = {"ok": True, "path": str(target), "text": f"Appended to {target.name}."}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("append_file", {"path": str(target), "ok": result.get("ok")}, cfg)
    return result


def make_dir(path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        blocked = guard_path(path, write=True, config=load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    target = Path(path).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
        result = {"ok": True, "path": str(target), "text": f"Created folder {target.name}."}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("make_dir", {"path": str(target), "ok": result.get("ok")}, cfg)
    return result


def rename_path(src: str, dst: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        sb = load_sandbox_config()
        for path in (src, dst):
            blocked = guard_path(path, write=True, config=sb)
            if blocked:
                return deny(blocked)
    except Exception:
        pass
    source = Path(src).expanduser()
    dest = Path(dst).expanduser()
    try:
        if not source.exists():
            return {"ok": False, "error": f"Path not found: {source}"}
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
        result = {"ok": True, "src": str(source), "dst": str(dest), "text": f"Renamed to {dest.name}."}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("rename_path", {"src": str(source), "dst": str(dest), "ok": result.get("ok")}, cfg)
    return result


def copy_path(src: str, dst: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_path, load_sandbox_config

        sb = load_sandbox_config()
        blocked_src = guard_path(src, write=False, config=sb)
        if blocked_src:
            return deny(blocked_src)
        blocked_dst = guard_path(dst, write=True, config=sb)
        if blocked_dst:
            return deny(blocked_dst)
    except Exception:
        pass
    source = Path(src).expanduser()
    dest = Path(dst).expanduser()
    try:
        if not source.exists():
            return {"ok": False, "error": f"Path not found: {source}"}
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)
        result = {"ok": True, "src": str(source), "dst": str(dest), "text": f"Copied to {dest.name}."}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("copy_path", {"src": str(source), "dst": str(dest), "ok": result.get("ok")}, cfg)
    return result


def observe_game(
    title: str = "game",
    seconds: float = 8.0,
    note: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.learn.gameplay import observe_game as _observe

        result = _observe(title=title, seconds=seconds, note=note)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("observe_game", {"title": title, "ok": result.get("ok")}, cfg)
    return result


def label_game_action(action: str, title: str = "", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.learn.gameplay import label_last_action

        result = label_last_action(action=action, title=title)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("label_game_action", {"action": action, "ok": result.get("ok")}, cfg)
    return result


def game_coach(title: str, note: str = "", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from navine.learn.gameplay import coach_advice

    text = coach_advice(title, note)
    return {"ok": True, "title": title, "text": text}


def train_domain(
    domain: str = "games",
    steps: int = 150,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    name = (domain or "games").strip().lower().replace(" ", "")
    if name in ("game", "gameplay", "games"):
        name = "games"
    if name in ("code", "coding"):
        name = "coding"
    if name in ("mm", "multimodal", "multimodaltext", "multimodal-text"):
        name = "multimodal-text"
    steps = max(10, min(int(steps or 150), 2000))
    try:
        if name == "image":
            from navine.image.train import train as train_image
            from navine.utils.tier import modality_config_name

            train_image(config_path=modality_config_name("image"), finetune=True, finetune_steps=steps)
        elif name == "video":
            from navine.video.train import train as train_video
            from navine.utils.tier import modality_config_name

            train_video(config_path=modality_config_name("video"), finetune=True, finetune_steps=steps)
        elif name == "text":
            from navine.text.train import train as train_text
            from navine.utils.tier import modality_config_name

            train_text(config_path=modality_config_name("text"), max_steps=steps)
        elif name == "all":
            from navine.train.registry import train_all

            train_all(steps=steps)
        else:
            from navine.train.registry import TRAINING_TYPES, train_type

            if name not in TRAINING_TYPES:
                return {"ok": False, "error": f"Unknown train domain: {domain}"}
            train_type(name, steps=steps)
        result = {
            "ok": True,
            "domain": name,
            "steps": steps,
            "text": f"Finished training for {name} ({steps} steps).",
        }
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "domain": name, "steps": steps}
    audit("train_domain", {"domain": name, "steps": steps, "ok": result.get("ok")}, cfg)
    return result


def send_keys(keys: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.desktop.input_control import press_hotkey

        result = press_hotkey(keys)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("send_keys", {"keys": keys, "ok": result.get("ok")}, cfg)
    return result


def desktop_type(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_capability, load_sandbox_config

        blocked = guard_capability("desktop_input", load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    try:
        from navine.desktop.input_control import type_text

        result = type_text(text)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("desktop_type", {"ok": result.get("ok"), "chars": len(text or "")}, cfg)
    return result


def desktop_click(
    x: int,
    y: int,
    button: str = "left",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_capability, load_sandbox_config

        blocked = guard_capability("desktop_input", load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    try:
        from navine.desktop.input_control import click_at

        result = click_at(x, y, button=button)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("desktop_click", {"ok": result.get("ok"), "x": x, "y": y}, cfg)
    return result


def screen_describe(question: str = "", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.desktop.vision import describe_screen

        result = describe_screen(question=question or None)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("screen_describe", {"ok": result.get("ok")}, cfg)
    return result


def delete_file(path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_delete, load_sandbox_config

        blocked = guard_delete(path, load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    target = Path(path).expanduser()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {target}"}
    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        result = {"ok": True, "deleted": str(target)}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("delete_file", {"path": str(target), "ok": result.get("ok")}, cfg)
    return result


def screenshot(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.desktop.capture import capture_screen

        result = capture_screen()
    except Exception as exc:
        result = {"ok": False, "error": f"{exc}. Install: pip install pillow"}
    audit("screenshot", {"ok": result.get("ok")}, cfg)
    return result


def clipboard_read(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        import pyperclip

        return {"ok": True, "content": pyperclip.paste()}
    except Exception as exc:
        return {"ok": False, "error": f"{exc}. Install: pip install pyperclip"}


def clipboard_write(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_capability, load_sandbox_config

        blocked = guard_capability("clipboard", load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    try:
        import pyperclip

        pyperclip.copy(text)
        result = {"ok": True, "copied_chars": len(text)}
    except Exception as exc:
        result = {"ok": False, "error": f"{exc}. Install: pip install pyperclip"}
    audit("clipboard_write", {"ok": result.get("ok")}, cfg)
    return result


def media_control(action: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.sandbox import deny, guard_capability, load_sandbox_config

        blocked = guard_capability("media_control", load_sandbox_config())
        if blocked:
            return deny(blocked)
    except Exception:
        pass
    if platform.system() != "Windows":
        return {"ok": False, "error": "media_control currently supports Windows only"}
    key_map = {
        "playpause": 0xB3,
        "play": 0xB3,
        "pause": 0xB3,
        "next": 0xB0,
        "previous": 0xB1,
        "prev": 0xB1,
        "stop": 0xB2,
        "volup": 0xAF,
        "voldown": 0xAE,
        "mute": 0xAD,
    }
    code = key_map.get(action.lower())
    if code is None:
        return {"ok": False, "error": f"Unknown media action: {action}"}
    try:
        import ctypes

        ctypes.windll.user32.keybd_event(code, 0, 0, 0)
        ctypes.windll.user32.keybd_event(code, 0, 2, 0)
        result = {"ok": True, "media_action": action}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("media_control", {"action": action, "ok": result.get("ok")}, cfg)
    return result


def discord_webhook_send(
    content: str,
    url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.integrations.discord_webhook import send_webhook
        from navine.utils.brand import load_brand

        brand = load_brand()
        result = send_webhook(
            content,
            webhook_url=url,
            username=str(brand.get("name") or "Navine AI - Python"),
        )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("discord_webhook", {"ok": result.get("ok"), "chars": len(content or "")}, cfg)
    return result


def discord_webhook_set(url: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.integrations.discord_webhook import set_webhook_url

        result = set_webhook_url(url)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("discord_webhook_set", {"ok": result.get("ok"), "configured": result.get("configured")}, cfg)
    return result


def discord_webhook_status(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_assistant_config()
    try:
        from navine.integrations.discord_webhook import webhook_status

        result = webhook_status()
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    audit("discord_webhook_status", {"ok": result.get("ok"), "configured": result.get("configured")}, cfg)
    return result
