import re
from typing import Any, Callable, Dict, Optional

from navine.assistant import actions

ConfirmCallback = Optional[Callable[[str, Dict[str, Any]], bool]]


def _strip_quotes(value: str) -> str:
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def parse_intent(text: str) -> Dict[str, Any]:
    raw = text.strip()
    lowered = raw.lower().strip()

    try:
        from navine.integrations.discord_webhook import parse_webhook_request

        hook = parse_webhook_request(raw)
        if hook:
            content, url = hook
            return {"action": "discord_webhook", "args": {"content": content, "url": url or ""}}
    except Exception:
        pass
    if lowered.startswith("webhook set ") or lowered.startswith("set discord webhook "):
        url = raw.split(" ", 2)[-1].strip() if lowered.startswith("webhook set ") else raw.split(" ", 3)[-1].strip()
        return {"action": "discord_webhook_set", "args": {"url": url}}
    if lowered in ("webhook status", "discord webhook status", "show webhook"):
        return {"action": "discord_webhook_status", "args": {}}

    if any(
        k in lowered
        for k in (
            "what time",
            "what's the time",
            "whats the time",
            "current time",
            "tell me the time",
            "time is it",
        )
    ):
        return {"action": "datetime_now", "args": {"kind": "time"}}

    if any(
        k in lowered
        for k in (
            "what day",
            "what's the date",
            "whats the date",
            "current date",
            "what is the date",
            "today's date",
            "todays date",
        )
    ):
        return {"action": "datetime_now", "args": {"kind": "date"}}

    if any(
        k in lowered
        for k in (
            "what can you do",
            "what do you do",
            "your capabilities",
            "help me",
            "list your features",
            "what are you able",
        )
    ) and "open" not in lowered:
        return {"action": "capabilities", "args": {}}

    m_observe = re.match(
        r"^(?:could you |please |can you )?"
        r"(?:observe|watch|record|learn from|learn)\s+(?:me\s+)?(?:play(?:ing)?\s+)?"
        r"(.+?)(?:\s+for\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?)?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_observe and any(
        k in lowered
        for k in (
            "observe",
            "watch me",
            "watch play",
            "record game",
            "learn from",
            "learn fortnite",
            "learn mario",
            "learn minecraft",
        )
    ):
        title = m_observe.group(1).strip()
        title = re.sub(r"^(me playing|playing|gameplay of|game)\s+", "", title, flags=re.I).strip()
        secs = float(m_observe.group(2) or 8)
        if title:
            return {"action": "observe_game", "args": {"title": title, "seconds": secs}}

    if any(
        k in lowered
        for k in (
            "how do i play",
            "tips for",
            "coach me",
            "help me play",
            "how to win in",
            "strategy for",
            "guide for playing",
        )
    ):
        title = re.sub(
            r"^(?:could you |please |can you )?(?:how do i play|tips for|coach me(?: on| for)?|"
            r"help me play|how to win in|strategy for|guide for playing)\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip(" ?.")
        if title:
            return {"action": "game_coach", "args": {"title": title}}

    m_train = re.match(
        r"^(?:please |could you |can you )?(?:start\s+)?train(?:ing)?\s+"
        r"(?:the\s+)?"
        r"(game\s*play|gameplay|games?|chat|coding|code|general|math|creative|multimodal(?:-text)?|"
        r"image|video|text|all)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if m_train or re.search(r"\btrain(?:ing)?\b.+\b(game\s*play|gameplay|games?)\b", lowered):
        domain = "games"
        if m_train:
            domain = m_train.group(1).lower().replace(" ", "")
        else:
            for candidate, mapped in (
                ("coding", "coding"),
                ("code", "coding"),
                ("chat", "chat"),
                ("image", "image"),
                ("video", "video"),
                ("math", "math"),
                ("creative", "creative"),
                ("general", "general"),
                ("multimodal", "multimodal-text"),
                ("gameplay", "games"),
                ("gameplay", "games"),
                ("games", "games"),
                ("game", "games"),
                ("text", "text"),
                ("all", "all"),
            ):
                if re.search(rf"\b{candidate}\b", lowered):
                    domain = mapped
                    break
        if domain in ("gameplay", "game"):
            domain = "games"
        steps_m = re.search(r"\b(\d+)\s*steps?\b", lowered)
        steps = int(steps_m.group(1)) if steps_m else 150
        return {"action": "train_domain", "args": {"domain": domain, "steps": steps}}

    m_label = re.match(
        r"^(?:please )?(?:label|mark|log)\s+(?:last\s+)?(?:action\s+)?(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_label and ("label" in lowered or "log action" in lowered or "mark action" in lowered):
        return {"action": "label_game_action", "args": {"action": m_label.group(1).strip()}}

    m_type = re.match(
        r"^(?:please |could you |can you )?(?:desktop\s+)?(?:type|type out|enter text)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_type:
        return {"action": "desktop_type", "args": {"text": m_type.group(1).strip()}}

    m_click = re.match(
        r"^(?:please |could you |can you )?(?:desktop\s+)?click(?:\s+(?:at|on))?\s+"
        r"(?:x\s*=\s*)?(-?\d+)\s*(?:,|\s+)\s*(?:y\s*=\s*)?(-?\d+)"
        r"(?:\s+(left|right|middle))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_click:
        return {
            "action": "desktop_click",
            "args": {
                "x": int(m_click.group(1)),
                "y": int(m_click.group(2)),
                "button": (m_click.group(3) or "left").lower(),
            },
        }

    m_keys = re.match(
        r"^(?:please |could you )?(?:press|hit|tap)\s+(?:the\s+)?(?:key(?:s)?\s+)?(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_keys:
        keys = m_keys.group(1).strip()
        if keys and "screenshot" not in lowered:
            return {"action": "send_keys", "args": {"keys": keys}}

    m_create = re.match(
        r"^(?:could you |please |can you )?(?:create|make|write|new)\s+"
        r"(?:a\s+)?(?:new\s+)?file\s+(.+?)\s+(?:with|containing|saying|that says|content)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_create:
        return {
            "action": "file_write",
            "args": {
                "path": _strip_quotes(m_create.group(1)),
                "content": _strip_quotes(m_create.group(2)),
            },
        }

    m_write_to = re.match(
        r"^(?:could you |please |can you )?(?:write|put|save)\s+(.+?)\s+(?:to|into|in)\s+(?:file\s+)?(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_write_to and not lowered.startswith(("write a ", "write me ", "write some ")):
        content = _strip_quotes(m_write_to.group(1))
        path = _strip_quotes(m_write_to.group(2))
        looks_like_path = " " not in path.strip() or any(
            path.lower().endswith(ext)
            for ext in (
                ".txt",
                ".md",
                ".py",
                ".json",
                ".yaml",
                ".yml",
                ".csv",
                ".log",
                ".html",
                ".css",
                ".js",
                ".ts",
            )
        )
        if path and content and looks_like_path:
            return {"action": "file_write", "args": {"path": path, "content": content}}

    m_mkdir = re.match(
        r"^(?:could you |please |can you )?(?:make|create)\s+(?:a\s+)?(?:new\s+)?"
        r"(?:folder|directory|dir)\s+(?:called\s+|named\s+)?(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_mkdir:
        return {"action": "make_dir", "args": {"path": _strip_quotes(m_mkdir.group(1))}}

    m_append = re.match(
        r"^(?:could you |please |can you )?(?:append|add)\s+(.+?)\s+(?:to\s+(?:file\s+)?|onto\s+)(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_append:
        return {
            "action": "file_append",
            "args": {
                "content": _strip_quotes(m_append.group(1)),
                "path": _strip_quotes(m_append.group(2)),
            },
        }

    m_rename = re.match(
        r"^(?:could you |please |can you )?(?:rename|move)\s+(.+?)\s+(?:to|as)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_rename and ("rename" in lowered or re.match(r"^move\s+\S+\s+to\s+\S+$", lowered)):
        return {
            "action": "file_rename",
            "args": {
                "src": _strip_quotes(m_rename.group(1)),
                "dst": _strip_quotes(m_rename.group(2)),
            },
        }

    m_copy = re.match(
        r"^(?:could you |please |can you )?copy\s+(.+?)\s+to\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_copy:
        return {
            "action": "file_copy",
            "args": {
                "src": _strip_quotes(m_copy.group(1)),
                "dst": _strip_quotes(m_copy.group(2)),
            },
        }

    url_match = re.search(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|io|dev|ai|gov|edu)\b\S*)", raw)
    if any(lowered.startswith(p) for p in ("open website", "go to", "browse", "open url")) or (
        "open" in lowered and url_match
    ):
        if url_match:
            return {"action": "open_url", "args": {"url": url_match.group(0)}}

    if lowered.startswith(("run ", "execute ", "cmd ", "shell ")):
        command = raw.split(" ", 1)[1] if " " in raw else ""
        return {"action": "run_command", "args": {"command": command}}
    if lowered.startswith("`") and raw.endswith("`"):
        return {"action": "run_command", "args": {"command": raw.strip("`")}}

    soft_open = re.match(
        r"^(?:could you |can you |please |kindly )?(?:open app|launch|start app|open program|open|start)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if soft_open:
        target = soft_open.group(1).strip()
        if target and not url_match:
            return {"action": "open_app", "args": {"target": target}}

    bare = re.fullmatch(
        r"(spotify|chrome|discord|steam|notepad|explorer|edge|firefox|code|vscode|calculator|youtube|fortnite)",
        lowered,
    )
    if bare:
        return {"action": "open_app", "args": {"target": bare.group(1)}}

    if any(k in lowered for k in ("system info", "system status", "how is my pc", "cpu usage", "memory usage", "disk space", "how is my computer")):
        return {"action": "system_info", "args": {}}

    if lowered.startswith(("list ", "ls ", "show folder", "list dir", "list files")):
        path = re.sub(
            r"^(list files in|list dir|list files|list|ls|show folder)\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
        return {"action": "list_dir", "args": {"path": path or "."}}

    if lowered.startswith(("read file", "read ", "cat ", "show file")):
        path = re.sub(r"^(read file|read|cat|show file)\s*", "", raw, flags=re.IGNORECASE).strip()
        return {"action": "file_read", "args": {"path": path}}

    if lowered.startswith(("delete ", "remove ", "rm ")):
        path = re.sub(r"^(delete|remove|rm)\s*", "", raw, flags=re.IGNORECASE).strip()
        return {"action": "file_delete", "args": {"path": path}}

    if any(
        k in lowered
        for k in (
            "look at my screen",
            "look at the screen",
            "see my screen",
            "see the screen",
            "what's on my screen",
            "whats on my screen",
            "what is on my screen",
            "describe my screen",
            "describe the screen",
            "what do you see on my screen",
            "can you see my screen",
        )
    ):
        return {"action": "screen_describe", "args": {"question": raw}}

    if "screenshot" in lowered or "screen shot" in lowered or "capture screen" in lowered or "take a picture of the screen" in lowered:
        return {"action": "screenshot", "args": {}}

    if any(
        k in lowered
        for k in (
            "play music",
            "pause music",
            "play pause",
            "next track",
            "previous track",
            "mute",
            "volume up",
            "volume down",
            "turn the volume",
        )
    ):
        media = "playpause"
        if "next" in lowered:
            media = "next"
        elif "previous" in lowered or "prev" in lowered:
            media = "previous"
        elif "mute" in lowered:
            media = "mute"
        elif "volume up" in lowered or "louder" in lowered:
            media = "volup"
        elif "volume down" in lowered or "quieter" in lowered:
            media = "voldown"
        elif "pause" in lowered:
            media = "pause"
        return {"action": "media_control", "args": {"action": media}}

    if lowered.startswith(("copy to clipboard", "clipboard copy")):
        payload = re.sub(r"^(copy to clipboard|clipboard copy)\s*", "", raw, flags=re.IGNORECASE).strip()
        return {"action": "clipboard_write", "args": {"text": payload}}
    m_clip = re.match(r"^(?:could you |please )?copy (.+) to (?:the )?clipboard$", raw, flags=re.IGNORECASE)
    if m_clip:
        return {"action": "clipboard_write", "args": {"text": m_clip.group(1).strip()}}
    if "read clipboard" in lowered or "what is on my clipboard" in lowered or "what's on my clipboard" in lowered:
        return {"action": "clipboard_read", "args": {}}

    return {"action": "unknown", "args": {"text": raw}}


_CAPABILITY_MAP = {
    "run_command": "run_command",
    "open_app": "open_app",
    "open_url": "open_url",
    "system_info": "system_info",
    "list_dir": "file_read",
    "file_read": "file_read",
    "file_write": "file_write",
    "file_append": "file_write",
    "file_delete": "file_delete",
    "make_dir": "file_write",
    "file_rename": "file_write",
    "file_copy": "file_write",
    "screenshot": "screenshot",
    "screen_describe": "screenshot",
    "media_control": "media_control",
    "clipboard_read": "clipboard",
    "clipboard_write": "clipboard",
    "datetime_now": "system_info",
    "capabilities": "system_info",
    "observe_game": "screenshot",
    "label_game_action": "file_write",
    "game_coach": "system_info",
    "send_keys": "desktop_input",
    "desktop_type": "desktop_input",
    "desktop_click": "desktop_input",
    "train_domain": "run_command",
    "discord_webhook": "open_url",
    "discord_webhook_set": "open_url",
    "discord_webhook_status": "system_info",
}


def _describe(action: str, args: Dict[str, Any]) -> str:
    if action == "run_command":
        return f"Run shell command: {args.get('command')}"
    if action == "open_app":
        return f"Open application/file: {args.get('target')}"
    if action == "open_url":
        return f"Open URL in browser: {args.get('url')}"
    if action == "file_delete":
        return f"DELETE path: {args.get('path')}"
    if action == "file_write":
        return f"Write file: {args.get('path')}"
    if action == "file_append":
        return f"Append to file: {args.get('path')}"
    if action == "make_dir":
        return f"Create folder: {args.get('path')}"
    if action == "file_rename":
        return f"Rename {args.get('src')} to {args.get('dst')}"
    if action == "file_copy":
        return f"Copy {args.get('src')} to {args.get('dst')}"
    if action == "screenshot":
        return "Capture a screenshot of your screen"
    if action == "screen_describe":
        return "Capture the screen and describe what is visible"
    if action == "observe_game":
        return f"Observe gameplay for {args.get('title')} ({args.get('seconds', 8)}s)"
    if action == "send_keys":
        return f"Press keys: {args.get('keys')}"
    if action == "desktop_type":
        return f"Type text: {str(args.get('text') or '')[:80]}"
    if action == "desktop_click":
        return f"Click at ({args.get('x')}, {args.get('y')})"
    if action == "media_control":
        return f"Media control: {args.get('action')}"
    if action == "clipboard_write":
        return "Write text to clipboard"
    if action == "datetime_now":
        return "Read the current time or date"
    if action == "capabilities":
        return "List assistant capabilities"
    if action == "game_coach":
        return f"Coach tips for {args.get('title')}"
    if action == "label_game_action":
        return f"Label action: {args.get('action')}"
    if action == "train_domain":
        return f"Run training for {args.get('domain')} ({args.get('steps', 150)} steps)"
    if action == "discord_webhook":
        return f"Send Discord webhook message ({len(str(args.get('content') or ''))} chars)"
    if action == "discord_webhook_set":
        return "Save Discord webhook URL"
    if action == "discord_webhook_status":
        return "Show Discord webhook status"
    return action


def execute_intent(
    intent: Dict[str, Any],
    confirm: ConfirmCallback = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or actions.load_assistant_config()
    action = intent.get("action", "unknown")
    args = intent.get("args", {})

    if action == "unknown":
        return {"ok": False, "error": "Could not map request to an action", "text": args.get("text")}

    try:
        from navine.sandbox import deny, guard_capability, guard_delete, guard_path, guard_shell, load_sandbox_config

        sb = load_sandbox_config()
        if action == "run_command":
            blocked = guard_shell(args.get("command", ""), sb)
            if blocked:
                return deny(blocked)
        elif action in ("file_write", "file_append", "make_dir"):
            blocked = guard_path(args.get("path", ""), write=True, config=sb)
            if blocked:
                return deny(blocked)
        elif action == "file_delete":
            blocked = guard_delete(args.get("path", ""), sb)
            if blocked:
                return deny(blocked)
        elif action in ("file_rename", "file_copy"):
            for key in ("path", "src", "dst"):
                if args.get(key):
                    blocked = guard_path(args.get(key), write=True, config=sb)
                    if blocked:
                        return deny(blocked)
        elif action in ("open_app", "open_url", "desktop_input", "media_control"):
            cap = "open_app" if action == "open_app" else "open_url" if action == "open_url" else "desktop_input" if action == "desktop_input" else "media_control"
            blocked = guard_capability(cap, sb)
            if blocked:
                return deny(blocked)
    except Exception:
        pass

    capability = _CAPABILITY_MAP.get(action)
    if capability:
        allowed = actions.capability_enabled(capability, cfg)
        if capability == "desktop_input":
            try:
                from navine.desktop.config import input_enabled

                allowed = allowed or input_enabled()
            except Exception:
                pass
            try:
                from navine.sandbox import guard_capability, load_sandbox_config

                if guard_capability("desktop_input", load_sandbox_config()):
                    allowed = False
            except Exception:
                pass
        if capability == "screenshot":
            try:
                from navine.desktop.config import screen_enabled

                allowed = allowed and screen_enabled()
            except Exception:
                pass
        if not allowed:
            if capability == "desktop_input":
                return {
                    "ok": False,
                    "error": "Desktop input is sandboxed/disabled. Contained mode blocks keyboard and mouse control.",
                    "sandboxed": True,
                }
            if capability == "screenshot":
                return {
                    "ok": False,
                    "error": "Screen capture is disabled. Enable Allow screen in the web UI or configs/desktop.yaml.",
                }
            return {"ok": False, "error": f"Capability disabled by sandbox/config: {capability}", "sandboxed": True}

    consent_cfg = cfg.get("consent") or {}
    read_only = actions.is_read_only(action)
    needs_confirm = bool(consent_cfg.get("require_confirmation", True)) and not (
        read_only and consent_cfg.get("auto_approve_read_only", True)
    )

    if action == "run_command":
        danger = actions.is_dangerous(args.get("command", ""), cfg)
        if danger:
            return {"ok": False, "error": f"Blocked dangerous command pattern: {danger}"}

    if needs_confirm:
        if confirm is None:
            return {"ok": False, "error": "Confirmation required but no confirm handler provided", "pending": _describe(action, args)}
        approved = confirm(_describe(action, args), intent)
        if not approved:
            return {"ok": False, "cancelled": True, "action": action}

    dispatch = {
        "run_command": lambda: actions.run_command(args.get("command", ""), cfg),
        "open_app": lambda: actions.open_app(args.get("target", ""), cfg),
        "open_url": lambda: actions.open_url(args.get("url", ""), cfg),
        "system_info": lambda: actions.system_info(cfg),
        "list_dir": lambda: actions.list_dir(args.get("path", "."), cfg),
        "file_read": lambda: actions.read_file(args.get("path", ""), config=cfg),
        "file_write": lambda: actions.write_file(args.get("path", ""), args.get("content", ""), cfg),
        "file_append": lambda: actions.append_file(args.get("path", ""), args.get("content", ""), cfg),
        "file_delete": lambda: actions.delete_file(args.get("path", ""), cfg),
        "make_dir": lambda: actions.make_dir(args.get("path", ""), cfg),
        "file_rename": lambda: actions.rename_path(args.get("src", ""), args.get("dst", ""), cfg),
        "file_copy": lambda: actions.copy_path(args.get("src", ""), args.get("dst", ""), cfg),
        "screenshot": lambda: actions.screenshot(cfg),
        "screen_describe": lambda: actions.screen_describe(args.get("question", ""), cfg),
        "media_control": lambda: actions.media_control(args.get("action", "playpause"), cfg),
        "clipboard_read": lambda: actions.clipboard_read(cfg),
        "clipboard_write": lambda: actions.clipboard_write(args.get("text", ""), cfg),
        "datetime_now": lambda: actions.datetime_now(args.get("kind", "both"), cfg),
        "capabilities": lambda: actions.capabilities_summary(cfg),
        "observe_game": lambda: actions.observe_game(
            args.get("title", "game"),
            float(args.get("seconds", 8)),
            args.get("note", ""),
            cfg,
        ),
        "label_game_action": lambda: actions.label_game_action(args.get("action", ""), args.get("title", ""), cfg),
        "game_coach": lambda: actions.game_coach(args.get("title", "game"), args.get("note", ""), cfg),
        "send_keys": lambda: actions.send_keys(args.get("keys", ""), cfg),
        "desktop_type": lambda: actions.desktop_type(args.get("text", ""), cfg),
        "desktop_click": lambda: actions.desktop_click(
            int(args.get("x", 0)),
            int(args.get("y", 0)),
            args.get("button", "left"),
            cfg,
        ),
        "train_domain": lambda: actions.train_domain(
            args.get("domain", "games"),
            int(args.get("steps", 150) or 150),
            cfg,
        ),
        "discord_webhook": lambda: actions.discord_webhook_send(
            args.get("content", ""),
            args.get("url") or None,
            cfg,
        ),
        "discord_webhook_set": lambda: actions.discord_webhook_set(args.get("url", ""), cfg),
        "discord_webhook_status": lambda: actions.discord_webhook_status(cfg),
    }
    handler = dispatch.get(action)
    if handler is None:
        return {"ok": False, "error": f"No handler for action: {action}"}
    result = handler()
    result["action"] = action
    return result


def handle(text: str, confirm: ConfirmCallback = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    intent = parse_intent(text)
    return execute_intent(intent, confirm=confirm, config=config)
