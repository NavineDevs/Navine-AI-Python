from typing import Optional, Set, Tuple
import re

from navine.code.fallbacks import (
    looks_like_valid_code,
    wants_gui_code,
    wants_keylogger,
    wants_multi_tool,
)
from navine.code.format import extract_code_block
from navine.code.prompts import detect_language

LANGUAGE_LABELS = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "rust": "Rust",
    "go": "Go",
    "java": "Java",
    "cpp": "C++",
    "csharp": "C#",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "php": "PHP",
    "ruby": "Ruby",
    "sql": "SQL",
    "bash": "Bash",
    "batch": "Batch",
    "bat": "Batch",
    "lua": "Lua",
    "scala": "Scala",
    "haskell": "Haskell",
    "r": "R",
    "dart": "Dart",
    "elixir": "Elixir",
    "perl": "Perl",
    "matlab": "MATLAB",
}

EXTERNAL_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)",
    re.MULTILINE,
)
BANNED_ROOTS = {
    "numpy",
    "np",
    "torch",
    "tensorflow",
    "jax",
    "requests",
    "httpx",
    "aiohttp",
    "bs4",
    "beautifulsoup4",
    "selenium",
    "pandas",
    "sklearn",
    "cv2",
    "PIL",
    "pillow",
    "pygame",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "flask",
    "fastapi",
    "django",
    "openai",
    "transformers",
    "diffusers",
}
DISCORD_BOT_ALLOW = {"discord"}


def _language_label(language: str) -> str:
    return LANGUAGE_LABELS.get(language, language)


def _force_python_cli() -> bool:
    try:
        from navine.utils.brand import is_python_only

        if is_python_only():
            return True
    except Exception:
        pass
    return True


def wants_discord_webhook(task: str) -> bool:
    lower = (task or "").lower()
    if wants_discord_bot(task):
        return False
    if "webhook" in lower and ("discord" in lower or "send" in lower or "post" in lower):
        return True
    if re.search(r"discord\.com/api/webhooks", lower):
        return True
    if re.search(r"\b(send|post|push)\b.{0,40}\bwebhook\b", lower):
        return True
    if "webhook" in lower and any(k in lower for k in ("message", "notify", "alert", "http")):
        return True
    if "discord" in lower and re.search(r"\b(send|post|push|notify|message)\b", lower):
        return True
    return False


def wants_discord_bot(task: str) -> bool:
    lower = (task or "").lower()
    return bool(
        re.search(
            r"\b(discord\s*bot|discord\.py|bot\s+token|slash\s*command|"
            r"on_message|discord\.client|discord\.ext|make\s+a\s+bot\b.*\bdiscord|"
            r"discord\b.*\b(bot|client)\b)\b",
            lower,
        )
    )


def _allowed_extra_imports(task: str) -> Set[str]:
    allowed: Set[str] = set()
    if wants_discord_webhook(task) or wants_discord_bot(task):
        allowed.update({"urllib", "http", "ssl", "json", "argparse", "sys", "os", "base64"})
    if wants_discord_bot(task):
        allowed |= DISCORD_BOT_ALLOW
    return allowed


def _is_usable_code(code: str, lang: str, task: str = "") -> bool:
    cleaned = str(code or "").strip()
    if not cleaned:
        return False
    if lang == "python" and _has_banned_imports(cleaned, task):
        return False
    return looks_like_valid_code(cleaned, lang)


def _has_banned_imports(code: str, task: str = "") -> bool:
    allowed = _allowed_extra_imports(task)
    for match in EXTERNAL_IMPORT_RE.finditer(code or ""):
        root = match.group(1).split(".", 1)[0]
        if root in allowed:
            continue
        if root in BANNED_ROOTS or root in DISCORD_BOT_ALLOW:
            return True
    return False


def _webhook_cli_template(task: str) -> str:
    return '''#!/usr/bin/env python3
import argparse
import json
import sys
from urllib import error, request


def send_discord_webhook(url: str, content: str, username: str = "") -> int:
    payload = {"content": content[:1900]}
    if username:
        payload["username"] = username[:80]
    body = json.dumps(payload).encode("utf-8")
    post_url = url if "wait=" in url else url + ("&" if "?" in url else "?") + "wait=true"
    req = request.Request(
        post_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "NavineAI-WebhookCLI/1.0",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=25) as resp:
            print("ok", getattr(resp, "status", 200))
            return 0
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        print("HTTP", exc.code, detail, file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Send a message to a Discord webhook")
    parser.add_argument("--url", required=True, help="Discord webhook URL")
    parser.add_argument("--message", "-m", required=True, help="Message text")
    parser.add_argument("--username", default="", help="Optional webhook display name")
    opts = parser.parse_args(argv)
    return send_discord_webhook(opts.url.strip(), opts.message, opts.username)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _discord_bot_template(task: str) -> str:
    return '''#!/usr/bin/env python3
"""Discord bot CLI (requires: pip install discord.py). Run locally with your bot token."""
import argparse
import os
import sys

try:
    import discord
    from discord.ext import commands
except ImportError:
    print("Install discord.py first: pip install discord.py", file=sys.stderr)
    raise SystemExit(2)


def build_bot(prefix: str = "!"):
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=prefix, intents=intents)

    @bot.event
    async def on_ready():
        print(f"logged in as {bot.user}")

    @bot.command(name="ping")
    async def ping(ctx):
        await ctx.send("pong")

    @bot.command(name="echo")
    async def echo(ctx, *, text: str = ""):
        await ctx.send(text or "(empty)")

    return bot


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a simple Discord bot")
    parser.add_argument("--token", default=os.environ.get("DISCORD_BOT_TOKEN", ""), help="Bot token or set DISCORD_BOT_TOKEN")
    parser.add_argument("--prefix", default="!", help="Command prefix")
    opts = parser.parse_args(argv)
    token = (opts.token or "").strip()
    if not token:
        print("Missing bot token. Pass --token or set DISCORD_BOT_TOKEN.", file=sys.stderr)
        return 1
    bot = build_bot(opts.prefix)
    bot.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _ask_model(task: str, language: str, temperature: float = 0.16, max_new_tokens: int = 896) -> str:
    from navine.text.infer import generate

    label = _language_label(language)
    extra = ""
    if language == "python":
        if wants_discord_bot(task):
            extra += (
                " Write a Discord bot using discord.py (commands.Bot). "
                "Use argparse for --token / DISCORD_BOT_TOKEN. Include on_ready and at least one command. "
                "Mention pip install discord.py in a top docstring only."
            )
        elif wants_discord_webhook(task):
            extra += (
                " Write a Python CLI that POSTs to a Discord webhook using ONLY urllib.request + json (stdlib). "
                "Accept --url and --message via argparse. No requests library."
            )
        else:
            extra += (
                " Prefer Python standard library only. "
                "No third-party packages (no requests, numpy, pygame, pillow, flask, etc.). "
                "Discord webhooks: use urllib.request. Discord bots: discord.py is allowed when asked. "
                "Write a CLI script: argparse or sys.argv, print results to stdout. "
                "Runnable with: python script.py"
            )
    prompt = (
        "### System: You are Navine AI - Python, an expert programmer. "
        f"Write complete, correct, runnable {label} code for the user task. "
        f"Return only the {label} code inside a single ```{language} code block. "
        "Use real indentation with spaces. "
        "No placeholders, no TODO stubs, no pseudo-code, no explanations."
        f"{extra}\n"
        f"### User: {task}\n"
        "### Assistant:\n```{language}\n"
    )
    return generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        mode="code",
        num_candidates=1,
    )


def _cli_stub(task: str) -> str:
    safe = re.sub(r"[^\w\s\-]", "", (task or "task").strip())[:60] or "task"
    return (
        "#!/usr/bin/env python3\n"
        "import argparse\n"
        "import sys\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser(description=%r)\n"
        "    parser.add_argument('args', nargs='*', help='optional arguments')\n"
        "    opts = parser.parse_args(argv)\n"
        "    print(%r)\n"
        "    if opts.args:\n"
        "        print('args:', ' '.join(opts.args))\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    ) % (safe, f"Navine CLI: {safe}")


def generate_code(task: str, language: Optional[str] = None) -> Tuple[str, str]:
    force_py = _force_python_cli()
    lang = "python" if force_py else (language or detect_language(task) or "python")
    if force_py:
        lang = "python"

    if wants_discord_bot(task):
        return _discord_bot_template(task), "python"
    if wants_discord_webhook(task):
        return _webhook_cli_template(task), "python"

    attempts = (
        (0.10, 1280),
        (0.18, 1400),
        (0.28, 1200),
        (0.08, 1024),
        (0.40, 1152),
        (0.55, 1024),
    )
    best_partial = ""
    for temperature, tokens in attempts:
        try:
            raw = _ask_model(task, lang, temperature=temperature, max_new_tokens=tokens)
            code = extract_code_block(raw) or str(raw or "").strip()
            code = code.strip()
            if code.startswith("```"):
                code = extract_code_block(code) or code
            if _is_usable_code(code, lang, task):
                return code, lang
            if code and len(code) > len(best_partial) and not _has_banned_imports(code, task):
                best_partial = code
        except Exception:
            continue

    if best_partial and looks_like_valid_code(best_partial, lang):
        return best_partial, lang

    if wants_keylogger(task) or wants_multi_tool(task):
        from navine.code.fallbacks import build_keylogger, build_multi_tool_batch

        if wants_keylogger(task):
            code, code_lang = build_keylogger(task)
            if code_lang == "python" and not _has_banned_imports(code, task):
                return code, "python"
        if wants_multi_tool(task):
            code, code_lang = build_multi_tool_batch(task)
            if code_lang == "python" and not _has_banned_imports(code, task):
                return code, "python"

    if wants_gui_code(task) and lang == "python":
        from navine.code.fallbacks import build_gui_for_task

        built = build_gui_for_task(task)
        if built and looks_like_valid_code(built[0], built[1]) and not _has_banned_imports(built[0], task):
            if "tkinter" in built[0] or "Tk(" in built[0]:
                return built[0], "python"

    return _cli_stub(task), "python"
