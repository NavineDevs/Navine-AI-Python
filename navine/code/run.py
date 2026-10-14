from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

MAX_OUTPUT_CHARS = 12000
DEFAULT_TIMEOUT = 20.0

BLOCKED_PATTERNS: List[str] = [
    r"\bos\.system\b",
    r"\bos\.popen\b",
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
    r"\bos\.rmdir\b",
    r"\bos\.removedirs\b",
    r"\bos\.rename\b",
    r"\bos\.replace\b",
    r"\bos\.chmod\b",
    r"\bos\.chown\b",
    r"\bos\.startfile\b",
    r"\bos\.execl",
    r"\bos\.execv",
    r"\bos\.spawn",
    r"\bsubprocess\b",
    r"\bshutil\.rmtree\b",
    r"\bshutil\.move\b",
    r"\bshutil\.copytree\b",
    r"\bpathlib\.Path\s*\([^)]*\)\.unlink\b",
    r"\bpathlib\.Path\s*\([^)]*\)\.rmdir\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bsocket\b",
    r"\brequests\b",
    r"\bhttpx\b",
    r"\bctypes\b",
    r"\bwinreg\b",
    r"\bpsutil\b",
    r"\bwebbrowser\b",
    r"\bpyautogui\b",
    r"\bkeyboard\b",
    r"\bmouse\b",
    r"\breg\s+add\b",
    r"\breg\s+delete\b",
    r"\bpowershell\b",
    r"\bcmd\.exe\b",
]

ALWAYS_BLOCKED_MODULES = {
    "subprocess",
    "socket",
    "ctypes",
    "cffi",
    "winreg",
    "multiprocessing",
    "requests",
    "httpx",
    "aiohttp",
    "psutil",
    "webbrowser",
    "pickle",
    "shelve",
    "importlib",
    "numpy",
    "torch",
    "cv2",
    "PIL",
    "pillow",
    "pygame",
    "pandas",
    "sklearn",
    "tensorflow",
    "jax",
    "flask",
    "fastapi",
    "django",
    "bs4",
    "selenium",
}

# discord.py bots need a local install; server run blocks the package but code gen is allowed.
LOCAL_ONLY_MODULES = {"discord"}


def _stdlib_names() -> Set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()) or set())
    if not names:
        names = {
            "abc",
            "argparse",
            "array",
            "ast",
            "asyncio",
            "base64",
            "binascii",
            "bisect",
            "builtins",
            "calendar",
            "cmath",
            "collections",
            "copy",
            "csv",
            "dataclasses",
            "datetime",
            "decimal",
            "enum",
            "functools",
            "hashlib",
            "heapq",
            "html",
            "http",
            "io",
            "itertools",
            "json",
            "math",
            "operator",
            "os",
            "pathlib",
            "pprint",
            "queue",
            "random",
            "re",
            "ssl",
            "statistics",
            "string",
            "struct",
            "sys",
            "tempfile",
            "textwrap",
            "threading",
            "time",
            "traceback",
            "typing",
            "urllib",
            "uuid",
            "warnings",
            "xml",
            "zlib",
        }
    names.update({"urllib", "http", "ssl", "json", "email", "base64", "__future__"})
    return names


def _imported_roots(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    roots: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.append((alias.name or "").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue
            if node.module:
                roots.append(node.module.split(".")[0])
    return [r for r in roots if r]


def external_import_violations(code: str) -> List[str]:
    stdlib = _stdlib_names()
    bad: List[str] = []
    for root in _imported_roots(code):
        if root in ALWAYS_BLOCKED_MODULES:
            bad.append(root)
            continue
        if root in LOCAL_ONLY_MODULES:
            bad.append(root)
            continue
        if root not in stdlib:
            bad.append(root)
    return sorted(set(bad))


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(output truncated)"


def is_safe_python(code: str) -> Tuple[bool, str]:
    try:
        from navine.sandbox import sandbox_enabled

        if not sandbox_enabled():
            return True, ""
    except Exception:
        pass
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Sandbox blocked unsafe code pattern: {pattern.strip()}"
    roots = _imported_roots(code)
    local_only = sorted({r for r in roots if r in LOCAL_ONLY_MODULES})
    if local_only:
        return (
            False,
            "Discord bot code uses "
            + ", ".join(local_only)
            + ". Download the script and run locally: pip install discord.py && python script.py --token YOUR_BOT_TOKEN. "
            "For webhooks only, ask for a Discord webhook CLI (stdlib urllib) which can run here.",
        )
    external = external_import_violations(code)
    if external:
        return (
            False,
            "Python run is stdlib-only CLI mode (no external libraries). Blocked imports: "
            + ", ".join(external)
            + ". Webhook scripts may use urllib.request. Discord bots need local discord.py.",
        )
    return True, ""


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, object]:
    safe, reason = is_safe_python(code)
    if not safe:
        return {
            "ok": False,
            "stdout": "",
            "stderr": reason,
            "exit_code": 1,
            "language": "python",
            "sandboxed": True,
            "stdlib_only": True,
        }
    if not code.strip():
        return {
            "ok": False,
            "stdout": "",
            "stderr": "No code to run",
            "exit_code": 1,
            "language": "python",
            "stdlib_only": True,
        }
    with tempfile.TemporaryDirectory(prefix="navine_sandbox_") as tmp:
        script_path = Path(tmp) / "main.py"
        script_path.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-u", "-S", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env={
                    "PYTHONPATH": "",
                    "PYTHONNOUSERSITE": "1",
                    "PATH": str(Path(sys.executable).parent),
                },
            )
            stdout = _truncate(completed.stdout or "")
            stderr = _truncate(completed.stderr or "")
            ok = completed.returncode == 0
            if completed.returncode != 0 and not stderr.strip() and not stdout.strip():
                stderr = f"Process exited with code {completed.returncode}"
            return {
                "ok": ok,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": int(completed.returncode),
                "language": "python",
                "sandboxed": True,
                "stdlib_only": True,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Timed out after {timeout:.0f}s",
                "exit_code": -1,
                "language": "python",
                "sandboxed": True,
                "stdlib_only": True,
            }
        except Exception as exc:
            return {
                "ok": False,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -1,
                "language": "python",
                "sandboxed": True,
                "stdlib_only": True,
            }


def run_code(code: str, language: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, object]:
    lang = (language or "python").strip().lower()
    if lang in {"python", "py"}:
        return run_python(code, timeout=timeout)
    return {
        "ok": False,
        "stdout": "",
        "stderr": f"Server run is Python stdlib-only. {lang} is not executed here; download and run locally.",
        "exit_code": 1,
        "language": lang,
        "sandboxed": True,
        "stdlib_only": True,
    }
