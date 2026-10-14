from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path
from typing import List, Tuple

from navine.code.format import extract_code_block
from navine.code.generate import generate_code


TASKS: List[Tuple[str, str]] = [
    ("fizzbuzz", "Write a Python function fizzbuzz(n) that returns a list of strings from 1 to n using FizzBuzz rules."),
    ("json_parse", "Write Python code that parses JSON string '{\"a\": 1}' into a dict named data."),
    ("file_io", "Write Python code that writes hello to out.txt then reads it back into text."),
    ("rest_handler", "Write a Python function handle_get(path) that returns 'ok' for path '/' else 'not found'."),
    ("hello_gui", "Write a Python tkinter GUI Hello World window with a Label and mainloop."),
]


def _try_compile(code: str) -> bool:
    if not code or len(code.strip()) < 8:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        path = handle.name
    try:
        py_compile.compile(path, doraise=True)
        return True
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    passed = 0
    for name, task in TASKS:
        code, lang = generate_code(task, language="python")
        block = extract_code_block(code, "python") if "```" in code else code
        ok = _try_compile(block)
        passed += int(ok)
        print(f"{name}: compile={'ok' if ok else 'fail'} lang={lang} chars={len(block)}")
    print(f"coding_eval {passed}/{len(TASKS)}")


if __name__ == "__main__":
    main()
