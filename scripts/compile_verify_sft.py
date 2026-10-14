from __future__ import annotations

import json
import py_compile
import tempfile
from pathlib import Path

from navine.code.format import extract_code_block
from navine.utils.paths import get_project_root

SEEDS = [
    (
        "Write a Python function add(a, b) that returns a + b.",
        "def add(a, b):\n    return a + b\n",
    ),
    (
        "Write a Python function is_even(n) that returns True if n is even.",
        "def is_even(n):\n    return n % 2 == 0\n",
    ),
    (
        "Write a Python function fizzbuzz(n) that returns a list of FizzBuzz strings from 1 to n.",
        "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n",
    ),
]


def _compiles(code: str) -> bool:
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
    out = get_project_root() / "data" / "train" / "coding" / "verified.jsonl"
    added = 0
    with out.open("a", encoding="utf-8") as handle:
        for prompt, code in SEEDS:
            if not _compiles(code):
                continue
            handle.write(
                json.dumps(
                    {
                        "language": "python",
                        "prompt": prompt,
                        "code": code,
                        "source": "compile_verified",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            added += 1
    print(f"verified_sft_added={added} path={out}")


if __name__ == "__main__":
    main()
