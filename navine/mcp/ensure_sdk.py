"""Ensure MCP SDK is installed for the current AI brand venv."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "pip", "install", "mcp>=1.28,<3"]
    print("Installing MCP SDK:", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code != 0:
        return code
    try:
        pkg = __package__.split(".", 1)[0] if __package__ else "navine"
        mod = __import__(f"{pkg}.mcp.client", fromlist=["status"])
        print(mod.status())
    except Exception as exc:
        print("MCP import check failed:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
