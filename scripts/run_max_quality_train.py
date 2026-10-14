import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "scripts" / "run_custom_only_train.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([str(PYTHON), str(SCRIPT)], cwd=str(ROOT)))
