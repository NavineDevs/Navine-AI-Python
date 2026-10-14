from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    py = ROOT / "venv" / "Scripts" / "python.exe"
    steps = 400
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    cmd = [str(py), "-m", "navine.cli", "train", "text", "--steps", str(steps)]
    print("chat SFT", cmd)
    subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
