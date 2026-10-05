import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navine.utils.training_lock import clear_stale_lock, training_lock_active, wait_for_training_unlock


def _log(msg: str, log_path: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def main() -> int:
    log_dir = ROOT / "logs" / "teacher_loop"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "watcher.log"
    for name in ("image", "video"):
        clear_stale_lock(name)
        if training_lock_active(name):
            _log(f"{name} lock active; waiting for release", log_path)
            wait_for_training_unlock(name, poll_seconds=90, progress=lambda m: _log(m, log_path))
    _log("Launching learn teacher-loop --count 5 --max-cycles 20 --model sd_turbo", log_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "navine.cli",
            "learn",
            "teacher-loop",
            "--count",
            "5",
            "--max-cycles",
            "20",
            "--model",
            "sd_turbo",
        ],
        cwd=str(ROOT),
    )
    _log(f"Teacher loop finished with exit code {result.returncode}", log_path)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
