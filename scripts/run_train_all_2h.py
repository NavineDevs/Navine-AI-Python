import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_LOG = LOG_DIR / "train_all_2h.out.log"
ERR_LOG = LOG_DIR / "train_all_2h.err.log"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
BLOCK_SECONDS = 30 * 60

STEPS = [
    (
        "text",
        [str(PYTHON), "-m", "navine.cli", "train", "text"],
    ),
    (
        "image_custom",
        [
            str(PYTHON),
            "-c",
            "from navine.image.train import train; train(config_path='image_custom', finetune=True, finetune_steps=1200)",
        ],
    ),
    (
        "video",
        [
            str(PYTHON),
            "-c",
            "from navine.video.train import train; train(config_path='video', finetune=True, finetune_steps=300)",
        ],
    ),
    (
        "registry_all",
        [str(PYTHON), "-m", "navine.cli", "train", "all", "--steps", "200"],
    ),
]


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _run_step(label: str, cmd: list[str], timeout: int) -> int:
    _log(f"START {label} ({timeout // 60}m): {' '.join(cmd)}")
    start = time.time()
    try:
        with OUT_LOG.open("a", encoding="utf-8") as out, ERR_LOG.open("a", encoding="utf-8") as err:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                timeout=timeout,
                stdout=out,
                stderr=err,
            )
        code = int(proc.returncode)
    except subprocess.TimeoutExpired:
        _log(f"TIMEBOX {label} hit {timeout // 60}m limit")
        code = 124
    except Exception as exc:
        _log(f"ERROR {label}: {exc}")
        code = 1
    elapsed = int(time.time() - start)
    _log(f"EXIT {label} code={code} elapsed={elapsed // 60}m{elapsed % 60}s")
    return code


def main() -> int:
    if not PYTHON.exists():
        _log("ERROR: venv python not found")
        return 1
    _log("TRAIN_2H_BEGIN")
    deadline = time.time() + (2 * 60 * 60)
    for label, cmd in STEPS:
        remaining = int(deadline - time.time())
        if remaining <= 60:
            _log(f"SKIP {label}: less than 1 minute left")
            break
        timeout = min(BLOCK_SECONDS, remaining)
        _run_step(label, cmd, timeout)
    _log("TRAIN_2H_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
