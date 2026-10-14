import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "train_parallel_1h"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
DURATION_SECONDS = 60 * 60
WORKER_SCRIPT = ROOT / "scripts" / "_parallel_train_worker.py"


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "main.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _clear_stale_locks() -> None:
    sys.path.insert(0, str(ROOT))
    from navine.utils.training_lock import clear_stale_lock

    for name in ("image", "image_lora", "video", "voice"):
        if clear_stale_lock(name):
            _log(f"Cleared stale lock: {name}")


def main() -> int:
    if not PYTHON.exists():
        _log("ERROR: venv python not found")
        return 1
    if not WORKER_SCRIPT.exists():
        _log(f"ERROR: worker script missing: {WORKER_SCRIPT}")
        return 1

    _clear_stale_locks()
    deadline = time.time() + DURATION_SECONDS
    deadline_iso = datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat()
    _log(f"PARALLEL_1H_BEGIN deadline={deadline_iso}")

    workers = [
        "text_stack",
        "text_enterprise",
        "image_stack",
        "image_lora",
        "video",
        "voice_misc",
    ]

    device_map = {
        "text_stack": "cpu",
        "text_enterprise": "cpu",
        "image_stack": "cuda",
        "image_lora": "cuda",
        "video": "cuda",
        "voice_misc": "cpu",
    }

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NAVINE_TRAIN_DEADLINE"] = str(deadline)
    env["NAVINE_PROJECT_ROOT"] = str(ROOT)
    env["NAVINE_PARALLEL_TRAIN"] = "1"

    processes = []
    for name in workers:
        out_path = LOG_DIR / f"{name}.out.log"
        err_path = LOG_DIR / f"{name}.err.log"
        out_handle = out_path.open("a", encoding="utf-8")
        err_handle = err_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [str(PYTHON), "-u", str(WORKER_SCRIPT), name],
            cwd=str(ROOT),
            env={**env, "NAVINE_DEVICE_MODE": device_map.get(name, "cuda")},
            stdout=out_handle,
            stderr=err_handle,
        )
        processes.append((name, proc, out_handle, err_handle))
        _log(f"START worker={name} pid={proc.pid} device={device_map.get(name, 'cuda')}")

    while time.time() < deadline:
        alive = [name for name, proc, _, _ in processes if proc.poll() is None]
        if not alive:
            break
        time.sleep(15)

    _log("PARALLEL_1H_TIMEBOX reached, stopping workers")
    for name, proc, out_handle, err_handle in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        code = proc.returncode if proc.returncode is not None else -9
        _log(f"STOP worker={name} code={code}")
        out_handle.close()
        err_handle.close()

    _log("PARALLEL_1H_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
