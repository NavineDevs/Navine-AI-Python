"""Train every modality/specialist to the 8GB / ~800M+ param hardware limit."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"
LOG_DIR = ROOT / "logs" / "limit_train"
LOG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE = LOG_DIR / "queue.log"
LOCK = LOG_DIR / "marathon.lock"

JOBS = [
    ("text_code", ["train", "text-code", "--steps", "4000", "--require-cuda"]),
    ("text_enterprise", ["train", "text", "--steps", "4000", "--require-cuda"]),
    ("coding", ["train", "coding", "--steps", "2000", "--require-cuda"]),
    ("chat", ["train", "chat", "--steps", "1600", "--require-cuda"]),
    ("thinking", ["train", "thinking", "--steps", "1400", "--require-cuda"]),
    ("detective", ["train", "detective", "--steps", "1200", "--require-cuda"]),
    ("osint", ["train", "osint", "--steps", "1200", "--require-cuda"]),
    ("math", ["train", "math", "--steps", "1200", "--require-cuda"]),
    ("creative", ["train", "creative", "--steps", "1200", "--require-cuda"]),
    ("general", ["train", "general", "--steps", "1400", "--require-cuda"]),
    ("nsfw", ["train", "nsfw", "--steps", "1400", "--require-cuda"]),
    ("unrestricted", ["train", "unrestricted", "--steps", "1400", "--require-cuda"]),
    ("hitboyx23", ["train", "hitboyx23", "--steps", "1600", "--require-cuda"]),
    ("hitboyx23_python", ["train", "hitboyx23_python", "--steps", "1600", "--require-cuda"]),
    ("multimodal_text", ["train", "multimodal-text", "--steps", "1000", "--require-cuda"]),
    ("games", ["train", "games", "--steps", "800", "--require-cuda"]),
    ("image", ["train", "image", "--steps", "2500", "--require-cuda"]),
    ("video", ["train", "video", "--steps", "1500", "--require-cuda"]),
    ("voice", ["train", "voice", "--steps", "200", "--require-cuda"]),
    ("deepfake", ["-c", "from navine.deepfake.train import train_deepfake; train_deepfake(steps=500)"]),
]


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    with QUEUE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def free_space_gb() -> float:
    usage = shutil.disk_usage(str(ROOT))
    return usage.free / (1024**3)


def prepare_image_800m() -> None:
    ckpt = ROOT / "checkpoints" / "image_enterprise_v2"
    init = ckpt / "latest_800m_init.pt"
    latest = ckpt / "latest.pt"
    if init.exists() and (not latest.exists() or latest.stat().st_size < init.stat().st_size * 0.5):
        shutil.copy2(init, latest)
        log(f"Prepared image latest.pt from {init.name} ({init.stat().st_size:,} bytes)")


def run_job(name: str, args: list[str]) -> int:
    if free_space_gb() < 4.0:
        log(f"SKIP {name}: only {free_space_gb():.2f} GB free")
        return 2
    out = LOG_DIR / f"{name}.out.log"
    err = LOG_DIR / f"{name}.err.log"
    if args[0] == "-c":
        cmd = [str(PY), "-u", *args]
    else:
        cmd = [str(PY), "-u", "-m", "navine.cli", *args]
    log(f"START {name} free_gb={free_space_gb():.2f} {' '.join(args[:6])}")
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    with out.open("ab") as out_f, err.open("ab") as err_f:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f, env=env)
        code = proc.wait()
    log(f"END {name} code={code} free_gb={free_space_gb():.2f}")
    return int(code)


def sync_all() -> None:
    sync = ROOT / "scripts" / "sync_all_six.py"
    if not sync.exists():
        return
    log("SYNC begin (code+configs)")
    code = subprocess.call(
        [str(PY), str(sync), "--code", "--configs"],
        cwd=str(ROOT),
    )
    log(f"SYNC code/configs end code={code}")
    if free_space_gb() >= 25.0:
        log("SYNC checkpoints begin")
        code = subprocess.call(
            [str(PY), str(sync), "--checkpoints"],
            cwd=str(ROOT),
        )
        log(f"SYNC checkpoints end code={code}")
    else:
        log(f"SKIP checkpoint sync: only {free_space_gb():.2f} GB free")


def acquire_lock() -> bool:
    for _attempt in range(3):
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n")
            return True
        except FileExistsError:
            try:
                old_pid = int(LOCK.read_text(encoding="utf-8").strip().splitlines()[0])
            except Exception:
                old_pid = None
            if old_pid:
                try:
                    import ctypes

                    kernel = ctypes.windll.kernel32
                    handle = kernel.OpenProcess(0x1000, False, int(old_pid))
                    if handle:
                        kernel.CloseHandle(handle)
                        return False
                except Exception:
                    pass
            try:
                LOCK.unlink()
            except Exception:
                time.sleep(0.5)
                continue
            time.sleep(0.2)
    return False


def release_lock() -> None:
    try:
        if LOCK.exists():
            LOCK.unlink()
    except Exception:
        pass


def main() -> int:
    if not acquire_lock():
        print("Another limit-train marathon is already running.", flush=True)
        return 0
    try:
        return _main_locked()
    finally:
        release_lock()


def _main_locked() -> int:
    argv = [a for a in sys.argv[1:] if a]
    loop = "--loop" in argv
    continue_on_fail = "--stop-on-fail" not in argv
    start_from = ""
    for a in argv:
        if a.startswith("--"):
            continue
        start_from = a.strip().lower()
        break
    log(
        f"LIMIT TRAIN BEGIN params_tier=gpt3_1b (~1.027B unique; stretch 1.3-1.8B) "
        f"push=hard cpu_adamw loop={loop} start_from={start_from or 'first'} "
        f"free_gb={free_space_gb():.2f}"
    )
    prepare_image_800m()
    round_idx = 0
    all_failures = []
    while True:
        round_idx += 1
        log(f"ROUND {round_idx} begin")
        started = not start_from
        failures = []
        for name, args in JOBS:
            if not started:
                if name.lower() == start_from or start_from in name.lower():
                    started = True
                else:
                    log(f"SKIP {name}")
                    continue
            code = run_job(name, args)
            if code != 0:
                failures.append((name, code))
                all_failures.append((round_idx, name, code))
                if not continue_on_fail:
                    log(f"STOP after failure in {name}")
                    return code
                log(f"CONTINUE after failure in {name}")
            time.sleep(2)
        sync_all()
        log(f"ROUND {round_idx} done failures={failures or 'none'}")
        if not loop:
            break
        start_from = ""
        time.sleep(5)
    log(f"LIMIT TRAIN DONE failures={all_failures or 'none'}")
    return 0 if not all_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
