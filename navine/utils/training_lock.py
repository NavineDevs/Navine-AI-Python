import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict

from navine.utils.paths import get_project_root


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

            process_query_limited_information = 0x1000
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            still_active = 259
            if not ok:
                return True
            return exit_code.value == still_active
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def clear_stale_lock(name: str) -> bool:
    lock_path = get_project_root() / "logs" / f".{name}_training.lock"
    if not lock_path.exists():
        return False
    try:
        pid = int(lock_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        lock_path.unlink(missing_ok=True)
        return True
    if not _pid_alive(pid):
        lock_path.unlink(missing_ok=True)
        return True
    return False

_thread_locks: Dict[str, threading.RLock] = {}
_depth: Dict[str, int] = defaultdict(int)


@contextmanager
def training_lock(name: str):
    thread_lock = _thread_locks.setdefault(name, threading.RLock())
    thread_lock.acquire()
    _depth[name] += 1
    root = get_project_root() / "logs"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{name}_training.lock"
    file_acquired = False
    if _depth[name] == 1:
        clear_stale_lock(name)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            file_acquired = True
        except FileExistsError:
            _depth[name] -= 1
            thread_lock.release()
            raise RuntimeError(
                f"Navine AI - Python {name} training is already running. Wait for it to finish before starting another."
            ) from None
    try:
        yield lock_path
    finally:
        if file_acquired and lock_path.exists():
            lock_path.unlink()
        _depth[name] -= 1
        thread_lock.release()


def training_lock_active(name: str) -> bool:
    clear_stale_lock(name)
    lock_path = get_project_root() / "logs" / f".{name}_training.lock"
    return lock_path.exists()


def wait_for_training_unlock(name: str, poll_seconds: int = 60, progress=None) -> None:
    import time

    lock_path = get_project_root() / "logs" / f".{name}_training.lock"
    while lock_path.exists():
        if progress:
            progress(f"Waiting for {name} training to finish...")
        time.sleep(max(5, poll_seconds))
