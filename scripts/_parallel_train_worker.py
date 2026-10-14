import os
import sys
import time
import traceback
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("NAVINE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def _deadline() -> float:
    return float(os.environ.get("NAVINE_TRAIN_DEADLINE", time.time() + 3600))


def _remaining() -> float:
    return max(0.0, _deadline() - time.time())


def _log(name: str, msg: str) -> None:
    print(f"[{name}] {msg}", flush=True)


def _steps_for_remaining(step_seconds: float = 2.5, cap: int = 400) -> int:
    remaining = _remaining()
    if remaining <= 30:
        return 0
    estimate = int(remaining / step_seconds)
    return max(40, min(cap, estimate))


def _clear_locks() -> None:
    from navine.utils.training_lock import clear_stale_lock

    for lock_name in ("image", "image_lora", "video", "voice"):
        clear_stale_lock(lock_name)


def _worker_text_stack(name: str) -> None:
    from navine.device_manager import configure_compute_environment
    from navine.train.registry import train_all
    from navine.train.nsfw import train as train_nsfw
    from navine.train.unrestricted import train as train_unrestricted
    from navine.text.train import train as train_text

    configure_compute_environment()
    _clear_locks()
    cycle = 0
    while _remaining() > 45:
        cycle += 1
        steps = _steps_for_remaining(step_seconds=1.5, cap=250)
        if steps <= 0:
            break
        _log(name, f"cycle {cycle} text chunk steps={steps}")
        try:
            train_text(max_steps=steps)
        except Exception as exc:
            _log(name, f"text error: {exc}")
        if _remaining() <= 45:
            break
        _log(name, f"cycle {cycle} registry steps={steps}")
        try:
            train_all(steps=steps)
        except Exception as exc:
            _log(name, f"registry error: {exc}")
        if _remaining() <= 45:
            break
        chunk = min(steps, 180)
        try:
            train_nsfw(steps=chunk)
        except Exception as exc:
            _log(name, f"nsfw error: {exc}")
        try:
            train_unrestricted(steps=chunk)
        except Exception as exc:
            _log(name, f"unrestricted error: {exc}")


def _worker_text_enterprise(name: str) -> None:
    from navine.device_manager import configure_compute_environment
    from navine.text.train import train as train_text

    configure_compute_environment()
    _clear_locks()
    cycle = 0
    while _remaining() > 45:
        cycle += 1
        steps = _steps_for_remaining(step_seconds=2.0, cap=300)
        if steps <= 0:
            break
        _log(name, f"cycle {cycle} text_enterprise steps={steps}")
        try:
            train_text("text_enterprise", max_steps=steps)
        except Exception as exc:
            _log(name, f"text_enterprise error: {exc}")
            _log(name, traceback.format_exc())
            break


def _worker_image_stack(name: str) -> None:
    from navine.device_manager import configure_compute_environment
    from navine.image.train import train as train_image

    configure_compute_environment()
    _clear_locks()
    plans = [
        ("image_nsfw", "image"),
        ("image_custom", "image_custom"),
        ("image_enterprise", "image_enterprise"),
    ]
    cycle = 0
    while _remaining() > 60:
        cycle += 1
        for label, config_path in plans:
            if _remaining() <= 60:
                break
            steps = _steps_for_remaining(step_seconds=2.5, cap=200)
            if steps <= 0:
                break
            _log(name, f"cycle {cycle} {label} finetune steps={steps}")
            try:
                train_image(config_path=config_path, finetune=True, finetune_steps=steps)
            except Exception as exc:
                _log(name, f"{label} error: {exc}")


def _worker_image_lora(name: str) -> None:
    from navine.device_manager import configure_compute_environment
    from navine.image.lora_custom import train_custom_lora

    configure_compute_environment()
    cycle = 0
    while _remaining() > 45:
        _clear_locks()
        cycle += 1
        steps = _steps_for_remaining(step_seconds=2.0, cap=300)
        if steps <= 0:
            break
        _log(name, f"cycle {cycle} custom lora steps={steps}")
        try:
            train_custom_lora(steps=steps)
        except Exception as exc:
            _log(name, f"lora error: {exc}")
            time.sleep(5)


def _worker_video(name: str) -> None:
    from navine.device_manager import configure_compute_environment
    from navine.video.train import train as train_video

    configure_compute_environment()
    _clear_locks()
    cycle = 0
    while _remaining() > 45:
        cycle += 1
        steps = _steps_for_remaining(step_seconds=1.8, cap=250)
        if steps <= 0:
            break
        _log(name, f"cycle {cycle} video finetune steps={steps}")
        try:
            train_video(config_path="video", finetune=True, finetune_steps=steps)
        except Exception as exc:
            _log(name, f"video error: {exc}")
        if _remaining() <= 45:
            break
        _log(name, f"cycle {cycle} video_enterprise finetune steps={steps}")
        try:
            train_video(config_path="video_enterprise", finetune=True, finetune_steps=steps)
        except Exception as exc:
            _log(name, f"video_enterprise error: {exc}")


def _worker_voice_misc(name: str) -> None:
    from navine.device_manager import configure_compute_environment
    from navine.voice.train import train_voice

    configure_compute_environment()
    _clear_locks()
    _log(name, "voice calibration")
    try:
        train_voice(steps=4)
    except Exception as exc:
        _log(name, f"voice error: {exc}")
    _log(name, "voice worker idle until deadline")
    while _remaining() > 5:
        time.sleep(30)


WORKERS = {
    "text_stack": _worker_text_stack,
    "text_enterprise": _worker_text_enterprise,
    "image_stack": _worker_image_stack,
    "image_lora": _worker_image_lora,
    "video": _worker_video,
    "voice_misc": _worker_voice_misc,
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: parallel_train_worker.py <worker_name>", flush=True)
        return 1
    name = sys.argv[1]
    root = _root()
    sys.path.insert(0, str(root))
    os.chdir(root)
    worker = WORKERS.get(name)
    if not worker:
        print(f"Unknown worker: {name}", flush=True)
        return 1
    _log(name, f"worker begin remaining={int(_remaining())}s device={os.environ.get('NAVINE_DEVICE_MODE', 'auto')}")
    worker(name)
    _log(name, "worker end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
