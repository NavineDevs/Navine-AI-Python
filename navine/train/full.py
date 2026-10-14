import time
from datetime import datetime, timezone
from typing import Optional

from navine.utils.paths import get_project_root


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    try:
        log_path = get_project_root() / "logs" / "train_full_status.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def train_full(steps: Optional[int] = None, language: Optional[str] = None) -> None:
    _log("TRAIN_FULL_BEGIN")
    started = time.time()

    try:
        from navine.nsfw.config import is_nsfw_enabled

        _log(f"NSFW enabled={is_nsfw_enabled()}")
    except Exception as exc:
        _log(f"NSFW status check failed: {exc}")

    jobs = []

    def _add(name, fn):
        jobs.append((name, fn))

    def run_text():
        from navine.text.train import train

        train()

    def run_nsfw_text():
        from navine.train.nsfw import train

        train(steps=steps or 300)

    def run_unrestricted():
        from navine.train.unrestricted import train

        train(steps=steps or 200)

    def run_registry():
        from navine.train.registry import train_all

        train_all(language=language, steps=steps or 150)

    def run_image_nsfw():
        try:
            from navine.learn.image_learn import ingest_learned_images, ingest_nsfw_images

            ingest_learned_images()
            ingest_nsfw_images()
        except Exception:
            pass
        from navine.image.train import train

        train(config_path="image", finetune=True, finetune_steps=steps or 800)

    def run_image_custom():
        from navine.image.train import train

        train(config_path="image_custom", finetune=True, finetune_steps=steps or 600)

    def run_image_lora():
        from navine.image.lora_custom import train_custom_lora

        train_custom_lora(steps=steps or 400)

    def run_video():
        from navine.video.train import train

        train(config_path="video", finetune=True, finetune_steps=steps or 200)

    def run_voice():
        from navine.voice.train import train_voice

        train_voice(steps=steps or 4, language=language)

    _add("text", run_text)
    _add("nsfw_text", run_nsfw_text)
    _add("unrestricted_text", run_unrestricted)
    _add("registry_all", run_registry)
    _add("image_nsfw", run_image_nsfw)
    _add("image_custom", run_image_custom)
    _add("image_lora", run_image_lora)
    _add("video", run_video)
    _add("voice", run_voice)

    for name, fn in jobs:
        _log(f"START {name}")
        t0 = time.time()
        try:
            fn()
            _log(f"DONE {name} in {int(time.time() - t0)}s")
        except Exception as exc:
            _log(f"ERROR {name}: {exc}")

    _log(f"TRAIN_FULL_DONE total={int(time.time() - started)}s")
