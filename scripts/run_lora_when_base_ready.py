import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navine.image.lora_custom import train_custom_lora

BASE_LOG = Path(__file__).resolve().parents[1] / "logs" / "image_custom_train.log"
DONE_MARKER = "Training complete."


def base_finished() -> bool:
    if not BASE_LOG.exists():
        return False
    try:
        text = BASE_LOG.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return DONE_MARKER in text


def main() -> None:
    print("Navine AI: waiting for base image training to finish before LoRA...")
    while not base_finished():
        time.sleep(60)
    time.sleep(10)
    print("Navine AI: base training finished. Starting custom-model LoRA training.")
    report = train_custom_lora(progress=print)
    print(f"Navine AI: LoRA training done: {report}")


if __name__ == "__main__":
    main()
