import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navine.image.train import train

if __name__ == "__main__":
    steps = 500
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    train(
        config_path="image_enterprise_v2_mix",
        finetune=True,
        finetune_steps=steps,
        require_cuda=True,
    )
