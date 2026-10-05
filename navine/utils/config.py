from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(name: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and (data.get("profiles") or data.get("size_tier") or data.get("model_size")):
        try:
            from navine.text.arch import apply_size_tier

            data = apply_size_tier(data)
        except Exception:
            pass
    return data


def load_train_config(name: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "train" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
