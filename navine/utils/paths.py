from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_checkpoint_dir(model_type: str) -> Path:
    path = get_project_root() / "checkpoints" / model_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(model_type: str) -> Path:
    return get_project_root() / "data" / model_type


def get_output_dir(model_type: str) -> Path:
    path = get_project_root() / "outputs" / model_type
    path.mkdir(parents=True, exist_ok=True)
    return path
