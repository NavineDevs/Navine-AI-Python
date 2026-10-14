"""Post-train smoke eval for custom-only video generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.paths import get_output_dir, get_project_root
from navine.utils.tier import resolve_checkpoint_dir
from navine.video.train import VIDEO_PROMPTS


def _frame_variance(video_path: Path) -> float:
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        prev = None
        total = 0.0
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev is not None:
                total += float(np.mean(np.abs(gray - prev)))
                count += 1
            prev = gray
        cap.release()
        return total / max(1, count)
    except Exception:
        return 0.0


def run_video_smoke_eval(
    model=None,
    device=None,
    config: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    use_loaded_model: bool = False,
) -> Dict[str, Any]:
    from navine.device_manager import get_device
    from navine.utils.tier import load_modality_config

    config = config or load_modality_config("video")
    device = device or get_device()
    out_root = Path(output_dir) if output_dir else get_output_dir("video")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_checkpoint_dir("video", config) / "latest.pt"
    results: Dict[str, Any] = {"ok": True, "items": {}}

    if use_loaded_model and model is not None:
        import torch
        from PIL import Image

        from navine.video.infer import _save_video

        model.eval()
        for tag, prompt in VIDEO_PROMPTS.items():
            path = out_root / f"generated_{tag}.mp4"
            seed = 42 + hash(tag) % 1000
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            frames_tensor = model.generate(text=prompt, device=device, seed=seed)
            imgs = []
            for i in range(frames_tensor.size(1)):
                frame = frames_tensor[0, i].cpu()
                frame = ((frame + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                imgs.append(Image.fromarray(frame))
            _save_video(imgs, path, fps=int(config.get("inference", {}).get("fps", 12)), config=config)
            exists = path.exists()
            size_ok = exists and path.stat().st_size > 4096
            variance = _frame_variance(path) if exists else 0.0
            min_var = float(config.get("inference", {}).get("min_frame_variance", 0.001))
            valid = exists and size_ok and variance >= min_var
            results["items"][tag] = {"path": str(path), "valid": valid, "variance": variance}
            if not valid:
                results["ok"] = False
            print(f"smoke {tag}: valid={valid} variance={variance:.5f} -> {path}")
    else:
        from navine.video.infer import generate as generate_video

        for tag, prompt in VIDEO_PROMPTS.items():
            path = out_root / f"generated_{tag}.mp4"
            generate_video(
                prompt,
                output_path=str(path),
                config_name=None,
                seed=42 + hash(tag) % 1000,
                auto_ingest=False,
            )
            exists = path.exists()
            size_ok = exists and path.stat().st_size > 4096
            variance = _frame_variance(path) if exists else 0.0
            min_var = float(config.get("inference", {}).get("min_frame_variance", 0.001))
            valid = exists and size_ok and variance >= min_var
            meta = {
                "prompt": prompt,
                "generation_mode": "video_checkpoint",
                "reference_used": False,
                "custom_trained_only": True,
                "checkpoint": str(ckpt_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output": str(path),
                "valid": valid,
                "frame_variance": variance,
                "category_tag": tag,
            }
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            results["items"][tag] = {"path": str(path), "valid": valid, "variance": variance}
            if not valid:
                results["ok"] = False
            print(f"smoke {tag}: valid={valid} variance={variance:.5f} -> {path}")
    report = get_project_root() / "logs" / "video_smoke_eval.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Video smoke eval ok={results['ok']} report={report}")
    return results


if __name__ == "__main__":
    run_video_smoke_eval()
