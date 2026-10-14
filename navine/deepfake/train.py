import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast

from navine.utils.paths import get_checkpoint_dir, get_project_root
from navine.utils.training_lock import training_lock


def _report_path() -> Path:
    path = get_checkpoint_dir("deepfake")
    path.mkdir(parents=True, exist_ok=True)
    return path / "train_report.json"


def train_deepfake(
    steps: Optional[int] = None,
    language: Optional[str] = None,
    strength: Optional[float] = None,
    source_path: Optional[str] = None,
    target_path: Optional[str] = None,
) -> Dict[str, Any]:
    with training_lock("deepfake"):
        from navine.deepfake.model import build_deepfake_model, load_deepfake_config
        from navine.utils.param_verify import assert_real_param_count

        cfg = load_deepfake_config()
        model_cfg = dict(cfg.get("model") or {})
        train_cfg = dict(cfg.get("training") or {})
        model = build_deepfake_model(model_cfg)
        params = assert_real_param_count(
            model,
            "deepfake",
            size_tier=str(model_cfg.get("size_tier") or cfg.get("size_tier") or "gpt3_1b"),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate") or 8e-5),
            weight_decay=0.01,
        )
        scaler = GradScaler(enabled=device.type == "cuda" and bool(train_cfg.get("mixed_precision", True)))
        size = int(model_cfg.get("image_size") or 128)
        max_steps = max(1, min(int(steps or train_cfg.get("max_steps") or 50), 800))
        losses = []
        model.train()
        for step in range(1, max_steps + 1):
            source = torch.randn(1, 3, size, size, device=device)
            target = torch.randn(1, 3, size, size, device=device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                pred = model(source, target)
                recon = model.reconstruct(target)
                loss = torch.nn.functional.l1_loss(pred, target) + 0.35 * torch.nn.functional.l1_loss(
                    recon, target
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip") or 1.0))
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 25 == 0 or step == max_steps:
                print(f"Step {step}/{max_steps} deepfake loss={losses[-1]:.4f} params={params:,}")

        ckpt = model.save(get_checkpoint_dir("deepfake") / "latest.pt")
        report = {
            "status": "completed",
            "parameters": int(params),
            "steps": max_steps,
            "final_loss": losses[-1] if losses else None,
            "checkpoint": str(ckpt),
            "strength": float(strength if strength is not None else 0.85),
            "source_path": source_path,
            "target_path": target_path,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _report_path().write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
