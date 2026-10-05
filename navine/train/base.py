import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from navine.text.model import NavineTextModel, build_text_model
from navine.text.tokenizer import NavineTokenizer
from navine.text.train import TextDataset, collate_fn
from navine.utils.config import load_train_config
from navine.utils.tier import active_tier, load_modality_config, resolve_checkpoint_dir


def _resolve_finetune_bundle(train_config_name: str) -> Tuple[Dict[str, Any], Path]:
    from navine.text.arch import apply_size_tier

    train_config = load_train_config(train_config_name)
    if active_tier() == "enterprise":
        base_config = apply_size_tier(load_modality_config("text"))
        config: Dict[str, Any] = dict(train_config)
        config["model"] = dict(base_config["model"])
        config["size_tier"] = base_config.get("size_tier", "medium")
        if base_config.get("profiles"):
            config["profiles"] = dict(base_config.get("profiles") or {})
        base_training = dict(base_config.get("training") or {})
        train_training = dict(train_config.get("training") or {})
        config["training"] = {**base_training, **train_training}
        if base_config.get("tokenizer"):
            config["tokenizer"] = dict(base_config.get("tokenizer") or {})
        ckpt_dir = resolve_checkpoint_dir("text", base_config)
        return apply_size_tier(config), ckpt_dir
    return apply_size_tier(train_config), resolve_checkpoint_dir("text", load_modality_config("text"))


def finetune_texts(
    config_name: str,
    texts: List[str],
    desc: str = "Fine-tuning Navine AI - Python",
    max_steps: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    require_cuda: bool = False,
) -> None:
    if not texts:
        raise ValueError("No training data found.")
    config, ckpt_dir = _resolve_finetune_bundle(config_name)
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    from navine.device_manager import (
        get_autocast_context,
        get_device,
        get_dataloader_kwargs,
        print_train_device_banner,
        to_device,
    )

    compute = print_train_device_banner(require_cuda=require_cuda)
    device = get_device()
    loader_kwargs = get_dataloader_kwargs()
    random.shuffle(texts)
    split = max(1, int(len(texts) * (1 - config["data"]["val_split"])))
    train_texts = texts[:split]
    val_texts = texts[split:] if split < len(texts) else texts[:1]
    from navine.text.arch import (
        archive_text_checkpoint,
        architectures_match,
        build_or_upgrade_text_model,
    )

    tok_path = ckpt_dir / "tokenizer.json"
    ckpt_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    use_bpe = bool((config.get("tokenizer") or {}).get("use_bpe", True))
    yaml_model = dict(config.get("model") or {})
    max_seq = int(yaml_model.get("max_seq_len") or 512)

    resume_ok = False
    if tok_path.exists() and ckpt_path.exists():
        try:
            probe = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            saved_cfg = dict((probe or {}).get("config") or {})
            resume_ok = architectures_match(saved_cfg, yaml_model)
        except Exception:
            resume_ok = False

    if resume_ok:
        tokenizer = NavineTokenizer.load(tok_path)
        model, payload = NavineTextModel.load_checkpoint(ckpt_path, device, target_config=config)
        model = model.to(device)
        print(f"Navine AI - Python fine-tuning from checkpoint: {ckpt_path}")
        ckpt_cfg = model.architecture_config()
        print(
            f"Loaded NN | layers={ckpt_cfg.get('n_layers')} d={ckpt_cfg.get('d_model')} "
            f"swiglu={ckpt_cfg.get('use_swiglu')} rms={ckpt_cfg.get('use_rms_norm')}"
        )
        max_seq = int(ckpt_cfg.get("max_seq_len") or max_seq)
    else:
        source_ckpt = ckpt_path if ckpt_path.exists() else None
        if source_ckpt is not None:
            try:
                probe = torch.load(str(source_ckpt), map_location="cpu", weights_only=False)
                saved_probe = dict((probe or {}).get("config") or {})
                already_large = int(saved_probe.get("d_model") or 0) >= int(yaml_model.get("d_model") or 0)
            except Exception:
                already_large = False
            if not already_large:
                archive = archive_text_checkpoint(ckpt_dir, label="archive_small_58m")
                if archive and (archive / "latest.pt").exists():
                    print(f"Archived prior weights before architecture upgrade: {archive}")
                    source_ckpt = archive / "latest.pt"
        if tok_path.exists():
            tokenizer = NavineTokenizer.load(tok_path)
            print(f"Navine AI - Python: reusing tokenizer ({len(tokenizer.token_to_id)} tokens).")
        else:
            tokenizer = NavineTokenizer(config["model"]["vocab_size"], use_bpe=use_bpe)
            print(
                f"Navine AI - Python: building {'BPE' if use_bpe else 'word'} tokenizer "
                f"for size_tier={config.get('size_tier', 'medium')}..."
            )
            tokenizer.build(train_texts + val_texts, use_bpe=use_bpe)
        model, meta = build_or_upgrade_text_model(
            config,
            vocab_size=len(tokenizer.token_to_id),
            checkpoint_path=source_ckpt,
            device=device,
            force_fresh=False,
        )
        if meta.get("fresh"):
            print("Navine AI - Python: training medium neural net from scratch (architecture bump).")
        else:
            print(
                f"Navine AI - Python: upgraded weights "
                f"({meta.get('transferred', 0)} tensors transferred, "
                f"skipped={meta.get('skipped', 0)}, note={meta.get('note')})."
            )

    arch = model.architecture_config()
    print(
        f"{desc} | Parameters: {model.count_parameters():,} | Device: {device} | "
        f"tier={config.get('size_tier', 'medium')} "
        f"arch L{arch['n_layers']}/D{arch['d_model']}/H{arch['n_heads']} "
        f"swiglu={arch['use_swiglu']} rms={arch['use_rms_norm']} rope={arch['use_rope']}"
    )
    if compute.get("dataloader"):
        print(f"Hybrid compute | CPU threads: {compute.get('cpu_threads')} | DataLoader: {compute.get('dataloader')}")

    train_cfg = dict(config.get("training") or {})
    sft_mask = bool(train_cfg.get("sft_mask", True))
    d_model = int(yaml_model.get("d_model") or 0)
    seq_cap = int(
        train_cfg.get("train_max_seq_len")
        or train_cfg.get("train_seq_len")
        or min(max_seq, 256 if d_model >= 896 else max_seq)
    )
    train_max_seq = max(32, min(max_seq, seq_cap))
    if train_max_seq < max_seq:
        print(f"Navine AI - Python: capping train seq {max_seq} -> {train_max_seq} for VRAM safety")
    if device.type == "cuda" and d_model >= 896 and not train_cfg.get("optimizer"):
        train_cfg["optimizer"] = "cpu_adamw"
    from navine.train.batch import apply_auto_batch
    from navine.text.train import _build_text_optimizer, _optimizer_step

    batch_info = apply_auto_batch(train_cfg, arch, modality="text")
    batch_size = batch_info["batch_size"]
    grad_accum = batch_info["grad_accum_steps"]
    train_ds = TextDataset(train_texts, tokenizer, train_max_seq, sft_mask=sft_mask)
    val_ds = TextDataset(val_texts, tokenizer, train_max_seq, sft_mask=sft_mask)
    pad_collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        **loader_kwargs,
    )
    optimizer, opt_kind = _build_text_optimizer(model, train_cfg, device)
    tokenizer.save(tok_path)
    step = 0
    steps_limit = max_steps if max_steps is not None else train_cfg["max_steps"]
    warmup = int(train_cfg.get("warmup_steps", max(20, steps_limit // 20)))
    grad_accum = max(1, int(grad_accum))
    use_amp = device.type == "cuda" and bool(train_cfg.get("mixed_precision", True))
    use_scaler = use_amp and opt_kind not in ("cpu_adamw",)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best_val = float("inf")

    def lr_at(step_idx: int) -> float:
        base = float(train_cfg["learning_rate"])
        if step_idx < warmup:
            return base * (step_idx + 1) / max(warmup, 1)
        progress = (step_idx - warmup) / max(steps_limit - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    if bool(train_cfg.get("gradient_checkpointing")):
        model.gradient_checkpointing = True
        print("Navine AI - Python: gradient checkpointing enabled")

    model.train()
    pbar = tqdm(total=steps_limit, desc=desc)
    optimizer.zero_grad(set_to_none=True)
    while step < steps_limit:
        for x, y in train_loader:
            x, y = to_device(x, device), to_device(y, device)
            with get_autocast_context(device):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
                loss = loss / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % grad_accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                _optimizer_step(optimizer, scaler, use_scaler, opt_kind)
                optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", lr=f"{lr_at(step - 1):.2e}")
            if step == 1 or step % 5 == 0 or step >= steps_limit:
                print(f"PROGRESS steps={step}/{steps_limit}", flush=True)
                try:
                    from navine.api.train_progress_helper import report_train_progress

                    report_train_progress(
                        stage=str(desc or "training"),
                        steps=step,
                        steps_total=steps_limit,
                        last_line=f"PROGRESS steps={step}/{steps_limit}",
                        force=(step == 1 or step >= steps_limit),
                    )
                except Exception:
                    pass
            if step % int(train_cfg.get("eval_interval", 100)) == 0:
                print(f"Evaluating at step {step}...", flush=True)
                try:
                    from navine.api.train_progress_helper import report_train_progress

                    report_train_progress(
                        stage=str(desc or "training"),
                        steps=step,
                        steps_total=steps_limit,
                        label=f"{desc or 'train'} eval",
                        last_line=f"Evaluating at step {step}...",
                        force=True,
                    )
                except Exception:
                    pass
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = to_device(vx, device), to_device(vy, device)
                        with get_autocast_context(device):
                            v_logits = model(vx)
                            v_loss = torch.nn.functional.cross_entropy(
                                v_logits.reshape(-1, v_logits.size(-1)),
                                vy.reshape(-1),
                                ignore_index=tokenizer.pad_id,
                            )
                        val_losses.append(float(v_loss.item()))
                mean_val = sum(val_losses) / max(len(val_losses), 1)
                print(f"Step {step} | Val Loss: {mean_val:.4f}")
                if mean_val < best_val:
                    best_val = mean_val
                    from navine.text.train import _safe_save

                    _safe_save(model, best_path, device, required=False)
                model.train()
            if step % int(train_cfg.get("save_interval", 50)) == 0 or step == steps_limit:
                from navine.text.train import _safe_save

                _safe_save(model, ckpt_path, device, required=(step >= steps_limit))
            if step >= steps_limit:
                break
    pbar.close()
    from navine.text.train import _safe_save

    _safe_save(model, ckpt_path, device, required=True)
    if best_val < float("inf") and not best_path.exists():
        _safe_save(model, best_path, device, required=False)
    from navine.text.infer import clear_model_cache

    clear_model_cache()
    print(f"Navine AI - Python training complete. Checkpoint: {ckpt_path}")
    if best_path.exists():
        print(f"Best validation checkpoint: {best_path}")
