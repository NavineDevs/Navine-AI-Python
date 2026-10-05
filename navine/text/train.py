import json
import math
import random
import re
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from navine.text.model import NavineTextModel, build_text_model
from navine.text.tokenizer import NavineTokenizer
from navine.utils.config import load_config
from navine.utils.paths import get_project_root
from navine.utils.tier import resolve_checkpoint_dir

_ASSISTANT_MARKERS = (
    "### Assistant:",
    "### assistant:",
    "<|assistant|>",
    "\nAssistant:",
    "\nassistant:",
)


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


def assistant_loss_start(ids: Sequence[int], tokenizer: NavineTokenizer) -> int:
    best = -1
    for marker in _ASSISTANT_MARKERS:
        for variant in (marker, marker.strip()):
            mid = tokenizer.encode(variant, add_special=False)
            if not mid:
                continue
            pos = find_subsequence(ids, mid)
            if pos >= 0:
                start = pos + len(mid)
                if best < 0 or start < best:
                    best = start
    if best < 0:
        text = tokenizer.decode(list(ids), skip_special=False)
        for marker in _ASSISTANT_MARKERS:
            m = re.search(re.escape(marker), text, flags=re.IGNORECASE)
            if m:
                approx = int(len(ids) * (m.end() / max(len(text), 1)))
                best = min(len(ids) - 1, max(1, approx))
                break
    return best


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: NavineTokenizer,
        max_len: int,
        sft_mask: bool = True,
    ):
        self.samples = []
        self.loss_starts = []
        for text in texts:
            if not text.strip():
                continue
            ids = tokenizer.encode(text)
            if len(ids) < 2:
                continue
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.samples.append(ids)
            start = assistant_loss_start(ids, tokenizer) if sft_mask else -1
            self.loss_starts.append(start)
            if len(self.samples) % 500 == 0:
                print(f"Navine AI - Python: encoded {len(self.samples)} sequences")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.tensor(self.samples[idx], dtype=torch.long), int(self.loss_starts[idx])


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = [item[0] if isinstance(item, tuple) else item for item in batch]
    starts = [item[1] if isinstance(item, tuple) else -1 for item in batch]
    max_len = max(x.size(0) for x in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, : seq.size(0)] = seq
    x = input_ids[:, :-1]
    y = input_ids[:, 1:].clone()
    for i, start in enumerate(starts):
        if start is None or start < 0:
            continue
        # y[t] predicts token at position t+1; supervise only assistant tokens.
        mask_until = max(0, int(start) - 1)
        if mask_until > 0:
            y[i, :mask_until] = pad_id
    return x, y


def load_training_texts(config) -> List[str]:
    root = get_project_root()
    texts = []
    data_cfg = config.get("data") or {}
    if data_cfg.get("use_data_pipeline"):
        try:
            from navine.text.data_pipeline import load_text_sources, pack_chunks

            folders = data_cfg.get("pipeline_folders") or ["data/text", "data/train"]
            piped = pack_chunks(load_text_sources(folders=folders), max_len=config["model"]["max_seq_len"])
            texts.extend(piped)
        except Exception:
            pass
    train_file = data_cfg.get("train_file")
    if train_file:
        train_path = root / train_file
        if train_path.exists():
            content = train_path.read_text(encoding="utf-8")
            if "### User:" in content:
                blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
                texts.extend(blocks)
            else:
                texts.extend(content.splitlines())
    code_file = data_cfg.get("code_file")
    if code_file:
        code_path = root / code_file
        if code_path.exists():
            import json
            for line in code_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                texts.append(entry.get("prompt", ""))
                texts.append(entry.get("code", ""))
    chat_path = root / "data" / "text" / "instruction_chat.txt"
    train_path = root / train_file if train_file else None
    if chat_path.exists() and (train_path is None or chat_path != train_path):
        content = chat_path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        texts.extend(blocks)
    learned = config["data"].get("learned_file")
    if learned:
        learned_path = root / learned
        if learned_path.exists():
            texts.extend(learned_path.read_text(encoding="utf-8").splitlines())
    if data_cfg.get("use_multilingual", True):
        try:
            from navine.text.multilingual import load_multilingual_pretrain_lines, load_multilingual_sft_blocks

            texts.extend(load_multilingual_pretrain_lines())
            texts.extend(load_multilingual_sft_blocks())
        except Exception:
            pass
    return texts


def _merge_config(base: dict, overrides: Optional[dict]) -> dict:
    if not overrides:
        return base
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


def _build_text_optimizer(model, train_cfg: dict, device: torch.device):
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg["weight_decay"])
    name = str(train_cfg.get("optimizer") or "adamw").strip().lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name in ("adamw8bit", "bnb_adamw", "adam8bit"):
        try:
            import bitsandbytes as bnb

            print("Navine AI - Python: using bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd), "adamw8bit"
        except Exception as exc:
            print(f"Navine AI - Python: AdamW8bit unavailable ({exc}); using cpu_adamw")
            name = "cpu_adamw"
    if name in ("sgd", "sgdm"):
        print("Navine AI - Python: using SGD momentum optimizer")
        return (
            torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True),
            "sgd",
        )
    if name in ("cpu_adamw", "adamw_cpu", "offload_adamw"):
        cpu_params = []
        link = []
        for p in params:
            master = p.detach().to("cpu", dtype=torch.float32).clone().requires_grad_(True)
            cpu_params.append(master)
            link.append((p, master))
        opt = torch.optim.AdamW(cpu_params, lr=lr, weight_decay=wd)
        opt._navine_cpu_link = link  # type: ignore[attr-defined]
        print("Navine AI - Python: using CPU-offloaded AdamW (GPU holds weights/grads only)")
        return opt, "cpu_adamw"
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd), "adamw"


def _optimizer_step(optimizer, scaler, use_scaler: bool, opt_kind: str) -> None:
    if opt_kind == "cpu_adamw":
        link = getattr(optimizer, "_navine_cpu_link", None) or []
        for gpu_p, cpu_p in link:
            if gpu_p.grad is None:
                cpu_p.grad = None
                continue
            cpu_p.grad = gpu_p.grad.detach().to("cpu", dtype=torch.float32)
        optimizer.step()
        for gpu_p, cpu_p in link:
            gpu_p.data.copy_(cpu_p.data.to(device=gpu_p.device, dtype=gpu_p.dtype))
        return
    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def train(
    config_path: str = "text",
    max_steps: Optional[int] = None,
    fresh: bool = False,
    require_cuda: bool = False,
    config_overrides: Optional[dict] = None,
) -> None:
    from navine.utils.tier import modality_config_name
    from navine.text.arch import apply_size_tier, archive_text_checkpoint, architectures_match

    if config_path in ("text", "text_enterprise"):
        config_path = modality_config_name("text") if config_path == "text" else config_path
    config = _merge_config(apply_size_tier(load_config(config_path)), config_overrides)
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
    texts = load_training_texts(config)
    if not texts:
        raise ValueError("No training data found. Check data/text/ files.")
    random.shuffle(texts)
    split = int(len(texts) * (1 - config["data"]["val_split"]))
    train_texts = texts[:split] if split > 0 else texts
    val_texts = texts[split:] if split > 0 else texts[:1]
    ckpt_dir = resolve_checkpoint_dir("text", config)
    use_bpe = bool((config.get("tokenizer") or {}).get("use_bpe", True))
    existing = ckpt_dir / "latest.pt"
    need_fresh = bool(fresh)
    if existing.exists() and not need_fresh:
        try:
            payload = torch.load(str(existing), map_location="cpu", weights_only=False)
            saved_cfg = dict((payload or {}).get("config") or {})
            if not architectures_match(saved_cfg, config.get("model") or {}):
                need_fresh = True
                print("Navine AI - Python: yaml architecture differs from checkpoint; forcing fresh medium rebuild")
        except Exception:
            need_fresh = True
    if need_fresh:
        archive = archive_text_checkpoint(ckpt_dir, label=f"archive_{config.get('size_tier', 'prev')}")
        if archive:
            print(f"Archived prior weights: {archive}")
        print(f"Navine AI - Python: fresh rebuild into {ckpt_dir} (size_tier={config.get('size_tier', 'medium')})")
    tok_path = ckpt_dir / "tokenizer.json"
    if (not need_fresh) and tok_path.exists():
        tokenizer = NavineTokenizer.load(tok_path)
        print(f"Navine AI - Python: loaded tokenizer from {tok_path} (vocab={len(tokenizer.token_to_id)})")
    else:
        tokenizer = NavineTokenizer(config["model"]["vocab_size"], use_bpe=use_bpe)
        print(f"Navine AI - Python: building {'BPE' if use_bpe else 'word'} tokenizer (vocab={config['model']['vocab_size']})...")
        tok_corpus = train_texts + val_texts
        try:
            from navine.device_manager import get_device_settings, detect_hardware

            use_all = bool(get_device_settings().get("use_all_system_ram", False))
            ram = float((detect_hardware() or {}).get("system_ram_gb") or 0)
            tok_cap = 50000 if use_all and ram >= 14 else 12000
        except Exception:
            tok_cap = 12000
        if len(tok_corpus) > tok_cap:
            tok_corpus = tok_corpus[:tok_cap]
        tokenizer.build(tok_corpus, use_bpe=use_bpe)
    actual_vocab = len(tokenizer.token_to_id)
    model_cfg = config["model"].copy()
    model_cfg["vocab_size"] = actual_vocab
    if (not need_fresh) and existing.exists():
        model, _payload = NavineTextModel.load_checkpoint(existing, device)
        model = model.to(device)
        print(f"Navine AI - Python: resumed weights from {existing}")
    else:
        model = build_text_model(model_cfg, actual_vocab).to(device)
    arch = model.architecture_config()
    print(
        f"Navine Neural Text | params={model.count_parameters():,} | device={device} | "
        f"tier={config.get('size_tier', 'medium')} layers={arch['n_layers']} d={arch['d_model']} "
        f"rope={arch['use_rope']} swiglu={arch['use_swiglu']} rms={arch['use_rms_norm']} tie={arch['tie_embeddings']}"
    )
    from navine.utils.param_verify import assert_real_param_count

    assert_real_param_count(model, f"text:{config_path}", size_tier=str(config.get("size_tier") or ""))
    train_cfg = config.get("training") or {}
    if bool(train_cfg.get("gradient_checkpointing")):
        model.gradient_checkpointing = True
        print("Navine AI - Python: gradient checkpointing enabled")
    if compute.get("dataloader"):
        print(f"Hybrid compute | CPU threads: {compute.get('cpu_threads')} | DataLoader: {compute.get('dataloader')}")
    sft_mask = bool(config.get("training", {}).get("sft_mask", True))
    train_seq = int(
        train_cfg.get("train_seq_len")
        or train_cfg.get("train_max_seq_len")
        or model_cfg["max_seq_len"]
    )
    train_seq = max(64, min(train_seq, int(model_cfg["max_seq_len"])))
    from navine.train.batch import apply_auto_batch

    arch = model.architecture_config()
    batch_info = apply_auto_batch(train_cfg, arch, modality="text")
    batch_size = batch_info["batch_size"]
    grad_accum = batch_info["grad_accum_steps"]
    root = get_project_root()
    data_cfg = config.get("data") or {}
    use_bins = bool(data_cfg.get("use_token_bins")) and not sft_mask
    train_bin = root / str(data_cfg.get("token_train_bin") or "data/tiny_gpt/train.bin")
    val_bin = root / str(data_cfg.get("token_val_bin") or "data/tiny_gpt/val.bin")
    if use_bins and train_bin.exists() and val_bin.exists():
        from navine.text.story_pipeline import TokenBinDataset

        train_ds = TokenBinDataset(train_bin, train_seq)
        val_ds = TokenBinDataset(val_bin, train_seq)

        def bin_collate(batch):
            xs = torch.stack([item[0] for item in batch])
            ys = torch.stack([item[1] for item in batch])
            return xs, ys

        pad_collate = bin_collate
        print(f"Navine AI - Python: using token bins | train={train_bin.name} val={val_bin.name}")
    else:
        train_ds = TextDataset(train_texts, tokenizer, train_seq, sft_mask=sft_mask)
        val_ds = TextDataset(val_texts, tokenizer, train_seq, sft_mask=sft_mask)
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
    optimizer, opt_kind = _build_text_optimizer(model, config["training"], device)
    warmup = int(config["training"].get("warmup_steps", 50))
    steps_limit = int(max_steps) if max_steps is not None else int(config["training"]["max_steps"])
    grad_accum = max(1, int(grad_accum))
    writer = None
    if config["training"].get("tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(get_project_root() / "logs" / "text_tensorboard"))
        except Exception:
            writer = None

    def lr_at(step_idx: int) -> float:
        base = config["training"]["learning_rate"]
        if step_idx < warmup:
            return base * (step_idx + 1) / max(warmup, 1)
        progress = (step_idx - warmup) / max(steps_limit - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    tokenizer.save(ckpt_dir / "tokenizer.json")
    step = 0
    best_val = float("inf")
    use_amp = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    use_scaler = use_amp and opt_kind not in ("cpu_adamw",)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if bool(config["training"].get("torch_compile", False)) and device.type == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("Navine AI - Python: torch.compile enabled for text model")
        except Exception as exc:
            print(f"Navine AI - Python: torch.compile skipped ({exc})")
    model.train()
    pbar = tqdm(total=steps_limit, desc="Training Navine AI - Python Text")
    accum_loss = 0.0
    try:
        while step < steps_limit:
            for x, y in train_loader:
                x, y = to_device(x, device), to_device(y, device)
                with get_autocast_context(device):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_id
                    )
                    loss = loss / grad_accum
                accum_loss += loss.item()
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % grad_accum == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr_at(step)
                    _optimizer_step(optimizer, scaler, use_scaler, opt_kind)
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{accum_loss:.4f}")
                if step == 1 or step % 5 == 0 or step >= steps_limit:
                    print(f"PROGRESS steps={step}/{steps_limit}", flush=True)
                    try:
                        from navine.api.train_progress_helper import report_train_progress

                        report_train_progress(
                            stage="text",
                            steps=step,
                            steps_total=steps_limit,
                            last_line=f"PROGRESS steps={step}/{steps_limit}",
                            force=(step == 1 or step >= steps_limit),
                        )
                    except Exception:
                        pass
                if writer is not None:
                    writer.add_scalar("train/loss", accum_loss, step)
                accum_loss = 0.0
                if step % config["training"]["eval_interval"] == 0:
                    print(f"Evaluating at step {step}...", flush=True)
                    try:
                        from navine.api.train_progress_helper import report_train_progress

                        report_train_progress(
                            stage="text",
                            steps=step,
                            steps_total=steps_limit,
                            label="text eval",
                            last_line=f"Evaluating at step {step}...",
                            force=True,
                        )
                    except Exception:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model.eval()
                    val_losses = []
                    max_eval_batches = int(train_cfg.get("max_eval_batches") or 8)
                    try:
                        with torch.no_grad():
                            for bi, (vx, vy) in enumerate(val_loader):
                                if bi >= max_eval_batches:
                                    break
                                vx, vy = to_device(vx, device), to_device(vy, device)
                                with get_autocast_context(device):
                                    v_logits = model(vx)
                                    v_loss = torch.nn.functional.cross_entropy(
                                        v_logits.reshape(-1, v_logits.size(-1)),
                                        vy.reshape(-1),
                                        ignore_index=tokenizer.pad_id,
                                    )
                                val_losses.append(float(v_loss.item()))
                                del v_logits, v_loss, vx, vy
                        if val_losses:
                            mean_val = sum(val_losses) / len(val_losses)
                            print(f"Step {step} | Val Loss: {mean_val:.4f}")
                            if mean_val < best_val:
                                best_val = mean_val
                                _safe_save(model, ckpt_dir / "best.pt", device, required=False)
                            if writer is not None:
                                writer.add_scalar("val/loss", mean_val, step)
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        if "out of memory" in msg or "cuda" in msg:
                            print(f"Navine AI - Python: skipping eval at step {step} ({exc})")
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        else:
                            raise
                    model.train()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if step % config["training"]["save_interval"] == 0 or step == steps_limit:
                    _safe_save(model, ckpt_dir / "latest.pt", device, required=(step >= steps_limit))
                if step >= steps_limit:
                    break
    except KeyboardInterrupt:
        _safe_save(model, ckpt_dir / "latest.pt", device, required=False)
        print("Interrupted. Checkpoint saved.")
    pbar.close()
    if writer is not None:
        writer.close()
    _safe_save(model, ckpt_dir / "latest.pt", device, required=True)
    print(f"Training complete. Checkpoint saved to {ckpt_dir / 'latest.pt'}")


def _safe_save(model: NavineTextModel, path, device, required: bool = True) -> bool:
    was_cuda = str(device).startswith("cuda")
    if was_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    try:
        model.save_checkpoint(path)
        return True
    except (MemoryError, RuntimeError, OSError) as exc:
        print(f"Navine AI - Python: checkpoint save skipped ({exc})")
        if required:
            raise
        return False
    finally:
        if was_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
