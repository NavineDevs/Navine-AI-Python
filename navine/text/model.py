import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, head_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0).unsqueeze(0)
    cos = emb.cos().to(dtype=x.dtype)
    sin = emb.sin().to(dtype=x.dtype)
    return x * cos + _rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def make_norm(dim: int, use_rms: bool) -> nn.Module:
    if use_rms:
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            positions = torch.arange(position_offset, position_offset + t, device=x.device)
            q = _apply_rope(q, positions, self.head_dim)
            k = _apply_rope(k, positions, self.head_dim)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)
        drop = float(self.dropout.p) if self.training else 0.0
        try:
            if past_kv is None:
                attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
            else:
                attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=False)
            out = attn.transpose(1, 2).reshape(b, t, c)
            return self.out(out), present
        except Exception:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                if past_kv is not None and mask.size(-1) != k.size(2):
                    mask = torch.ones(b, 1, t, k.size(2), device=x.device)
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, c)
            return self.out(out), present


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_ff = d_ff
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        hidden = int(2 * d_ff / 3)
        hidden = max(d_model, ((hidden + 63) // 64) * 64)
        self.d_ff = hidden
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_rope: bool = False,
        use_swiglu: bool = False,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.ln1 = make_norm(d_model, use_rms_norm)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.ln2 = make_norm(d_model, use_rms_norm)
        self.ff = SwiGLU(d_model, d_ff, dropout) if use_swiglu else FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present = self.attn(self.ln1(x), mask, past_kv, position_offset)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, present


class NavineTextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_kv_cache: bool = False,
        use_swiglu: bool = False,
        use_rms_norm: bool = False,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_kv_cache = use_kv_cache
        self.use_swiglu = bool(use_swiglu)
        self.use_rms_norm = bool(use_rms_norm)
        self.tie_embeddings = bool(tie_embeddings)
        self.dropout_p = float(dropout)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.gradient_checkpointing = False
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    use_rope=use_rope,
                    use_swiglu=self.use_swiglu,
                    use_rms_norm=self.use_rms_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = make_norm(d_model, self.use_rms_norm)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if self.tie_embeddings:
            self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            if isinstance(block.ff, SwiGLU):
                nn.init.normal_(block.ff.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
            elif isinstance(block.ff, FeedForward):
                nn.init.normal_(block.ff.net[-2].weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        idx: torch.Tensor,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        b, t = idx.shape
        position_offset = 0
        if past_kvs:
            position_offset = past_kvs[0][0].size(2)
        if t > self.max_seq_len:
            idx = idx[:, -self.max_seq_len :]
            t = self.max_seq_len
        x = self.token_emb(idx)
        if not self.use_rope:
            pos = torch.arange(position_offset, position_offset + t, device=idx.device)
            pos = pos.clamp(max=self.max_seq_len - 1).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        mask = None
        if not past_kvs:
            mask = self._causal_mask(t, idx.device)
        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        use_ckpt = bool(self.gradient_checkpointing) and self.training and past_kvs is None
        for layer_idx, block in enumerate(self.blocks):
            past_kv = past_kvs[layer_idx] if past_kvs else None
            if use_ckpt:

                def _run(module, hidden, attn_mask, offset):
                    out, present = module(hidden, attn_mask, None, offset)
                    return out, present

                x, present = torch.utils.checkpoint.checkpoint(
                    _run,
                    block,
                    x,
                    mask,
                    position_offset,
                    use_reentrant=False,
                )
            else:
                x, present = block(x, mask, past_kv, position_offset)
            new_kvs.append(present)
        x = self.ln_f(x)
        logits = self.head(x)
        self._last_kvs = new_kvs
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        frequency_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        self.eval()
        generated = idx.clone()
        past_kvs = None
        use_cache = bool(self.use_kv_cache)
        prompt_len = generated.size(1)
        token_counts: Dict[int, int] = {}
        for tid in generated[0].tolist():
            token_counts[tid] = token_counts.get(tid, 0) + 1
        for _ in range(max_new_tokens):
            if use_cache and past_kvs is not None:
                idx_cond = generated[:, -1:]
            else:
                idx_cond = (
                    generated
                    if generated.size(1) <= self.max_seq_len
                    else generated[:, -self.max_seq_len :]
                )
            logits = self(idx_cond, past_kvs=past_kvs if use_cache else None)[:, -1, :]
            if use_cache:
                past_kvs = getattr(self, "_last_kvs", None)
            if repetition_penalty and repetition_penalty > 1.0:
                for token_id in set(generated[0].tolist()):
                    val = logits[0, token_id]
                    logits[0, token_id] = torch.where(
                        val > 0,
                        val / repetition_penalty,
                        val * repetition_penalty,
                    )
            if frequency_penalty and frequency_penalty > 0:
                for token_id, count in token_counts.items():
                    if count > 0:
                        logits[0, token_id] = logits[0, token_id] - frequency_penalty * count
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                seq = generated[0].tolist()
                if len(seq) >= no_repeat_ngram_size - 1:
                    banned = set()
                    n = no_repeat_ngram_size
                    prefix = tuple(seq[-(n - 1) :])
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i : i + n - 1]) == prefix:
                            banned.add(seq[i + n - 1])
                    for token_id in banned:
                        logits[0, token_id] = float("-inf")
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            tid = int(next_token.item())
            token_counts[tid] = token_counts.get(tid, 0) + 1
            if eos_id is not None and (next_token == eos_id).all():
                if generated.size(1) - prompt_len >= max(min_new_tokens, 0):
                    break
        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_config(self) -> Dict[str, Any]:
        first = self.blocks[0]
        if isinstance(first.ff, SwiGLU):
            d_ff = int(first.ff.d_ff)
        else:
            d_ff = int(first.ff.d_ff if hasattr(first.ff, "d_ff") else first.ff.net[0].out_features)
        return {
            "vocab_size": self.token_emb.num_embeddings,
            "d_model": self.token_emb.embedding_dim,
            "n_heads": first.attn.n_heads,
            "n_layers": len(self.blocks),
            "d_ff": d_ff if not self.use_swiglu else self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_p,
            "use_rope": self.use_rope,
            "use_kv_cache": self.use_kv_cache,
            "use_swiglu": self.use_swiglu,
            "use_rms_norm": self.use_rms_norm,
            "tie_embeddings": self.tie_embeddings,
        }

    def save_checkpoint(self, path, tokenizer_meta=None, extra=None):
        from pathlib import Path
        import gc
        import os
        import shutil
        import time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        cpu_state = {}
        for key, value in self.state_dict().items():
            cpu_state[key] = value.detach().to("cpu", dtype=value.dtype).contiguous()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        payload = {
            "model_state": cpu_state,
            "config": self.architecture_config(),
            "parameters": int(self.count_parameters()),
        }
        if extra:
            payload["extra"] = extra
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.stem}_{os.getpid()}_{int(time.time())}.tmp.pt")
        try:
            torch.save(payload, tmp)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise
        del payload
        del cpu_state
        gc.collect()
        last_err = None
        for attempt in range(12):
            try:
                os.replace(str(tmp), str(target))
                return
            except PermissionError as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
            except OSError as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        alt = target.with_name(target.stem + f"_write_{os.getpid()}.pt")
        try:
            shutil.copy2(str(tmp), str(alt))
            print(f"Navine AI - Python: checkpoint locked; wrote {alt}")
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        except Exception:
            pass
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"failed to save checkpoint to {target}")

    @classmethod
    def load_checkpoint(cls, path, device="cpu", target_config: Optional[Dict[str, Any]] = None) -> Tuple["NavineTextModel", dict]:
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(payload["config"])
        cfg.setdefault("use_rope", False)
        cfg.setdefault("use_kv_cache", False)
        cfg.setdefault("use_swiglu", False)
        cfg.setdefault("use_rms_norm", False)
        cfg.setdefault("tie_embeddings", False)
        cfg.setdefault("dropout", 0.1)
        state = payload["model_state"]
        if not cfg.get("use_swiglu") and any(k.startswith("blocks.0.ff.gate") for k in state):
            cfg["use_swiglu"] = True
        if not cfg.get("use_rms_norm") and any(k == "ln_f.weight" and "ln_f.bias" not in state for k in state):
            if "ln_f.bias" not in state:
                cfg["use_rms_norm"] = True
        if target_config:
            from navine.text.arch import architectures_match, build_or_upgrade_text_model

            want = dict(target_config.get("model") or target_config)
            want.setdefault("vocab_size", cfg.get("vocab_size"))
            if not architectures_match(cfg, want):
                vocab = int(want.get("vocab_size") or cfg.get("vocab_size") or 12000)
                model, meta = build_or_upgrade_text_model(
                    {"model": want},
                    vocab_size=vocab,
                    checkpoint_path=path,
                    device=device,
                    force_fresh=False,
                )
                payload = dict(payload)
                payload["upgrade_meta"] = meta
                payload["config"] = model.architecture_config()
                return model, payload
        model = cls(**cfg)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            incompatible = model.load_state_dict(state, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            if len(missing) > max(8, len(state) // 3):
                print("Navine AI - Python: checkpoint architecture mismatch is too large; using loaded shapes as-is where possible.")
        return model, payload


def build_text_model(config: Dict[str, Any], vocab_size: int) -> NavineTextModel:
    model_cfg = dict(config.get("model") or config)
    model_cfg["vocab_size"] = vocab_size
    model_cfg.setdefault("use_rope", False)
    model_cfg.setdefault("use_kv_cache", False)
    model_cfg.setdefault("use_swiglu", False)
    model_cfg.setdefault("use_rms_norm", False)
    model_cfg.setdefault("tie_embeddings", False)
    allowed = {
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "max_seq_len",
        "dropout",
        "use_rope",
        "use_kv_cache",
        "use_swiglu",
        "use_rms_norm",
        "tie_embeddings",
    }
    clean = {k: model_cfg[k] for k in allowed if k in model_cfg}
    return NavineTextModel(**clean)
