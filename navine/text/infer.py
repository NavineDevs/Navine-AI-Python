from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from navine.device_manager import configure_compute_environment, get_autocast_context, get_device
from navine.text.model import NavineTextModel
from navine.text.tokenizer import NavineTokenizer
from navine.utils.config import load_config
from navine.utils.tier import load_modality_config, resolve_checkpoint_dir

_ModelBundle = Tuple[NavineTextModel, NavineTokenizer, dict, torch.device]
_MODEL_CACHE: Dict[str, _ModelBundle] = {}
_ENV_CONFIGURED = False


def _ensure_compute_env() -> None:
    global _ENV_CONFIGURED
    if _ENV_CONFIGURED:
        return
    configure_compute_environment()
    _ENV_CONFIGURED = True


def _resolve_paths(checkpoint: Optional[str], config_name: Optional[str] = None):
    if config_name:
        config = load_config(config_name)
    elif checkpoint and "hitboyx23_ai_python" in Path(checkpoint).as_posix():
        try:
            config = load_config("hitboyx23_ai_python")
        except FileNotFoundError:
            config = load_config("text_enterprise")
    elif checkpoint and "hitboyx23" in Path(checkpoint).as_posix():
        try:
            config = load_config("hitboyx23_ai")
        except FileNotFoundError:
            config = load_config("text_enterprise")
    elif checkpoint and "text_code" in Path(checkpoint).as_posix():
        try:
            config = load_config("text_code")
        except FileNotFoundError:
            config = load_config("text_enterprise")
    elif checkpoint and "text_enterprise" in Path(checkpoint).as_posix():
        config = load_config("text_enterprise")
    else:
        config = load_modality_config("text")
    ckpt_dir = resolve_checkpoint_dir("text", config)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        tok_path = ckpt_path.parent / "tokenizer.json"
    else:
        ckpt_path = ckpt_dir / "latest.pt"
        tok_path = ckpt_dir / "tokenizer.json"
    return config, ckpt_path, tok_path


def resolve_text_checkpoint_for_mode(mode: Optional[str] = None) -> Optional[Path]:
    from navine.utils.paths import get_checkpoint_dir
    from navine.text.arch import preferred_text_checkpoint

    if mode and str(mode).lower() in ("code", "coding", "text_code", "text-code"):
        code_ckpt = preferred_text_checkpoint("text_code", prefer_medium=True)
        if code_ckpt is not None and code_ckpt.exists():
            return code_ckpt
    try:
        from navine.utils.brand import default_text_model

        preferred = preferred_text_checkpoint(default_text_model(), prefer_medium=True)
        if preferred is not None and preferred.exists():
            return preferred
    except Exception:
        pass
    return None


def _cache_key(checkpoint: Optional[str], config_name: Optional[str] = None) -> str:
    _, ckpt_path, tok_path = _resolve_paths(checkpoint, config_name)
    device = str(get_device())
    mtime = 0
    try:
        if ckpt_path.exists():
            mtime = int(ckpt_path.stat().st_mtime_ns)
    except Exception:
        mtime = 0
    return f"{ckpt_path.resolve()}|{tok_path.resolve()}|{device}|{mtime}"


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def _load_model_and_tokenizer(
    checkpoint: Optional[str] = None,
    config_name: Optional[str] = None,
) -> _ModelBundle:
    key = _cache_key(checkpoint, config_name)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    # Drop stale keys when checkpoint was rewritten by online adapt.
    stale = [k for k in list(_MODEL_CACHE.keys()) if k.split("|")[0] == str(_resolve_paths(checkpoint, config_name)[1].resolve())]
    for k in stale:
        _MODEL_CACHE.pop(k, None)

    _ensure_compute_env()
    config, ckpt_path, tok_path = _resolve_paths(checkpoint, config_name)
    if not ckpt_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            f"No trained Navine AI - Python text model found at {ckpt_path}. Run: python -m navine.text.train"
        )
    device = get_device()
    from navine.text.arch import architectures_match, preferred_text_checkpoint

    preferred = preferred_text_checkpoint(
        str(config.get("checkpoint_namespace") or ckpt_path.parent.name),
        prefer_medium=True,
    )
    if preferred is not None and preferred.exists():
        ckpt_path = preferred
        candidate_tok = preferred.parent / "tokenizer.json"
        if candidate_tok.exists():
            tok_path = candidate_tok
        elif preferred.parent.name == "medium":
            parent_tok = preferred.parent.parent / "tokenizer.json"
            if parent_tok.exists():
                tok_path = parent_tok

    tokenizer = NavineTokenizer.load(tok_path)
    yaml_model = dict(config.get("model") or {})
    yaml_model["vocab_size"] = len(tokenizer.token_to_id)
    namespace = str(config.get("checkpoint_namespace") or ckpt_path.parent.name)
    load_candidates = []
    if preferred is not None:
        load_candidates.append(preferred)
    if ckpt_path not in load_candidates:
        load_candidates.append(ckpt_path)
    for alt in (
        ckpt_path.parent / "best.pt",
        ckpt_path.parent / "archive_small_58m" / "latest.pt",
        preferred_text_checkpoint(namespace, prefer_medium=False),
    ):
        if alt is not None and alt.exists() and alt not in load_candidates:
            load_candidates.append(alt)

    last_err: Optional[Exception] = None
    model = None
    payload = None
    for candidate in load_candidates:
        try:
            model, payload = NavineTextModel.load_checkpoint(candidate, device)
            ckpt_path = candidate
            candidate_tok = candidate.parent / "tokenizer.json"
            if candidate_tok.exists():
                tok_path = candidate_tok
                tokenizer = NavineTokenizer.load(tok_path)
            break
        except Exception as exc:
            last_err = exc
            continue
    if model is None:
        raise RuntimeError(
            f"Failed to load text checkpoint for {namespace}. "
            f"Last error: {last_err}"
        ) from last_err
    saved = dict((payload or {}).get("config") or {})
    if yaml_model and not architectures_match(saved, yaml_model):
        from navine.text.arch import build_or_upgrade_text_model

        model, meta = build_or_upgrade_text_model(
            config,
            vocab_size=len(tokenizer.token_to_id),
            checkpoint_path=ckpt_path,
            device=device,
            force_fresh=False,
        )
        print(
            f"Navine AI - Python infer: size_tier={config.get('size_tier', 'medium')} "
            f"upgrade transferred={meta.get('transferred')} fresh={meta.get('fresh')}"
        )
    model = model.to(device)
    model.eval()
    bundle = (model, tokenizer, config, device)
    _MODEL_CACHE[key] = bundle
    return bundle


def _sample_next_token(
    logits: torch.Tensor,
    generated: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty > 1.0:
        for token_id in set(generated[0].tolist()):
            val = logits[0, token_id]
            logits[0, token_id] = torch.where(
                val > 0,
                val / repetition_penalty,
                val * repetition_penalty,
            )
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _decode_settings(config: dict, mode: Optional[str]) -> dict:
    infer_cfg = dict(config.get("inference") or {})
    mode = (mode or "chat").lower()
    modes = dict(infer_cfg.get("modes") or {})
    overrides = dict(modes.get(mode) or {})
    return {
        "max_new_tokens": int(overrides.get("max_new_tokens", infer_cfg.get("max_new_tokens", 256))),
        "min_new_tokens": int(overrides.get("min_new_tokens", infer_cfg.get("min_new_tokens", 4))),
        "temperature": float(overrides.get("temperature", infer_cfg.get("temperature", 0.7))),
        "top_k": overrides.get("top_k", infer_cfg.get("top_k", 50)),
        "top_p": overrides.get("top_p", infer_cfg.get("top_p", 0.9)),
        "repetition_penalty": float(
            overrides.get("repetition_penalty", infer_cfg.get("repetition_penalty", 1.12))
        ),
        "frequency_penalty": float(
            overrides.get("frequency_penalty", infer_cfg.get("frequency_penalty", 0.2))
        ),
        "no_repeat_ngram_size": int(
            overrides.get("no_repeat_ngram_size", infer_cfg.get("no_repeat_ngram_size", 4))
        ),
        "num_candidates": int(overrides.get("num_candidates", infer_cfg.get("num_candidates", 1))),
    }


def generate_stream(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    checkpoint: Optional[str] = None,
) -> Iterator[str]:
    model, tokenizer, config, device = _load_model_and_tokenizer(checkpoint)
    settings = _decode_settings(config, "chat")
    max_new_tokens = max_new_tokens or settings["max_new_tokens"]
    temperature = temperature if temperature is not None else settings["temperature"]
    top_k = top_k if top_k is not None else settings["top_k"]
    top_p = top_p if top_p is not None else settings["top_p"]
    repetition_penalty = settings["repetition_penalty"]
    min_new_tokens = settings["min_new_tokens"]
    ids = tokenizer.encode(prompt, add_special=False)
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = generated.size(1)
    for _step in range(max_new_tokens):
        cond = generated if generated.size(1) <= model.max_seq_len else generated[:, -model.max_seq_len :]
        with get_autocast_context(device):
            logits = model(cond)[:, -1, :]
        next_token = _sample_next_token(
            logits,
            generated,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
        generated = torch.cat([generated, next_token], dim=1)
        piece = tokenizer.decode([next_token.item()])
        yield piece
        if (
            tokenizer.eos_id is not None
            and next_token.item() == tokenizer.eos_id
            and generated.size(1) - prompt_len >= min_new_tokens
        ):
            break


def _generate_once(
    model: NavineTextModel,
    tokenizer: NavineTokenizer,
    device: torch.device,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
    min_new_tokens: int,
    frequency_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    ids = tokenizer.encode(prompt, add_special=False)
    vocab_limit = int(model.token_emb.num_embeddings)
    ids = [i for i in ids if 0 <= i < vocab_limit]
    if not ids:
        ids = [tokenizer.pad_id if tokenizer.pad_id is not None else 0]
    prompt_len = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with get_autocast_context(device):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=tokenizer.eos_id,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    new_ids = output_ids[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids)


def generate(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    checkpoint: Optional[str] = None,
    config_name: Optional[str] = None,
    mode: Optional[str] = None,
    num_candidates: Optional[int] = None,
) -> str:
    from navine.utils.inference_policy import is_custom_only

    if not is_custom_only():
        try:
            from navine.text.opensource import generate_opensource, is_opensource_enabled

            if is_opensource_enabled():
                oss = generate_opensource(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if oss and oss.strip():
                    return oss.strip()
        except Exception:
            pass

    if checkpoint is None and mode and str(mode).lower() in ("code", "coding"):
        preferred = resolve_text_checkpoint_for_mode("code")
        if preferred is not None:
            checkpoint = str(preferred)
            if config_name is None:
                config_name = "text_code"
    if checkpoint is None:
        preferred = resolve_text_checkpoint_for_mode(mode)
        if preferred is not None:
            checkpoint = str(preferred)

    model, tokenizer, config, device = _load_model_and_tokenizer(checkpoint, config_name=config_name)
    settings = _decode_settings(config, mode)
    max_new_tokens = max_new_tokens or settings["max_new_tokens"]
    temperature = temperature if temperature is not None else settings["temperature"]
    top_k = top_k if top_k is not None else settings["top_k"]
    top_p = top_p if top_p is not None else settings["top_p"]
    rep = settings["repetition_penalty"]
    min_new = settings["min_new_tokens"]
    freq = settings["frequency_penalty"]
    ngram = settings["no_repeat_ngram_size"]
    n_cand = max(1, int(num_candidates if num_candidates is not None else settings["num_candidates"]))

    temps = [temperature]
    if n_cand > 1:
        temps = [
            max(0.15, temperature * 0.85),
            temperature,
            min(1.1, temperature * 1.15),
        ][:n_cand]
        while len(temps) < n_cand:
            temps.append(temperature)

    texts: List[str] = []
    for temp in temps[:n_cand]:
        texts.append(
            _generate_once(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep,
                min_new_tokens=min_new,
                frequency_penalty=freq,
                no_repeat_ngram_size=ngram,
            )
        )

    if len(texts) == 1:
        return texts[0]

    from navine.text.quality import pick_best_candidate

    return pick_best_candidate(texts)


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(generate(prompt))
