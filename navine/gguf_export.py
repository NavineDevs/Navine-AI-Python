from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from navine.utils.paths import get_checkpoint_dir, get_project_root

NAVINE_JINJA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}### System: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}### User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Assistant:{% endif %}"
)

NAVINE_OLLAMA_TEMPLATE = '''TEMPLATE """{{- if .System }}### System: {{ .System }}
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}### User: {{ .Content }}
{{ else if eq .Role "assistant" }}### Assistant: {{ .Content }}
{{ end }}{{- end }}### Assistant:"""'''


def gguf_dir() -> Path:
    path = get_project_root() / "models" / "gguf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _np(t: torch.Tensor, transpose: bool = False) -> np.ndarray:
    arr = t.detach().float().cpu().contiguous().numpy()
    if transpose and arr.ndim == 2:
        arr = np.ascontiguousarray(arr.T)
    return arr.astype(np.float16, copy=False)


def _token_lists(tok_path: Path) -> Dict[str, Any]:
    data = json.loads(tok_path.read_text(encoding="utf-8"))
    id_to_token = {int(k): v for k, v in (data.get("id_to_token") or {}).items()}
    token_to_id = data.get("token_to_id") or {}
    n = int(data.get("vocab_size") or (max(id_to_token) + 1 if id_to_token else 0))
    tokens: List[str] = []
    for idx in range(n):
        tok = id_to_token.get(idx)
        if tok is None:
            for k, v in token_to_id.items():
                if int(v) == idx:
                    tok = k
                    break
        tokens.append(tok if tok is not None else f"<unk_{idx}>")
    merges = [" ".join(pair) for pair in (data.get("merges") or [])]
    return {
        "tokens": tokens,
        "merges": merges,
        "bos": int(token_to_id.get("<bos>", 2)),
        "eos": int(token_to_id.get("<eos>", 3)),
        "unk": int(token_to_id.get("<unk>", 1)),
        "pad": int(token_to_id.get("<pad>", 0)),
        "use_bpe": bool(data.get("use_bpe", True)),
    }


def export_text_gguf(name: str, out_path: Optional[Path] = None) -> Path:
    import gguf

    ckpt_dir = get_checkpoint_dir(name)
    pt = ckpt_dir / "latest.pt"
    tok_path = ckpt_dir / "tokenizer.json"
    if not tok_path.exists():
        tok_path = get_checkpoint_dir("text_enterprise") / "tokenizer.json"
    if not pt.exists() or not tok_path.exists():
        raise FileNotFoundError(f"Missing weights or tokenizer for {name}")
    payload = torch.load(str(pt), map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config") or {})
    state = payload.get("model_state") or payload
    vocab = int(cfg.get("vocab_size") or state["token_emb.weight"].shape[0])
    d_model = int(cfg.get("d_model") or state["token_emb.weight"].shape[1])
    n_heads = int(cfg.get("n_heads") or 10)
    n_layers = int(cfg.get("n_layers") or 10)
    max_seq = int(cfg.get("max_seq_len") or 1024)
    head_dim = d_model // n_heads
    gate = state.get("blocks.0.ff.gate.weight")
    d_ff = int(gate.shape[0]) if gate is not None else int(cfg.get("d_ff") or d_model * 4)
    tok = _token_lists(tok_path)
    dest = Path(out_path) if out_path else gguf_dir() / f"{name}.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(dest), arch="llama")
    writer.add_name(f"navine-{name}")
    writer.add_description("Navine AI - Python custom local LLM. Run with Navine Engine or Ollama-compatible API.")
    writer.add_block_count(n_layers)
    writer.add_context_length(max_seq)
    writer.add_embedding_length(d_model)
    writer.add_feed_forward_length(d_ff)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(vocab)
    writer.add_file_type(int(gguf.GGMLQuantizationType.F16))
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tok["tokens"])
    if tok["merges"]:
        writer.add_token_merges(tok["merges"])
    writer.add_bos_token_id(tok["bos"])
    writer.add_eos_token_id(tok["eos"])
    writer.add_unk_token_id(tok["unk"])
    writer.add_pad_token_id(tok["pad"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_chat_template(NAVINE_JINJA_CHAT_TEMPLATE)
    writer.add_string("navine.engine", "Navine Engine")
    writer.add_string("navine.checkpoint", name)
    writer.add_string("navine.layout", "llama-f16")
    writer.add_bool("navine.tie_embeddings", bool(cfg.get("tie_embeddings", True)))
    writer.add_bool("navine.use_rope", bool(cfg.get("use_rope", True)))
    writer.add_bool("navine.use_swiglu", bool(cfg.get("use_swiglu", True)))
    writer.add_tensor("token_embd.weight", _np(state["token_emb.weight"], transpose=False))
    writer.add_tensor("output_norm.weight", _np(state["ln_f.weight"], transpose=False))
    if not bool(cfg.get("tie_embeddings", True)) and "head.weight" in state:
        writer.add_tensor("output.weight", _np(state["head.weight"], transpose=True))
    for layer in range(n_layers):
        prefix = f"blocks.{layer}"
        qkv_w = state[f"{prefix}.attn.qkv.weight"]
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", _np(state[f"{prefix}.ln1.weight"]))
        writer.add_tensor(f"blk.{layer}.ffn_norm.weight", _np(state[f"{prefix}.ln2.weight"]))
        writer.add_tensor(f"blk.{layer}.attn_q.weight", _np(q_w, transpose=True))
        writer.add_tensor(f"blk.{layer}.attn_k.weight", _np(k_w, transpose=True))
        writer.add_tensor(f"blk.{layer}.attn_v.weight", _np(v_w, transpose=True))
        writer.add_tensor(f"blk.{layer}.attn_output.weight", _np(state[f"{prefix}.attn.out.weight"], transpose=True))
        writer.add_tensor(f"blk.{layer}.ffn_gate.weight", _np(state[f"{prefix}.ff.gate.weight"], transpose=True))
        writer.add_tensor(f"blk.{layer}.ffn_up.weight", _np(state[f"{prefix}.ff.up.weight"], transpose=True))
        writer.add_tensor(f"blk.{layer}.ffn_down.weight", _np(state[f"{prefix}.ff.down.weight"], transpose=True))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    _write_modelfile(name, dest)
    return dest


def export_modality_gguf(name: str, modality: str, out_path: Optional[Path] = None) -> Path:
    import gguf

    pt = get_checkpoint_dir(name) / "latest.pt"
    if not pt.exists():
        raise FileNotFoundError(f"Missing weights for {name}")
    payload = torch.load(str(pt), map_location="cpu", weights_only=False)
    state = payload.get("model_state") or payload
    cfg = payload.get("config") or {}
    dest = Path(out_path) if out_path else gguf_dir() / f"{name}.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(dest), arch=f"navine-{modality}")
    writer.add_name(f"navine-{name}")
    writer.add_description(f"Navine AI - Python {modality} weights in GGUF. Run with Navine Engine.")
    writer.add_string("navine.engine", "Navine Engine")
    writer.add_string("navine.checkpoint", name)
    writer.add_string("navine.modality", modality)
    writer.add_string("navine.layout", "pytorch-f16")
    writer.add_string("navine.config_json", json.dumps(cfg, default=str))
    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        safe = key.replace(".", "-")
        writer.add_tensor(f"navine.{safe}", _np(tensor, transpose=False))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    _write_modelfile(name, dest)
    return dest


def modelfile_body(name: str, gguf_filename: str, system_label: Optional[str] = None) -> str:
    label = system_label or f"Navine AI - Python ({name})"
    return "\n".join(
        [
            f"FROM ./{gguf_filename}",
            NAVINE_OLLAMA_TEMPLATE,
            "PARAMETER temperature 0.7",
            "PARAMETER top_p 0.9",
            "PARAMETER top_k 40",
            "PARAMETER num_ctx 1024",
            "PARAMETER stop \"### User:\"",
            "PARAMETER stop \"### Assistant:\"",
            f"SYSTEM You are {label}, a local custom model.",
            "",
        ]
    )


def _write_modelfile(name: str, gguf_path: Path) -> Path:
    path = gguf_dir() / f"{name}.Modelfile"
    path.write_text(modelfile_body(name, gguf_path.name), encoding="utf-8")
    return path


def write_modelfile_at(path: Path, name: str, gguf_filename: str, system_label: Optional[str] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(modelfile_body(name, gguf_filename, system_label=system_label), encoding="utf-8")
    return path


def export_all_gguf() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    mapping = [
        ("text_enterprise", "text"),
        ("text_code", "text"),
        ("hitboyx23_ai", "text"),
        ("hitboyx23_ai_python", "text"),
        ("image_enterprise", "image"),
        ("video_enterprise", "video"),
    ]
    for name, modality in mapping:
        try:
            if modality == "text":
                path = export_text_gguf(name)
            else:
                path = export_modality_gguf(name, modality)
            rows.append({"name": name, "ok": True, "gguf": str(path), "bytes": path.stat().st_size})
        except Exception as exc:
            rows.append({"name": name, "ok": False, "error": str(exc)})
    index = {"format": "gguf", "models": rows}
    (gguf_dir() / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def _stale_gguf_names() -> List[str]:
    import time

    stale: List[str] = []
    now = time.time()
    for name, _modality in (
        ("text_enterprise", "text"),
        ("text_code", "text"),
        ("hitboyx23_ai", "text"),
        ("hitboyx23_ai_python", "text"),
        ("image_enterprise", "image"),
        ("video_enterprise", "video"),
    ):
        pt = get_checkpoint_dir(name) / "latest.pt"
        gf = gguf_dir() / f"{name}.gguf"
        if not pt.exists():
            continue
        if now - pt.stat().st_mtime < 20:
            continue
        if not gf.exists() or pt.stat().st_mtime > gf.stat().st_mtime:
            stale.append(name)
    return stale


def watch_gguf(poll_seconds: int = 180) -> None:
    import time

    while True:
        names = _stale_gguf_names()
        if names:
            export_all_gguf()
        time.sleep(max(30, int(poll_seconds)))


def verify_gguf(path: Path) -> Dict[str, Any]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    arch_field = reader.fields.get("general.architecture")
    name_field = reader.fields.get("general.name")
    arch = arch_field.contents() if arch_field is not None else None
    name = name_field.contents() if name_field is not None else None
    return {
        "ok": True,
        "path": str(path),
        "name": name,
        "architecture": arch,
        "tensors": int(len(reader.tensors)),
        "bytes": int(path.stat().st_size),
    }


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export Navine AI - Python checkpoints to GGUF")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    sub.add_parser("list")
    ver = sub.add_parser("verify")
    ver.add_argument("name", nargs="?", default=None)
    watch_p = sub.add_parser("watch")
    watch_p.add_argument("--poll", type=int, default=180)
    args = parser.parse_args(argv)
    if args.cmd == "export":
        print(json.dumps(export_all_gguf(), indent=2))
        return 0
    if args.cmd == "watch":
        watch_gguf(poll_seconds=int(args.poll))
        return 0
    if args.cmd == "list":
        rows = []
        for path in sorted(gguf_dir().glob("*.gguf")):
            rows.append({"name": path.stem, "path": str(path), "bytes": path.stat().st_size})
        print(json.dumps(rows, indent=2))
        return 0
    names = [args.name] if args.name else ["text_enterprise", "text_code", "hitboyx23_ai", "hitboyx23_ai_python", "image_enterprise", "video_enterprise"]
    ok = True
    report = []
    for name in names:
        path = gguf_dir() / f"{name}.gguf"
        if not path.exists():
            report.append({"name": name, "ok": False, "error": f"missing {path}"})
            ok = False
            continue
        try:
            report.append(verify_gguf(path))
        except Exception as exc:
            report.append({"name": name, "ok": False, "error": str(exc)})
            ok = False
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
