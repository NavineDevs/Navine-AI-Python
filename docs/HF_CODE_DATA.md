# Hugging Face & open code data (for local Navine training)

## Rule

Navine is **custom-only for model weights**: do **not** load foundation models (Qwen, Llama, CodeLlama, etc.) as the brain.

Hugging Face is used for **open datasets** (instruction/code pairs), not inference APIs.

Policy blocks `api-inference.huggingface.co` but allows dataset hosts (`huggingface.co`, LFS CDNs).

## Command

```text
python -m navine.cli learn hf-code --list
python -m navine.cli learn hf-code --max 1500
python -m navine.cli learn hf-code --only code_alpaca,python_code_18k
python -m navine.cli train coding --steps 200
```

Also included in:

```text
python -m navine.cli learn coding-all
```

## Datasets configured

| Alias | Source | Notes |
|-------|--------|-------|
| code_alpaca | GitHub JSON mirror of CodeAlpaca 20k | Reliable fallback |
| hf_codealpaca_20k | HuggingFaceH4/CodeAlpaca_20K | Needs `datasets` |
| python_code_18k | iamtarun/python_code_instructions_18k_alpaca | Python instruct |
| python_codes_25k | flytech/python-codes-25k | Python tasks |
| magicoder_oss | ise-uiuc/Magicoder-OSS-Instruct-75K | OSS-Instruct style |
| evol_instruct_code | nickrosh/Evol-Instruct-Code-80k-v1 | Evol-Instruct code |
| stack_smol_xs | bigcode/the-stack-smol-xs | Real repo snippets |

Catalog output: `data/learn/huggingface/HF_CODE_CATALOG.md`

## Stronger / larger HF options (optional manual)

These are larger; keep custom_only — download subsets with streaming:

- [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (~5M, large)
- [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) (competitive + reasoning)
- [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (pretrain corpora)
- [codeparrot/github-code](https://huggingface.co/datasets/codeparrot/github-code)
- StackOverflow dumps via existing `learn` stackoverflow connectors

## Outputs

Rows append to `data/train/coding/<language>.jsonl` as:

```json
{"language":"python","prompt":"...","code":"...","source":"hf:..."}
```
