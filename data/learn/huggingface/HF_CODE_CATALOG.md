# Hugging Face / open code datasets used for Navine

Data only (instruction/code pairs). Custom text NN weights stay local.

| Alias | Dataset | Fetched | Added | Error |
|-------|---------|---------|-------|-------|
| code_alpaca | sahil280114/codealpaca | 579 | 0 |  |
| hf_codealpaca_20k | HuggingFaceH4/CodeAlpaca_20K | 579 | 0 |  |
| python_code_18k | iamtarun/python_code_instructions_18k_alpaca | 600 | 0 |  |
| python_codes_25k | flytech/python-codes-25k | 599 | 0 |  |
| magicoder_oss | ise-uiuc/Magicoder-OSS-Instruct-75K | 600 | 0 |  |
| evol_instruct_code | nickrosh/Evol-Instruct-Code-80k-v1 | 600 | 0 |  |
| stack_smol_xs | bigcode/the-stack-smol-xs | 0 | 0 | Dataset scripts are no longer supported, but found the-stack-smol-xs.py |

## How to refresh

```text
python -m navine.cli learn hf-code --max 1500
python -m navine.cli train coding --steps 200
```

Inference HF model APIs stay blocked by policy; hub dataset hosts are allowed for learning.