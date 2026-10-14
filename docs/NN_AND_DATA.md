# Navine AI — Neural Networks & Training Data

Local custom models only. No external foundation weights on the core path.

## Neural networks

| Modality | Code | Checkpoint | Train entry |
|----------|------|------------|-------------|
| Text (chat / SFT domains) | `navine/text/model.py` | `checkpoints/text_enterprise/` (enterprise tier) or `checkpoints/text/` | `python -m navine.cli train chat` (also coding, games, …) |
| Image diffusion | `navine/image/model.py` | `checkpoints/image_enterprise/` (enterprise) | `python -m navine.cli train image` |
| Video | `navine/video/model.py` | `checkpoints/video_enterprise/` (enterprise) | `python -m navine.cli train video` |
| Text code specialist | same text stack | `checkpoints/text_code/` | `python -m navine.cli train text-code` |
| Voice | `navine/voice/train.py` | `checkpoints/voice/` (calibration) | `python -m navine.cli train voice` |

### Text architecture (typical enterprise ckpt)

- Transformer decoder blocks (enterprise target: 10 layers × d_model 640 × 10 heads)
- RoPE positions, SwiGLU, RMSNorm, tied embeddings when configured
- BPE tokenizer under `checkpoints/.../tokenizer.json`
- Weights: `latest.pt`, `best.pt`
- Live param counts: see [LIVE_PARAM_COUNTS.md](LIVE_PARAM_COUNTS.md)

Optional coding specialist: `checkpoints/text_code/` (fork after enterprise SFT; preferred for code mode).

### Domain finetune targets (`train <name>`)

Runs on the shared text NN (same checkpoint), different data:

`general`, `chat`, `coding`, `creative`, `math`, `multimodal-text`, `games`, `nsfw`, `unrestricted`, `voice`

```text
python -m navine.cli train list
python -m navine.cli train all --steps 200
python -m navine.cli train games --steps 150
```

## Data layout (`data/train/`)

| Path | Format | Used by |
|------|--------|---------|
| `chat/*.txt`, `chat/*.md` | Dialogue blocks (`### User:` / `Human:`) | `train chat` |
| `coding/*.jsonl` | `{language, prompt, code}` | `train coding` |
| `creative/*` | Stories / prose | `train creative` |
| `math/*.jsonl` | Q/A | `train math` |
| `general/*.txt` | Corpus lines | `train general` |
| `multimodal-text/*.jsonl` | media + caption | `train multimodal-text` |
| `games/*.txt`, `games/*.md` | Strategy + observe logs | `train games` |
| `games/sessions/` | Screenshots + `session.json` | Live observe → games train |
| `nsfw/`, `unrestricted/` | Instruction blocks | matching train types |

Also:

- `data/text/instruction_chat.txt` — chat fallback corpus
- `docs/` — human docs (this file)
- Catalog: `data/train/NN_DATA_CATALOG.md`

## Voice assistant vs models

Voice is STT/TTS + assist intents (`navine/assistant/`), not a separate large spoken language model.  
Spoken answers use the text NN; file / game actions use the agent.

## Live game learning

1. Play Fortnite / Mario (or any title).
2. `python -m navine.cli assist` → `observe Fortnite for 10 seconds`
3. Optional: `label last action jump`
4. `python -m navine.cli train games`

## Navuryx twin

Same layout with package `navuryx` and brand strings. Checkpoints may be per-tree or shared depending on install.
