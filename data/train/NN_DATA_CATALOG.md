# Training data catalog

This file lists local training corpora for Navine custom neural nets.

## Text / chat

| File | Role |
|------|------|
| `chat/quality_core_dialogue.txt` | Core high-quality chat |
| `chat/voice_assistant_dialogue.txt` | Spoken short-reply style |
| `chat/file_ops_dialogue.txt` | Create/write/list files by voice |
| `chat/gameplay_learn_dialogue.txt` | Learn live games coach dialogue |
| `chat/games_dialogue.txt` | Built-in board games |
| `chat/owner_persona_dialogue.txt` | Owner persona |
| `chat/neural_reason_core.txt` | Reasoning style |
| `chat/nn_systems.md` | NN and system Q&A (markdown) |
| `chat/multimodal_ops.md` | Image/video/chat usage dialogue |
| `chat/instructions.txt` | Base instructions |
| `../text/instruction_chat.txt` | Fallback instruction chat |

## Coding

| File | Role |
|------|------|
| `coding/*.jsonl` | Language-tagged code tasks |
| `coding/assistant_files.jsonl` | File helper / voice router code |

## Games

| File | Role |
|------|------|
| `games/strategy_corpus.txt` | Fortnite, Mario, FPS tips |
| `games/platform_fps_moba.txt` | Genre basics |
| `games/skill_drills.md` | Drills in markdown |
| `games/sessions/*/` | Screen capture observe sessions |

## Multimodal captions

| File | Role |
|------|------|
| `multimodal-text/captions.jsonl` | Image/video captions |
| `multimodal-text/captions_extra.jsonl` | Extra captions |
| `multimodal-text/game_frames.jsonl` | Captions from observe sessions |

## General / math / creative

| Path | Role |
|------|------|
| `general/*.txt` | Broad corpus |
| `math/*.jsonl` | Math Q&A |
| `creative/*` | Creative writing |

## How trainers load data

- Chat: oversamples selected dialogue files then finetunes text NN.
- Coding: all `*.jsonl` under `coding/`.
- Games: strategy + chat gameplay files + session-derived text.
- Multimodal-text: all caption JSONL files under `multimodal-text/`.

See `docs/NN_AND_DATA.md` for architecture maps.
