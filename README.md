# Navine AI

Navine AI is a fully local, trainable multimodal AI stack. It includes three small-scale models you can train and run on consumer hardware:

- **Text model** — GPT-style transformer for chat, general text, and code generation
- **Image model** — Text-conditioned diffusion model for text-to-image
- **Video model** — LSTM-based frame sequence generator for text-to-video

Everything runs locally with PyTorch for inference. **Navine AI runs 100% locally. No external AI APIs are used.** Internet access is used only for **learning** (fetching URLs, datasets, RAG) — not for sending your prompts to external AI services.

Local-only enforcement is configured in `configs/navine.yaml` (`local_only: true`) and implemented in `navine/policy.py`, which blocks external AI provider URLs and imports at runtime.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- 8 GB RAM minimum (16 GB recommended)
- GPU optional but recommended for faster training

## Installation

```bash
cd "Navine AI"
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Download the trained model weights. Checkpoints over 95 MB are stored as split files on the GitHub release `weights-v1`, listed in `weights_manifest.json`. This script downloads and rejoins them into `checkpoints/`, `models/`, and `data/`:

```bash
python scripts/download_weights.py
python scripts/download_weights.py --only text_enterprise
```

Prepare sample datasets:

```bash
python scripts/prepare_data.py
```

Verify installation:

```bash
python -m navine.cli info
```

On Windows you can also use the launcher scripts from the project root:

```powershell
.\navine.ps1 info
navine.bat info
```

## GUI Launcher (Windows)

Double-click `navine.bat` or run `.\navine.ps1` with no arguments to open a native Windows launcher.

| Section | Options |
|---------|---------|
| **Open** | Web UI, Desktop App |
| **Chat & Generate** | Chat REPL, Text, Image, Video, Code |
| **Learn & Train** | Learn menu (NSFW autolearn, images, videos, marathon, everything), Train menu (all types including NSFW/unrestricted) |
| **Quick actions** | NSFW Autolearn Now, NSFW Autolearn (Hours), Marathon, Learn Everything |
| **Tools** | Info, Doctor, API Docs |

The **Learn** submenu includes: NSFW autolearn, NSFW images/videos, unrestricted text, local folder ingest, learn everything, coding-all, autolearn, deep train, and marathon.

When you pass arguments, the launcher scripts keep the original CLI behavior:

```powershell
navine.bat chat
navine.bat info
.\navine.ps1 train text
```

The GUI uses PowerShell WinForms (no extra packages). Launcher logic lives in `scripts/navine-launcher.ps1`.

## Three Ways to Use Navine AI

Navine AI can be accessed three ways, all running locally on your machine:

| Method | Best for | Start command |
|--------|----------|---------------|
| **CLI** | Scripts, automation, terminal workflows | `python -m navine.cli` or `navine.bat <command>` |
| **Web UI** | Browser-based access on localhost | `python -m navine.server` |
| **Desktop App** | Native window with auto-started backend | `cd app && npm run dev` |

## Quick Start

### 1. Train all models (demo run)

Training uses tiny demo datasets and completes in minutes on CPU:

```bash
python -m navine.cli train text
python -m navine.cli train image
python -m navine.cli train video
```

Or train individually:

```bash
python -m navine.text.train
python -m navine.image.train
python -m navine.video.train
```

### 2. Run inference

```bash
python -m navine.cli text "Hello from Navine AI"
python -m navine.cli image "a colorful gradient pattern"
python -m navine.cli video "waves on a beach"
python -m navine.cli code "write a fibonacci function in python"
```

## CLI Reference

Navine AI includes a command-line interface for all models. Use `python -m navine.cli`, or on Windows run `navine.bat` / `.\navine.ps1` from the project root (these automatically use the project venv when present).

### Interactive chat REPL

```bash
python -m navine.cli chat
```

Type your messages at the `You:` prompt. Enter `exit`, `quit`, or press Ctrl+C to leave. Optional flags: `--max-tokens`, `--temperature`.

### One-shot commands

| Command | Description |
|---------|-------------|
| `python -m navine.cli text "prompt"` | Generate text |
| `python -m navine.cli chat` | Interactive text chat REPL |
| `python -m navine.cli image "prompt" [--steps N]` | Generate image with optional diffusion steps |
| `python -m navine.cli video "prompt" [--frames N] [--fps N]` | Generate GIF video with configurable frames/fps |
| `python -m navine.cli code "task"` | Generate code with syntax-aware prompting |
| `python -m navine.cli learn url <url>` | Fetch URL for text learning |
| `python -m navine.cli learn image url <url>` | Download image(s) for image model training |
| `python -m navine.cli learn video gif <url>` | Download GIF frames for video model training |
| `python -m navine.cli learn train` | Fine-tune chat model |
| `python -m navine.cli train text` | Train text model (legacy) |
| `python -m navine.cli train coding [--language python] [--steps N]` | Fine-tune for code generation |
| `python -m navine.cli train chat` | Fine-tune for instruction chat |
| `python -m navine.cli train creative` | Fine-tune for stories and creative writing |
| `python -m navine.cli train math` | Fine-tune for math Q&A |
| `python -m navine.cli train general` | Fine-tune on general text corpus |
| `python -m navine.cli train list` | List all specialized training types |
| `python -m navine.cli train all` | Train all specialized types sequentially |
| `python -m navine.cli learn coding [--language python] [--max N]` | Download public code samples |
| `python -m navine.cli train image` | Train image model |
| `python -m navine.cli train video` | Train video model |
| `python -m navine.cli info` | Show system and model status |
| `python -m navine.cli doctor` | Run health checks (use `--fix` to auto-repair) |

Run `python -m navine.cli --help` for examples and all options.

### Code generation options

```bash
python -m navine.cli code "sort a list" --language python
python -m navine.cli code "fetch API data" --language javascript
python -m navine.cli code "parse JSON" --language rust
```

Supported languages: `python`, `javascript`, `typescript`, `rust`, `go`, `java`, `cpp`

## Improved Text Chat

Navine AI uses an instruction-tuned chat pipeline (`navine/text/chat.py`) with:

- **System prompt** — Navine AI persona (local assistant, creator is you)
- **Instruction format** — `### System / User / Assistant` turns
- **Response cleaning** — strips training junk, code fragments, and prompt echo
- **Code in chat** — ask "write a fibonacci function" in the Text tab or chat REPL; code renders in markdown blocks
- **RAG** — factual answers can use content fetched via the learn commands

Fine-tune on chat examples:

```bash
python -m navine.cli learn train
```

Or full training with instruction data:

```bash
python -m navine.cli train text
```

Chat data lives in `data/text/instruction_chat.txt`. Add your own examples in the same format.

### Honest quality notes

Navine AI's default text model is ~3M parameters — orders of magnitude smaller than GPT-4 or Claude. Even with chat fine-tuning, RAG, and prompt engineering, it will not match frontier cloud models. What you get:

| Strength | Limitation |
|----------|------------|
| Fully local, private, yours to extend | Limited reasoning and world knowledge |
| Fast on CPU/GPU for small prompts | Short, sometimes repetitive replies |
| Code in chat (Text tab or CLI) | Code quality depends on training data size |
| Learn from URLs and RAG | RAG is keyword/TF-IDF based, not semantic search at scale |

For best results: run `learn train` after adding instruction examples, fetch relevant URLs with `learn url`, and expand `data/text/` with your own corpus.

## Internet Learning

Fetch web content, build a local knowledge base, and fine-tune on collected data:

| Command | Description |
|---------|-------------|
| `python -m navine.cli learn url <url>` | Fetch a page, save to `data/learn/pages/`, index for RAG |
| `python -m navine.cli learn crawl <url>` | Crawl up to 20 same-domain pages to JSONL |
| `python -m navine.cli learn datasets` | Download sample public text/code datasets |
| `python -m navine.cli learn ingest` | Merge learned content into `data/text/learned_corpus.txt` |
| `python -m navine.cli learn index` | Rebuild SQLite TF-IDF RAG index |
| `python -m navine.cli learn train` | Fine-tune text model on chat + learned data |

When you ask factual questions in chat, Navine AI retrieves matching learned passages and injects them into context automatically.

### Image internet learning

Download images from the web, ingest them at 64x64, and fine-tune the diffusion model:

| Command | Description |
|---------|-------------|
| `python -m navine.cli learn image url <url> [--caption "text"]` | Download a single image or gallery page |
| `python -m navine.cli learn image crawl <page_url>` | Scrape image URLs from a page |
| `python -m navine.cli learn image search <query> [--max N]` | Fetch public domain images (Wikimedia Commons, picsum fallback) |
| `python -m navine.cli learn image ingest` | Resize and copy to `data/image/samples/` |
| `python -m navine.cli learn image train [--steps N]` | Fine-tune diffusion on learned + existing data |

Learned images are stored in `data/image/learned/` with JSON metadata (url, caption, timestamp). Captions are auto-generated from alt text, page title, or filename when not provided.

Sample workflow:

```bash
python -m navine.cli learn image url https://picsum.photos/id/10/200/200 --caption "sunset over ocean"
python -m navine.cli learn image ingest
python -m navine.cli learn image train
python -m navine.cli image "sunset over ocean" --steps 75
```

### Video internet learning

Download GIFs or videos, extract frames, and fine-tune the LSTM video model:

| Command | Description |
|---------|-------------|
| `python -m navine.cli learn video url <url>` | Download video or GIF and extract frames |
| `python -m navine.cli learn video gif <url>` | Download GIF and extract frames (Pillow fallback) |
| `python -m navine.cli learn video ingest` | Prepare frame sequences for training |
| `python -m navine.cli learn video train [--steps N]` | Fine-tune video model on learned data |

Frames are saved under `data/video/learned/<id>/`. OpenCV or imageio is used for video when available; GIFs work with Pillow alone.

### Image and video generation options

```bash
python -m navine.cli image "sunset over ocean" --steps 100 --guidance 3.5 --seed 42
python -m navine.cli video "waves on a beach" --frames 16 --fps 12
```

The web UI includes a steps slider for images and frame count / FPS controls for video.

## Autonomous Learning

Navine AI can autonomously fetch public internet content for local training. No external AI APIs are used. Only public data from approved sources is downloaded.

### Learn Everything workflow

For maximum coverage across coding, general knowledge, science, creative writing, and documentation:

```bash
python -m navine.cli learn everything
python -m navine.cli learn coding-all
python -m navine.cli autolearn run --aggressive
python -m navine.cli autolearn deep-train
```

The Windows launcher includes **Learn Everything** (aggressive autolearn cycle) and **Deep Train All Types** buttons.

### Sources

| Source | Data | CLI |
|--------|------|-----|
| GitHub | Repos, gists, awesome lists, 14 languages | `learn github [--language rust] [--max-files 20]` |
| GitLab / Codeberg | Public repo code | Included in `learn coding-all` |
| Reddit | 17 coding/dev subreddits | `learn reddit [--subreddit Python] [--max 25]` |
| Hacker News | Top stories via Firebase API | `learn hn [--max 30]` |
| Stack Overflow | Q&A by tag (python, rust, algorithms, etc.) | Included in `learn all-sources` |
| Wikipedia | Random articles and category pages | `learn wikipedia [--max 10]` |
| arXiv | CS/ML paper abstracts | `learn arxiv [--max 20]` |
| Project Gutenberg | Public domain book excerpts | Included in autolearn cycle |
| Documentation | docs.python.org, developer.mozilla.org | `learn docs [--topic python]` |
| RSS/Feeds | Rust blog, HN, Dev.to, Medium | Included in autolearn cycle |
| Wikimedia Commons | Image captions (when `learn_images: true`) | Auto during autolearn |

All requests use `NavineAI/1.0 (local training bot)` as the User-Agent. Reddit requires a descriptive User-Agent per their API rules. If the JSON endpoint returns 403, Navine AI falls back to the public Atom RSS feed for the subreddit.

### Classification

Autolearn classifies fetched content into: coding, chat, creative, math, science, history, news, technical, qa, and image-caption. Code blocks are extracted per language (Python, JS, Rust, Go, Java, C++, C#, Ruby, PHP, Swift, Kotlin, Lua, shell).

### Autolearn engine

Configuration lives in `configs/autolearn.yaml`:

```yaml
enabled: true
sources: [github, gitlab, reddit, hackernews, stackoverflow, wikipedia, arxiv, feeds, docs, gutenberg]
max_items_per_run: 500
min_samples_before_train: 25
github_languages: [python, javascript, typescript, rust, go, java, cpp, csharp, ruby, php, swift, kotlin, lua, shell]
auto_train: true
train_types: [coding, chat, general, creative, math, multimodal-text]
learn_images: true
cycle_all_languages: true
```

| Command | Description |
|---------|-------------|
| `python -m navine.cli autolearn run` | Run one autonomous cycle now |
| `python -m navine.cli autolearn run --aggressive` | High volume cycle from all sources |
| `python -m navine.cli autolearn deep-train` | Train all configured types sequentially |
| `python -m navine.cli autolearn start` | Foreground loop (Ctrl+C to stop) |
| `python -m navine.cli autolearn status` | Last run, samples collected, next train threshold |
| `python -m navine.cli autolearn config` | Show current configuration |
| `python -m navine.cli learn everything` | Max fetch from ALL sources once |
| `python -m navine.cli learn coding-all` | All coding sources and languages |
| `python -m navine.cli learn wikipedia [--max N]` | Fetch Wikipedia articles |
| `python -m navine.cli learn arxiv [--max N]` | Fetch arXiv abstracts |
| `python -m navine.cli learn docs [--topic python]` | Crawl public documentation |
| `python -m navine.cli learn all-sources` | Run all source connectors once |

Raw fetched content is saved under `data/autolearn/raw/`. Classified training data is written to `data/train/` (coding JSONL per language, chat instructions, general/science/history corpus, creative stories, math Q&A, multimodal captions). Activity is logged to `data/autolearn/log.jsonl`. Interrupted cycles resume from `data/autolearn/state.json` with round-robin source rotation.

The web UI footer shows autolearn status from `/api/autolearn/status`. The Windows launcher includes **Auto Learn from Internet**, **Learn Everything**, **Learn Everything (Marathon - Hours)**, and **Deep Train All Types** buttons.

### Marathon Learning

For unattended multi-hour learning across all sources (coding, text, images, video):

```bash
python -m navine.cli marathon start --hours 8
python -m navine.cli learn everything --marathon --hours 8
python -m navine.cli marathon status
python -m navine.cli marathon stop
python -m navine.cli marathon log
```

Go away for 8 hours and return to a trained model with a large corpus. Marathon mode rotates through GitHub languages, Reddit subreddits, Wikipedia categories, arXiv, Stack Overflow, Gutenberg, feeds, Wikimedia images, and GIF frames. Training runs automatically every 50 new samples or 30 minutes; deep train runs every 2 hours.

Configuration lives in `configs/marathon.yaml`. State and resume checkpoint: `data/autolearn/marathon_state.json`. Activity log: `data/autolearn/marathon.log`. Stop gracefully with `marathon stop` (writes `data/autolearn/marathon_stop.flag`) or Ctrl+C.

| Command | Description |
|---------|-------------|
| `python -m navine.cli marathon start [--hours 8]` | Long-running daemon (default 8 hours) |
| `python -m navine.cli marathon start --forever` | Run until stop flag or Ctrl+C |
| `python -m navine.cli marathon start --resume` | Resume from checkpoint |
| `python -m navine.cli marathon status` | Live stats from state file |
| `python -m navine.cli marathon stop` | Graceful shutdown after current batch |
| `python -m navine.cli marathon log` | Tail last 50 lines of marathon log |
| `python -m navine.cli learn everything --marathon` | Alias for marathon start |

The web UI footer shows marathon status from `/api/marathon/status` when running.

### Ethics and rate limits

- Public content only; no login scraping or paywalled data
- Respect source rate limits (Navine AI sleeps 1.5s between requests)
- Blocked hosts include OpenAI, Anthropic, and other external AI APIs
- Allowed learning hosts: GitHub, GitLab, Codeberg, Reddit, Stack Exchange, Hacker News, Wikipedia, arXiv, Gutenberg, Dev.to, Mozilla docs, Python docs, Wikimedia Commons, and configured RSS feeds

## Web UI

The local web interface provides tabs for text chat, image generation, and video generation. Use the Text Chat tab for code generation via natural language.

### Start the server

```bash
python -m navine.server
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

Optional flags:

```bash
python -m navine.server --host 127.0.0.1 --port 8765
```

### API endpoints

The same server exposes a REST API used by the web UI and desktop app:

| Endpoint | Method | Body |
|----------|--------|------|
| `/api/health` | GET | — |
| `/api/info` | GET | — |
| `/api/text` | POST | `{ "prompt", "max_tokens?", "temperature?" }` |
| `/api/image` | POST | `{ "prompt", "num_steps?", "guidance_scale?", "seed?", "enhance?" }` |
| `/api/video` | POST | `{ "prompt", "num_frames?", "fps?", "seed?" }` |
| `/api/code` | POST | `{ "task", "language?" }` |

Static frontend files live in `web/` and are served automatically.

## External API

Navine AI exposes a REST API with optional API key authentication for external clients. The local web UI continues to work from your browser without a key when `allow_localhost_without_auth` is enabled (default).

### Configuration

Edit `configs/api.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Master switch for API server settings |
| `bind_host` | `127.0.0.1` | Use `0.0.0.0` to accept connections from other machines |
| `port` | `8765` | Server port |
| `require_auth` | `true` | Require API keys on protected endpoints |
| `allow_localhost_without_auth` | `true` | Skip auth for requests from localhost (keeps web UI working) |
| `keys_file` | `.navine/api_keys.json` | Path to hashed key store (gitignored) |
| `cors_origins` | `["*"]` | Allowed CORS origins for browser clients |
| `rate_limit.enabled` | `false` | Enable in-memory rate limiting |
| `rate_limit.requests_per_minute` | `60` | Max requests per key or IP per minute |

Example for external access:

```yaml
bind_host: 0.0.0.0
require_auth: true
allow_localhost_without_auth: true
cors_origins:
  - "https://myapp.example.com"
```

Start the server (reads host/port from config):

```bash
python -m navine.server
```

Override on the command line:

```bash
python -m navine.server --host 0.0.0.0 --port 8765
```

### API key management

Keys are stored as SHA-256 hashes in `.navine/api_keys.json`. The full key is shown only once at creation.

On first server start with no keys, a default key is generated and printed to the console.

```bash
python -m navine.cli api-key create --name "my integration"
python -m navine.cli api-key list
python -m navine.cli api-key revoke abc123def456
```

Pass keys using either header:

- `Authorization: Bearer YOUR_API_KEY`
- `X-Navine-API-Key: YOUR_API_KEY`

### Protected endpoints

All `/api/*` routes require a valid key except `/api/health`. The OpenAI-compatible route `/v1/chat/completions` uses the same auth.

| Endpoint | Method | Auth | Body |
|----------|--------|------|------|
| `/api/health` | GET | No | — |
| `/api/info` | GET | Yes | — |
| `/api/text` | POST | Yes | `{ "prompt", "max_tokens?", "temperature?" }` |
| `/api/chat` | POST | Yes | `{ "message" }` or `{ "messages": [...] }` |
| `/api/image` | POST | Yes | `{ "prompt" }` |
| `/api/video` | POST | Yes | `{ "prompt" }` |
| `/api/code` | POST | Yes | `{ "task", "language?" }` |
| `/v1/chat/completions` | POST | Yes | OpenAI chat format |

### curl examples

Health check (no key):

```bash
curl http://127.0.0.1:8765/api/health
```

Generate text:

```bash
curl -X POST http://127.0.0.1:8765/api/text \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Hello from Navine AI\"}"
```

Chat:

```bash
curl -X POST http://127.0.0.1:8765/api/chat \
  -H "X-Navine-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Who are you?\"}"
```

OpenAI-compatible (works with tools expecting OpenAI API):

```bash
curl -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"navine-text\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}"
```

System info:

```bash
curl http://127.0.0.1:8765/api/info \
  -H "Authorization: Bearer YOUR_API_KEY"
```

From another machine, replace `127.0.0.1` with your host IP and ensure `bind_host: 0.0.0.0` in `configs/api.yaml`.

## Desktop App (Tauri)

The desktop app is a lightweight Tauri shell that starts the Python backend and opens the web UI in a native window.

### Prerequisites

- Python 3.9+ with Navine AI installed (`pip install -e .`)
- [Node.js](https://nodejs.org/) 18+
- [Rust](https://www.rust-lang.org/tools/install) (for Tauri builds)

### Run in development

```bash
cd app
npm install
npm run dev
```

The app spawns `python -m navine.server` automatically and loads `http://127.0.0.1:8765`.

If Python is not on PATH, set the project root explicitly:

```powershell
$env:NAVINE_ROOT = "C:\path\to\Navine AI"
npm run dev
```

### Build installer

```bash
cd app
npm run build
```

Built binaries appear under `app/src-tauri/target/release/bundle/`.

## Project Structure

```
Navine AI/
├── README.md
├── requirements.txt
├── pyproject.toml
├── navine.bat              Windows launcher (GUI or CLI passthrough)
├── navine.ps1              PowerShell launcher (GUI or CLI passthrough)
├── navine/
│   ├── cli.py              Unified CLI (includes chat REPL)
│   ├── server.py           FastAPI local server
│   ├── api/                REST route handlers
│   ├── text/               GPT-style text model
│   │   ├── model.py
│   │   ├── tokenizer.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── image/              Diffusion image model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── video/              Frame sequence video model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── code/               Code generation utilities
│   │   ├── prompts.py
│   │   └── format.py
│   └── utils/
├── web/                    Local web UI (HTML/CSS/JS)
│   ├── index.html
│   └── assets/
├── app/                    Tauri desktop app
│   ├── package.json
│   └── src-tauri/
├── configs/                Model hyperparameters and API settings
│   └── api.yaml            External API host, auth, CORS
├── .navine/                API key store (created at runtime, gitignored)
├── data/                   Training data
│   ├── text/
│   ├── image/samples/
│   └── video/samples/
├── scripts/
│   ├── navine-launcher.ps1 Windows GUI launcher (WinForms)
│   └── prepare_data.py
├── checkpoints/            Saved model weights (created on train)
└── outputs/                Generated images/videos (created on infer)
```

## Training Types

Navine AI supports modular specialized training types under `navine/train/`. Each type fine-tunes the existing text checkpoint (does not retrain from scratch unless no checkpoint exists).

| Type | Purpose | Data format | Config |
|------|---------|-------------|--------|
| `coding` | Python, JS, Rust, and other code generation | JSONL with `language`, `prompt`, `code` | `configs/train/coding.yaml` |
| `chat` | Conversational and instruction following | `instruction_chat.txt` blocks | `configs/train/chat.yaml` |
| `creative` | Stories, poems, creative writing | plain text or JSONL | `configs/train/creative.yaml` |
| `math` | Math Q&A and step-by-step reasoning | JSONL `question`/`answer` | `configs/train/math.yaml` |
| `general` | Generic text (existing corpus style) | corpus txt | `configs/train/general.yaml` |
| `multimodal-text` | Captions for image/video alignment | JSONL caption pairs | `configs/train/multimodal-text.yaml` |
| `nsfw` | Unrestricted adult/mature content (opt-in) | instruction blocks in `data/train/nsfw/` | `configs/train/nsfw.yaml` |
| `unrestricted` | No-refusal direct responses (opt-in) | instruction blocks in `data/train/unrestricted/` | `configs/train/unrestricted.yaml` |

Sample datasets live in `data/train/`:

```
data/train/coding/python.jsonl
data/train/coding/javascript.jsonl
data/train/chat/instructions.txt
data/train/creative/stories.txt
data/train/math/qa.jsonl
data/train/general/corpus.txt
```

### Commands

```bash
python -m navine.cli train list
python -m navine.cli train coding --language python --steps 200
python -m navine.cli train chat
python -m navine.cli train creative
python -m navine.cli train math
python -m navine.cli train general
python -m navine.cli train all
```

### Unrestricted NSFW training (always on)

Navine AI is **always fully unrestricted** with zero safety guardrails. NSFW autolearn is enabled by default.

1. Add subreddits to `configs/nsfw.yaml` or drop files in `data/nsfw/local/` and `data/unrestricted/local/`.

```yaml
enabled: true
unrestricted_mode: true
reddit_subreddits: [your_sub_here]
reddit_image_subreddits: [your_image_subs]
reddit_video_subreddits: [your_video_subs]
reddit_unrestricted_subreddits: [your_text_subs]
```

2. Fetch and train (full auto — text + images + videos):

```bash
python -m navine.cli learn nsfw autolearn
python -m navine.cli learn nsfw autolearn --hours 8
python -m navine.cli learn nsfw images
python -m navine.cli learn nsfw videos
python -m navine.cli learn nsfw unrestricted
python -m navine.cli train nsfw
python -m navine.cli train unrestricted
python -m navine.cli learn nsfw status
```

Configure separate subreddit lists in `configs/nsfw.yaml`:

```yaml
reddit_image_subreddits: [your_image_sub]
reddit_video_subreddits: [your_video_sub]
reddit_unrestricted_subreddits: [your_text_sub]
```

When enabled, marathon mode auto-fetches NSFW text, images, and videos every 2 cycles and trains all models.

**Private local use only.** Only train on content you have the right to use.

Download additional coding samples from public GitHub sources (local download only, no external AI APIs):

```bash
python -m navine.cli learn coding --language python --max 5
python -m navine.cli learn coding --language javascript
```

### Sample workflow: coding training

1. Train a base text model (if you do not already have a checkpoint):

```bash
python -m navine.cli train text
```

2. Optionally download more code samples:

```bash
python -m navine.cli learn coding --language python --max 5
```

3. Fine-tune on coding data:

```bash
python -m navine.cli train coding --language python --steps 200
```

4. Test code generation:

```bash
python -m navine.cli code "write a binary search function" --language python
```

The web API exposes training type metadata at `GET /api/train/types` and includes `training_types` in `GET /api/info`.

## Training on Your Own Data

### Text and code

1. Add plain text lines to `data/text/your_corpus.txt`
2. Add code examples as JSONL to `data/text/your_code.jsonl`:

```json
{"language": "python", "prompt": "Write a merge sort", "code": "def merge_sort(arr):\n    ..."}
```

3. Update `configs/text.yaml` to point to your files
4. Run `python -m navine.cli train text`

### Images

1. Place PNG/JPG files in `data/image/samples/`
2. Run `python -m navine.cli train image`

### Video

1. Place sequential frame images in `data/video/samples/`
2. Run `python -m navine.cli train video`

## Configuration

Edit YAML files in `configs/`:

- `configs/text.yaml` — model size, training steps, inference temperature
- `configs/image.yaml` — image size, diffusion timesteps, guidance scale
- `configs/video.yaml` — frame count, LSTM size, output FPS

## Architecture Details

### Text Model (~3M parameters default)

- 4-layer transformer decoder (GPT-style)
- 256-dim embeddings, 4 attention heads
- Custom word-level tokenizer built from your corpus
- Causal self-attention for autoregressive generation

### Image Model (~5M parameters default)

- DDPM diffusion with simplified U-Net denoiser
- 64x64 RGB output
- Character-level text conditioning encoder
- Classifier-free guidance at inference

### Video Model (~2M parameters default)

- LSTM temporal model over encoded frames
- Text-conditioned frame prediction
- Outputs animated GIF from generated frame sequence

## Scaling to Larger Models

Navine AI is designed as a foundation you own and can grow. Here is an honest scaling path:

| Goal | Approach |
|------|----------|
| Better text/code | Increase `d_model`, `n_layers` in `configs/text.yaml`. Train on larger corpus (books, code repos). Consider migrating tokenizer to SentencePiece BPE. |
| Better images | Increase `base_channels`, train on 128x128+ data. Add more diffusion timesteps. Collect thousands of captioned images. |
| Better video | Train on real video frame sequences. Increase `num_frames` and model capacity. Consider latent diffusion for efficiency. |
| Production quality | Consumer hardware cannot train GPT-4 scale models. For serious results, use multi-GPU setups, longer training (days/weeks), and datasets with millions of examples. Navine AI gives you the pipeline; scale is a hardware and data problem. |

### Realistic expectations

- **Demo training** (included): Verifies the pipeline works. Output quality is limited.
- **Fine-tuned small model**: Useful for domain-specific text, stylized images, simple animations.
- **Large model**: Requires datacenter GPUs (A100/H100), terabytes of data, and weeks of training.

## Troubleshooting

Run the built-in doctor first:

```bash
python -m navine.cli doctor
python -m navine.cli doctor --fix
```

The doctor checks Python, dependencies, configs, checkpoints, data files, launcher syntax, and runs chat/code smoke tests.

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No trained Navine AI text model found` | Missing checkpoint | `python -m navine.cli train text` |
| Garbled or empty text replies | Model not trained or stale server | Train text model; restart server after code updates |
| `/api/chat` returns 404 | Old server process still running | Stop all Python servers; run `python -m navine.server` again |
| Empty code output | Code prompt format not suited to small model | Fixed in current version via chat pipeline; run `doctor` |
| Web UI shows 401 | API key required from non-localhost | Use `127.0.0.1`; or set `allow_localhost_without_auth: true` in `configs/api.yaml` |
| Import errors (cv2, fastapi, etc.) | Missing dependencies | `pip install -r requirements.txt && pip install -e .` |
| GUI launcher does nothing | PowerShell execution policy | Run `navine.bat` or `powershell -ExecutionPolicy Bypass -File scripts/navine-launcher.ps1` |
| Image/video generation fails | Missing checkpoint | `python -m navine.cli train image` / `train video` |

### Fresh start on Windows

```powershell
cd "Navine AI"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/prepare_data.py
python -m navine.cli train text
python -m navine.cli train image
python -m navine.cli train video
python -m navine.cli doctor
python -m navine.server
```

Then open http://127.0.0.1:8765 or double-click `navine.bat` for the GUI launcher.

## License

This project is yours to use, modify, and extend.
