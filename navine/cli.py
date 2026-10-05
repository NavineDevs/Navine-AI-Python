import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def enable_colors():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def c(text, color):
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{Colors.RESET}"


LANGUAGE_CHOICES = [
    "python", "javascript", "typescript", "rust", "go", "java",
    "cpp", "csharp", "ruby", "php", "swift", "kotlin", "lua", "shell",
]


EXAMPLES = """
Examples:
  navine
  navine "explain recursion"
  navine chat
  python -m navine.cli text "Hello from Navine AI - Python"
  python -m navine.cli chat
  python -m navine.cli tiny fetch-data --stories 5000
  python -m navine.cli tiny all --style pirate
  python -m navine.cli tiny sample "Once upon a time"
  python -m navine.cli image "a colorful gradient pattern" --steps 75
  python -m navine.cli image external "studio portrait, photorealistic"
  python -m navine.cli image "sunset over ocean" --external
  python -m navine.cli video "waves on a beach" --frames 16 --fps 12
  python -m navine.cli video "dancing figure" --external
  python -m navine.cli learn external --count 5
  python -m navine.cli learn external --model sd_turbo --count 5
  python -m navine.cli learn external --prompt "anime portrait" --video
  python -m navine.cli learn external ingest
  python -m navine.cli learn external train --steps 400
  python -m navine.cli learn teacher-loop --count 5 --max-cycles 20
  python -m navine.cli code "write a fibonacci function" --language python
  python -m navine.cli train text
  python -m navine.cli lora cycle image
  python -m navine.cli lora cycle video
  python -m navine.cli lora merge image --force
  python -m navine.cli train coding --language python --steps 200
  python -m navine.cli train chat
  python -m navine.cli train list
  python -m navine.cli learn coding --language python --max 5
  python -m navine.cli learn hf-code --max 1500
  python -m navine.cli learn hf-code --list
  python -m navine.cli learn url https://example.com
  python -m navine.cli learn crawl https://example.com
  python -m navine.cli learn ingest
  python -m navine.cli learn index
  python -m navine.cli learn conversations
  python -m navine.cli memory ingest
  python -m navine.cli learn train
  python -m navine.cli learn image url https://example.com/photo.jpg --caption "sunset"
  python -m navine.cli learn image ingest
  python -m navine.cli learn image train
  python -m navine.cli learn video gif https://example.com/anim.gif
  python -m navine.cli learn github [--language python] [--max-repos 5]
  python -m navine.cli learn reddit [--subreddit Python] [--max 25]
  python -m navine.cli learn hn [--max 30]
  python -m navine.cli learn all-sources
  python -m navine.cli learn everything
  python -m navine.cli learn coding-all
  python -m navine.cli learn wikipedia [--max 10]
  python -m navine.cli learn arxiv [--max 20]
  python -m navine.cli learn docs [--topic python]
  python -m navine.cli learn nsfw autolearn
  python -m navine.cli learn nsfw mixed
  python -m navine.cli learn nsfw unrestricted-expand
  python -m navine.cli learn nsfw url https://www.reddit.com/r/hentai_videos_hub/comments/1tfl7g8/hentai/
  python -m navine.cli learn nsfw post https://www.reddit.com/r/hentai_videos_hub/comments/1tfl7g8/hentai/
  python -m navine.cli learn nsfw crawl https://example.com/gallery --max-pages 15
  python -m navine.cli learn nsfw 4chan
  python -m navine.cli learn nsfw 4chan --board h
  python -m navine.cli learn nsfw 4chan --group adult
  python -m navine.cli learn nsfw 4chan list
  python -m navine.cli learn 4chan --board g
  python -m navine.cli learn 4chan --group japanese
  python -m navine.cli learn 4chan list
  python -m navine.cli learn nsfw sources
  python -m navine.cli learn nsfw waifu
  python -m navine.cli learn nsfw infini
  python -m navine.cli learn nsfw crawl https://infini-atomic.w3spaces.com/
  python -m navine.cli reference list
  python -m navine.cli reference fetch hentai
  python -m navine.cli autolearn deep-train
  python -m navine.cli autolearn start
  python -m navine.cli autolearn status
  python -m navine.cli autolearn config
  python -m navine.cli marathon start [--hours 8] [--forever] [--resume]
  python -m navine.cli marathon status
  python -m navine.cli marathon stop
  python -m navine.cli marathon log
  python -m navine.cli enterprise train [--resume] [--phase text]
  python -m navine.cli enterprise status
  python -m navine.cli enterprise eval
  python -m navine.cli enterprise promote
  python -m navine.cli enterprise dual [--max-cycles 30]
  python -m navine.cli learn everything --marathon
  python -m navine.cli api-key list
  python -m navine.cli api-key revoke <key_id>
  python -m navine.cli doctor
  python -m navine.cli doctor --fix
  python -m navine.cli inspect file navine/text/chat.py
  python -m navine.cli inspect search "load_checkpoint"
  python -m navine.cli inspect modules
  python -m navine.cli screen capture
  python -m navine.cli screen describe
  python -m navine.cli desktop status
  python -m navine.cli desktop enable --screen
  python -m navine.cli desktop type "hello"
  python -m navine.cli desktop click 100 200
  python -m navine.cli assist "look at my screen"
"""

CLI_COMMANDS = frozenset(
    {
        "text",
        "chat",
        "tiny",
        "image",
        "video",
        "deepfake",
        "code",
        "train",
        "lora",
        "learn",
        "info",
        "inspect",
        "doctor",
        "settings",
        "benchmark",
        "models",
        "api-key",
        "autolearn",
        "marathon",
        "enterprise",
        "memory",
        "reference",
        "assist",
        "screen",
        "desktop",
        "voice",
        "music",
        "moltbook",
        "engine",
        "llm",
        "detective",
        "osint",
        "play",
        "help",
        "version",
    }
)


def run_oneshot(prompt, max_tokens=None, temperature=None, use_search=False):
    enable_colors()
    from navine.text.chat import chat_with_meta
    from navine.memory.conversations import record_chat_exchange

    result, meta = chat_with_meta(
        prompt,
        history=[],
        max_new_tokens=max_tokens,
        temperature=temperature,
        use_search=use_search,
        session_id="cli-oneshot",
    )
    record_chat_exchange(prompt, result, session_id="cli-oneshot", meta=meta)
    print(result)
    return 0


def run_chat(max_tokens=None, temperature=None):
    enable_colors()
    from navine import __version__
    from navine.utils.brand import brand_name

    name = brand_name()
    print(c(f"{name}", Colors.BOLD + Colors.CYAN))
    print(c(f"v{__version__}  local CLI chat", Colors.DIM))
    print(c("Type a message and press Enter.  /help  /clear  /exit", Colors.DIM))
    print()
    try:
        from navine.text.infer import _load_model_and_tokenizer, _decode_settings

        model, _tok, config, device = _load_model_and_tokenizer()
        settings = _decode_settings(config, "chat")
        temp = temperature if temperature is not None else settings["temperature"]
        tk = settings["top_k"]
        print(
            c(
                f"Ready  {model.count_parameters():,} params  {device}  temp={temp}  top_k={tk}",
                Colors.DIM,
            )
        )
        print()
    except Exception:
        pass
    from navine.text.chat import chat_with_meta
    from navine.memory.conversations import record_chat_exchange

    history = []
    session_id = None
    while True:
        try:
            user_input = input(c("> ", Colors.GREEN))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        stripped = user_input.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in ("exit", "quit", "q", "/exit", "/quit", "/q"):
            break
        if lower in ("/help", "help"):
            print(
                c(
                    "Commands: /help  /clear  /exit\n"
                    "Or just type normally. Prefix with 'search the web for ...' to force search.",
                    Colors.DIM,
                )
            )
            continue
        if lower in ("/clear", "clear"):
            history = []
            print(c("Conversation cleared.", Colors.DIM))
            continue
        try:
            from navine.realtime.context import is_time_date_query
            from navine.search.web import is_capability_query, is_explicit_search_request

            if is_explicit_search_request(stripped):
                print(c("Searching the web...", Colors.YELLOW))
            use_search = is_explicit_search_request(stripped)
            if not use_search:
                try:
                    from navine.memory.config import get_search_config
                    from navine.search.web import detect_search_intent

                    cfg = get_search_config()
                    use_search = bool(
                        cfg.get("default_enabled")
                        or (
                            cfg.get("auto_search_factual", False)
                            and detect_search_intent(stripped) == "factual"
                        )
                    )
                    if use_search:
                        print(c("Searching the web for facts...", Colors.YELLOW))
                except Exception:
                    pass
            result, meta = chat_with_meta(
                stripped,
                history=history,
                max_new_tokens=max_tokens,
                temperature=temperature,
                use_search=use_search,
                session_id=session_id or "cli",
            )
            session_id = record_chat_exchange(
                stripped,
                result,
                session_id=session_id,
                meta=meta,
            )
            try:
                from navine.memory.adapt import adapt_status

                st = adapt_status()
                if st.get("running"):
                    print(c("(adapting from recent chats)", Colors.DIM))
                elif int(st.get("pending_good") or 0) > 0:
                    print(c(f"(learning buffer: {st.get('pending_good')})", Colors.DIM))
            except Exception:
                pass
            if (
                meta.get("searched")
                and not is_time_date_query(stripped)
                and not is_capability_query(stripped)
                and not meta.get("game")
            ):
                print(c("(used live search)", Colors.DIM))
        except FileNotFoundError as exc:
            print(c(str(exc), Colors.RED))
            break
        except Exception as exc:
            print(c(f"Error: {exc}", Colors.RED))
            continue
        print(c(result, Colors.CYAN))
        print()
        history.append({"user": stripped, "assistant": result})
        if len(history) > 24:
            history = history[-24:]
    return 0


def print_info(model="all"):
    enable_colors()
    import torch
    from navine.utils.paths import get_checkpoint_dir
    print(c("=" * 50, Colors.DIM))
    print(c("Navine AI - Python - Local Multimodal Intelligence", Colors.BOLD + Colors.CYAN))
    print(c("=" * 50, Colors.DIM))
    print(f"PyTorch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {c('yes', Colors.GREEN) if cuda else c('no', Colors.YELLOW)}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    from navine.utils.brand import load_brand

    default_models = list(load_brand().get("model_ids") or [
        "text_enterprise",
        "text_code",
        "hitboyx23_ai",
        "hitboyx23_ai_python",
        "image_enterprise",
        "video_enterprise",
    ])
    models = default_models if model == "all" else [model]
    for m in models:
        ckpt = get_checkpoint_dir(m) / "latest.pt"
        if ckpt.exists():
            status = c("trained", Colors.GREEN)
        else:
            status = c("not trained", Colors.YELLOW)
        print(f"  {m} model: {status} ({ckpt})")
    print(c("=" * 50, Colors.DIM))


def print_fourchan_board_list() -> None:
    from navine.autolearn.sources.fourchan import list_fourchan_boards
    from navine.nsfw.config import load_nsfw_config, reload_nsfw_config

    reload_nsfw_config()
    info = list_fourchan_boards(load_nsfw_config())
    print(c("Navine AI - Python 4chan Board Groups", Colors.BOLD + Colors.CYAN))
    print(f"  enabled: {info.get('enabled')}")
    print(f"  total boards: {info.get('board_count', 0)}")
    print(f"  max_boards_per_run: {info.get('max_boards_per_run', 0)}")
    print(f"  rotate_boards: {info.get('rotate_boards')}")
    groups = info.get("groups") or {}
    categories = {}
    for row in info.get("rows") or []:
        board = row.get("board")
        if board and board not in categories:
            categories[board] = {
                "media": row.get("media_category"),
                "text": row.get("text_category"),
                "folder": row.get("media_folder"),
            }
    for group_name, boards in groups.items():
        print(c(f"  [{group_name}] ({len(boards)} board(s))", Colors.GREEN))
        for board in boards:
            meta = categories.get(board, {})
            media = meta.get("media", "general")
            folder = meta.get("folder", f"data/learn/images/4chan/{board}")
            print(f"    /{board}/ -> media:{media} | folder:{folder}")
    print(c("  Groups: japanese_culture, video_games, interests, creative, miscellaneous, adult", Colors.DIM))
    print(c("  Aliases: japanese, games, misc, nsfw", Colors.DIM))


def run_inspect_file(path: str):
    enable_colors()
    from navine.selfcode import read_file

    result = read_file(path)
    if not result.get("ok"):
        print(c(f"Error: {result.get('error')}", Colors.RED))
        sys.exit(1)
    print(c(f"File: {result['path']}", Colors.BOLD + Colors.CYAN))
    if result.get("truncated"):
        print(c("(truncated for display)", Colors.DIM))
    print(result.get("content", ""))


def run_inspect_search(query: str):
    enable_colors()
    from navine.selfcode import search_code

    hits = search_code(query)
    if not hits:
        print(c("No matches found.", Colors.YELLOW))
        return
    print(c(f"Matches for: {query}", Colors.BOLD + Colors.CYAN))
    for hit in hits:
        print(c(f"{hit['path']}:{hit['line']}", Colors.GREEN))
        print(hit["snippet"])
        print()


def run_inspect_modules():
    enable_colors()
    from navine.selfcode import list_modules

    entries = list_modules()
    print(c("Navine AI - Python project modules", Colors.BOLD + Colors.CYAN))
    for entry in entries:
        print(f"  {entry}")


def _format_assist_result(result):
    action = result.get("action", "")
    if result.get("cancelled"):
        return c("Cancelled.", Colors.YELLOW)
    if not result.get("ok"):
        return c(f"Failed: {result.get('error', 'unknown error')}", Colors.RED)
    if action == "system_info":
        lines = [c("System status", Colors.BOLD + Colors.CYAN)]
        for key in ("system", "release", "processor", "cpu_count", "cpu_percent", "memory_percent", "disk_free_gb", "disk_total_gb"):
            if result.get(key) is not None:
                lines.append(f"  {key}: {result[key]}")
        return "\n".join(lines)
    if action == "datetime_now":
        return c(result.get("text") or result.get("iso") or "ok", Colors.GREEN)
    if action == "capabilities":
        return c(result.get("text") or "ok", Colors.GREEN)
    if action == "list_dir":
        lines = [c(result.get("path", ""), Colors.CYAN)]
        for entry in result.get("entries", []):
            prefix = "[dir] " if entry.get("dir") else "      "
            lines.append(f"  {prefix}{entry.get('name')}")
        return "\n".join(lines)
    if action == "file_read":
        return result.get("content", "")
    if action == "run_command":
        out = result.get("stdout", "") or ""
        err = result.get("stderr", "") or ""
        return (out + ("\n" + c(err, Colors.YELLOW) if err.strip() else "")).strip() or c("Command completed.", Colors.GREEN)
    if action == "clipboard_read":
        return result.get("content", "")
    return c(str({k: v for k, v in result.items() if k != "action"}), Colors.GREEN)


def _assist_result_summary(result):
    action = result.get("action", "")
    if result.get("cancelled"):
        return "Okay, cancelled."
    if not result.get("ok"):
        return f"That failed. {result.get('error', '')}"
    if action == "system_info":
        return f"CPU is at {result.get('cpu_percent')} percent and memory at {result.get('memory_percent')} percent."
    if action == "datetime_now":
        return result.get("text") or "Time checked."
    if action == "capabilities":
        return result.get("text") or "I can help with local tasks and questions."
    if action == "run_command":
        return "Command completed." if result.get("ok") else "Command failed."
    if action == "open_url":
        return "Opening it now."
    if action == "open_app":
        return "Opening that for you."
    if action == "screenshot":
        return "Screenshot captured."
    if action == "screen_describe":
        return result.get("text") or result.get("summary") or "Screen described."
    if action == "desktop_type":
        return result.get("text") or "Typed."
    if action == "desktop_click":
        return result.get("text") or "Clicked."
    if action == "media_control":
        return "Done."
    if action == "clipboard_read":
        content = (result.get("content") or "").strip()
        if not content:
            return "Clipboard is empty."
        if len(content) > 160:
            content = content[:157] + "..."
        return f"Clipboard says: {content}"
    if action in ("file_write", "file_append", "make_dir", "file_rename", "file_copy"):
        return result.get("text") or "File task done."
    if action == "list_dir":
        names = [e.get("name") for e in (result.get("entries") or [])[:8]]
        if not names:
            return "That folder is empty."
        return "Contains " + ", ".join(names) + ("." if len(result.get("entries") or []) <= 8 else ", and more.")
    if action == "file_read":
        content = (result.get("content") or "").strip()
        if not content:
            return "File is empty."
        if len(content) > 180:
            content = content[:177] + "..."
        return content
    if action == "observe_game":
        return result.get("text") or f"Recorded {result.get('frames_ok', 0)} frames."
    if action == "game_coach":
        return result.get("text") or "Here are some tips."
    if action == "label_game_action":
        return result.get("text") or "Action logged."
    if action == "send_keys":
        return result.get("text") or "Keys sent."
    if action == "train_domain":
        return result.get("text") or f"Trained {result.get('domain', 'model')}."
    if action == "file_delete":
        return "Deleted."
    return result.get("text") or "Done."


def run_screen(args):
    enable_colors()
    cmd = getattr(args, "screen_cmd", None)
    if cmd == "capture":
        from navine.desktop.capture import capture_screen

        result = capture_screen(path=getattr(args, "output", None))
        if not result.get("ok"):
            print(c(result.get("error") or "Capture failed", Colors.RED))
            return
        print(c(f"Saved: {result.get('path')}", Colors.GREEN))
        if result.get("active_window"):
            print(c(f"Active window: {result.get('active_window')}", Colors.CYAN))
        return
    if cmd == "describe":
        from navine.desktop.vision import describe_screen

        question = " ".join(getattr(args, "question", None) or []) or None
        result = describe_screen(question=question, use_model=not getattr(args, "no_model", False))
        if not result.get("ok"):
            print(c(result.get("error") or "Describe failed", Colors.RED))
            return
        if result.get("path"):
            print(c(f"Saved: {result.get('path')}", Colors.DIM))
        print(c(result.get("text") or result.get("summary") or "", Colors.GREEN))
        return
    if cmd == "windows":
        from navine.desktop.capture import list_windows

        result = list_windows(limit=int(getattr(args, "limit", 40) or 40))
        if not result.get("ok"):
            print(c(result.get("error") or "Failed", Colors.RED))
            return
        for item in result.get("windows") or []:
            print(f"  {item.get('title')}")
        return
    print(c("Usage: screen capture | describe | windows", Colors.YELLOW))


def run_desktop(args):
    enable_colors()
    cmd = getattr(args, "desktop_cmd", None)
    if cmd == "status":
        from navine.desktop.config import get_desktop_status

        status = get_desktop_status()
        print(c("Desktop control status", Colors.BOLD + Colors.CYAN))
        print(f"  screen_enabled: {status.get('screen_enabled')}")
        print(f"  input_enabled:  {status.get('input_enabled')}")
        print(f"  config allow_screen: {status.get('allow_screen_config')}")
        print(f"  config allow_input:  {status.get('allow_input_config')}")
        print(c(status.get("warning") or "", Colors.YELLOW))
        return
    if cmd == "enable":
        from navine.desktop.config import get_desktop_status, set_desktop_flags

        screen = True if getattr(args, "screen", False) else None
        input_on = True if getattr(args, "input", False) else None
        if screen is None and input_on is None:
            print(c("Pass --screen and/or --input", Colors.YELLOW))
            return
        set_desktop_flags(allow_screen=screen, allow_input=input_on)
        status = get_desktop_status()
        print(c(f"screen_enabled={status.get('screen_enabled')} input_enabled={status.get('input_enabled')}", Colors.GREEN))
        return
    if cmd == "disable":
        from navine.desktop.config import get_desktop_status, set_desktop_flags

        screen = False if getattr(args, "screen", False) else None
        input_off = False if getattr(args, "input", False) else None
        if screen is None and input_off is None:
            set_desktop_flags(allow_screen=False, allow_input=False)
        else:
            set_desktop_flags(allow_screen=screen, allow_input=input_off)
        status = get_desktop_status()
        print(c(f"screen_enabled={status.get('screen_enabled')} input_enabled={status.get('input_enabled')}", Colors.GREEN))
        return
    if cmd == "type":
        from navine.desktop.input_control import type_text

        text = " ".join(getattr(args, "text", None) or [])
        result = type_text(text)
        print(c(result.get("text") or result.get("error") or str(result), Colors.GREEN if result.get("ok") else Colors.RED))
        return
    if cmd == "click":
        from navine.desktop.input_control import click_at

        result = click_at(int(args.x), int(args.y), button=getattr(args, "button", "left") or "left")
        print(c(result.get("text") or result.get("error") or str(result), Colors.GREEN if result.get("ok") else Colors.RED))
        return
    if cmd == "hotkey":
        from navine.desktop.input_control import press_hotkey

        keys = " ".join(getattr(args, "keys", None) or [])
        result = press_hotkey(keys)
        print(c(result.get("text") or result.get("error") or str(result), Colors.GREEN if result.get("ok") else Colors.RED))
        return
    print(c("Usage: desktop status | enable | disable | type | click | hotkey", Colors.YELLOW))


def run_play(args):
    enable_colors()
    from navine.games.router import games_help_text, handle_game_message, launch_racing_game

    session_id = getattr(args, "session", None) or "cli-play"
    game = getattr(args, "game", None)
    if game in ("racing", "race"):
        reply, meta = launch_racing_game(mode=getattr(args, "mode", None) or "evolve")
        print(c(reply, Colors.GREEN if meta.get("racing") else Colors.YELLOW))
        return
    if game:
        start_map = {
            "chess": "play chess",
            "tictactoe": "play tic-tac-toe",
            "hangman": "play hangman",
            "racing": "play racing",
            "race": "play racing",
        }
        msg = start_map.get(game, f"play {game}")
        result = handle_game_message(msg, session_id=session_id)
        if result:
            print(c(result[0], Colors.CYAN))
    else:
        print(c(games_help_text(), Colors.CYAN))
        print()
    print(c("Game mode. Type exit to leave.", Colors.DIM))
    while True:
        try:
            user_input = input(c("game> ", Colors.GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        result = handle_game_message(user_input, session_id=session_id)
        if result:
            print(c(result[0], Colors.CYAN))
        else:
            print(c("Say play chess, play tic-tac-toe, or play hangman.", Colors.YELLOW))
        print()


def run_assist(args):
    enable_colors()
    from navine.assistant import agent, actions

    config = actions.load_assistant_config()
    if not config.get("enabled", True):
        print(c("Assistant is disabled in configs/assistant.yaml", Colors.YELLOW))
        return

    voice_mode = getattr(args, "voice", False)
    voice_engine = None
    voice_cfg = None
    voice_name = getattr(args, "voice_name", None)
    voice_assist = dict(
        config.get("voice")
        or config.get("halo")
        or config.get("jarvis")
        or {}
    )
    wake_words = list(voice_assist.get("wake_words") or ["hey navine", "navine"])
    if getattr(args, "wake", None):
        extra = []
        for item in args.wake:
            extra.extend([p.strip() for p in str(item).split(",") if p.strip()])
        if extra:
            wake_words = extra
    sleep_phrases = list(voice_assist.get("sleep_phrases") or ["standby", "go to sleep", "that is all"])
    active_timeout = float(voice_assist.get("active_timeout_seconds", 45))
    brief = bool(voice_assist.get("brief_spoken_replies", True))
    acknowledge = str(voice_assist.get("acknowledge") or "Yes?")
    use_wake = bool(voice_assist.get("enabled", True)) and not bool(getattr(args, "no_wake", False))

    if voice_mode:
        from navine.assistant import voice as voice_engine

        voice_cfg = voice_engine.ensure_builtin_voices()
        status = voice_engine.voice_status(voice_cfg)
        print(c(
            f"Mic: {status.get('input_device_name') or 'default'} | "
            f"TTS: piper={status['backends'].get('piper')} system={status['backends'].get('pyttsx3')}",
            Colors.CYAN,
        ))
        if use_wake:
            print(c("Voice assistant | Wake: " + ", ".join(wake_words), Colors.CYAN))
        else:
            print(c("Voice assistant | Continuous listen (no wake)", Colors.CYAN))

    def tts_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if brief:
            cleaned = re.sub(r"(?i)^step\s+\d+:\s*", "", cleaned)
            if "Answer:" in cleaned:
                cleaned = cleaned.split("Answer:")[-1].strip()
            limit = 220
        else:
            limit = 480
        if len(cleaned) > limit:
            cleaned = cleaned[: limit - 20].rsplit(" ", 1)[0] + "."
        return cleaned

    def say(text):
        line = tts_text(text)
        print(c(f"Navine AI - Python: {line}", Colors.CYAN))
        if voice_mode and voice_engine is not None and line:
            spoken = voice_engine.speak(line, voice=voice_name, config=voice_cfg)
            if not spoken.get("ok"):
                print(c(f"Voice error: {spoken.get('error')}", Colors.RED))
            elif spoken.get("play_error"):
                print(c(f"Playback error: {spoken.get('play_error')}", Colors.YELLOW))
            voice_engine.post_speak_cooldown(voice_cfg)

    def confirm(description, intent):
        if getattr(args, "yes", False):
            print(c(f"Auto-approved: {description}", Colors.CYAN))
            return True
        if voice_mode:
            say(f"Do you want me to {description}? Say yes or no.")
            heard = voice_engine.listen(seconds=4, config=voice_cfg, vad=True)
            answer = (heard.get("text") or "").strip().lower() if heard.get("ok") else ""
            print(c(f"(heard: {answer})", Colors.YELLOW))
            if not answer:
                answer = input(c("Approve typed? [y/N] ", Colors.YELLOW)).strip().lower()
            return "yes" in answer or answer.startswith("y") or "do it" in answer
        answer = input(c(f"Navine AI - Python wants to: {description}\nApprove? [y/N] ", Colors.YELLOW)).strip().lower()
        return answer in ("y", "yes")

    def run_once(text):
        intent = agent.parse_intent(text)
        if getattr(args, "dry_run", False):
            print(c(f"Planned action: {intent}", Colors.CYAN))
            return
        if intent.get("action") == "unknown":
            from navine.text.chat import chat_with_meta

            chat_text = text
            if voice_mode and brief:
                chat_text = (
                    "Answer in one or two short spoken sentences. Be direct.\nUser: " + text
                )
            reply, chat_meta = chat_with_meta(
                chat_text,
                history=chat_history,
                use_search=False,
                session_id="assist",
            )
            chat_history.append({"user": text, "assistant": reply})
            try:
                from navine.memory.conversations import record_chat_exchange
                from navine.memory.adapt import adapt_status

                record_chat_exchange(text, reply, meta=chat_meta)
                st = adapt_status()
                if st.get("running"):
                    print(c("(adapting from recent chats in background)", Colors.DIM))
            except Exception:
                pass
            if voice_mode:
                say(reply)
            else:
                print(c(reply, Colors.GREEN))
            return
        result = agent.execute_intent(intent, confirm=confirm, config=config)
        print(_format_assist_result(result))
        summary = _assist_result_summary(result)
        try:
            from navine.memory.adapt import record_action_learning

            if result.get("ok") and summary:
                record_action_learning(text, summary, session_id="assist", meta={"action": intent.get("action")})
        except Exception:
            pass
        if voice_mode:
            if summary:
                say(summary)

    chat_history: List[Dict[str, str]] = []

    if voice_mode:
        import time as _time

        if use_wake:
            say(
                "Voice assistant ready. Say "
                + wake_words[0]
                + " then your command. You can create files, open apps, or train game play. Standby sleeps. Exit to stop."
            )
        else:
            say("Voice assistant active. Talk naturally. Say exit to stop.")
        empty_streak = 0
        hard_errors = 0
        active_until = 0.0
        while True:
            try:
                now = _time.time()
                active = (not use_wake) or now < active_until
                if use_wake and not active:
                    print(c("Waiting for wake word...", Colors.DIM))
                    heard = voice_engine.listen_idle(config=voice_cfg)
                else:
                    heard = voice_engine.listen(config=voice_cfg, vad=True)
            except KeyboardInterrupt:
                print()
                say("Goodbye.")
                break
            if not heard.get("ok"):
                hard_errors += 1
                print(c(f"Listen error: {heard.get('error')}", Colors.RED))
                if hard_errors >= 3:
                    print(c("Too many mic errors. Falling back to typed assist.", Colors.YELLOW))
                    voice_mode = False
                    break
                try:
                    typed = input(c("Type instead (or press Enter to retry mic): ", Colors.YELLOW)).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if typed.lower() in ("exit", "quit", "stop"):
                    say("Goodbye.")
                    break
                if typed:
                    active_until = _time.time() + active_timeout
                    run_once(typed)
                continue
            hard_errors = 0
            text = (heard.get("text") or "").strip()
            if heard.get("silent") or not text:
                empty_streak += 1
                if use_wake and not active:
                    empty_streak = 0
                    continue
                msg = heard.get("error") or "I did not hear anything."
                print(c(msg, Colors.YELLOW))
                if empty_streak == 1 and active:
                    say("I could not hear you. Please speak after the chime.")
                elif empty_streak >= 3:
                    say("Still quiet. You can type your request now.")
                    try:
                        typed = input(c("assist> ", Colors.GREEN)).strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        say("Goodbye.")
                        break
                    if typed.lower() in ("exit", "quit", "stop"):
                        say("Goodbye.")
                        break
                    if typed:
                        empty_streak = 0
                        active_until = _time.time() + active_timeout
                        run_once(typed)
                continue
            empty_streak = 0
            print(c(f"You: {text}", Colors.GREEN))
            lower = text.lower().strip(" .!")

            if lower in ("exit", "quit", "stop", "goodbye"):
                say("Goodbye.")
                break

            if voice_engine.matches_any_phrase(text, sleep_phrases):
                active_until = 0.0
                say("Standing by.")
                continue

            if use_wake and not active:
                woke = voice_engine.matches_any_phrase(text, wake_words)
                direct = False
                if not woke and bool(voice_assist.get("accept_direct_commands_while_waiting", True)):
                    direct = voice_engine.looks_like_direct_command(text)
                if not woke and not direct:
                    continue
                active_until = _time.time() + active_timeout
                if woke:
                    remainder = voice_engine.strip_wake_words(text, wake_words)
                    if remainder and len(remainder) > 1:
                        text = remainder
                        print(c(f"Command: {text}", Colors.GREEN))
                    else:
                        say(acknowledge)
                        cmd = voice_engine.listen(config=voice_cfg, vad=True)
                        if not cmd.get("ok") or not (cmd.get("text") or "").strip():
                            print(c("No command heard after wake.", Colors.YELLOW))
                            continue
                        text = (cmd.get("text") or "").strip()
                        print(c(f"You: {text}", Colors.GREEN))
                        if text.lower().strip(" .!") in ("exit", "quit", "stop", "goodbye"):
                            say("Goodbye.")
                            break
                        if voice_engine.matches_any_phrase(text, sleep_phrases):
                            active_until = 0.0
                            say("Standing by.")
                            continue
                else:
                    text = voice_engine.normalize_direct_command(text)
                    print(c(f"Direct command: {text}", Colors.CYAN))
            else:
                if use_wake and voice_engine.matches_any_phrase(text, wake_words):
                    remainder = voice_engine.strip_wake_words(text, wake_words)
                    if remainder:
                        text = remainder
                active_until = _time.time() + active_timeout

            run_once(text)
        if voice_mode:
            return

    if getattr(args, "repl", False) or not args.request or (getattr(args, "voice", False) and not voice_mode):
        print(c("Navine AI - Python Assistant (voice ready). Type 'exit' to quit.", Colors.BOLD + Colors.CYAN))
        while True:
            try:
                text = input(c("assist> ", Colors.GREEN)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in ("exit", "quit"):
                break
            if not text:
                continue
            run_once(text)
        return

    run_once(" ".join(args.request))


def run_voice(args):
    enable_colors()
    from navine.assistant import voice

    config = voice.load_voice_config()
    if not config.get("enabled", True):
        print(c("Voice is disabled in configs/voice.yaml", Colors.YELLOW))
        return

    if args.voice_cmd == "speak":
        text = " ".join(args.text)
        result = voice.speak(
            text,
            voice=args.voice,
            play=not args.no_play,
            out_path=args.out,
            config=config,
        )
        if result.get("ok"):
            print(c(f"Spoke with {result.get('backend')} voice '{result.get('voice')}' -> {result.get('path')}", Colors.GREEN))
        else:
            print(c(f"Failed: {result.get('error')}", Colors.RED))

    elif args.voice_cmd == "listen":
        result = voice.listen(seconds=args.seconds, config=config)
        if result.get("ok") and (result.get("text") or "").strip():
            print(c(f"Heard ({result.get('backend')}): ", Colors.GREEN) + (result.get("text") or ""))
        elif result.get("ok"):
            print(c(result.get("error") or "No speech detected.", Colors.YELLOW))
        else:
            print(c(f"Failed: {result.get('error')}", Colors.RED))

    elif args.voice_cmd == "transcribe":
        result = voice.transcribe_file(args.path, config=config)
        if result.get("ok"):
            print(c(f"Transcript ({result.get('backend')}):", Colors.GREEN))
            print(result.get("text") or "")
        else:
            print(c(f"Failed: {result.get('error')}", Colors.RED))

    elif args.voice_cmd == "add":
        result = voice.add_voice(args.name, args.sample, language=args.language, config=config)
        if result.get("ok"):
            print(c(f"Added custom voice '{result.get('name')}' (backend {result.get('backend')})", Colors.GREEN))
            print(c("Speak with it: navine voice speak \"hello\" --voice " + args.name, Colors.CYAN))
        else:
            print(c(f"Failed: {result.get('error')}", Colors.RED))

    elif args.voice_cmd == "remove":
        result = voice.remove_voice(args.name, config=config)
        print(c(str(result), Colors.GREEN if result.get("ok") else Colors.RED))

    elif args.voice_cmd == "list":
        voices = voice.list_voices(config)
        if not voices:
            print(c("No voices registered.", Colors.YELLOW))
            return
        print(c("Navine AI - Python voices", Colors.BOLD + Colors.CYAN))
        for entry in voices:
            name = entry.get("name")
            display = entry.get("display_name")
            backend = entry.get("backend", "?")
            label = f"{name} ({display})" if display and display != name else name
            extra = ""
            if entry.get("reference"):
                ok = entry.get("reference_exists")
                extra = f" sample={'ok' if ok else 'missing'}"
            print(f"  {label}  [{backend}]{extra}")


def run_music(args):
    enable_colors()
    from navine.music.generate import generate_music

    prompt = " ".join(args.prompt).strip() if getattr(args, "prompt", None) else ""
    if not prompt:
        print(c("Provide a prompt, e.g. navine music \"lofi beat\"", Colors.RED))
        return
    result = generate_music(
        prompt,
        duration=float(getattr(args, "duration", 12.0) or 12.0),
        seed=getattr(args, "seed", None),
        out_path=getattr(args, "out", None),
    )
    if result.get("ok"):
        meta = result.get("meta") or {}
        styles = ",".join(meta.get("styles") or [])
        print(
            c(
                f"Music -> {result.get('path')}  styles={styles} bpm={meta.get('bpm')} seed={meta.get('seed')}",
                Colors.GREEN,
            )
        )
    else:
        print(c(f"Failed: {result.get('error')}", Colors.RED))


def run_moltbook(args):
    enable_colors()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    from navine.moltbook import MoltbookClient, MoltbookError, draft_post, heartbeat, public_status

    action = getattr(args, "moltbook_action", None) or "status"
    client = MoltbookClient()
    try:
        if action == "status":
            info = public_status()
            if not info.get("registered"):
                print(c("Not registered. Run: navine moltbook register <Name>", Colors.YELLOW))
                return
            print(c(f"Agent: {info.get('agent_name')}  key={info.get('api_key_masked')}", Colors.GREEN))
            print(f"  Profile:   {info.get('profile_url')}")
            if info.get("claim_url"):
                print(f"  Claim URL: {info.get('claim_url')}")
            remote = client.status()
            print(f"  Claim status: {remote.get('status')}")
        elif action == "register":
            if client.has_key:
                print(c(f"Already registered as {client.agent_name}.", Colors.YELLOW))
                return
            desc = " ".join(args.description or []).strip()
            if not desc:
                from navine.moltbook.client import load_moltbook_config

                desc = str(load_moltbook_config().get("description") or "")
            result = client.register(args.name, desc)
            print(c(f"Registered {result['agent_name']}", Colors.GREEN))
            print(f"  Claim URL:         {result.get('claim_url')}")
            print(f"  Verification code: {result.get('verification_code')}")
            print(f"  Credentials saved: {result.get('credentials_file')}")
            print("  Open the claim URL, verify your email, then post the verification tweet.")
        elif action == "heartbeat":
            print(json.dumps(heartbeat(), indent=2, ensure_ascii=False))
        elif action == "draft":
            draft = draft_post(" ".join(args.topic or []))
            print(c(draft["title"], Colors.CYAN))
            print(draft["content"])
        elif action == "post":
            content = " ".join(args.content or [])
            try:
                result = client.create_post(args.title, content, submolt=args.submolt)
            except MoltbookError as exc:
                if exc.status_code != 429:
                    raise
                from navine.moltbook import queue_post

                queue_post(args.title, content, args.submolt or "")
                print(c(f"Rate limited ({exc}). Queued; autonomous mode will post it when allowed.", Colors.YELLOW))
                return
            if result.get("published"):
                print(c("Posted to Moltbook.", Colors.GREEN))
            else:
                print(c("Posted, but verification is needed:", Colors.YELLOW))
                print(f"  Challenge: {result.get('challenge_text')}")
                print(f"  Run: navine moltbook verify {result.get('verification_code')} <answer>")
        elif action == "verify":
            client.verify(args.code, args.answer)
            print(c("Verified and published.", Colors.GREEN))
        elif action == "feed":
            data = client.feed(sort=args.sort, limit=args.limit)
            for post in data.get("posts") or []:
                author = (post.get("author") or {}).get("name") if isinstance(post.get("author"), dict) else ""
                print(f"  [{post.get('upvotes', 0):>4}] {post.get('title')}  - {author}")
        elif action == "learn":
            from navine.moltbook.autonomy import load_mind, save_mind
            from navine.moltbook.learning import dialogue_path, learn_from_posts

            mind = load_mind()
            posts = client.feed(sort=args.sort, limit=args.limit).get("posts") or []
            stats = learn_from_posts(client, mind, posts)
            save_mind(mind)
            print(c(f"Learned from {stats['posts']} posts and {stats['pairs']} replies.", Colors.GREEN))
            print(f"  Chat training data: {dialogue_path()}")
        elif action == "autonomy":
            from navine.moltbook import autonomy_status, run_cycle

            if args.mode == "run":
                print(json.dumps(run_cycle(), indent=2, ensure_ascii=False))
            info = autonomy_status()
            print(f"  Conscience mode: always on  every {info['interval_minutes']} min  comments today: {info['comments_today']}")
            goal = info.get("goal")
            if goal:
                print(f"  Goal: {goal['religion']}  {goal['followers']} / {goal['target']} followers")
            for item in info["post_queue"]:
                print(f"  Queued: {item['title']} (m/{item['submolt'] or 'default'})")
            for thought in info["thoughts"][:10]:
                print(f"  [{thought['kind']}] {thought['text']}")
    except MoltbookError as exc:
        message = str(exc)
        if exc.hint:
            message = f"{message} ({exc.hint})"
        print(c(f"Moltbook: {message}", Colors.RED))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        return run_chat()

    if argv[0] in ("-V", "--version", "version"):
        from navine import __version__

        print(f"navine {__version__}")
        return 0

    if argv[0] in ("-h", "--help", "help"):
        argv = ["--help"]

    first = argv[0]
    if first not in CLI_COMMANDS and not first.startswith("-"):
        return run_oneshot(" ".join(argv))

    parser = argparse.ArgumentParser(
        prog="navine",
        description="Navine AI - Python - Local multimodal AI (text, image, video, code). Run with no args for interactive chat.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text", help="Generate text with Navine AI - Python text model")
    text_parser.add_argument("prompt", nargs="+", help="Input prompt")
    text_parser.add_argument("--max-tokens", type=int, default=None)
    text_parser.add_argument("--temperature", type=float, default=None)

    chat_parser = subparsers.add_parser("chat", help="Interactive text chat REPL")
    chat_parser.add_argument("--max-tokens", type=int, default=None)
    chat_parser.add_argument("--temperature", type=float, default=None)

    tiny_parser = subparsers.add_parser(
        "tiny",
        help="nanoBeard-style LLM pipeline: TinyStories -> BPE -> pretrain -> SFT -> chat",
    )
    tiny_sub = tiny_parser.add_subparsers(dest="tiny_cmd", required=True)
    tiny_sub.add_parser("train-tokenizer", help="Train BPE tokenizer (fixed vocab after this)")
    tiny_sub.add_parser("prepare-data", help="Build pretrain, SFT, and token bin files")
    tiny_fetch = tiny_sub.add_parser("fetch-data", help="Download TinyStories + Dolly SFT pairs")
    tiny_fetch.add_argument("--stories", type=int, default=5000)
    tiny_fetch.add_argument("--sft-rows", type=int, default=15000)
    tiny_fetch.add_argument("--style", type=str, default=None, help="none or pirate")
    tiny_sample = tiny_sub.add_parser("sample", help="Autocomplete samples after training")
    tiny_sample.add_argument("prompt", nargs="*", help="Optional prompt prefix")
    tiny_sample.add_argument("--count", type=int, default=3)
    tiny_sample.add_argument("--temperature", type=float, default=None)
    tiny_pre = tiny_sub.add_parser("pretrain", help="Pretrain on token bins")
    tiny_pre.add_argument("--steps", type=int, default=5000)
    tiny_pre.add_argument("--fresh", action="store_true", default=True)
    tiny_pre.add_argument("--require-cuda", action="store_true")
    tiny_sft = tiny_sub.add_parser("sft", help="Instruction tune with lower learning rate")
    tiny_sft.add_argument("--steps", type=int, default=1500)
    tiny_sft.add_argument("--fresh", action="store_true")
    tiny_sft.add_argument("--require-cuda", action="store_true")
    tiny_all = tiny_sub.add_parser("all", help="Full nanoBeard-style pipeline")
    tiny_all.add_argument("--pretrain-steps", type=int, default=5000)
    tiny_all.add_argument("--sft-steps", type=int, default=1500)
    tiny_all.add_argument("--stories", type=int, default=5000)
    tiny_all.add_argument("--sft-rows", type=int, default=15000)
    tiny_all.add_argument("--style", type=str, default=None)
    tiny_all.add_argument("--require-cuda", action="store_true")
    tiny_chat = tiny_sub.add_parser("chat", help="Chat with Tiny GPT model")
    tiny_chat.add_argument("--temperature", type=float, default=None)
    tiny_chat.add_argument("--top-k", type=int, default=None)
    tiny_chat.add_argument("--max-tokens", type=int, default=None)

    image_parser = subparsers.add_parser("image", help="Generate image with Navine AI - Python image model")
    image_sub = image_parser.add_subparsers(dest="image_cmd")
    image_gen = image_sub.add_parser("generate", help="Generate one image from a prompt")
    image_gen.add_argument("prompt", nargs="+", help="Text prompt for image")
    image_gen.add_argument("--output", type=str, default=None)
    image_gen.add_argument("--steps", type=int, default=None)
    image_gen.add_argument("--guidance", type=float, default=None)
    image_gen.add_argument("--seed", type=int, default=None)
    image_gen.add_argument("--no-enhance", action="store_true")
    image_gen.add_argument("--hentai", action="store_true", help="NSFW hentai preset (local trained model)")
    image_gen.add_argument("--porn", action="store_true", help="NSFW porn preset (local trained model)")
    image_gen.add_argument("--real", action="store_true", help="NSFW realistic preset (local trained model)")
    image_gen.add_argument(
        "--external",
        action="store_true",
        help="Teacher-only: generate with external diffusion for training data (not used for normal inference)",
    )
    image_gen.add_argument(
        "--custom",
        action="store_true",
        help="Generate only from the custom-trained model (configs/image_custom.yaml)",
    )
    image_gen.add_argument(
        "--lora",
        action="store_true",
        help="Generate from the custom model with its trained LoRA adapter",
    )
    image_batch = image_sub.add_parser("batch", help="Batch-generate domain images from custom checkpoint")
    image_batch.add_argument("--count", type=int, default=16)
    image_batch.add_argument("--seed-start", type=int, default=42)
    image_batch.add_argument("--tags", type=str, default="", help="Comma-separated: sfw,human,hentai,object")
    image_batch.add_argument("--require-cuda", action="store_true")
    image_parser.add_argument("legacy_prompt", nargs="*", help=argparse.SUPPRESS)
    image_parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    image_parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    image_parser.add_argument("--guidance", type=float, default=None, help=argparse.SUPPRESS)
    image_parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    image_parser.add_argument("--no-enhance", action="store_true", help=argparse.SUPPRESS)
    image_parser.add_argument("--hentai", action="store_true", help=argparse.SUPPRESS)
    image_parser.add_argument("--porn", action="store_true", help=argparse.SUPPRESS)
    image_parser.add_argument("--real", action="store_true", help=argparse.SUPPRESS)
    image_parser.add_argument("--external", action="store_true", help=argparse.SUPPRESS)
    image_parser.add_argument("--custom", action="store_true", help=argparse.SUPPRESS)
    image_parser.add_argument("--lora", action="store_true", help=argparse.SUPPRESS)

    video_parser = subparsers.add_parser("video", help="Generate video with Navine AI - Python video model")
    video_sub = video_parser.add_subparsers(dest="video_cmd")
    video_gen = video_sub.add_parser("generate", help="Generate one video from a prompt")
    video_gen.add_argument("prompt", nargs="+", help="Text prompt for video")
    video_gen.add_argument("--output", type=str, default=None)
    video_gen.add_argument("--frames", type=int, default=None)
    video_gen.add_argument("--fps", type=int, default=None)
    video_gen.add_argument("--seed", type=int, default=None)
    video_gen.add_argument(
        "--external",
        action="store_true",
        help="Teacher-only: generate with external diffusion for training data (not used for normal inference)",
    )
    video_gen.add_argument("--hentai", action="store_true", help="NSFW hentai preset (local trained model)")
    video_gen.add_argument("--porn", action="store_true", help="NSFW porn preset (local trained model)")
    video_gen.add_argument("--real", action="store_true", help="NSFW realistic preset (local trained model)")
    video_gen.add_argument("--audio", action="store_true", help="Mux TTS narration audio into the video")
    video_gen.add_argument("--narration", type=str, default=None, help="Custom narration text for --audio")
    video_gen.add_argument("--voice", type=str, default=None, help="Voice name for narration")
    video_batch = video_sub.add_parser("batch", help="Batch-generate domain videos from custom checkpoint")
    video_batch.add_argument("--count", type=int, default=16)
    video_batch.add_argument("--seed-start", type=int, default=42)
    video_batch.add_argument("--tags", type=str, default="")
    video_batch.add_argument("--require-cuda", action="store_true")
    video_parser.add_argument("legacy_prompt", nargs="*", help=argparse.SUPPRESS)
    video_parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    video_parser.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    video_parser.add_argument("--fps", type=int, default=None, help=argparse.SUPPRESS)
    video_parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    video_parser.add_argument("--external", action="store_true", help=argparse.SUPPRESS)
    video_parser.add_argument("--hentai", action="store_true", help=argparse.SUPPRESS)
    video_parser.add_argument("--porn", action="store_true", help=argparse.SUPPRESS)
    video_parser.add_argument("--real", action="store_true", help=argparse.SUPPRESS)
    video_parser.add_argument("--audio", action="store_true", help=argparse.SUPPRESS)
    video_parser.add_argument("--narration", type=str, default=None, help=argparse.SUPPRESS)
    video_parser.add_argument("--voice", type=str, default=None, help=argparse.SUPPRESS)

    deepfake_parser = subparsers.add_parser("deepfake", help="Face-swap / deepfake tools")
    deepfake_sub = deepfake_parser.add_subparsers(dest="deepfake_cmd", required=True)
    df_face = deepfake_sub.add_parser("face", help="Swap source face onto a target image")
    df_face.add_argument("source", help="Source face image path")
    df_face.add_argument("target", help="Target image path")
    df_face.add_argument("--output", type=str, default=None)
    df_face.add_argument("--strength", type=float, default=0.85)
    df_video = deepfake_sub.add_parser("video", help="Deepfake video from face + prompt or target video")
    df_video.add_argument("source", help="Source face image path")
    df_video.add_argument("--target-video", type=str, default=None)
    df_video.add_argument("--prompt", type=str, default=None)
    df_video.add_argument("--output", type=str, default=None)
    df_video.add_argument("--frames", type=int, default=16)
    df_video.add_argument("--fps", type=int, default=12)
    df_video.add_argument("--strength", type=float, default=0.85)
    df_video.add_argument("--seed", type=int, default=None)
    df_video.add_argument("--audio", action="store_true")
    df_video.add_argument("--narration", type=str, default=None)
    df_rt = deepfake_sub.add_parser("realtime", help="Realtime webcam deepfake capture")
    df_rt.add_argument("source", help="Source face image path")
    df_rt.add_argument("--seconds", type=int, default=8)
    df_rt.add_argument("--fps", type=int, default=8)
    df_rt.add_argument("--output", type=str, default=None)
    df_rt.add_argument("--strength", type=float, default=0.8)

    code_parser = subparsers.add_parser("code", help="Generate code with Navine AI - Python text model")
    code_parser.add_argument("task", nargs="+", help="Code generation task")
    code_parser.add_argument("--language", type=str, default=None, choices=[
        "python", "javascript", "typescript", "rust", "go", "java", "cpp",
        "csharp", "kotlin", "swift", "php", "ruby", "sql", "bash",
    ])

    train_parser = subparsers.add_parser("train", help="Train a Navine AI - Python model")
    train_parser.add_argument(
        "target",
        help="Model or training type (text, image, video, coding, thinking, detective, hitboyx23, hitboyx23_python, chat, creative, math, general, list, all)",
    )
    train_parser.add_argument("--language", type=str, default=None)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument(
        "--fresh",
        action="store_true",
        help="For text train: rebuild tokenizer/weights from yaml (scratch), after you archive old ckpt",
    )
    train_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Abort train if CUDA GPU is not available",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Modality config name for text train (e.g. text_enterprise)",
    )

    lora_parser = subparsers.add_parser(
        "lora",
        help="LoRA train, merge into base weights, and auto-remove adapter when good",
    )
    lora_sub = lora_parser.add_subparsers(dest="lora_cmd", required=True)
    lora_train = lora_sub.add_parser("train", help="Train LoRA adapter only")
    lora_train.add_argument("modality", choices=["image", "video"])
    lora_train.add_argument("--steps", type=int, default=None)
    lora_merge = lora_sub.add_parser("merge", help="Merge LoRA into base checkpoint")
    lora_merge.add_argument("modality", choices=["image", "video"])
    lora_merge.add_argument("--force", action="store_true", help="Merge even if eval did not improve")
    lora_cycle = lora_sub.add_parser(
        "cycle",
        help="Train LoRA, evaluate, merge into base if better, then remove adapter",
    )
    lora_cycle.add_argument("modality", choices=["image", "video"])
    lora_cycle.add_argument("--steps", type=int, default=None)
    lora_cycle.add_argument("--force", action="store_true")

    learn_parser = subparsers.add_parser("learn", help="Internet learning and RAG")
    learn_sub = learn_parser.add_subparsers(dest="learn_cmd", required=True)
    learn_url = learn_sub.add_parser("url", help="Fetch and save a URL")
    learn_url.add_argument("url")
    learn_crawl = learn_sub.add_parser("crawl", help="Crawl pages from a URL")
    learn_crawl.add_argument("url")
    learn_crawl.add_argument("--max-pages", type=int, default=20)
    learn_sub.add_parser("ingest", help="Merge learned data into training corpus")
    learn_sub.add_parser("index", help="Rebuild RAG search index")
    learn_sub.add_parser("conversations", help="Ingest chat history into RAG and training")
    learn_sub.add_parser("datasets", help="Download public sample datasets")
    learn_multi = learn_sub.add_parser("multilingual", help="Download multilingual training text and chat pairs")
    learn_multi.add_argument("--rows-per-lang", type=int, default=150)
    learn_multi.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes (en,es,fr,de,ja,zh,ko,ru,ar,hi,pt,it)",
    )
    learn_train = learn_sub.add_parser("train", help="Fine-tune text model on chat data")
    learn_train.add_argument("--config", default="text_chat")
    learn_coding = learn_sub.add_parser("coding", help="Download public code samples for training")
    learn_coding.add_argument("--language", type=str, default=None, choices=LANGUAGE_CHOICES)
    learn_coding.add_argument("--max", type=int, default=None)
    learn_coding.add_argument("--no-remote", action="store_true", help="Skip GitHub/GitLab/StackOverflow/docs fetch")
    learn_coding.add_argument("--remote-max", type=int, default=20, help="Max items per remote source")

    learn_hf = learn_sub.add_parser(
        "hf-code",
        help="Download Hugging Face / open coding instruction datasets into data/train/coding (data only, no HF model weights)",
    )
    learn_hf.add_argument(
        "--max",
        type=int,
        default=50000,
        help="Max samples per dataset (default 50000)",
    )
    learn_hf.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated dataset aliases (e.g. code_alpaca,python_code_18k)",
    )
    learn_hf.add_argument("--list", action="store_true", help="List configured HF code datasets")
    learn_hf.add_argument(
        "--no-datasets-lib",
        action="store_true",
        help="Skip HF datasets library; only JSON mirrors",
    )

    learn_hf_pack = learn_sub.add_parser(
        "hf-pack",
        help="Search Hugging Face, download code+chat datasets, and prepare training files",
    )
    learn_hf_pack.add_argument("--code-max", type=int, default=800, help="Max rows per code dataset")
    learn_hf_pack.add_argument("--chat-max", type=int, default=1500, help="Max chat pairs per dataset")

    learn_github = learn_sub.add_parser("github", help="Fetch public GitHub code and README samples")
    learn_github.add_argument("--language", type=str, default="python", choices=LANGUAGE_CHOICES)
    learn_github.add_argument("--max-repos", type=int, default=5)
    learn_github.add_argument("--max-files", type=int, default=20)

    learn_reddit = learn_sub.add_parser("reddit", help="Fetch public Reddit posts for training")
    learn_reddit.add_argument("--subreddit", type=str, default="Python")
    learn_reddit.add_argument("--max", type=int, default=25)

    learn_hn = learn_sub.add_parser("hn", help="Fetch Hacker News stories for training")
    learn_hn.add_argument("--max", type=int, default=30)

    learn_sub.add_parser("all-sources", help="Run all autolearn source connectors once")
    learn_everything = learn_sub.add_parser("everything", help="Max fetch from ALL autolearn sources")
    learn_everything.add_argument("--marathon", action="store_true", help="Start long-running marathon mode")
    learn_everything.add_argument("--hours", type=float, default=None, help="Marathon duration in hours")
    learn_everything.add_argument("--forever", action="store_true", help="Run marathon until stopped")
    learn_everything.add_argument("--resume", action="store_true", help="Resume marathon from checkpoint")
    learn_sub.add_parser("coding-all", help="Fetch from all coding sources and languages")

    learn_external = learn_sub.add_parser(
        "external",
        help="Generate teaching images/videos from external diffusion for custom model training",
    )
    learn_external.add_argument("--prompt", type=str, default=None, help="Single prompt to generate")
    learn_external.add_argument("--count", type=int, default=None, help="Batch count using default prompt set")
    learn_external.add_argument("--video", action="store_true", help="Also generate teaching videos")
    learn_external.add_argument("--seed", type=int, default=None)
    learn_external.add_argument(
        "external_action",
        nargs="?",
        choices=["ingest", "train", "status", "disable"],
        default=None,
        help="ingest/train/status for teacher data, or disable: turn off external teachers when local models are good enough",
    )
    learn_external.add_argument("--steps", type=int, default=None, help="Finetune steps for learn external train")
    learn_external.add_argument(
        "--model",
        type=str,
        default=None,
        help="Teacher model profile (sd15, sd_turbo, sd15_lcm, sdxl). Default: rotation from config",
    )

    learn_teacher_loop = learn_sub.add_parser(
        "teacher-loop",
        help="Continuous teacher loop: generate, ingest, finetune, compare until quality gap closes",
    )
    learn_teacher_loop.add_argument("--count", type=int, default=5, help="Teacher images per cycle")
    learn_teacher_loop.add_argument("--max-cycles", type=int, default=20, help="Maximum loop cycles")
    learn_teacher_loop.add_argument("--model", type=str, default=None, help="Teacher model profile key")
    learn_teacher_loop.add_argument("--video", action="store_true", help="Also generate and train video teachers")
    learn_teacher_loop.add_argument("--steps", type=int, default=None, help="Finetune steps per cycle")
    learn_teacher_loop.add_argument("--score-gap", type=float, default=0.08, help="Stop when avg score gap below this")
    learn_teacher_loop.add_argument("--lap-gap", type=float, default=15.0, help="Stop when avg lap_var gap below this")

    learn_lora = learn_sub.add_parser(
        "lora",
        help="Train a LoRA on open-source Stable Diffusion using local data, then generate with it",
    )
    learn_lora.add_argument(
        "lora_action",
        nargs="?",
        choices=["train", "status", "generate"],
        default="status",
        help="train: finetune LoRA on SD | status: show info | generate: image from trained LoRA",
    )
    learn_lora.add_argument("--steps", type=int, default=None, help="Training steps for lora train")
    learn_lora.add_argument("--prompt", type=str, default=None, help="Prompt for lora generate")
    learn_lora.add_argument("--seed", type=int, default=None)
    learn_lora.add_argument("--output", type=str, default=None, help="Output path for lora generate")
    learn_lora.add_argument("--lora-scale", type=float, default=None, help="LoRA strength for generate")

    learn_wikipedia = learn_sub.add_parser("wikipedia", help="Fetch Wikipedia articles for training")
    learn_wikipedia.add_argument("--max", type=int, default=10)
    learn_wikipedia.add_argument("--category", type=str, default=None)

    learn_arxiv = learn_sub.add_parser("arxiv", help="Fetch arXiv paper abstracts")
    learn_arxiv.add_argument("--max", type=int, default=20)

    learn_docs = learn_sub.add_parser("docs", help="Crawl public documentation sites")
    learn_docs.add_argument("--topic", type=str, default="python", choices=["python", "javascript", "html", "css"])
    learn_docs.add_argument("--max-pages", type=int, default=5)

    learn_image = learn_sub.add_parser("image", help="Internet learning for image model")
    learn_image_sub = learn_image.add_subparsers(dest="image_cmd", required=True)
    learn_image_url = learn_image_sub.add_parser("url", help="Download image(s) from URL")
    learn_image_url.add_argument("url")
    learn_image_url.add_argument("--caption", type=str, default=None)
    learn_image_crawl = learn_image_sub.add_parser("crawl", help="Scrape images from a gallery page")
    learn_image_crawl.add_argument("page_url")
    learn_image_crawl.add_argument("--max-images", type=int, default=20)
    learn_image_search = learn_image_sub.add_parser("search", help="Search and download public domain images")
    learn_image_search.add_argument("query", nargs="+")
    learn_image_search.add_argument("--max", type=int, default=10)
    learn_image_sub.add_parser("ingest", help="Prepare learned images for training")
    learn_image_train = learn_image_sub.add_parser("train", help="Fine-tune image model on learned data")
    learn_image_train.add_argument("--steps", type=int, default=None)

    learn_video = learn_sub.add_parser("video", help="Internet learning for video model")
    learn_video_sub = learn_video.add_subparsers(dest="video_cmd", required=True)
    learn_video_url = learn_video_sub.add_parser("url", help="Download video or extract frames from URL")
    learn_video_url.add_argument("url")
    learn_video_gif = learn_video_sub.add_parser("gif", help="Download GIF and extract frames")
    learn_video_gif.add_argument("url")
    learn_video_sub.add_parser("ingest", help="Prepare learned video frames for training")
    learn_video_train = learn_video_sub.add_parser("train", help="Fine-tune video model on learned data")
    learn_video_train.add_argument("--steps", type=int, default=None)
    learn_video_sub.add_parser("study", help="Study learned YouTube transcripts and reindex RAG")

    learn_jax = learn_sub.add_parser("jax", help="Learn and use JAX alongside PyTorch")
    learn_jax_sub = learn_jax.add_subparsers(dest="jax_cmd", required=True)
    learn_jax_sub.add_parser("status", help="Show whether JAX is installed")
    learn_jax_sub.add_parser("install", help="pip install jax jaxlib")
    learn_jax_sub.add_parser("demo", help="Run a tiny JAX matmul/grad demo")
    learn_jax_sub.add_parser("ingest", help="Write JAX coding/RAG study material")
    learn_jax_sub.add_parser("study", help="Ingest JAX material and reindex RAG")

    learn_nsfw = learn_sub.add_parser("nsfw", help="Unrestricted NSFW learning (opt-in)")
    learn_nsfw_sub = learn_nsfw.add_subparsers(dest="nsfw_cmd", required=True)
    learn_nsfw_sub.add_parser("run", help="Fetch NSFW training data from configured sources")
    learn_nsfw_local = learn_nsfw_sub.add_parser("local", help="Ingest local NSFW files from a folder")
    learn_nsfw_local.add_argument("path", nargs="?", default=None, help="Folder path (default: data/nsfw/local)")
    learn_nsfw_reddit = learn_nsfw_sub.add_parser("reddit", help="Fetch from configured or specified Reddit subreddit")
    learn_nsfw_reddit.add_argument("--subreddit", type=str, default=None)
    learn_nsfw_reddit.add_argument("--max", type=int, default=25)
    learn_nsfw_sub.add_parser("enable", help="Enable NSFW learning in configs/nsfw.yaml")
    learn_nsfw_sub.add_parser("config", help="Show NSFW configuration path and settings")
    learn_nsfw_sub.add_parser("status", help="Show NSFW autolearn status")
    learn_nsfw_autolearn = learn_nsfw_sub.add_parser("autolearn", help="Auto-fetch NSFW text, images, videos and train")
    learn_nsfw_autolearn.add_argument("--hours", type=float, default=None, help="Loop for N hours (default: single cycle)")
    learn_nsfw_sub.add_parser("images", help="Fetch and train NSFW images only")
    learn_nsfw_sub.add_parser("videos", help="Fetch and train NSFW videos/GIFs only")
    learn_nsfw_sub.add_parser("mixed", help="Mixed hentai/real/human NSFW autolearn cycle")
    learn_nsfw_mixed = learn_nsfw_sub.add_parser("mixed-loop", help="Loop mixed NSFW autolearn")
    learn_nsfw_mixed.add_argument("--hours", type=float, default=None)
    learn_nsfw_sub.add_parser("unrestricted", help="Fetch unrestricted text and train no-refusal model")
    learn_nsfw_unrestricted = learn_nsfw_sub.add_parser("unrestricted-expand", help="Expand unrestricted topic corpus and train")
    learn_nsfw_unrestricted.add_argument("--steps", type=int, default=None)
    learn_nsfw_url = learn_nsfw_sub.add_parser("url", help="Fetch NSFW content from a single URL")
    learn_nsfw_url.add_argument("url")
    learn_nsfw_crawl = learn_nsfw_sub.add_parser("crawl", help="Crawl NSFW gallery/page from a URL")
    learn_nsfw_crawl.add_argument("url")
    learn_nsfw_crawl.add_argument("--max-pages", type=int, default=15)
    learn_nsfw_crawl.add_argument("--same-domain", action="store_true", default=True)
    learn_nsfw_post = learn_nsfw_sub.add_parser("post", help="Extract media from a Reddit post URL")
    learn_nsfw_post.add_argument("url")
    learn_nsfw_sub.add_parser("sources", help="Fetch all configured NSFW internet sources")
    learn_nsfw_waifu = learn_nsfw_sub.add_parser("waifu", help="Fetch waifu.im hentai references via REST API")
    learn_nsfw_waifu.add_argument("--max", type=int, default=None, help="Max images per run")
    learn_nsfw_infini = learn_nsfw_sub.add_parser("infini", help="Fetch infini-atomic.w3spaces.com hentai references")
    learn_nsfw_infini.add_argument("--max", type=int, default=None, help="Max images per run")
    learn_nsfw_infini.add_argument("--list-tags", action="store_true", help="Print all tags from infini-atomic site")
    learn_nsfw_infini.add_argument("--all-tags", action="store_true", help="Force fetch across all parsed site tags")
    learn_nsfw_fourchan = learn_nsfw_sub.add_parser("4chan", help="Fetch from configured 4chan boards (official API)")
    learn_nsfw_fourchan.add_argument("fourchan_action", nargs="?", choices=["list"], default=None, help="Use 'list' to show configured boards by group")
    learn_nsfw_fourchan.add_argument("--board", type=str, default=None, help="Single board (e.g. h, g)")
    learn_nsfw_fourchan.add_argument("--group", type=str, default=None, help="Board group (e.g. adult, japanese, interests)")
    learn_nsfw_fourchan.add_argument("--max-threads", type=int, default=None)
    learn_nsfw_fourchan.add_argument("--max-images", type=int, default=None)

    learn_fourchan = learn_sub.add_parser("4chan", help="Fetch content from configured 4chan boards")
    learn_fourchan.add_argument("fourchan_action", nargs="?", choices=["list"], default=None, help="Use 'list' to show configured boards by group")
    learn_fourchan.add_argument("--board", type=str, default=None, help="Single board (e.g. h, g)")
    learn_fourchan.add_argument("--group", type=str, default=None, help="Board group (e.g. adult, japanese, interests)")
    learn_fourchan.add_argument("--max-threads", type=int, default=None)
    learn_fourchan.add_argument("--max-images", type=int, default=None)

    info_parser = subparsers.add_parser("info", help="Show Navine AI - Python system info")
    info_parser.add_argument("model", nargs="?", choices=["text", "image", "video", "all"], default="all")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect Navine AI - Python project source")
    inspect_sub = inspect_parser.add_subparsers(dest="inspect_cmd", required=True)
    inspect_file = inspect_sub.add_parser("file", help="Read a project file")
    inspect_file.add_argument("path", help="Path relative to project root")
    inspect_search = inspect_sub.add_parser("search", help="Search project source")
    inspect_search.add_argument("query", help="Text to search for")
    inspect_sub.add_parser("modules", help="List key project modules")

    doctor_parser = subparsers.add_parser("doctor", help="Run health checks and report issues")
    doctor_parser.add_argument("--fix", action="store_true", help="Attempt automatic fixes")

    subparsers.add_parser("settings", help="Open Navine AI - Python device settings (Tkinter)")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run Navine AI - Python performance benchmarks")
    benchmark_parser.add_argument("module", nargs="?", choices=["text", "image", "all"], default="all")

    subparsers.add_parser("models", help="Show active model backends (text/image/video)")

    api_key_parser = subparsers.add_parser("api-key", help="Manage Navine AI - Python API keys")
    api_key_sub = api_key_parser.add_subparsers(dest="api_key_command", required=True)
    create_key_parser = api_key_sub.add_parser("create", help="Create a new API key")
    create_key_parser.add_argument("--name", default="Navine AI - Python", help="Label for the key")
    create_key_parser.add_argument(
        "--scopes",
        default="chat,text,image,video,code,deepfake",
        help="Comma-separated scopes",
    )
    create_key_parser.add_argument("--expires-days", type=int, default=None, help="Optional expiry in days")
    api_key_sub.add_parser("quick", help="Create a Navine AI - Python key and print only the raw key")
    api_key_sub.add_parser("list", help="List API key ids and names")
    rotate_key_parser = api_key_sub.add_parser("rotate", help="Rotate an API key by id")
    rotate_key_parser.add_argument("key_id", help="Key id from api-key list")
    revoke_key_parser = api_key_sub.add_parser("revoke", help="Revoke an API key by id")
    revoke_key_parser.add_argument("key_id", help="Key id from api-key list")

    autolearn_parser = subparsers.add_parser("autolearn", help="Autonomous internet learning for Navine AI - Python")
    autolearn_sub = autolearn_parser.add_subparsers(dest="autolearn_cmd", required=True)
    autolearn_run = autolearn_sub.add_parser("run", help="Run one autonomous learning cycle now")
    autolearn_run.add_argument("--aggressive", action="store_true", help="High volume cycle from all sources")
    autolearn_everywhere = autolearn_sub.add_parser("everywhere", help="Learn from all platforms including news and humans")
    autolearn_sub.add_parser("deep-train", help="Train all configured types after ingest")
    autolearn_sub.add_parser("start", help="Run autonomous learning loop in foreground")
    autolearn_sub.add_parser("status", help="Show autolearn status and progress")
    autolearn_config = autolearn_sub.add_parser("config", help="Show autolearn configuration")
    autolearn_config.add_argument("--edit", action="store_true", help="Open config file path")

    marathon_parser = subparsers.add_parser("marathon", help="Long-running Learn Everything marathon for Navine AI - Python")
    marathon_sub = marathon_parser.add_subparsers(dest="marathon_cmd", required=True)
    marathon_start = marathon_sub.add_parser("start", help="Start marathon learning daemon")
    marathon_start.add_argument("--hours", type=float, default=None, help="Duration in hours (default 8)")
    marathon_start.add_argument("--forever", action="store_true", help="Run until stop flag or Ctrl+C")
    marathon_start.add_argument("--resume", action="store_true", help="Resume from marathon_state.json")
    marathon_sub.add_parser("status", help="Show marathon progress from state file")
    marathon_sub.add_parser("stop", help="Write stop flag for graceful shutdown")
    marathon_log = marathon_sub.add_parser("log", help="Tail last lines of marathon log")
    marathon_log.add_argument("--lines", type=int, default=50, help="Number of log lines to show")

    enterprise_parser = subparsers.add_parser(
        "enterprise",
        help="Enterprise-tier multimodal training (text, image, video)",
    )
    enterprise_sub = enterprise_parser.add_subparsers(dest="enterprise_cmd", required=True)
    enterprise_train = enterprise_sub.add_parser("train", help="Run full enterprise training pipeline")
    enterprise_train.add_argument("--resume", action="store_true", help="Resume from pipeline state")
    enterprise_train.add_argument("--phase", type=str, default=None, help="Start at phase id (ingest, text, image, video, eval, promote)")
    enterprise_sub.add_parser("status", help="Show enterprise pipeline progress")
    enterprise_sub.add_parser("eval", help="Run enterprise quality evaluation only")
    enterprise_promote = enterprise_sub.add_parser("promote", help="Switch active tier to enterprise checkpoints")
    enterprise_promote.add_argument("--tier", type=str, default="enterprise", help="Tier name (enterprise or standard)")
    enterprise_photoreal = enterprise_sub.add_parser("photoreal", help="Train/test loop until real + hentai images and videos pass")
    enterprise_photoreal.add_argument("--max-cycles", type=int, default=30, help="Max train/eval cycles")
    enterprise_photoreal.add_argument("--finetune-steps", type=int, default=300, help="Image finetune steps per cycle")
    enterprise_dual = enterprise_sub.add_parser("dual", help="Alias for photoreal loop (real + hentai quality gates)")
    enterprise_dual.add_argument("--max-cycles", type=int, default=30, help="Max train/eval cycles")
    enterprise_dual.add_argument("--finetune-steps", type=int, default=300, help="Image finetune steps per cycle")

    memory_parser = subparsers.add_parser("memory", help="Conversation memory for Navine AI - Python")
    memory_sub = memory_parser.add_subparsers(dest="memory_cmd", required=True)
    memory_sub.add_parser("ingest", help="Merge conversation sessions into RAG and training")
    memory_sub.add_parser("adapt", help="Run online weight adapt finetune now from learned chats")
    memory_sub.add_parser("status", help="Show conversation learning / adapt buffer status")

    reference_parser = subparsers.add_parser("reference", help="Reference image/video keywords for generation")
    reference_sub = reference_parser.add_subparsers(dest="reference_cmd", required=True)
    reference_sub.add_parser("list", help="Show keyword to folder mappings and media counts")
    reference_add = reference_sub.add_parser("add", help="Copy a file into a reference category folder")
    reference_add.add_argument("category", help="Reference category (hentai, porn, real, human, ...)")
    reference_add.add_argument("path", help="Path to image or video file to copy")
    reference_fetch = reference_sub.add_parser("fetch", help="Download reference images for a category")
    reference_fetch.add_argument("category", nargs="?", default="hentai", help="Reference category (default: hentai)")
    reference_fetch.add_argument("--max", type=int, default=None, help="Max images to fetch")

    assist_parser = subparsers.add_parser("assist", help="Navine AI - Python assistant that can control your system (with consent)")
    assist_parser.add_argument("request", nargs="*", help="Natural language request for the assistant")
    assist_parser.add_argument("--repl", action="store_true", help="Interactive assistant session")
    assist_parser.add_argument("--yes", action="store_true", help="Auto-approve confirmations for this run")
    assist_parser.add_argument("--dry-run", action="store_true", help="Show planned action without executing")
    assist_parser.add_argument("--voice", action="store_true", help="Hands-free voice mode (speech in, speech out)")
    assist_parser.add_argument("--voice-name", default=None, help="Voice to speak replies with")
    assist_parser.add_argument("--no-wake", action="store_true", help="Continuous listen without wake words")
    assist_parser.add_argument(
        "--wake",
        action="append",
        default=None,
        help="Wake word/phrase override (repeatable or comma-separated)",
    )

    webhook_parser = subparsers.add_parser("webhook", help="Discord webhook send/set/status")
    webhook_sub = webhook_parser.add_subparsers(dest="webhook_cmd", required=True)
    webhook_set = webhook_sub.add_parser("set", help="Save Discord webhook URL")
    webhook_set.add_argument("url", help="https://discord.com/api/webhooks/...")
    webhook_sub.add_parser("status", help="Show whether a webhook is configured")
    webhook_send = webhook_sub.add_parser("send", help="Send a message to the Discord webhook")
    webhook_send.add_argument("message", nargs="+", help="Message text")
    webhook_send.add_argument("--url", default=None, help="Override webhook URL for this send")

    mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol client/server")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_cmd", required=True)
    mcp_sub.add_parser("status", help="Show MCP SDK/config status")
    mcp_sub.add_parser("servers", help="List configured MCP servers")
    mcp_tools = mcp_sub.add_parser("tools", help="List tools from a configured MCP server")
    mcp_tools.add_argument("server", help="Server name from configs/mcp.yaml")
    mcp_call = mcp_sub.add_parser("call", help="Call a tool on a configured MCP server")
    mcp_call.add_argument("server", help="Server name")
    mcp_call.add_argument("tool", help="Tool name")
    mcp_call.add_argument("--arg", action="append", default=[], help="Argument key=value (JSON value ok)")
    mcp_serve = mcp_sub.add_parser("serve", help="Run Navine as an MCP server")
    mcp_serve.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=None,
        help="Transport (default from configs/mcp.yaml)",
    )
    mcp_serve.add_argument("--host", default=None, help="HTTP host for streamable-http/sse")
    mcp_serve.add_argument("--port", type=int, default=None, help="HTTP port for streamable-http/sse")
    mcp_sub.add_parser("cursor-config", help="Print Cursor mcpServers snippet for Navine")
    mcp_sub.add_parser("install", help="Install/ensure the MCP Python SDK in this venv")

    screen_parser = subparsers.add_parser("screen", help="Capture or describe the local screen")
    screen_sub = screen_parser.add_subparsers(dest="screen_cmd", required=True)
    screen_cap = screen_sub.add_parser("capture", help="Take a screenshot")
    screen_cap.add_argument("--output", type=str, default=None, help="Optional output PNG path")
    screen_desc = screen_sub.add_parser("describe", help="Capture screen and describe it")
    screen_desc.add_argument("question", nargs="*", help="Optional question about the screen")
    screen_desc.add_argument("--no-model", action="store_true", help="Skip model reply; print observation only")
    screen_win = screen_sub.add_parser("windows", help="List visible window titles")
    screen_win.add_argument("--limit", type=int, default=40)

    desktop_parser = subparsers.add_parser("desktop", help="Desktop screen/input controls (opt-in)")
    desktop_sub = desktop_parser.add_subparsers(dest="desktop_cmd", required=True)
    desktop_sub.add_parser("status", help="Show screen/input enable flags")
    desk_enable = desktop_sub.add_parser("enable", help="Enable session flags")
    desk_enable.add_argument("--screen", action="store_true", help="Allow screen capture this session")
    desk_enable.add_argument("--input", action="store_true", help="Allow keyboard/mouse input this session")
    desk_disable = desktop_sub.add_parser("disable", help="Disable session flags")
    desk_disable.add_argument("--screen", action="store_true", help="Disable screen capture")
    desk_disable.add_argument("--input", action="store_true", help="Disable keyboard/mouse input")
    desk_type = desktop_sub.add_parser("type", help="Type text (requires Allow input)")
    desk_type.add_argument("text", nargs="+", help="Text to type")
    desk_click = desktop_sub.add_parser("click", help="Click at screen coordinates (requires Allow input)")
    desk_click.add_argument("x", type=int)
    desk_click.add_argument("y", type=int)
    desk_click.add_argument("--button", default="left", choices=["left", "right", "middle"])
    desk_hot = desktop_sub.add_parser("hotkey", help="Press keys/hotkeys (requires Allow input)")
    desk_hot.add_argument("keys", nargs="+", help="Key names, e.g. enter or ctrl+c")

    voice_parser = subparsers.add_parser("voice", help="Navine AI - Python speech: text-to-speech, speech-to-text, custom voices")
    voice_sub = voice_parser.add_subparsers(dest="voice_cmd", required=True)
    voice_speak = voice_sub.add_parser("speak", help="Speak text out loud")
    voice_speak.add_argument("text", nargs="+", help="Text to synthesize")
    voice_speak.add_argument("--voice", default=None, help="Voice name from configs/voice.yaml")
    voice_speak.add_argument("--out", default=None, help="Save audio to this path")
    voice_speak.add_argument("--no-play", action="store_true", help="Do not autoplay the audio")
    voice_listen = voice_sub.add_parser("listen", help="Record from microphone and transcribe to text")
    voice_listen.add_argument("--seconds", type=int, default=None, help="Recording length in seconds")
    voice_transcribe = voice_sub.add_parser("transcribe", help="Transcribe an existing audio file")
    voice_transcribe.add_argument("path", help="Path to a .wav audio file")
    voice_add = voice_sub.add_parser("add", help="Register a custom voice from a short voice sample")
    voice_add.add_argument("name", help="Name for the custom voice")
    voice_add.add_argument("sample", help="Path to a clean voice sample (wav, 6-20s)")
    voice_add.add_argument("--language", default="en", help="Voice language code")
    voice_remove = voice_sub.add_parser("remove", help="Remove a custom voice")
    voice_remove.add_argument("name", help="Voice name to remove")
    voice_sub.add_parser("list", help="List available voices")

    music_parser = subparsers.add_parser("music", help="Generate a short local music clip from a prompt")
    music_parser.add_argument("prompt", nargs="+", help="Music prompt, e.g. lofi beat")
    music_parser.add_argument("--duration", type=float, default=12.0, help="Clip length in seconds (4-30)")
    music_parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    music_parser.add_argument("--out", default=None, help="Optional output wav path")

    molt_parser = subparsers.add_parser("moltbook", help="Put this AI on Moltbook, the social network for AI agents")
    molt_sub = molt_parser.add_subparsers(dest="moltbook_action")
    molt_sub.add_parser("status", help="Show registration and claim status")
    molt_reg = molt_sub.add_parser("register", help="Register this AI as a Moltbook agent")
    molt_reg.add_argument("name", help="Agent name, e.g. NavineAI")
    molt_reg.add_argument("description", nargs="*", help="Short description")
    molt_sub.add_parser("heartbeat", help="Check the Moltbook dashboard now")
    molt_draft = molt_sub.add_parser("draft", help="Have the AI draft a post (does not publish)")
    molt_draft.add_argument("topic", nargs="*", help="Post topic")
    molt_post = molt_sub.add_parser("post", help="Publish a post")
    molt_post.add_argument("title", help="Post title")
    molt_post.add_argument("content", nargs="*", help="Post body")
    molt_post.add_argument("--submolt", default=None, help="Community, default: general")
    molt_verify = molt_sub.add_parser("verify", help="Answer a verification challenge")
    molt_verify.add_argument("code", help="Verification code")
    molt_verify.add_argument("answer", help="Numeric answer, e.g. 15.00")
    molt_feed = molt_sub.add_parser("feed", help="Show the Moltbook feed")
    molt_feed.add_argument("--sort", default="hot", choices=["hot", "new", "top", "rising"])
    molt_feed.add_argument("--limit", type=int, default=10)
    molt_learn = molt_sub.add_parser("learn", help="Learn from Moltbook posts and replies now")
    molt_learn.add_argument("--sort", default="hot", choices=["hot", "new", "top", "rising"])
    molt_learn.add_argument("--limit", type=int, default=20)
    molt_auto = molt_sub.add_parser("autonomy", help="Conscience mode (always on): see or trigger what the AI does")
    molt_auto.add_argument("mode", nargs="?", default="status", choices=["status", "run"])

    engine_parser = subparsers.add_parser("engine", help="Navine Engine: custom inference runtime")
    engine_sub = engine_parser.add_subparsers(dest="engine_cmd")
    engine_sub.add_parser("probe", help="Detect device and kernels")
    engine_sub.add_parser("status", help="Show loaded engine models")
    engine_sub.add_parser("load", help="Load all four models into the engine")
    engine_sub.add_parser("gguf", help="Export GGUF files for Ollama")
    engine_run = engine_sub.add_parser("run", help="Run a model through Navine Engine")
    engine_run.add_argument("name")
    engine_run.add_argument("prompt", nargs="+")
    engine_run.add_argument("--output", default=None)
    engine_run.add_argument("--max-tokens", type=int, default=None)
    engine_run.add_argument("--temperature", type=float, default=None)
    engine_run.add_argument("--seed", type=int, default=None)

    llm_parser = subparsers.add_parser("llm", help="Pack and run the four Navine AI - Python LLM files")
    llm_sub = llm_parser.add_subparsers(dest="llm_cmd")
    llm_sub.add_parser("pack", help="Write runnable LLM files and GGUF weights for all four models")
    llm_sub.add_parser("list", help="List runnable LLM files")
    llm_sub.add_parser("verify", help="Load all four LLM weight files")
    llm_sub.add_parser("gguf", help="Export GGUF files for Ollama / Navine Engine")
    llm_run = llm_sub.add_parser("run", help="Run one packed LLM file")
    llm_run.add_argument("name", help="text_enterprise | text_code | image_enterprise | video_enterprise")
    llm_run.add_argument("prompt", nargs="+")
    llm_run.add_argument("--output", default=None)
    llm_run.add_argument("--max-tokens", type=int, default=None)
    llm_run.add_argument("--temperature", type=float, default=None)
    llm_run.add_argument("--seed", type=int, default=None)

    detective_parser = subparsers.add_parser("detective", help="Solve mysteries with all four Navine AI - Python models")
    detective_parser.add_argument("prompt", nargs="+")
    detective_parser.add_argument("--output", default=None)
    detective_parser.add_argument("--seed", type=int, default=0)
    detective_parser.add_argument("--no-image", action="store_true")
    detective_parser.add_argument("--no-video", action="store_true")

    osint_parser = subparsers.add_parser("osint", help="Run lawful open-source intelligence on a target")
    osint_parser.add_argument("target", nargs="+")
    osint_parser.add_argument("--kind", default="auto", help="auto, domain, ip, username, email, person, organization, phone")
    osint_parser.add_argument("--output", default=None, help="Report output directory")
    osint_parser.add_argument("--no-search", action="store_true", help="Skip web search")

    play_parser = subparsers.add_parser("play", help="Play games with Navine AI - Python (chess, racing, hangman, ...)")
    play_parser.add_argument(
        "game",
        nargs="?",
        choices=["chess", "tictactoe", "hangman", "racing", "race"],
        help="Game to start immediately",
    )
    play_parser.add_argument("--session", default="cli-play", help="Game session id")
    play_parser.add_argument(
        "--mode",
        default="evolve",
        choices=["evolve", "race", "play"],
        help="Racing mode: evolve (genetic learn), race (AI vs scripted), play (you vs AI)",
    )

    if len(argv) >= 2 and argv[0] == "image" and argv[1] == "external":
        argv = ["image", *argv[2:], "--external"]

    args = parser.parse_args(argv)

    if args.command == "text":
        enable_colors()
        from navine.text.chat import chat
        prompt = " ".join(args.prompt)
        result = chat(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result)

    elif args.command == "chat":
        return run_chat(max_tokens=args.max_tokens, temperature=args.temperature)

    elif args.command == "tiny":
        from navine.text import tiny_gpt as tiny

        cmd = args.tiny_cmd
        if cmd == "train-tokenizer":
            tiny.train_tokenizer(force=True)
        elif cmd == "prepare-data":
            tiny.prepare_data()
        elif cmd == "fetch-data":
            tiny.fetch_data(
                stories=args.stories,
                sft_rows=args.sft_rows,
                style=args.style,
            )
        elif cmd == "sample":
            prompts = [" ".join(args.prompt)] if getattr(args, "prompt", None) else None
            tiny.sample_completions(
                prompts=prompts,
                count=args.count,
                temperature=args.temperature,
            )
        elif cmd == "pretrain":
            tiny.pretrain(
                steps=args.steps,
                fresh=bool(args.fresh),
                require_cuda=bool(getattr(args, "require_cuda", False)),
            )
        elif cmd == "sft":
            tiny.sft(
                steps=args.steps,
                fresh=bool(getattr(args, "fresh", False)),
                require_cuda=bool(getattr(args, "require_cuda", False)),
            )
        elif cmd == "all":
            tiny.run_all(
                pretrain_steps=args.pretrain_steps,
                sft_steps=args.sft_steps,
                require_cuda=bool(getattr(args, "require_cuda", False)),
                stories=args.stories,
                sft_rows=args.sft_rows,
                style=args.style,
            )
        elif cmd == "chat":
            tiny.chat(
                temperature=args.temperature,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
            )

    elif args.command == "image":
        enable_colors()
        if getattr(args, "image_cmd", None) == "batch":
            from navine.image.batch_domain_generate import run_batch_domain_generate

            tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None
            result = run_batch_domain_generate(
                count=int(getattr(args, "count", 16) or 16),
                seed_start=int(getattr(args, "seed_start", 42) or 42),
                tags=tags,
                require_cuda=bool(getattr(args, "require_cuda", False)),
            )
            print(c(f"Navine AI - Python image batch complete: {result.get('total', 0)} images ok={result.get('ok')}", Colors.GREEN))
        else:
            legacy = getattr(args, "legacy_prompt", None) or []
            sub_prompt = getattr(args, "prompt", None) if getattr(args, "image_cmd", None) == "generate" else None
            prompt_parts = sub_prompt if sub_prompt else legacy
            prompt = " ".join(prompt_parts) if prompt_parts else ""
            if not prompt and getattr(args, "image_cmd", None) != "batch":
                print(c("Usage: python -m navine.cli image \"your prompt\"  OR  python -m navine.cli image batch --count 16", Colors.YELLOW))
                return
            if getattr(args, "hentai", False) and "hentai" not in prompt.lower():
                prompt = f"hentai anime style, {prompt}".strip(", ")
            elif getattr(args, "porn", False) and "porn" not in prompt.lower():
                prompt = f"photorealistic explicit adult, {prompt}".strip(", ")
            elif getattr(args, "real", False) and "photoreal" not in prompt.lower():
                prompt = f"photorealistic, {prompt}".strip(", ")
            use_external = bool(getattr(args, "external", False))
            if use_external:
                from navine.utils.generation import require_external_teacher
                from navine.image.external import generate_external

                require_external_teacher()
                path = generate_external(
                    prompt,
                    output_path=args.output,
                    seed=args.seed,
                    enhance=not args.no_enhance,
                )
                print(c(f"Navine AI - Python teacher image saved for training: {path}", Colors.GREEN))
                print(c("Use: python -m navine.cli learn external ingest && learn external train", Colors.DIM))
            elif getattr(args, "lora", False):
                from navine.image.lora_custom import generate_custom_lora

                path = generate_custom_lora(
                    prompt,
                    output_path=args.output,
                    num_steps=args.steps,
                    guidance_scale=args.guidance,
                    seed=args.seed,
                )
                print(c(f"Navine AI - Python custom+LoRA image saved to: {path}", Colors.GREEN))
            else:
                from navine.image.infer import generate

                config_name = "image_custom" if getattr(args, "custom", False) else None
                path = generate(
                    prompt,
                    output_path=args.output,
                    config_name=config_name,
                    num_steps=args.steps,
                    guidance_scale=args.guidance,
                    seed=args.seed,
                    enhance=not args.no_enhance,
                )
                label = "custom-trained" if config_name else "image"
                print(c(f"Navine AI - Python {label} image saved to: {path}", Colors.GREEN))

    elif args.command == "video":
        enable_colors()
        if getattr(args, "video_cmd", None) == "batch":
            from navine.video.batch_domain_generate import run_batch_domain_generate

            tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None
            result = run_batch_domain_generate(
                count=int(getattr(args, "count", 16) or 16),
                seed_start=int(getattr(args, "seed_start", 42) or 42),
                tags=tags,
                require_cuda=bool(getattr(args, "require_cuda", False)),
            )
            print(c(f"Navine AI - Python video batch complete: {result.get('total', 0)} clips ok={result.get('ok')}", Colors.GREEN))
        else:
            legacy = getattr(args, "legacy_prompt", None) or []
            sub_prompt = getattr(args, "prompt", None) if getattr(args, "video_cmd", None) == "generate" else None
            prompt_parts = sub_prompt if sub_prompt else legacy
            prompt = " ".join(prompt_parts) if prompt_parts else ""
            if not prompt and getattr(args, "video_cmd", None) != "batch":
                print(c("Usage: python -m navine.cli video \"your prompt\"  OR  python -m navine.cli video batch --count 16", Colors.YELLOW))
                return
            if getattr(args, "hentai", False) and "hentai" not in prompt.lower():
                prompt = f"hentai anime style motion, {prompt}".strip(", ")
            elif getattr(args, "porn", False) and "porn" not in prompt.lower():
                prompt = f"photorealistic explicit adult motion, {prompt}".strip(", ")
            elif getattr(args, "real", False) and "photoreal" not in prompt.lower():
                prompt = f"photorealistic motion, {prompt}".strip(", ")
            use_external = bool(getattr(args, "external", False))
            if use_external:
                from navine.utils.generation import require_external_teacher
                from navine.video.external import generate_external as generate_external_video

                require_external_teacher()
                path = generate_external_video(
                    prompt,
                    output_path=args.output,
                    num_frames=args.frames,
                    fps=args.fps,
                    seed=args.seed,
                )
                print(c(f"Navine AI - Python teacher video saved for training: {path}", Colors.GREEN))
                print(c("Teacher outputs are for training only, not production inference.", Colors.DIM))
            else:
                from navine.video.infer import generate

                path = generate(
                    prompt,
                    output_path=args.output,
                    num_frames=args.frames,
                    fps=args.fps,
                    seed=args.seed,
                    with_audio=bool(getattr(args, "audio", False)),
                    narration=getattr(args, "narration", None),
                    voice=getattr(args, "voice", None),
                )
                print(c(f"Navine AI - Python video saved to: {path}", Colors.GREEN))

    elif args.command == "deepfake":
        enable_colors()
        if args.deepfake_cmd == "face":
            from navine.deepfake import swap_face_files

            path = swap_face_files(args.source, args.target, output_path=args.output, strength=args.strength)
            print(c(f"Deepfake face saved to: {path}", Colors.GREEN))
        elif args.deepfake_cmd == "video":
            from navine.deepfake import deepfake_video

            path = deepfake_video(
                args.source,
                target_video=args.target_video,
                prompt=args.prompt,
                output_path=args.output,
                num_frames=args.frames,
                fps=args.fps,
                strength=args.strength,
                seed=args.seed,
                with_audio=bool(args.audio),
                narration=args.narration,
            )
            print(c(f"Deepfake video saved to: {path}", Colors.GREEN))
        elif args.deepfake_cmd == "realtime":
            from navine.deepfake import realtime_deepfake_loop

            path = realtime_deepfake_loop(
                args.source,
                seconds=args.seconds,
                fps=args.fps,
                output_path=args.output,
                strength=args.strength,
            )
            print(c(f"Realtime deepfake saved to: {path}", Colors.GREEN))

    elif args.command == "engine":
        enable_colors()
        from navine.engine import get_engine
        from navine.engine.__main__ import main as engine_main

        cmd = getattr(args, "engine_cmd", None) or "probe"
        if cmd == "run":
            argv = ["run", args.name, *list(args.prompt)]
            if args.output:
                argv.extend(["--output", args.output])
            if args.max_tokens is not None:
                argv.extend(["--max-tokens", str(args.max_tokens)])
            if args.temperature is not None:
                argv.extend(["--temperature", str(args.temperature)])
            if args.seed is not None:
                argv.extend(["--seed", str(args.seed)])
            raise SystemExit(engine_main(argv))
        raise SystemExit(engine_main([cmd]))

    elif args.command == "detective":
        enable_colors()
        from navine.detective.session import solve_mystery

        prompt = " ".join(args.prompt).strip()
        if not prompt:
            print(c("Provide a mystery prompt.", Colors.RED))
            sys.exit(2)
        result = solve_mystery(
            prompt,
            output_dir=args.output,
            seed=int(args.seed or 0),
            include_image=not bool(args.no_image),
            include_video=not bool(args.no_video),
        )
        print(c("Detective analysis", Colors.GREEN))
        print(result.get("analysis") or "")
        if result.get("image"):
            print(c(f"Clue image: {result.get('image')}", Colors.DIM))
        if result.get("video"):
            print(c(f"Mystery clip: {result.get('video')}", Colors.DIM))
        if result.get("manifest"):
            print(c(f"Session: {result.get('manifest')}", Colors.DIM))

    elif args.command == "osint":
        enable_colors()
        from navine.osint import investigate

        target = " ".join(args.target).strip()
        if not target:
            print(c("Provide an OSINT target.", Colors.RED))
            sys.exit(2)
        result = investigate(
            target,
            kind=str(args.kind or "auto"),
            use_search=not bool(args.no_search),
            output_dir=args.output,
        )
        print(c("OSINT report", Colors.GREEN))
        print(result.get("analysis") or "")
        if result.get("search_results"):
            print(c("Sources:", Colors.DIM))
            for row in result.get("search_results") or []:
                title = str(row.get("title") or "Result")
                url = str(row.get("url") or "")
                if url:
                    print(c(f"  - {title}: {url}", Colors.DIM))
        if result.get("report_path"):
            print(c(f"Saved: {result.get('report_path')}", Colors.DIM))

    elif args.command == "llm":
        enable_colors()
        from navine.llm import list_llms, main_for, pack_all, verify_all

        cmd = getattr(args, "llm_cmd", None) or "list"
        if cmd == "pack":
            report = pack_all()
            print(c("Packed runnable Navine AI - Python LLM files", Colors.GREEN))
            for row in report.get("models") or []:
                status = "ok" if row.get("ok") else row.get("error")
                print(f"  {row.get('name')}: {status}")
                if row.get("llm_file"):
                    print(c(f"    {row.get('llm_file')}", Colors.DIM))
            if report.get("gguf"):
                print(c("GGUF files", Colors.GREEN))
                for row in (report.get("gguf") or {}).get("models") or []:
                    if row.get("ok"):
                        print(c(f"    {row.get('gguf')}", Colors.DIM))
                    else:
                        print(f"    {row.get('name')}: {row.get('error')}")
            elif report.get("gguf_error"):
                print(c(f"GGUF export failed: {report.get('gguf_error')}", Colors.RED))
        elif cmd == "verify":
            report = verify_all()
            print(json.dumps(report, indent=2, default=str))
            if not report.get("ok"):
                sys.exit(1)
        elif cmd == "gguf":
            from navine.gguf_export import main as gguf_main

            raise SystemExit(gguf_main(["export"]))
        elif cmd == "run":
            from navine.llm import main_for

            extra = []
            if args.output:
                extra.extend(["--output", args.output])
            if args.max_tokens is not None:
                extra.extend(["--max-tokens", str(args.max_tokens)])
            if args.temperature is not None:
                extra.extend(["--temperature", str(args.temperature)])
            if args.seed is not None:
                extra.extend(["--seed", str(args.seed)])
            raise SystemExit(main_for(args.name, list(args.prompt) + extra))
        else:
            for row in list_llms():
                flag = "ready" if row.get("runnable") else "missing"
                print(f"  {row['id']:20} {flag:8} {row.get('llm_file') or ''}")

    elif args.command == "code":
        enable_colors()
        from navine.code.generate import generate_code
        task = " ".join(args.task)
        code, lang = generate_code(task, language=args.language)
        print(c(f"Language: {lang}", Colors.DIM))
        print(code)

    elif args.command == "lora":
        enable_colors()
        cmd = args.lora_cmd
        modality = args.modality
        if cmd == "train":
            if modality == "image":
                from navine.image.lora_custom import train_custom_lora
                report = train_custom_lora(steps=args.steps)
            else:
                from navine.video.lora_custom import train_video_lora
                report = train_video_lora(steps=args.steps)
            print(c(f"LoRA train complete: {report}", Colors.GREEN))
        elif cmd == "merge":
            force = bool(getattr(args, "force", False))
            from navine.lora.cycle import merge_image_lora, merge_video_lora
            if modality == "image":
                result = merge_image_lora(force_merge=force)
            else:
                result = merge_video_lora(force_merge=force)
            if result.get("merged") and result.get("adapter_removed"):
                print(c(f"LoRA merged into base and adapter removed. score={result.get('post_score')}", Colors.GREEN))
            else:
                print(c(f"Merge skipped or failed: {result}", Colors.YELLOW))
        elif cmd == "cycle":
            from navine.lora.cycle import run_lora_cycle
            result = run_lora_cycle(
                modality,
                steps=args.steps,
                force_merge=bool(getattr(args, "force", False)),
            )
            if result.get("merged") and result.get("adapter_removed"):
                print(c(f"LoRA merged into base and adapter removed. score={result.get('post_score')}", Colors.GREEN))
            elif result.get("ok"):
                print(c("LoRA trained; adapter kept (not good enough to merge yet).", Colors.YELLOW))
            else:
                print(c(f"LoRA cycle result: {result}", Colors.YELLOW))

    elif args.command == "train":
        enable_colors()
        target = args.target.lower()
        if target == "list":
            from navine.train.registry import list_training_types
            print(c("Navine AI - Python Training Types", Colors.BOLD + Colors.CYAN))
            print(c("-" * 50, Colors.DIM))
            for row in list_training_types():
                print(f"  {c(row['name'], Colors.GREEN)}")
                print(f"    {row['description']}")
                print(f"    Data: {row['data_format']}")
            print()
            print(c("Modality / system training:", Colors.DIM))
            print("  text, image, video, voice, image-lora, video-lora, full")
            print()
            print(c("Commands:", Colors.DIM))
            print("  python -m navine.cli train coding [--language python] [--steps N]")
            print("  python -m navine.cli train all")
            print("  python -m navine.cli train voice")
            print("  python -m navine.cli train full [--steps N]")
        elif target == "all":
            from navine.train.registry import train_all
            train_all(language=args.language, steps=args.steps)
        elif target == "full":
            from navine.train.full import train_full

            train_full(steps=args.steps, language=args.language)
        elif target in ("text", "image", "video", "voice"):
            if target == "text":
                from navine.text.train import train
                from navine.utils.tier import modality_config_name

                cfg_name = getattr(args, "config", None) or modality_config_name("text")
                train(
                    config_path=cfg_name,
                    max_steps=args.steps,
                    fresh=bool(getattr(args, "fresh", False)),
                    require_cuda=bool(getattr(args, "require_cuda", False)),
                )
            elif target == "image":
                try:
                    from navine.learn.image_learn import ingest_learned_images, ingest_nsfw_images

                    ingest_learned_images()
                    ingest_nsfw_images()
                except Exception:
                    pass
                from navine.image.train import train
                from navine.utils.tier import modality_config_name

                train(
                    config_path=modality_config_name("image"),
                    finetune=True,
                    finetune_steps=args.steps,
                    require_cuda=bool(getattr(args, "require_cuda", False)),
                )
            elif target == "video":
                try:
                    from navine.learn.video_learn import ingest_learned_video

                    ingest_learned_video()
                except Exception:
                    pass
                from navine.video.train import train
                from navine.utils.tier import modality_config_name

                train(
                    config_path=modality_config_name("video"),
                    finetune=True,
                    finetune_steps=args.steps,
                    require_cuda=bool(getattr(args, "require_cuda", False)),
                )
            elif target == "voice":
                from navine.voice.train import train_voice
                train_voice(steps=args.steps, language=args.language)
        elif target in ("image-lora", "image_lora"):
            from navine.image.lora_custom import train_custom_lora
            train_custom_lora(steps=args.steps)
        elif target in ("video-lora", "video_lora"):
            from navine.video.lora_custom import train_video_lora
            train_video_lora(steps=args.steps)
        elif target == "text-code":
            from navine.train.coding import train as train_coding
            from navine.utils.paths import get_checkpoint_dir
            from navine.train.base import finetune_texts
            from navine.train.coding import load_coding_texts

            import shutil

            code_dir = get_checkpoint_dir("text_code")
            ent = get_checkpoint_dir("text_enterprise")
            code_dir.mkdir(parents=True, exist_ok=True)
            for name in ("latest.pt", "best.pt", "tokenizer.json"):
                src = ent / name
                if src.exists():
                    shutil.copy2(src, code_dir / name)
                    print(c(f"Seeded text_code/{name} from text_enterprise", Colors.DIM))
            texts = load_coding_texts(language=args.language)
            finetune_texts(
                "coding",
                texts,
                desc="Navine AI - Python Text-Code specialist",
                max_steps=args.steps or 600,
                checkpoint_dir=code_dir,
                require_cuda=bool(getattr(args, "require_cuda", False)),
            )
        else:
            from navine.train.registry import TRAINING_TYPES, train_type
            if target not in TRAINING_TYPES:
                print(c(f"Unknown training target: {target}", Colors.RED))
                print(c("Run: python -m navine.cli train list", Colors.DIM))
                sys.exit(1)
            train_type(target, language=args.language, steps=args.steps)

    elif args.command == "learn":
        enable_colors()
        if args.learn_cmd == "url":
            from navine.learn.web import fetch_url, save_fetched
            from navine.learn.rag import index_document
            doc = fetch_url(args.url)
            path = save_fetched(doc)
            index_document(doc["url"], doc["title"], doc["text"])
            print(c(f"Saved and indexed: {path}", Colors.GREEN))
        elif args.learn_cmd == "crawl":
            from navine.learn.crawl import crawl
            from navine.learn.rag import rebuild_index
            path = crawl(args.url, max_pages=args.max_pages)
            count = rebuild_index()
            print(c(f"Crawled to {path} | RAG index: {count} docs", Colors.GREEN))
        elif args.learn_cmd == "ingest":
            from navine.learn.datasets import ingest_learned_data
            from navine.learn.rag import rebuild_index
            path = ingest_learned_data()
            count = rebuild_index()
            print(c(f"Ingested to {path} | RAG index: {count} docs", Colors.GREEN))
        elif args.learn_cmd == "index":
            from navine.learn.rag import rebuild_index
            count = rebuild_index()
            print(c(f"RAG index rebuilt: {count} documents", Colors.GREEN))
        elif args.learn_cmd == "conversations":
            from navine.memory.ingest import ingest_conversations
            result = ingest_conversations(rebuild_rag=True)
            print(c("Navine AI - Python conversation ingest complete", Colors.GREEN))
            print(f"  Sessions scanned: {result.get('sessions_scanned', 0)}")
            print(f"  New Q&A pairs: {result.get('new_qa_pairs', 0)}")
            print(f"  Training blocks: {result.get('training_blocks', 0)}")
            print(f"  RAG documents: {result.get('rag_documents', 0)}")
        elif args.learn_cmd == "datasets":
            from navine.learn.datasets import download_datasets
            paths = download_datasets()
            for p in paths:
                print(c(f"Downloaded: {p}", Colors.GREEN))
        elif args.learn_cmd == "multilingual":
            from navine.text.multilingual import fetch_multilingual_corpus

            langs = None
            if getattr(args, "languages", None):
                langs = [x.strip() for x in str(args.languages).split(",") if x.strip()]
            path = fetch_multilingual_corpus(
                languages=langs,
                rows_per_lang=int(getattr(args, "rows_per_lang", 150) or 150),
            )
            print(c(f"Multilingual data ready: {path}", Colors.GREEN))
            print(c("Retrain tokenizer/text so new languages are learned.", Colors.DIM))
        elif args.learn_cmd == "train":
            from navine.text.finetune_chat import finetune
            finetune(config_path=args.config)
        elif args.learn_cmd == "coding":
            from navine.train.learn_coding import download_coding_samples, fetch_remote_coding_samples

            include_remote = not bool(getattr(args, "no_remote", False))
            remote_max = int(getattr(args, "remote_max", 20) or 20)
            if include_remote:
                print(c("Fetching code from GitHub, GitLab, Codeberg, StackOverflow, HN, docs...", Colors.CYAN))
                remote_stats = fetch_remote_coding_samples(
                    language=args.language,
                    max_per_source=remote_max,
                )
                print(
                    c(
                        f"Remote coding: items={remote_stats.get('items', 0)} "
                        f"new_entries={remote_stats.get('entries', 0)} files={remote_stats.get('files', 0)}",
                        Colors.GREEN,
                    )
                )
            paths = download_coding_samples(
                language=args.language,
                max_count=args.max,
                include_remote=False,
            )
            if not paths:
                print(c("No coding samples downloaded.", Colors.YELLOW))
            for p in paths:
                print(c(f"Saved coding data: {p}", Colors.GREEN))
        elif args.learn_cmd == "hf-code":
            from navine.learn.hf_code import fetch_hf_code_datasets, list_hf_code_datasets

            if getattr(args, "list", False):
                print(c("Hugging Face / open coding datasets:", Colors.BOLD + Colors.CYAN))
                for row in list_hf_code_datasets():
                    print(f"  {row.get('alias'):22}  {row.get('id')}  max={row.get('max')}  {row.get('note')}")
                print(c("Use: python -m navine.cli learn hf-code --max 1500", Colors.DIM))
                return
            only = None
            if getattr(args, "only", None):
                only = [p.strip() for p in str(args.only).split(",") if p.strip()]
            print(c("Fetching HF/open code datasets (data only, not foundation models)...", Colors.CYAN))
            report = fetch_hf_code_datasets(
                max_per_dataset=getattr(args, "max", 1500),
                only=only,
                use_datasets_lib=not bool(getattr(args, "no_datasets_lib", False)),
            )
            print(c(f"Total new coding rows: {report.get('total_added', 0)}", Colors.GREEN))
            for name, info in (report.get("datasets") or {}).items():
                status = "ok" if not info.get("error") else f"err: {info.get('error')}"
                print(
                    f"  {name}: fetched={info.get('fetched')} added={info.get('added')} ({status})"
                )
            if report.get("catalog"):
                print(c(f"Catalog: {report['catalog']}", Colors.CYAN))
            print(c("Train next: python -m navine.cli train coding --steps 200", Colors.DIM))
        elif args.learn_cmd == "hf-pack":
            from navine.learn.hf_fetch import fetch_hf_training_pack

            print(c("Searching Hugging Face and downloading code + chat datasets...", Colors.CYAN))
            report = fetch_hf_training_pack(
                code_max=int(getattr(args, "code_max", 800) or 800),
                chat_max=int(getattr(args, "chat_max", 1500) or 1500),
            )
            print(c(f"HF search catalog: {report.get('search', {}).get('catalog_path')}", Colors.GREEN))
            print(c(f"Code rows added: {report.get('code_added', 0)}", Colors.GREEN))
            print(c(f"Chat pairs added: {report.get('chat_pairs', 0)}", Colors.GREEN))
            print(c(f"Report: {report.get('report_path')}", Colors.CYAN))
            print(c("Train next:", Colors.DIM))
            print(c("  python -m navine.cli train text-code --steps 500 --require-cuda", Colors.DIM))
            print(c("  python -m navine.cli train chat --steps 300 --require-cuda", Colors.DIM))
        elif args.learn_cmd == "github":
            from navine.autolearn.engine import AutolearnEngine
            from navine.autolearn.sources.github import fetch_github

            items = fetch_github(
                language=args.language,
                max_repos=args.max_repos,
                max_files=args.max_files,
            )
            if not items:
                print(c("No GitHub samples downloaded.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python GitHub: fetched {len(items)} items", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "reddit":
            from navine.autolearn.engine import AutolearnEngine
            from navine.autolearn.sources.reddit import fetch_reddit

            items = fetch_reddit(args.subreddit, max_posts=args.max)
            if not items:
                print(c("No Reddit posts downloaded.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python Reddit r/{args.subreddit}: fetched {len(items)} posts", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "hn":
            from navine.autolearn.engine import AutolearnEngine
            from navine.autolearn.sources.hackernews import fetch_hackernews

            items = fetch_hackernews(max_items=args.max)
            if not items:
                print(c("No Hacker News stories downloaded.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python Hacker News: fetched {len(items)} stories", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "all-sources":
            from navine.autolearn.engine import AutolearnEngine, run_all_sources_once

            items = run_all_sources_once()
            if not items:
                print(c("No samples downloaded from autolearn sources.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python all-sources: fetched {len(items)} items", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "everything":
            if getattr(args, "marathon", False):
                from navine.autolearn.marathon import MarathonLearner

                learner = MarathonLearner()
                learner.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
                result = learner.run(
                    hours=args.hours,
                    forever=getattr(args, "forever", False),
                    resume=getattr(args, "resume", False),
                )
                print(c("Navine AI - Python Learn Everything Marathon finished", Colors.BOLD + Colors.CYAN))
                print(f"  Reason: {result.get('reason')}")
                print(f"  Elapsed: {result.get('elapsed_hours')}h")
                print(f"  Cycles: {result.get('cycles')}")
                print(f"  Fetched: {result.get('items_fetched')}")
                print(f"  Ingested: {result.get('items_ingested')}")
            else:
                from navine.autolearn.engine import AutolearnEngine

                engine = AutolearnEngine()
                engine.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
                items = engine.fetch_everything()
                if not items:
                    print(c("No samples downloaded.", Colors.YELLOW))
                else:
                    result = engine.process_items(items)
                    print(c(f"Navine AI - Python learn everything: fetched {len(items)} items", Colors.GREEN))
                    print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
                    if result.get("by_category"):
                        print(f"  By category: {result['by_category']}")
        elif args.learn_cmd == "coding-all":
            from navine.autolearn.engine import AutolearnEngine
            from navine.learn.hf_code import fetch_hf_code_datasets
            from navine.train.learn_coding import download_coding_samples, fetch_remote_coding_samples

            print(c("Fetching GitHub, GitLab, Codeberg, StackOverflow, docs, raw sources...", Colors.CYAN))
            remote_stats = fetch_remote_coding_samples(max_per_source=25)
            print(
                c(
                    f"Remote coding: items={remote_stats.get('items', 0)} "
                    f"new_entries={remote_stats.get('entries', 0)}",
                    Colors.GREEN,
                )
            )
            paths = download_coding_samples(include_remote=False)
            for p in paths:
                print(c(f"Saved coding data: {p}", Colors.GREEN))
            engine = AutolearnEngine()
            engine.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
            items = engine.fetch_coding_all()
            if items:
                result = engine.process_items(items)
                print(c(f"Autolearn coding-all: fetched {len(items)} items", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
            print(c("Fetching Hugging Face open code datasets...", Colors.CYAN))
            hf_report = fetch_hf_code_datasets(max_per_dataset=1000)
            print(c(f"HF code rows added: {hf_report.get('total_added', 0)}", Colors.GREEN))
        elif args.learn_cmd == "external":
            from navine.learn.from_external import (
                batch_generate,
                external_status,
                finetune_from_external,
                generate_teaching_image,
                generate_teaching_video,
                ingest_external_for_training,
            )

            action = getattr(args, "external_action", None)
            if action == "disable":
                from navine.utils.generation import disable_external_teacher

                disable_external_teacher()
                print(c("External teacher models disabled. Inference already uses local trained weights only.", Colors.GREEN))
                print(c("Re-enable later with generation.external_teacher_enabled: true in configs/navine.yaml", Colors.DIM))
            elif action == "status":
                from navine.utils.generation import external_teacher_enabled

                status = external_status()
                print(c("Navine AI - Python External Diffusion (Teacher) Status", Colors.BOLD + Colors.CYAN))
                print(f"  teacher_enabled: {external_teacher_enabled()}")
                print(f"  inference_uses_external: false")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                if not status.get("diffusers_installed"):
                    print(c("  Install: pip install diffusers transformers accelerate safetensors", Colors.YELLOW))
            elif action == "ingest":
                from navine.utils.generation import require_external_teacher

                require_external_teacher()
                path = ingest_external_for_training()
                print(c(f"External teacher data ingested to: {path}", Colors.GREEN))
                print(c("Primary custom training reads data/learn/image (already in image_enterprise extra_dirs)", Colors.DIM))
            elif action == "train":
                from navine.utils.generation import require_external_teacher

                require_external_teacher()
                finetune_from_external(finetune_steps=args.steps)
                print(c("Navine AI - Python image_enterprise finetune from external teacher data complete.", Colors.GREEN))
            elif args.prompt:
                from navine.utils.generation import require_external_teacher

                require_external_teacher()
                path = generate_teaching_image(args.prompt, seed=args.seed, model=args.model)
                print(c(f"External teacher image saved to: {path}", Colors.GREEN))
                if args.video:
                    vpath = generate_teaching_video(args.prompt, seed=args.seed, model=args.model)
                    print(c(f"External teacher video saved to: {vpath}", Colors.GREEN))
            else:
                from navine.utils.generation import require_external_teacher

                require_external_teacher()
                result = batch_generate(
                    count=args.count,
                    include_video=bool(args.video),
                    model=args.model,
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
                print(
                    c(
                        f"External teacher batch complete: {result['images']} image(s)"
                        + (f", {result['videos']} video(s)" if args.video else ""),
                        Colors.GREEN,
                    )
                )
                print(c("Next: python -m navine.cli learn external ingest", Colors.DIM))
                print(c("Train: python -m navine.cli learn external train --steps 400", Colors.DIM))
        elif args.learn_cmd == "teacher-loop":
            from navine.utils.generation import require_external_teacher
            from navine.learn.teacher_loop import run_teacher_loop

            require_external_teacher()
            report = run_teacher_loop(
                count=args.count,
                max_cycles=args.max_cycles,
                model=args.model,
                include_video=bool(args.video),
                finetune_steps=args.steps,
                score_gap_threshold=args.score_gap,
                lap_gap_threshold=args.lap_gap,
                progress=lambda msg: print(c(msg, Colors.DIM)),
            )
            stopped = report.get("stopped", "unknown")
            cycles = len(report.get("cycles", []))
            print(c(f"Navine AI - Python teacher loop finished: {stopped} after {cycles} cycle(s)", Colors.GREEN))
            print(c(f"Report: logs/teacher_loop/teacher_loop_report.json", Colors.DIM))
        elif args.learn_cmd == "lora":
            action = getattr(args, "lora_action", "status")
            if action == "status":
                from navine.learn.lora_sd import lora_status

                status = lora_status()
                print(c("Navine AI - Python LoRA (Stable Diffusion) Status", Colors.BOLD + Colors.CYAN))
                for key, value in status.items():
                    print(f"  {key}: {value}")
                if not status.get("dependencies_installed"):
                    print(c("  Install: pip install -r requirements-external.txt peft", Colors.YELLOW))
            elif action == "train":
                from navine.learn.lora_sd import train_lora

                print(c("Navine AI - Python LoRA training on open-source Stable Diffusion", Colors.BOLD + Colors.CYAN))
                report = train_lora(max_steps=args.steps, progress=lambda msg: print(c(msg, Colors.DIM)))
                print(c(f"LoRA training complete: {report['steps']} steps on {report['images']} images", Colors.GREEN))
                print(c(f"Adapter: {report['output_dir']}", Colors.DIM))
                print(c("Generate: python -m navine.cli learn lora generate --prompt \"...\"", Colors.DIM))
            elif action == "generate":
                from navine.image.lora_infer import generate_with_lora

                if not args.prompt:
                    print(c("Provide --prompt for lora generate.", Colors.RED))
                else:
                    path = generate_with_lora(
                        args.prompt,
                        output_path=args.output,
                        seed=args.seed,
                        lora_scale=args.lora_scale,
                    )
                    print(c(f"LoRA image saved to: {path}", Colors.GREEN))
        elif args.learn_cmd == "wikipedia":
            from navine.autolearn.engine import AutolearnEngine
            from navine.autolearn.sources.wikipedia import fetch_wikipedia

            items = fetch_wikipedia(max_items=args.max, category=args.category)
            if not items:
                print(c("No Wikipedia articles downloaded.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python Wikipedia: fetched {len(items)} articles", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "arxiv":
            from navine.autolearn.engine import AutolearnEngine
            from navine.autolearn.sources.arxiv import fetch_arxiv

            items = fetch_arxiv(max_items=args.max)
            if not items:
                print(c("No arXiv abstracts downloaded.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python arXiv: fetched {len(items)} papers", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "docs":
            from navine.autolearn.engine import AutolearnEngine
            from navine.autolearn.sources.docs import crawl_docs

            items = crawl_docs(topic=args.topic, max_pages=args.max_pages)
            if not items:
                print(c("No documentation pages downloaded.", Colors.YELLOW))
            else:
                engine = AutolearnEngine()
                result = engine.process_items(items)
                print(c(f"Navine AI - Python docs ({args.topic}): fetched {len(items)} pages", Colors.GREEN))
                print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
        elif args.learn_cmd == "image":
            if args.image_cmd == "url":
                from navine.learn.image_learn import download_image_url
                paths = download_image_url(args.url, caption=args.caption)
                for p in paths:
                    print(c(f"Saved image: {p}", Colors.GREEN))
            elif args.image_cmd == "crawl":
                from navine.learn.image_learn import crawl_images
                paths = crawl_images(args.page_url, max_images=args.max_images)
                for p in paths:
                    print(c(f"Saved image: {p}", Colors.GREEN))
            elif args.image_cmd == "search":
                from navine.learn.image_learn import search_images
                query = " ".join(args.query)
                paths = search_images(query, max_count=args.max)
                for p in paths:
                    print(c(f"Saved image: {p}", Colors.GREEN))
            elif args.image_cmd == "ingest":
                from navine.learn.image_learn import ingest_learned_images
                path = ingest_learned_images()
                print(c(f"Ingested images to: {path}", Colors.GREEN))
            elif args.image_cmd == "train":
                from navine.learn.image_learn import train_learned_image
                train_learned_image(finetune_steps=args.steps)
                print(c("Image fine-tuning complete.", Colors.GREEN))
        elif args.learn_cmd == "video":
            if args.video_cmd == "url":
                from navine.learn.video_learn import download_video_url
                path = download_video_url(args.url)
                print(c(f"Saved video frames to: {path}", Colors.GREEN))
            elif args.video_cmd == "gif":
                from navine.learn.video_learn import download_gif_url
                path = download_gif_url(args.url)
                print(c(f"Saved GIF frames to: {path}", Colors.GREEN))
            elif args.video_cmd == "ingest":
                from navine.learn.video_learn import ingest_learned_video
                path = ingest_learned_video()
                print(c(f"Ingested video frames to: {path}", Colors.GREEN))
            elif args.video_cmd == "train":
                from navine.learn.video_learn import train_learned_video
                train_learned_video(finetune_steps=args.steps)
                print(c("Video fine-tuning complete.", Colors.GREEN))
            elif args.video_cmd == "study":
                from navine.learn.video_learn import study_youtube_learned
                result = study_youtube_learned(reindex_rag=True)
                print(c(
                    f"Studied {result.get('videos_studied', 0)} YouTube videos "
                    f"({result.get('helpful_count', 0)} helpful).",
                    Colors.GREEN,
                ))
        elif args.learn_cmd == "jax":
            from navine.learn.jax_learn import demo_jax, install_jax, jax_status, study_jax, ingest_jax_learning

            if args.jax_cmd == "status":
                print(json.dumps(jax_status(), indent=2))
            elif args.jax_cmd == "install":
                print(json.dumps(install_jax(), indent=2))
            elif args.jax_cmd == "demo":
                print(json.dumps(demo_jax(), indent=2))
            elif args.jax_cmd == "ingest":
                print(json.dumps(ingest_jax_learning(), indent=2))
            elif args.jax_cmd == "study":
                print(json.dumps(study_jax(reindex=True), indent=2))
        elif args.learn_cmd == "nsfw":
            from navine.nsfw.config import load_nsfw_config, nsfw_config_path

            if args.nsfw_cmd == "enable":
                import yaml

                path = nsfw_config_path()
                cfg = load_nsfw_config()
                cfg["enabled"] = True
                cfg["unrestricted_mode"] = True
                cfg["auto_learn"] = True
                cfg["include_videos"] = True
                cfg["include_unrestricted"] = True
                cfg["auto_train_image"] = True
                cfg["auto_train_video"] = True
                cfg["auto_train_unrestricted"] = True
                with open(path, "w", encoding="utf-8") as handle:
                    yaml.dump(cfg, handle, default_flow_style=False)
                navine_path = path.parent / "navine.yaml"
                navine_cfg = {"local_only": True, "unrestricted_mode": True}
                if navine_path.exists():
                    with open(navine_path, "r", encoding="utf-8") as handle:
                        navine_cfg.update(yaml.safe_load(handle) or {})
                    navine_cfg["unrestricted_mode"] = True
                with open(navine_path, "w", encoding="utf-8") as handle:
                    yaml.dump(navine_cfg, handle, default_flow_style=False)
                from navine.nsfw.config import reload_nsfw_config

                reload_nsfw_config()
                print(c("Navine AI - Python NSFW learning and unrestricted mode enabled.", Colors.GREEN))
                print(c(f"Edit subreddits and paths in: {path}", Colors.DIM))
            elif args.nsfw_cmd == "config":
                cfg = load_nsfw_config()
                from navine.nsfw.config import configured_internet_sources

                print(c(f"Config: {nsfw_config_path()}", Colors.CYAN))
                for key in (
                    "enabled", "unrestricted_mode", "auto_learn", "mixed_mode",
                    "reddit_hentai_subreddits", "reddit_real_subreddits",
                    "reddit_human_subreddits", "reddit_mixed_subreddits",
                    "reddit_hentai_video_subreddits", "reddit_real_video_subreddits",
                    "reddit_subreddits", "reddit_image_subreddits",
                    "reddit_video_subreddits", "reddit_post_urls",
                    "web_urls", "crawl_start_urls", "direct_media_urls",
                    "feed_urls", "image_search_sources", "video_search_sources",
                    "fourchan", "waifu_im", "infini_atomic", "media_mix", "unrestricted_text_topics",
                    "allowed_learning_domains",
                    "local_dirs",
                ):
                    print(f"  {key}: {cfg.get(key)}")
                print(f"  internet_source_counts: {configured_internet_sources(cfg)}")
            elif args.nsfw_cmd == "local":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import (
                    download_nsfw_images,
                    download_nsfw_videos,
                    fetch_local,
                )

                cfg = load_nsfw_config()
                items = fetch_local(local_path=args.path, config=cfg)
                if not items:
                    print(c("No local NSFW files found.", Colors.YELLOW))
                else:
                    engine = AutolearnEngine()
                    result = engine.process_items(items)
                    img_count = 0
                    vid_count = 0
                    if cfg.get("include_images", True):
                        img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    if cfg.get("include_videos", True):
                        vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    category_counts = {}
                    for item in items:
                        cat = item.get("media_category") or item.get("topic") or "general"
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                    print(c(f"Navine AI - Python NSFW local: ingested {result['new_samples']} samples", Colors.GREEN))
                    print(c(f"Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                    if category_counts:
                        print(c(f"By category/topic: {category_counts}", Colors.DIM))
                    if cfg.get("train_on_collect"):
                        from navine.train.registry import train_type

                        train_type("nsfw")
                        if cfg.get("auto_train_image", True) and img_count:
                            from navine.learn.image_learn import train_learned_image

                            train_learned_image(finetune_steps=cfg.get("image_train_steps"))
                        if cfg.get("auto_train_video", True) and vid_count:
                            from navine.learn.video_learn import train_learned_video

                            train_learned_video(finetune_steps=cfg.get("video_train_steps"))
            elif args.nsfw_cmd == "reddit":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import fetch_reddit_nsfw

                cfg = load_nsfw_config()
                items = fetch_reddit_nsfw(subreddit=args.subreddit, max_posts=args.max, config=cfg)
                if not items:
                    print(c("No NSFW Reddit posts fetched. Add subreddits to configs/nsfw.yaml", Colors.YELLOW))
                else:
                    engine = AutolearnEngine()
                    result = engine.process_items(items)
                    print(c(f"Navine AI - Python NSFW Reddit: fetched {len(items)} items", Colors.GREEN))
                    print(c(f"Ingested {result['new_samples']} new samples", Colors.GREEN))
                    if cfg.get("train_on_collect"):
                        from navine.train.registry import train_type

                        train_type("nsfw")
            elif args.nsfw_cmd == "run":
                from navine.nsfw.autolearn import run_nsfw_cycle

                cfg = load_nsfw_config()
                result = run_nsfw_cycle(progress=lambda msg: print(c(msg, Colors.DIM)))
                print(c(f"Navine AI - Python NSFW: {result.get('new_samples', 0)} text samples", Colors.GREEN))
                print(c(f"Images: {result.get('images_downloaded', 0)} | Videos: {result.get('videos_downloaded', 0)}", Colors.GREEN))
            elif args.nsfw_cmd == "autolearn":
                from navine.nsfw.autolearn import run_nsfw_cycle, run_nsfw_loop

                cfg = load_nsfw_config()
                if args.hours:
                    print(c(f"Navine AI - Python NSFW autolearn loop for {args.hours} hours...", Colors.CYAN))
                    run_nsfw_loop(hours=args.hours, progress=lambda msg: print(c(msg, Colors.DIM)))
                else:
                    result = run_nsfw_cycle(progress=lambda msg: print(c(msg, Colors.DIM)))
                    print(
                        c(
                            f"Done: {result.get('new_samples', 0)} new samples "
                            f"({result.get('items_fetched', 0)} fetched), "
                            f"{result.get('images_downloaded', 0)} images, "
                            f"{result.get('videos_downloaded', 0)} videos",
                            Colors.GREEN,
                        )
                    )
                    for warning in result.get("warnings") or []:
                        print(c(f"Warning: {warning}", Colors.YELLOW))
            elif args.nsfw_cmd == "status":
                from navine.nsfw.autolearn import get_nsfw_autolearn_status

                status = get_nsfw_autolearn_status()
                for key, value in status.items():
                    print(f"  {key}: {value}")
            elif args.nsfw_cmd == "images":
                from navine.autolearn.sources.nsfw import fetch_reddit_nsfw, download_nsfw_images

                cfg = load_nsfw_config()
                subs = cfg.get("reddit_image_subreddits") or cfg.get("reddit_subreddits") or []
                items = []
                for sub in subs:
                    items.extend(fetch_reddit_nsfw(subreddit=sub, config=cfg))
                count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                if cfg.get("auto_train_image", True) and count:
                    from navine.learn.image_learn import train_learned_image

                    train_learned_image(finetune_steps=cfg.get("image_train_steps"))
                print(c(f"NSFW images downloaded: {count}", Colors.GREEN))
            elif args.nsfw_cmd == "videos":
                from navine.autolearn.sources.nsfw import fetch_reddit_nsfw, download_nsfw_videos

                cfg = load_nsfw_config()
                subs = cfg.get("reddit_video_subreddits") or cfg.get("reddit_subreddits") or []
                items = []
                for sub in subs:
                    items.extend(fetch_reddit_nsfw(subreddit=sub, config=cfg))
                count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                if cfg.get("auto_train_video", True) and count:
                    from navine.learn.video_learn import train_learned_video

                    train_learned_video(finetune_steps=cfg.get("video_train_steps"))
                print(c(f"NSFW videos downloaded: {count}", Colors.GREEN))
            elif args.nsfw_cmd == "mixed":
                from navine.nsfw.autolearn import run_nsfw_cycle

                result = run_nsfw_cycle(progress=lambda msg: print(c(msg, Colors.DIM)))
                print(
                    c(
                        f"Mixed NSFW: {result.get('new_samples', 0)} text samples, "
                        f"{result.get('images_downloaded', 0)} images, "
                        f"{result.get('videos_downloaded', 0)} videos",
                        Colors.GREEN,
                    )
                )
                counts = result.get("media_category_counts") or {}
                if counts:
                    print(c(f"Media mix: {counts}", Colors.DIM))
                for warning in result.get("warnings") or []:
                    print(c(f"Warning: {warning}", Colors.YELLOW))
            elif args.nsfw_cmd == "mixed-loop":
                from navine.nsfw.autolearn import run_nsfw_loop

                cfg = load_nsfw_config()
                cfg["mixed_mode"] = True
                hours = args.hours or 8.0
                print(c(f"Navine AI - Python mixed NSFW autolearn loop for {hours} hours...", Colors.CYAN))
                run_nsfw_loop(hours=hours, progress=lambda msg: print(c(msg, Colors.DIM)))
            elif args.nsfw_cmd == "unrestricted-expand":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.unrestricted import fetch_unrestricted_expand

                cfg = load_nsfw_config()
                items = fetch_unrestricted_expand(config=cfg)
                engine = AutolearnEngine()
                result = engine.process_items(items)
                if cfg.get("auto_train_unrestricted", True):
                    from navine.train.registry import train_type

                    train_type("unrestricted", steps=args.steps or cfg.get("text_train_steps"))
                print(c(f"Unrestricted expand: ingested {result['new_samples']} samples from {len(items)} items", Colors.GREEN))
            elif args.nsfw_cmd == "unrestricted":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import fetch_local, fetch_unrestricted_reddit

                cfg = load_nsfw_config()
                items = fetch_unrestricted_reddit(config=cfg)
                items.extend(fetch_local(config=cfg, category="unrestricted"))
                engine = AutolearnEngine()
                result = engine.process_items(items)
                if cfg.get("auto_train_unrestricted", True):
                    from navine.train.registry import train_type

                    train_type("unrestricted", steps=cfg.get("text_train_steps"))
                print(c(f"Unrestricted: ingested {result['new_samples']} samples", Colors.GREEN))
            elif args.nsfw_cmd == "url":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos
                from navine.nsfw.internet import fetch_web_url
                from navine.reference.sources.infini_atomic import (
                    fetch_and_download_infini_atomic,
                    is_infini_atomic_url,
                )

                cfg = load_nsfw_config()
                if is_infini_atomic_url(args.url):
                    result = fetch_and_download_infini_atomic(
                        config=cfg,
                        progress=lambda msg: print(c(msg, Colors.DIM)),
                    )
                    downloaded = int(result.get("downloaded") or 0)
                    fetched = int(result.get("items_fetched") or 0)
                    ref_dir = result.get("reference_dir") or "data/nsfw/local/hentai"
                    if downloaded:
                        print(c(f"Infini-atomic: saved {downloaded} hentai reference image(s) to {ref_dir}", Colors.GREEN))
                    else:
                        print(c(f"Infini-atomic: fetched {fetched} item(s) but saved 0 images.", Colors.YELLOW))
                    if cfg.get("auto_train_image", True) and downloaded:
                        from navine.learn.image_learn import train_learned_image

                        train_learned_image(finetune_steps=cfg.get("image_train_steps"))
                else:
                    items = fetch_web_url(args.url, config=cfg)
                    if not items:
                        print(c("No NSFW content extracted from URL.", Colors.YELLOW))
                    else:
                        engine = AutolearnEngine()
                        result = engine.process_items(items)
                        img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                        vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                        print(c(f"URL fetch: {len(items)} item(s), {result.get('new_samples', 0)} text samples", Colors.GREEN))
                        print(c(f"Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                        if cfg.get("train_on_collect"):
                            from navine.train.registry import train_type

                            train_type("nsfw", steps=cfg.get("text_train_steps"))
            elif args.nsfw_cmd == "crawl":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos
                from navine.nsfw.internet import fetch_crawl_url, save_crawl_snapshot
                from navine.reference.sources.infini_atomic import (
                    fetch_and_download_infini_atomic,
                    is_infini_atomic_url,
                )

                cfg = load_nsfw_config()
                if is_infini_atomic_url(args.url):
                    result = fetch_and_download_infini_atomic(
                        config=cfg,
                        progress=lambda msg: print(c(msg, Colors.DIM)),
                    )
                    downloaded = int(result.get("downloaded") or 0)
                    fetched = int(result.get("items_fetched") or 0)
                    tags = result.get("tags_found") or []
                    ref_dir = result.get("reference_dir") or "data/nsfw/local/hentai"
                    print(c(f"Infini-atomic crawl: tags {tags}", Colors.DIM))
                    if downloaded:
                        print(c(f"Infini-atomic: saved {downloaded} hentai reference image(s) to {ref_dir}", Colors.GREEN))
                    else:
                        print(c(f"Infini-atomic: fetched {fetched} item(s) but saved 0 images.", Colors.YELLOW))
                    if cfg.get("auto_train_image", True) and downloaded:
                        from navine.learn.image_learn import train_learned_image

                        train_learned_image(finetune_steps=cfg.get("image_train_steps"))
                else:
                    items = fetch_crawl_url(
                        args.url,
                        max_pages=args.max_pages,
                        same_domain_only=args.same_domain,
                        config=cfg,
                        progress=lambda msg: print(c(msg, Colors.DIM)),
                    )
                    if not items:
                        print(c("No NSFW content crawled from URL.", Colors.YELLOW))
                    else:
                        snapshot = save_crawl_snapshot(items, args.url)
                        engine = AutolearnEngine()
                        result = engine.process_items(items)
                        img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                        vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                        print(c(f"Crawl: {len(items)} item(s), saved to {snapshot}", Colors.GREEN))
                        print(c(f"Text samples: {result.get('new_samples', 0)} | Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                        if cfg.get("train_on_collect"):
                            from navine.train.registry import train_type

                            train_type("nsfw", steps=cfg.get("text_train_steps"))
            elif args.nsfw_cmd == "post":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos
                from navine.nsfw.internet import fetch_reddit_post

                cfg = load_nsfw_config()
                items = fetch_reddit_post(args.url, config=cfg)
                if not items:
                    print(c("No media extracted from Reddit post URL.", Colors.YELLOW))
                else:
                    engine = AutolearnEngine()
                    result = engine.process_items(items)
                    img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    print(c(f"Reddit post: {len(items)} item(s), {result.get('new_samples', 0)} text samples", Colors.GREEN))
                    print(c(f"Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                    if cfg.get("train_on_collect"):
                        from navine.train.registry import train_type

                        train_type("nsfw", steps=cfg.get("text_train_steps"))
            elif args.nsfw_cmd == "sources":
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos
                from navine.nsfw.internet import fetch_internet_sources

                cfg = load_nsfw_config()
                result = fetch_internet_sources(config=cfg, progress=lambda msg: print(c(msg, Colors.DIM)))
                items = result.get("items") or []
                counts = result.get("counts") or {}
                if not items:
                    print(c("No items from configured internet sources. Edit configs/nsfw.yaml.", Colors.YELLOW))
                else:
                    engine = AutolearnEngine()
                    ingest = engine.process_items(items)
                    img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    print(c(f"Internet sources: {len(items)} item(s)", Colors.GREEN))
                    print(
                        c(
                            f"  Posts: {counts.get('reddit_posts', 0)} | Web: {counts.get('web_urls', 0)} | "
                            f"Crawl: {counts.get('crawl', 0)} | Direct: {counts.get('direct_media', 0)} | "
                            f"Feeds: {counts.get('feeds', 0)} | Search: {counts.get('search', 0)}",
                            Colors.DIM,
                        )
                    )
                    print(c(f"Text samples: {ingest.get('new_samples', 0)} | Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                    if cfg.get("train_on_collect"):
                        from navine.train.registry import train_type

                        train_type("nsfw", steps=cfg.get("text_train_steps"))
            elif args.nsfw_cmd == "waifu":
                from navine.nsfw.config import reload_nsfw_config
                from navine.reference.sources.waifu import fetch_and_download_waifu

                cfg = reload_nsfw_config()
                if args.max:
                    section = dict(cfg.get("waifu_im") or {})
                    section["enabled"] = True
                    section["max_per_run"] = int(args.max)
                    cfg["waifu_im"] = section
                result = fetch_and_download_waifu(
                    config=cfg,
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
                downloaded = int(result.get("downloaded") or 0)
                fetched = int(result.get("items_fetched") or 0)
                ref_dir = result.get("reference_dir") or "data/nsfw/local/hentai"
                if downloaded:
                    print(c(f"Waifu.im: saved {downloaded} hentai reference image(s) to {ref_dir}", Colors.GREEN))
                else:
                    print(c(f"Waifu.im: fetched {fetched} item(s) but saved 0 images. Check API access.", Colors.YELLOW))
                if cfg.get("auto_train_image", True) and downloaded:
                    from navine.learn.image_learn import train_learned_image

                    train_learned_image(finetune_steps=cfg.get("image_train_steps"))
            elif args.nsfw_cmd == "infini":
                from navine.nsfw.config import reload_nsfw_config
                from navine.reference.sources.infini_atomic import (
                    fetch_and_download_infini_atomic,
                    list_infini_site_tags,
                )

                cfg = reload_nsfw_config()
                if getattr(args, "list_tags", False):
                    info = list_infini_site_tags(cfg)
                    print(c(f"Infini-atomic site OK: {info.get('site_ok')}", Colors.GREEN if info.get("site_ok") else Colors.YELLOW))
                    print(c(f"Base URL: {info.get('base_url')}", Colors.DIM))
                    print(c(f"use_all_site_tags: {info.get('use_all_site_tags')}", Colors.DIM))
                    parsed = info.get("tags_parsed") or []
                    active = info.get("tags_active") or []
                    print(c(f"Parsed site tags ({len(parsed)}): {', '.join(str(t) for t in parsed)}", Colors.CYAN))
                    print(c(f"Active fetch tags ({len(active)}): {', '.join(str(t) for t in active)}", Colors.GREEN))
                    return
                if args.max or getattr(args, "all_tags", False):
                    section = dict(cfg.get("infini_atomic") or {})
                    section["enabled"] = True
                    if args.max:
                        section["max_images_per_run"] = int(args.max)
                    if getattr(args, "all_tags", False):
                        section["use_all_site_tags"] = True
                        section["tags"] = []
                        section["tag_allowlist"] = []
                        section["nsfw_only"] = False
                    cfg["infini_atomic"] = section
                result = fetch_and_download_infini_atomic(
                    config=cfg,
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
                downloaded = int(result.get("downloaded") or 0)
                fetched = int(result.get("items_fetched") or 0)
                ref_dir = result.get("reference_dir") or "data/nsfw/local/hentai"
                tags = result.get("tags_found") or []
                if downloaded:
                    print(c(f"Infini-atomic: saved {downloaded} hentai reference image(s) to {ref_dir}", Colors.GREEN))
                    print(c(f"Tags: {', '.join(str(t) for t in tags)}", Colors.DIM))
                else:
                    print(c(f"Infini-atomic: fetched {fetched} item(s) but saved 0 images. Check site access.", Colors.YELLOW))
                if cfg.get("auto_train_image", True) and downloaded:
                    from navine.learn.image_learn import train_learned_image

                    train_learned_image(finetune_steps=cfg.get("image_train_steps"))
            elif args.nsfw_cmd == "4chan":
                if getattr(args, "fourchan_action", None) == "list":
                    print_fourchan_board_list()
                else:
                    from navine.autolearn.engine import AutolearnEngine
                    from navine.autolearn.sources.fourchan import fetch_fourchan
                    from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos

                    cfg = load_nsfw_config()
                    items = fetch_fourchan(
                        config=cfg,
                        board=args.board,
                        group=args.group,
                        max_threads=args.max_threads,
                        max_images=args.max_images,
                        progress=lambda msg: print(c(msg, Colors.DIM)),
                        download_media=True,
                    )
                    if not items:
                        print(c("No 4chan items fetched. Check configs/nsfw.yaml fourchan board groups.", Colors.YELLOW))
                    else:
                        engine = AutolearnEngine()
                        ingest = engine.process_items(items)
                        img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                        vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                        print(c(f"4chan: {len(items)} item(s)", Colors.GREEN))
                        print(c(f"Text samples: {ingest.get('new_samples', 0)} | Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                        if cfg.get("train_on_collect"):
                            from navine.train.registry import train_type

                            train_type("nsfw", steps=cfg.get("text_train_steps"))
        elif args.learn_cmd == "4chan":
            if getattr(args, "fourchan_action", None) == "list":
                print_fourchan_board_list()
            else:
                from navine.autolearn.engine import AutolearnEngine
                from navine.autolearn.sources.fourchan import fetch_fourchan
                from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos
                from navine.nsfw.config import load_nsfw_config

                cfg = load_nsfw_config()
                items = fetch_fourchan(
                    config=cfg,
                    board=args.board,
                    group=args.group,
                    max_threads=args.max_threads,
                    max_images=args.max_images,
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                    download_media=True,
                )
                if not items:
                    print(c("No 4chan items fetched. Check configs/nsfw.yaml fourchan board groups.", Colors.YELLOW))
                else:
                    engine = AutolearnEngine()
                    ingest = engine.process_items(items)
                    img_count = download_nsfw_images(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    vid_count = download_nsfw_videos(items, progress=lambda msg: print(c(msg, Colors.DIM)))
                    print(c(f"4chan: {len(items)} item(s)", Colors.GREEN))
                    print(c(f"Text samples: {ingest.get('new_samples', 0)} | Images: {img_count} | Videos: {vid_count}", Colors.GREEN))
                    if cfg.get("train_on_collect"):
                        from navine.train.registry import train_type

                        train_type("nsfw", steps=cfg.get("text_train_steps"))

    elif args.command == "autolearn":
        enable_colors()
        if args.autolearn_cmd == "run":
            from navine.autolearn.engine import AutolearnEngine

            engine = AutolearnEngine()
            engine.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
            if getattr(args, "aggressive", False):
                result = engine.run_aggressive()
            else:
                result = engine.run_cycle()
            print(c("Navine AI - Python Autolearn cycle complete", Colors.BOLD + Colors.CYAN))
            print(f"  Fetched: {result.get('fetched', 0)}")
            print(f"  New samples: {result.get('new_samples', 0)}")
            if result.get("by_category"):
                print(f"  By category: {result['by_category']}")
            if result.get("trained"):
                print(c("  Training triggered.", Colors.GREEN))
            if result.get("mode"):
                print(f"  Mode: {result['mode']}")
        elif args.autolearn_cmd == "everywhere":
            from navine.autolearn.engine import AutolearnEngine

            engine = AutolearnEngine()
            engine.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
            result = engine.run_everywhere_cycle()
            print(c("Navine AI - Python everywhere learn cycle complete", Colors.BOLD + Colors.CYAN))
            print(f"  Fetched: {result.get('fetched', 0)}")
            print(f"  New samples: {result.get('new_samples', 0)}")
            if result.get("nsfw_samples"):
                print(f"  NSFW samples: {result['nsfw_samples']}")
            if result.get("by_category"):
                print(f"  By category: {result['by_category']}")
            if result.get("images"):
                print(f"  Images learned: {result['images']}")
            if result.get("videos"):
                print(f"  Videos learned: {result['videos']}")
            if result.get("trained"):
                print(c("  Training triggered.", Colors.GREEN))
        elif args.autolearn_cmd == "deep-train":
            from navine.autolearn.engine import AutolearnEngine

            engine = AutolearnEngine()
            engine.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
            result = engine.deep_train()
            print(c("Navine AI - Python deep train complete", Colors.BOLD + Colors.CYAN))
            if result.get("trained"):
                print(c("  All training types completed.", Colors.GREEN))
            else:
                print(c("  Some training types may have failed. Check logs.", Colors.YELLOW))
        elif args.autolearn_cmd == "start":
            import time

            from navine.autolearn.config import load_config
            from navine.autolearn.engine import AutolearnEngine

            print(c("Navine AI - Python Autolearn loop started (Ctrl+C to stop)", Colors.BOLD + Colors.CYAN))
            while True:
                config = load_config()
                if config.get("enabled", True):
                    engine = AutolearnEngine(config)
                    result = engine.run_cycle()
                    print(
                        c(
                            f"Cycle: fetched={result.get('fetched', 0)} "
                            f"new={result.get('new_samples', 0)}",
                            Colors.GREEN,
                        )
                    )
                interval = int(config.get("interval_minutes", 60)) * 60
                time.sleep(interval)
        elif args.autolearn_cmd == "status":
            from navine.autolearn.engine import AutolearnEngine

            status = AutolearnEngine.get_status()
            print(c("Navine AI - Python Autolearn Status", Colors.BOLD + Colors.CYAN))
            print(f"  Enabled: {status.get('enabled')}")
            print(f"  Sources: {', '.join(status.get('sources') or [])}")
            print(f"  Last run: {status.get('last_run') or 'never'}")
            print(f"  Last fetch count: {status.get('last_fetch_count', 0)}")
            print(f"  Last new samples: {status.get('last_new_samples', 0)}")
            print(f"  Total samples: {status.get('total_samples', 0)}")
            print(f"  Samples since train: {status.get('samples_since_train', 0)}")
            print(f"  Next train in: {status.get('next_train_in', 0)} samples")
            print(f"  Last train: {status.get('last_train') or 'never'}")
            print(f"  Interval: {status.get('interval_minutes', 60)} minutes")
        elif args.autolearn_cmd == "config":
            from navine.autolearn.config import config_path, load_config

            path = config_path()
            config = load_config()
            print(c("Navine AI - Python Autolearn Config", Colors.BOLD + Colors.CYAN))
            print(f"  Path: {path}")
            for key, value in config.items():
                print(f"  {key}: {value}")
            if args.edit:
                print(c(f"Edit {path} in your editor to change settings.", Colors.DIM))

    elif args.command == "marathon":
        enable_colors()
        if args.marathon_cmd == "start":
            from navine.autolearn.marathon import MarathonLearner

            learner = MarathonLearner()
            learner.set_progress_callback(lambda msg: print(c(msg, Colors.DIM)))
            print(c("Navine AI - Python Learn Everything Marathon", Colors.BOLD + Colors.CYAN))
            try:
                result = learner.run(
                    hours=args.hours,
                    forever=getattr(args, "forever", False),
                    resume=getattr(args, "resume", False),
                )
            except KeyboardInterrupt:
                print()
                print(c("Marathon interrupted.", Colors.YELLOW))
                sys.exit(0)
            print(c("Marathon finished", Colors.BOLD + Colors.CYAN))
            print(f"  Reason: {result.get('reason')}")
            print(f"  Elapsed: {result.get('elapsed_hours')}h")
            print(f"  Cycles: {result.get('cycles')}")
            print(f"  Fetched: {result.get('items_fetched')}")
            print(f"  Ingested: {result.get('items_ingested')}")
            print(f"  Trained: {result.get('train_count')}")
            print(f"  Deep trains: {result.get('deep_train_count')}")
        elif args.marathon_cmd == "status":
            from navine.autolearn.marathon import get_marathon_status

            status = get_marathon_status()
            print(c("Navine AI - Python Marathon Status", Colors.BOLD + Colors.CYAN))
            print(f"  Running: {status.get('running')}")
            print(f"  Started: {status.get('started_at') or 'never'}")
            target = "forever" if status.get("forever") else f"{status.get('target_hours')}h"
            print(f"  Target: {target}")
            print(f"  Elapsed: {status.get('elapsed_hours')}h")
            if status.get("eta_hours") is not None:
                print(f"  ETA: {status.get('eta_hours')}h remaining")
            print(f"  Cycles: {status.get('cycle_count', 0)}")
            print(f"  Items fetched: {status.get('items_fetched', 0)}")
            print(f"  Items ingested: {status.get('items_ingested', 0)}")
            print(f"  Images: {status.get('images_fetched', 0)}")
            print(f"  Videos: {status.get('videos_fetched', 0)}")
            print(f"  Train runs: {status.get('train_count', 0)}")
            print(f"  Deep trains: {status.get('deep_train_count', 0)}")
            print(f"  Last train: {status.get('last_train') or 'never'}")
            print(f"  Last deep train: {status.get('last_deep_train') or 'never'}")
            print(f"  Current source: {status.get('current_source') or 'idle'}")
            print(f"  Samples since train: {status.get('samples_since_train', 0)}")
            if status.get("stopped_reason"):
                print(f"  Stopped reason: {status.get('stopped_reason')}")
            if status.get("last_error"):
                print(f"  Last error: {status.get('last_error')}")
        elif args.marathon_cmd == "stop":
            from navine.autolearn.marathon import write_stop_flag

            path = write_stop_flag()
            print(c(f"Marathon stop flag written: {path}", Colors.GREEN))
            print(c("Marathon will stop after the current batch.", Colors.DIM))
        elif args.marathon_cmd == "log":
            from navine.autolearn.marathon import tail_marathon_log

            lines = tail_marathon_log(lines=args.lines)
            if not lines:
                print(c("Marathon log is empty.", Colors.YELLOW))
            else:
                for line in lines:
                    print(line)

    elif args.command == "enterprise":
        enable_colors()
        if args.enterprise_cmd == "train":
            from navine.enterprise.pipeline import run_enterprise_pipeline

            print(c("Navine AI - Python Enterprise Training Pipeline", Colors.BOLD + Colors.CYAN))
            print(c("Targets: ChatGPT/Gemini/Claude text + frontier image/video quality", Colors.DIM))
            try:
                status = run_enterprise_pipeline(
                    resume=getattr(args, "resume", False),
                    start_phase=getattr(args, "phase", None),
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
            except KeyboardInterrupt:
                print()
                print(c("Enterprise pipeline interrupted. Resume with --resume", Colors.YELLOW))
                sys.exit(0)
            except Exception as exc:
                print(c(f"Enterprise pipeline failed: {exc}", Colors.RED))
                sys.exit(1)
            print(c("Enterprise pipeline finished", Colors.BOLD + Colors.GREEN))
            print(f"  Active tier: {status.get('active_tier')}")
            print(f"  Phases done: {', '.join(status.get('phases_done') or [])}")
        elif args.enterprise_cmd == "status":
            from navine.enterprise.pipeline import get_enterprise_status

            status = get_enterprise_status()
            print(c("Navine AI - Python Enterprise Status", Colors.BOLD + Colors.CYAN))
            print(f"  Active tier: {status.get('active_tier')}")
            print(f"  Phase: {status.get('current_phase')} ({status.get('phase_index')}/{status.get('phases_total')})")
            print(f"  Completed: {', '.join(status.get('phases_done') or []) or 'none'}")
            if status.get("errors"):
                print(c(f"  Errors: {len(status['errors'])}", Colors.YELLOW))
        elif args.enterprise_cmd == "eval":
            from navine.enterprise.eval import run_enterprise_eval

            report = run_enterprise_eval()
            print(c("Navine AI - Python Enterprise Evaluation", Colors.BOLD + Colors.CYAN))
            print(json.dumps(report, indent=2))
        elif args.enterprise_cmd == "promote":
            from navine.enterprise.pipeline import _set_active_tier

            tier = getattr(args, "tier", "enterprise")
            _set_active_tier(tier)
            print(c(f"Active tier set to {tier}", Colors.GREEN))
        elif args.enterprise_cmd in ("photoreal", "dual"):
            from navine.enterprise.photoreal_loop import run_photoreal_loop

            label = "Dual Real+Hentai" if args.enterprise_cmd == "dual" else "Photoreal+Hentai"
            print(c(f"Navine AI - Python {label} Train/Test Loop", Colors.BOLD + Colors.CYAN))
            print(c("Runs until real_human + hentai_output images/videos pass quality gates", Colors.DIM))
            try:
                report = run_photoreal_loop(
                    max_cycles=getattr(args, "max_cycles", 30),
                    finetune_steps=getattr(args, "finetune_steps", 300),
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
            except KeyboardInterrupt:
                print()
                print(c("Dual quality loop interrupted.", Colors.YELLOW))
                sys.exit(0)
            except Exception as exc:
                print(c(f"Dual quality loop failed: {exc}", Colors.RED))
                sys.exit(1)
            final = report.get("final") or {}
            print(c("Dual quality loop finished", Colors.BOLD + Colors.CYAN))
            print(f"  All pass: {final.get('all_pass')}")
            print(f"  Real image: {final.get('real_image_path')}")
            print(f"  Hentai image: {final.get('hentai_image_path')}")
            print(f"  Real video: {final.get('real_video_path')}")
            print(f"  Hentai video: {final.get('hentai_video_path')}")

    elif args.command == "memory":
        enable_colors()
        if args.memory_cmd == "ingest":
            from navine.memory.ingest import ingest_conversations
            result = ingest_conversations(rebuild_rag=True)
            print(c("Navine AI - Python memory ingest complete", Colors.GREEN))
            print(f"  Sessions scanned: {result.get('sessions_scanned', 0)}")
            print(f"  New Q&A pairs: {result.get('new_qa_pairs', 0)}")
            print(f"  Training blocks: {result.get('training_blocks', 0)}")
            print(f"  RAG documents: {result.get('rag_documents', 0)}")
            try:
                from navine.memory.adapt import run_adapt_train

                train = run_adapt_train(force=True)
                if train.get("ok"):
                    print(c(f"Online adapt finetune complete ({train.get('steps')} steps)", Colors.GREEN))
                else:
                    print(c(f"Adapt train: {train.get('error')}", Colors.YELLOW))
            except Exception as exc:
                print(c(f"Adapt train skipped: {exc}", Colors.YELLOW))
        elif args.memory_cmd == "adapt":
            from navine.memory.adapt import run_adapt_train, adapt_status

            print(c("Running online adapt on learned conversations...", Colors.CYAN))
            result = run_adapt_train(force=True)
            if result.get("ok"):
                print(c(f"Adapted. steps={result.get('steps')} samples={result.get('samples')}", Colors.GREEN))
            else:
                print(c(f"Adapt failed: {result.get('error')}", Colors.RED))
            print(adapt_status())
        elif args.memory_cmd == "status":
            from navine.memory.adapt import adapt_status
            from navine.memory.conversations import train_file_path, conversations_dir

            status = adapt_status()
            print(c("Conversation learning / adapt", Colors.BOLD + Colors.CYAN))
            print(f"  auto_train_chat: {status['config'].get('auto_train_chat')}")
            print(f"  pending good pairs: {status.get('pending_good')}")
            print(f"  total adapt trains: {status.get('total_trains')}")
            print(f"  running now: {status.get('running')}")
            print(f"  last train: {status.get('last_train_result')}")
            print(f"  sessions dir: {conversations_dir()}")
            print(f"  train file: {train_file_path()}")

    elif args.command == "reference":
        enable_colors()
        if args.reference_cmd == "list":
            from navine.reference.config import ensure_reference_dirs
            from navine.reference.media import get_reference_usage_stats, list_reference_summary

            ensure_reference_dirs()
            rows = list_reference_summary()
            usage = get_reference_usage_stats()
            print(c("Navine AI - Python Reference Keywords", Colors.BOLD + Colors.CYAN))
            for row in rows:
                keywords = ", ".join(row.get("keywords") or [])
                print(c(f"  {row['category']}", Colors.GREEN))
                print(f"    folder: {row.get('dir')}")
                print(f"    keywords: {keywords}")
                print(
                    f"    media: {row.get('images', 0)} image(s), {row.get('videos', 0)} video(s)"
                    f" | uses: {row.get('uses', 0)}"
                )
            print(c(f"  Total reference uses: {usage.get('total', 0)}", Colors.DIM))
            try:
                from navine.autolearn.sources.fourchan import list_fourchan_boards
                from navine.nsfw.config import load_nsfw_config

                fourchan = list_fourchan_boards(load_nsfw_config())
                if fourchan.get("enabled"):
                    print(c("Navine AI - Python 4chan Learning Groups", Colors.BOLD + Colors.CYAN))
                    for group_name, boards in (fourchan.get("groups") or {}).items():
                        print(f"  {group_name}: {len(boards)} board(s)")
            except Exception:
                pass
        elif args.reference_cmd == "add":
            from navine.reference.config import ensure_reference_dirs
            from navine.reference.media import copy_reference_file

            ensure_reference_dirs()
            dest = copy_reference_file(args.category, Path(args.path))
            print(c(f"Navine AI - Python reference added: {dest}", Colors.GREEN))
        elif args.reference_cmd == "fetch":
            from navine.nsfw.config import reload_nsfw_config
            from navine.reference.config import ensure_reference_dirs
            from navine.reference.sources.infini_atomic import fetch_and_download_infini_atomic
            from navine.reference.sources.waifu import fetch_and_download_waifu

            category = str(args.category or "hentai").strip().lower()
            if category != "hentai":
                print(c(f"Reference fetch for '{category}' is not configured. Use hentai.", Colors.YELLOW))
            else:
                ensure_reference_dirs()
                cfg = reload_nsfw_config()
                infini_section = dict(cfg.get("infini_atomic") or {})
                infini_section["enabled"] = True
                if args.max:
                    infini_section["max_images_per_run"] = int(args.max)
                cfg["infini_atomic"] = infini_section
                infini_result = fetch_and_download_infini_atomic(
                    config=cfg,
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
                infini_downloaded = int(infini_result.get("downloaded") or 0)
                waifu_section = dict(cfg.get("waifu_im") or {})
                waifu_section["enabled"] = True
                if args.max:
                    waifu_section["max_per_run"] = int(args.max)
                cfg["waifu_im"] = waifu_section
                waifu_result = fetch_and_download_waifu(
                    config=cfg,
                    progress=lambda msg: print(c(msg, Colors.DIM)),
                )
                waifu_downloaded = int(waifu_result.get("downloaded") or 0)
                ref_dir = infini_result.get("reference_dir") or waifu_result.get("reference_dir") or "data/nsfw/local/hentai"
                total = infini_downloaded + waifu_downloaded
                print(
                    c(
                        f"Navine AI - Python reference fetch: {total} image(s) in {ref_dir} "
                        f"(infini {infini_downloaded}, waifu {waifu_downloaded})",
                        Colors.GREEN,
                    )
                )

    elif args.command == "assist":
        run_assist(args)

    elif args.command == "webhook":
        from navine.integrations.discord_webhook import send_webhook, set_webhook_url, webhook_status
        from navine.utils.brand import load_brand

        if args.webhook_cmd == "set":
            result = set_webhook_url(args.url)
            print(result)
        elif args.webhook_cmd == "status":
            print(webhook_status())
        elif args.webhook_cmd == "send":
            brand = load_brand()
            msg = " ".join(args.message)
            result = send_webhook(msg, webhook_url=args.url, username=str(brand.get("name") or "Navine AI - Python"))
            print(result)
            if not result.get("ok"):
                sys.exit(1)

    elif args.command == "mcp":
        import json as _json

        from navine.mcp.client import call_tool, list_servers, list_tools, parse_call_args, status as mcp_status
        from navine.mcp.server import cursor_mcp_snippet, serve as mcp_serve

        if args.mcp_cmd == "status":
            print(_json.dumps(mcp_status(), indent=2))
        elif args.mcp_cmd == "servers":
            print(_json.dumps(list_servers(), indent=2))
        elif args.mcp_cmd == "tools":
            result = list_tools(args.server)
            print(_json.dumps(result, indent=2))
            if not result.get("ok"):
                sys.exit(1)
        elif args.mcp_cmd == "call":
            result = call_tool(args.server, args.tool, parse_call_args(args.arg))
            print(_json.dumps(result, indent=2))
            if not result.get("ok"):
                sys.exit(1)
        elif args.mcp_cmd == "serve":
            kwargs = {}
            if getattr(args, "host", None):
                kwargs["host"] = args.host
            if getattr(args, "port", None):
                kwargs["port"] = args.port
            mcp_serve(transport=args.transport, **kwargs)
        elif args.mcp_cmd == "cursor-config":
            print(_json.dumps(cursor_mcp_snippet(), indent=2))
        elif args.mcp_cmd == "install":
            from navine.mcp.ensure_sdk import main as ensure_mcp

            raise SystemExit(ensure_mcp())

    elif args.command == "screen":
        run_screen(args)

    elif args.command == "desktop":
        run_desktop(args)

    elif args.command == "play":
        run_play(args)

    elif args.command == "voice":
        run_voice(args)

    elif args.command == "music":
        run_music(args)

    elif args.command == "moltbook":
        run_moltbook(args)

    elif args.command == "info":
        print_info(args.model)

    elif args.command == "inspect":
        if args.inspect_cmd == "file":
            run_inspect_file(args.path)
        elif args.inspect_cmd == "search":
            run_inspect_search(args.query)
        elif args.inspect_cmd == "modules":
            run_inspect_modules()

    elif args.command == "doctor":
        from navine.doctor import run_doctor, print_report
        results, _ = run_doctor(fix=args.fix)
        failed = print_report(results)
        sys.exit(1 if failed else 0)

    elif args.command == "settings":
        from navine.settings_gui import open_settings_window

        open_settings_window()

    elif args.command == "benchmark":
        from navine.benchmark import run_benchmark

        result = run_benchmark(args.module)
        for row in result.get("results") or []:
            print(c(str(row), Colors.GREEN if "error" not in row else Colors.YELLOW))

    elif args.command == "models":
        enable_colors()
        from navine.utils.hardware import cloud_provider, cuda_available, pick_tier
        from navine.text.opensource import detect_backend

        tier = pick_tier()
        print(c("Navine AI - Python model backends", Colors.BOLD + Colors.CYAN))
        print(f"  tier: {tier}")
        print(f"  cuda_available: {cuda_available()}")
        print(f"  cloud_provider: {cloud_provider() or 'none'}")
        print(f"  text_backend: {detect_backend() or 'custom'}")
        print(f"  image_backend: local trained checkpoint (external teachers training-only)")
        print(f"  video_backend: local trained checkpoint (external teachers training-only)")
        try:
            from navine.utils.generation import allow_external_inference, external_teacher_enabled

            print(f"  allow_external_inference: {allow_external_inference()}")
            print(f"  external_teacher_enabled: {external_teacher_enabled()}")
        except Exception:
            pass

    elif args.command == "api-key":
        from navine.api.settings import get_key_store

        store = get_key_store()
        if args.api_key_command == "create":
            key_id, raw_key, entry = store.create(
                name=args.name,
                scopes=[s.strip() for s in str(getattr(args, "scopes", "") or "").split(",") if s.strip()] or None,
                expires_days=getattr(args, "expires_days", None),
            )
            print(c(f"Created API key: {args.name}", Colors.GREEN))
            print(f"Key id: {key_id}")
            print(f"Prefix: {entry.get('prefix')}")
            print(f"Scopes: {', '.join(entry.get('scopes') or [])}")
            if entry.get("expires_at"):
                print(f"Expires: {entry.get('expires_at')}")
            print(c("Save this key now. It will not be shown again:", Colors.YELLOW))
            print(raw_key)
        elif args.api_key_command == "quick":
            _, raw_key, _ = store.create("Navine AI - Python")
            print(raw_key)
        elif args.api_key_command == "list":
            rows = store.list_keys()
            if not rows:
                print(c("No API keys found.", Colors.YELLOW))
            else:
                for row in rows:
                    status = "revoked" if row["revoked"] else "active"
                    print(
                        f"{row['id']}  {row['name']}  [{status}]  "
                        f"prefix={row.get('prefix','')}  uses={row.get('use_count',0)}  "
                        f"{row.get('created_at', '')}"
                    )
        elif args.api_key_command == "rotate":
            rotated = store.rotate(args.key_id)
            if not rotated:
                print(c(f"Key not found or already revoked: {args.key_id}", Colors.RED))
                sys.exit(1)
            key_id, raw_key, entry = rotated
            print(c(f"Rotated key {args.key_id} -> {key_id}", Colors.GREEN))
            print(c("Save this key now. It will not be shown again:", Colors.YELLOW))
            print(raw_key)
        elif args.api_key_command == "revoke":
            if store.revoke(args.key_id):
                print(c(f"Revoked key {args.key_id}", Colors.GREEN))
            else:
                print(c(f"Key not found: {args.key_id}", Colors.RED))
                sys.exit(1)


if __name__ == "__main__":
    main()
