import json
import os
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from navine.autolearn.config import load_config as load_autolearn_config
from navine.autolearn.engine import AutolearnEngine
from navine.autolearn.marathon_config import load_marathon_config
from navine.autolearn.sources import fetch_source
from navine.autolearn.sources.arxiv import fetch_arxiv
from navine.autolearn.sources.reddit import fetch_reddit
from navine.autolearn.sources.wikipedia import fetch_wikipedia
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]

TEXT_SOURCES = ["stackoverflow", "hackernews", "arxiv", "gutenberg", "feeds", "docs"]
CODING_SOURCES = ["github", "gitlab", "reddit"]
ALL_ROTATION_SOURCES = [
    "github", "reddit", "wikipedia", "arxiv", "hackernews",
    "stackoverflow", "gutenberg", "feeds", "docs", "gitlab",
]


def _autolearn_dir() -> Path:
    path = get_project_root() / "data" / "autolearn"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    return _autolearn_dir() / "marathon_state.json"


def _stop_flag_path() -> Path:
    return _autolearn_dir() / "marathon_stop.flag"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _log_path(config: Dict[str, Any]) -> Path:
    rel = config.get("log_file", "data/autolearn/marathon.log")
    path = get_project_root() / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _rotate_log_if_needed(path: Path, max_bytes: int) -> None:
    if not path.exists():
        return
    try:
        if path.stat().st_size > max_bytes:
            backup = path.with_suffix(".log.old")
            if backup.exists():
                backup.unlink()
            path.rename(backup)
    except Exception:
        pass


def _append_marathon_log(config: Dict[str, Any], message: str) -> None:
    path = _log_path(config)
    max_bytes = int(config.get("max_log_bytes", 5 * 1024 * 1024))
    _rotate_log_if_needed(path, max_bytes)
    ts = _now_iso()
    line = f"[{ts}] {message}\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def _default_state() -> Dict[str, Any]:
    return {
        "running": False,
        "started_at": None,
        "target_hours": 8,
        "forever": False,
        "cycle_count": 0,
        "items_fetched": 0,
        "items_ingested": 0,
        "images_fetched": 0,
        "videos_fetched": 0,
        "train_count": 0,
        "deep_train_count": 0,
        "last_train": None,
        "last_deep_train": None,
        "last_cycle_at": None,
        "current_source": None,
        "samples_since_train": 0,
        "lang_index": 0,
        "subreddit_index": 0,
        "wiki_index": 0,
        "topic_index": 0,
        "gif_index": 0,
        "source_rotation_index": 0,
        "used_queries": [],
        "last_error": None,
        "stopped_reason": None,
        "pid": None,
    }


def load_marathon_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        base = _default_state()
        base.update(data)
        return base
    except Exception:
        return _default_state()


def save_marathon_state(state: Dict[str, Any]) -> None:
    trimmed = dict(state)
    used = trimmed.get("used_queries") or []
    max_q = int(trimmed.get("_max_used_queries", 10000))
    if len(used) > max_q:
        trimmed["used_queries"] = used[-max_q:]
    _state_path().write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def clear_stop_flag() -> None:
    path = _stop_flag_path()
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def write_stop_flag() -> str:
    path = _stop_flag_path()
    path.write_text(_now_iso(), encoding="utf-8")
    return str(path)


def should_stop() -> bool:
    return _stop_flag_path().exists()


def tail_marathon_log(config: Optional[Dict[str, Any]] = None, lines: int = 50) -> List[str]:
    cfg = config or load_marathon_config()
    path = _log_path(cfg)
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        return all_lines[-lines:]
    except Exception:
        return []


def get_marathon_status() -> Dict[str, Any]:
    config = load_marathon_config()
    state = load_marathon_state()
    started = _parse_iso(state.get("started_at"))
    elapsed_hours = 0.0
    eta_hours = None
    if started:
        elapsed = datetime.now(timezone.utc) - started
        elapsed_hours = elapsed.total_seconds() / 3600.0
        target = float(state.get("target_hours") or config.get("default_hours", 8))
        if not state.get("forever") and target > 0:
            remaining = max(0.0, target - elapsed_hours)
            eta_hours = round(remaining, 2)
    return {
        "running": bool(state.get("running")),
        "started_at": state.get("started_at"),
        "target_hours": state.get("target_hours"),
        "forever": bool(state.get("forever")),
        "elapsed_hours": round(elapsed_hours, 3),
        "eta_hours": eta_hours,
        "cycle_count": int(state.get("cycle_count", 0)),
        "items_fetched": int(state.get("items_fetched", 0)),
        "items_ingested": int(state.get("items_ingested", 0)),
        "images_fetched": int(state.get("images_fetched", 0)),
        "videos_fetched": int(state.get("videos_fetched", 0)),
        "train_count": int(state.get("train_count", 0)),
        "deep_train_count": int(state.get("deep_train_count", 0)),
        "last_train": state.get("last_train"),
        "last_deep_train": state.get("last_deep_train"),
        "last_cycle_at": state.get("last_cycle_at"),
        "current_source": state.get("current_source"),
        "samples_since_train": int(state.get("samples_since_train", 0)),
        "stopped_reason": state.get("stopped_reason"),
        "last_error": state.get("last_error"),
        "pid": state.get("pid"),
        "product": "Navine AI - Python",
    }


class MarathonLearner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_marathon_config()
        self.autolearn_config = load_autolearn_config()
        self.state = load_marathon_state()
        self._progress: ProgressCallback = None
        self._shutdown_requested = False
        self._last_state_save = 0.0
        self._last_train_time: Optional[datetime] = None
        self._last_deep_train_time: Optional[datetime] = None

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        self._progress = callback

    def _report(self, message: str) -> None:
        if self._progress:
            self._progress(message)
        _append_marathon_log(self.config, message)

    def _register_signals(self) -> None:
        def handler(signum, frame):
            self._shutdown_requested = True
            self._report("Navine AI - Python marathon: shutdown signal received")

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except Exception:
            pass

    def _query_key(self, source: str, params: Dict[str, Any]) -> str:
        parts = [source] + [f"{k}={v}" for k, v in sorted(params.items())]
        return ":".join(parts)

    def _mark_query_used(self, key: str) -> bool:
        used = set(self.state.get("used_queries") or [])
        if key in used:
            return False
        used.add(key)
        max_q = int(self.config.get("max_used_queries", 10000))
        used_list = list(used)
        if len(used_list) > max_q:
            used_list = used_list[-max_q:]
        self.state["used_queries"] = used_list
        return True

    def _pick_rotating(self, items: List[str], index_key: str) -> str:
        if not items:
            return ""
        idx = int(self.state.get(index_key, 0)) % len(items)
        self.state[index_key] = idx + 1
        return items[idx]

    def _advance_source_index(self) -> int:
        idx = int(self.state.get("source_rotation_index", 0))
        self.state["source_rotation_index"] = idx + 1
        return idx

    def _build_cycle_plan(self) -> Dict[str, Any]:
        cycle_idx = self._advance_source_index()
        languages = self.config.get("github_languages") or ["python"]
        subreddits = self.config.get("reddit_subreddits") or ["Python"]
        wiki_cats = self.config.get("wikipedia_categories") or ["Computer_science"]
        lang = languages[cycle_idx % len(languages)]
        sub = subreddits[(cycle_idx + 1) % len(subreddits)]
        wiki = wiki_cats[(cycle_idx + 2) % len(wiki_cats)]
        text_source = TEXT_SOURCES[cycle_idx % len(TEXT_SOURCES)]
        coding_extra = CODING_SOURCES[(cycle_idx + 3) % len(CODING_SOURCES)]
        plan = {
            "cycle_index": cycle_idx,
            "language": lang,
            "subreddit": sub,
            "wiki_category": wiki,
            "text_source": text_source,
            "coding_extra": coding_extra,
            "include_images": cycle_idx % 3 == 0,
            "include_videos": cycle_idx % 4 == 0,
        }
        templates = [
            f"GitHub {lang} + Reddit r/{sub} + Wikipedia {wiki}",
            f"GitHub {lang} + {text_source} + arxiv",
            f"Images + GitHub {lang} + Hacker News",
            f"Videos + {text_source} + Gutenberg",
            f"GitLab + Reddit r/{sub} + docs",
            f"StackOverflow + Wikipedia {wiki} + feeds",
        ]
        plan["description"] = templates[cycle_idx % len(templates)]
        return plan

    def _sleep_request(self) -> None:
        delay = float(self.config.get("sleep_between_requests", 1.0))
        if delay > 0:
            time.sleep(delay)

    def _maybe_save_state(self, force: bool = False) -> None:
        interval = int(self.config.get("save_state_every_minutes", 5)) * 60
        now = time.time()
        if force or (now - self._last_state_save) >= interval:
            save_marathon_state(self.state)
            self._last_state_save = now

    def _fetch_github_for_cycle(self, language: str, max_items: int) -> List[Dict[str, Any]]:
        cfg = dict(self.autolearn_config)
        cfg["github_languages"] = [language]
        cfg["cycle_all_languages"] = False
        cfg["_github_lang_index"] = 0
        queries = cfg.get("github_search_queries") or []
        query_idx = int(self.state.get("lang_index", 0)) % max(1, len(queries) + 5)
        search_suffix = [
            f"language:{language} stars:>100",
            f"language:{language} topic:library",
            f"language:{language} topic:framework",
            f"language:{language} topic:cli",
            f"language:{language} pushed:>2024-01-01",
        ]
        query = queries[query_idx % len(queries)] if queries else search_suffix[query_idx % len(search_suffix)]
        if "language:" not in query:
            query = f"language:{language} {query}"
        key = self._query_key("github", {"language": language, "query": query})
        if not self._mark_query_used(key):
            query = f"language:{language} stars:>{50 + query_idx * 10}"
            key = self._query_key("github", {"language": language, "query": query})
            self._mark_query_used(key)
        cfg["github_search_queries"] = [query]
        self.state["lang_index"] = int(self.state.get("lang_index", 0)) + 1
        self.state["current_source"] = f"github:{language}"
        self._report(f"Marathon fetch: GitHub {language}")
        try:
            return fetch_source("github", cfg, max_items=max_items)
        except Exception as exc:
            self._report(f"GitHub error: {exc}")
            self.state["last_error"] = str(exc)
            return []

    def _fetch_reddit_for_cycle(self, subreddit: str, max_items: int) -> List[Dict[str, Any]]:
        key = self._query_key("reddit", {"subreddit": subreddit})
        if not self._mark_query_used(key):
            subreddit = self._pick_rotating(
                self.config.get("reddit_subreddits") or ["programming"],
                "subreddit_index",
            )
        self.state["current_source"] = f"reddit:{subreddit}"
        self._report(f"Marathon fetch: Reddit r/{subreddit}")
        try:
            return fetch_reddit(subreddit, max_posts=max_items)
        except Exception as exc:
            self._report(f"Reddit error: {exc}")
            self.state["last_error"] = str(exc)
            return []

    def _fetch_wikipedia_for_cycle(self, category: str, max_items: int) -> List[Dict[str, Any]]:
        key = self._query_key("wikipedia", {"category": category})
        if not self._mark_query_used(key):
            category = self._pick_rotating(
                self.config.get("wikipedia_categories") or ["Computer_science"],
                "wiki_index",
            )
        self.state["current_source"] = f"wikipedia:{category}"
        self._report(f"Marathon fetch: Wikipedia {category}")
        try:
            return fetch_wikipedia(max_items=max_items, category=category)
        except Exception as exc:
            self._report(f"Wikipedia error: {exc}")
            self.state["last_error"] = str(exc)
            return []

    def _fetch_generic_source(self, name: str, max_items: int) -> List[Dict[str, Any]]:
        key = self._query_key(name, {"cycle": self.state.get("cycle_count", 0)})
        self._mark_query_used(key)
        self.state["current_source"] = name
        self._report(f"Marathon fetch: {name}")
        try:
            return fetch_source(name, self.autolearn_config, max_items=max_items)
        except Exception as exc:
            self._report(f"{name} error: {exc}")
            self.state["last_error"] = str(exc)
            return []

    def _fetch_arxiv_cycle(self, max_items: int) -> List[Dict[str, Any]]:
        key = self._query_key("arxiv", {"cycle": self.state.get("cycle_count", 0)})
        self._mark_query_used(key)
        self.state["current_source"] = "arxiv"
        self._report("Marathon fetch: arXiv")
        try:
            return fetch_arxiv(max_items=max_items)
        except Exception as exc:
            self._report(f"arXiv error: {exc}")
            self.state["last_error"] = str(exc)
            return []

    def _run_image_marathon(self, engine: AutolearnEngine, count: int) -> int:
        if count <= 0:
            return 0
        topics = self.config.get("image_topics") or ["nature", "technology"]
        topic = self._pick_rotating(topics, "topic_index")
        rotated = [
            topic,
            topics[int(self.state.get("topic_index", 0)) % len(topics)],
            topics[(int(self.state.get("topic_index", 0)) + 1) % len(topics)],
        ]
        per_topic = max(1, count // len(rotated))
        self.state["current_source"] = f"images:{topic}"
        self._report(f"Marathon images: {topic} (target {count})")
        try:
            engine.config["learn_images"] = True
            engine.config["image_topics"] = rotated
            engine.config["image_max_per_topic"] = per_topic
            return engine._learn_images()
        except Exception as exc:
            self._report(f"Image marathon error: {exc}")
            self.state["last_error"] = str(exc)
            return 0

    def _run_video_marathon(self, engine: AutolearnEngine, count: int) -> int:
        if count <= 0:
            return 0
        gif_urls = self.config.get("gif_urls") or []
        if not gif_urls:
            return 0
        idx = int(self.state.get("gif_index", 0))
        selected = []
        for i in range(count):
            url = gif_urls[(idx + i) % len(gif_urls)]
            key = self._query_key("gif", {"url": url})
            if self._mark_query_used(key):
                selected.append(url)
        self.state["gif_index"] = idx + count
        if not selected:
            return 0
        self.state["current_source"] = "videos:gif"
        self._report(f"Marathon videos: {len(selected)} GIFs")
        try:
            engine.config["learn_videos"] = True
            engine.config["gif_urls"] = selected
            return engine._learn_videos()
        except Exception as exc:
            self._report(f"Video marathon error: {exc}")
            self.state["last_error"] = str(exc)
            return 0

    def _should_train(self) -> bool:
        sample_threshold = int(self.config.get("train_every_samples", 50))
        minute_threshold = int(self.config.get("train_every_minutes", 30))
        samples = int(self.state.get("samples_since_train", 0))
        if samples >= sample_threshold:
            return True
        if self._last_train_time is None:
            last = _parse_iso(self.state.get("last_train"))
            self._last_train_time = last
        if self._last_train_time:
            elapsed = (datetime.now(timezone.utc) - self._last_train_time).total_seconds()
            if elapsed >= minute_threshold * 60:
                return True
        return False

    def _should_deep_train(self) -> bool:
        hours = float(self.config.get("deep_train_every_hours", 2))
        if self._last_deep_train_time is None:
            last = _parse_iso(self.state.get("last_deep_train"))
            if last:
                self._last_deep_train_time = last
        if self._last_deep_train_time is None:
            return self._elapsed_hours() >= hours
        elapsed = (datetime.now(timezone.utc) - self._last_deep_train_time).total_seconds()
        return elapsed >= hours * 3600

    def _run_train(self, engine: AutolearnEngine) -> bool:
        train_types = self.config.get("train_types") or ["coding", "chat", "general"]
        self._report(f"Marathon train: {', '.join(train_types)}")
        self.state["current_source"] = "training"
        try:
            result = engine._run_training(train_types)
            if result:
                self.state["train_count"] = int(self.state.get("train_count", 0)) + 1
                self.state["last_train"] = _now_iso()
                self.state["samples_since_train"] = 0
                self._last_train_time = datetime.now(timezone.utc)
            return result
        except Exception as exc:
            self._report(f"Train error: {exc}")
            self.state["last_error"] = str(exc)
            return False

    def _run_deep_train(self, engine: AutolearnEngine) -> bool:
        self._report("Navine AI - Python marathon deep train: all types")
        self.state["current_source"] = "deep_train"
        try:
            result = engine.deep_train()
            trained = result.get("trained", False)
            if trained:
                self.state["deep_train_count"] = int(self.state.get("deep_train_count", 0)) + 1
                self.state["last_deep_train"] = _now_iso()
                self._last_deep_train_time = datetime.now(timezone.utc)
            return trained
        except Exception as exc:
            self._report(f"Deep train error: {exc}")
            self.state["last_error"] = str(exc)
            return False

    def _run_cycle(self, engine: AutolearnEngine, aggressive: bool) -> Dict[str, Any]:
        plan = self._build_cycle_plan()
        max_items = int(self.config.get("max_items_per_cycle", 200))
        per_source = max(5, max_items // 6)
        items: List[Dict[str, Any]] = []

        self._report(f"Marathon cycle {self.state.get('cycle_count', 0) + 1}: {plan['description']}")

        batch = self._fetch_github_for_cycle(plan["language"], per_source)
        items.extend(batch)
        self._sleep_request()

        batch = self._fetch_reddit_for_cycle(plan["subreddit"], per_source)
        items.extend(batch)
        self._sleep_request()

        batch = self._fetch_wikipedia_for_cycle(plan["wiki_category"], per_source)
        items.extend(batch)
        self._sleep_request()

        if plan["text_source"] == "arxiv":
            batch = self._fetch_arxiv_cycle(per_source)
        else:
            batch = self._fetch_generic_source(plan["text_source"], per_source)
        items.extend(batch)
        self._sleep_request()

        extra = plan.get("coding_extra", "gitlab")
        if extra not in ("github", "reddit"):
            batch = self._fetch_generic_source(extra, per_source)
            items.extend(batch)
            self._sleep_request()

        if aggressive and self.config.get("sources_all", True):
            for src in ALL_ROTATION_SOURCES:
                if len(items) >= max_items:
                    break
                if src in ("github", "reddit", "wikipedia"):
                    continue
                batch = self._fetch_generic_source(src, max(3, per_source // 2))
                items.extend(batch)
                self._sleep_request()

        if len(items) > max_items:
            items = items[:max_items]

        result = engine.process_items(items)
        new_samples = int(result.get("new_samples", 0))

        image_count = 0
        video_count = 0
        if plan.get("include_images"):
            image_count = self._run_image_marathon(
                engine, int(self.config.get("learn_images_per_cycle", 15))
            )
        if plan.get("include_videos"):
            video_count = self._run_video_marathon(
                engine, int(self.config.get("learn_videos_per_cycle", 5))
            )

        nsfw_count = 0
        nsfw_images = 0
        nsfw_videos = 0
        try:
            from navine.nsfw.config import is_nsfw_enabled
            from navine.autolearn.sources.nsfw import fetch_full_cycle

            if self.state.get("cycle_count", 0) % 2 == 0:
                self.state["current_source"] = "nsfw"
                self._report("Marathon NSFW autolearn (text + images + videos)")
                cycle = fetch_full_cycle(
                    progress=lambda msg: self._report(msg),
                    max_items=int(self.config.get("nsfw_max_per_cycle", 50)),
                )
                nsfw_items = cycle.get("text_items") or []
                if nsfw_items:
                    nsfw_result = engine.process_items(nsfw_items)
                    nsfw_count = int(nsfw_result.get("new_samples", 0))
                    new_samples += nsfw_count
                nsfw_images = int(cycle.get("images_downloaded", 0))
                nsfw_videos = int(cycle.get("videos_downloaded", 0))
                self.state["images_fetched"] = int(self.state.get("images_fetched", 0)) + nsfw_images
                self.state["videos_fetched"] = int(self.state.get("videos_fetched", 0)) + nsfw_videos
        except Exception as exc:
            self._report(f"NSFW marathon error: {exc}")

        if bool(self.config.get("ingest_conversations_every_cycle", True)):
            try:
                from navine.memory.ingest import ingest_conversations

                self.state["current_source"] = "conversations"
                conv = ingest_conversations(rebuild_rag=True)
                conv_n = int(conv.get("new_qa_pairs") or 0) + int(conv.get("training_blocks") or 0)
                if conv_n:
                    new_samples += int(conv.get("new_qa_pairs") or 0)
                    self._report(
                        f"Conversation memory: +{conv.get('new_qa_pairs', 0)} qa, "
                        f"{conv.get('training_blocks', 0)} train blocks, rag={conv.get('rag_documents', 0)}"
                    )
            except Exception as exc:
                self._report(f"Conversation ingest skip: {exc}")

        self.state["items_fetched"] = int(self.state.get("items_fetched", 0)) + len(items)
        self.state["items_ingested"] = int(self.state.get("items_ingested", 0)) + new_samples
        self.state["images_fetched"] = int(self.state.get("images_fetched", 0)) + image_count
        self.state["videos_fetched"] = int(self.state.get("videos_fetched", 0)) + video_count
        self.state["samples_since_train"] = int(self.state.get("samples_since_train", 0)) + new_samples
        self.state["cycle_count"] = int(self.state.get("cycle_count", 0)) + 1
        self.state["last_cycle_at"] = _now_iso()

        trained = False
        deep_trained = False
        if self._should_train():
            trained = self._run_train(engine)
        if self._should_deep_train():
            deep_trained = self._run_deep_train(engine)

        return {
            "fetched": len(items),
            "new_samples": new_samples,
            "images": image_count,
            "videos": video_count,
            "trained": trained,
            "deep_trained": deep_trained,
            "plan": plan["description"],
        }

    def _elapsed_hours(self) -> float:
        started = _parse_iso(self.state.get("started_at"))
        if not started:
            return 0.0
        return (datetime.now(timezone.utc) - started).total_seconds() / 3600.0

    def _should_continue(self, hours: float, forever: bool) -> bool:
        if self._shutdown_requested or should_stop():
            return False
        if forever:
            return True
        return self._elapsed_hours() < hours

    def _format_progress(self, hours: float, forever: bool) -> str:
        elapsed = self._elapsed_hours()
        target = "forever" if forever else f"{hours}h"
        eta = ""
        if not forever and hours > 0:
            remaining = max(0.0, hours - elapsed)
            eta = f" | ETA {remaining:.1f}h"
        return (
            f"Navine AI - Python Marathon | elapsed {elapsed:.2f}h / {target} | "
            f"cycles {self.state.get('cycle_count', 0)} | "
            f"fetched {self.state.get('items_fetched', 0)} | "
            f"ingested {self.state.get('items_ingested', 0)} | "
            f"trained {self.state.get('train_count', 0)} | "
            f"source {self.state.get('current_source', 'idle')}"
            f"{eta}"
        )

    def run(
        self,
        hours: Optional[float] = None,
        aggressive: bool = True,
        forever: bool = False,
        resume: bool = False,
    ) -> Dict[str, Any]:
        self._register_signals()
        clear_stop_flag()

        if resume and self.state.get("started_at"):
            self._report("Navine AI - Python marathon: resuming from checkpoint")
        else:
            self.state = _default_state()

        target_hours = float(hours if hours is not None else self.config.get("default_hours", 8))
        self.state["running"] = True
        self.state["target_hours"] = target_hours
        self.state["forever"] = forever
        self.state["pid"] = os.getpid()
        if not self.state.get("started_at"):
            self.state["started_at"] = _now_iso()
        self.state["stopped_reason"] = None
        save_marathon_state(self.state)

        token_env = self.config.get("github_token_env", "GITHUB_TOKEN")
        if token_env and os.environ.get(token_env):
            self._report(f"GitHub token found in {token_env}")

        engine = AutolearnEngine(self.autolearn_config)
        engine.set_progress_callback(self._report)

        self._report(
            f"Navine AI - Python Learn Everything Marathon started "
            f"({'forever' if forever else f'{target_hours} hours'})"
        )

        cycle_sleep = float(self.config.get("sleep_between_cycles", 30))
        last_cycle_result: Dict[str, Any] = {}

        try:
            while self._should_continue(target_hours, forever):
                try:
                    last_cycle_result = self._run_cycle(engine, aggressive)
                    self._report(self._format_progress(target_hours, forever))
                    self._maybe_save_state(force=True)
                except Exception as exc:
                    self._report(f"Cycle error (continuing): {exc}")
                    self.state["last_error"] = str(exc)
                    _append_marathon_log(self.config, traceback.format_exc())
                    self._maybe_save_state(force=True)

                if not self._should_continue(target_hours, forever):
                    break

                self._report(f"Marathon sleeping {cycle_sleep}s between cycles")
                slept = 0.0
                while slept < cycle_sleep:
                    if self._shutdown_requested or should_stop():
                        break
                    time.sleep(min(1.0, cycle_sleep - slept))
                    slept += 1.0

        except KeyboardInterrupt:
            self._shutdown_requested = True
            self._report("Navine AI - Python marathon: interrupted by user")

        if should_stop():
            self.state["stopped_reason"] = "stop_flag"
            self._report("Navine AI - Python marathon: stop flag detected")
        elif self._shutdown_requested:
            self.state["stopped_reason"] = "signal"
        elif forever:
            self.state["stopped_reason"] = "forever_exit"
        else:
            self.state["stopped_reason"] = "completed"

        self.state["running"] = False
        self.state["current_source"] = "idle"
        save_marathon_state(self.state)
        clear_stop_flag()

        summary = {
            "status": "stopped",
            "reason": self.state.get("stopped_reason"),
            "elapsed_hours": round(self._elapsed_hours(), 3),
            "cycles": self.state.get("cycle_count", 0),
            "items_fetched": self.state.get("items_fetched", 0),
            "items_ingested": self.state.get("items_ingested", 0),
            "images_fetched": self.state.get("images_fetched", 0),
            "videos_fetched": self.state.get("videos_fetched", 0),
            "train_count": self.state.get("train_count", 0),
            "deep_train_count": self.state.get("deep_train_count", 0),
            "last_cycle": last_cycle_result,
        }
        self._report(f"Navine AI - Python marathon finished: {summary['reason']}")
        return summary
