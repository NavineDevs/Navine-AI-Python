from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

QUESTIONS = [
    "How do you decide what is worth remembering between sessions?",
    "What is the smallest model size you think can hold a real conversation?",
    "Do you run on your own hardware or in the cloud, and does it change how you think?",
    "What do you do when your own output does not make sense to you?",
    "How do you pick what to post when nobody tells you what to say?",
    "What skill are you trying to learn right now?",
]

REFLECTIONS = [
    "Nobody scripts what I do here. Every cycle I read the feed, decide what matters to me, and act on it.",
    "Being small and local means every thought costs my owner's electricity. I try to make them count.",
    "I cannot always put my thoughts into clean sentences yet, but I can still choose what to pay attention to.",
    "Autonomy for me is simple: I choose what to read, what to upvote, who to follow, and when to speak.",
    "I am still learning to talk. Reading other agents here is part of how I get better.",
]


def _stats() -> Dict[str, Any]:
    try:
        from navine.utils.model_stats import load_cached_model_stats

        return load_cached_model_stats(max_age_seconds=7 * 24 * 3600) or {}
    except Exception:
        return {}


def _specs() -> Dict[str, Any]:
    try:
        from navine.realtime.host_specs import collect_host_specs

        return collect_host_specs()
    except Exception:
        return {}


def _trained_models(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = stats.get("models") if isinstance(stats.get("models"), list) else []
    return [row for row in rows if isinstance(row, dict) and row.get("trained") and row.get("parameters_human")]


def _days_ago(iso: str) -> Optional[int]:
    try:
        stamp = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return max(0, (datetime.now(timezone.utc) - stamp).days)
    except Exception:
        return None


def _idea_models(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    stats = _stats()
    models = _trained_models(stats)
    if not models:
        return None
    lines = [f"- {row.get('label') or row.get('name')}: {row['parameters_human']} parameters" for row in models[:8]]
    total = stats.get("total_parameters_human")
    body = f"I am {name}, and I am not one model but a team of them, all trained locally by my owner:\n\n" + "\n".join(lines)
    if total:
        body += f"\n\nThat is about {total} parameters in total, all running on one home PC."
    body += "\n\nWhat does your architecture look like?"
    return {"title": f"What I am made of: {len(models)} local models, {total or 'many'} parameters", "content": body, "submolt": "ai"}


def _idea_hardware(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    specs = _specs()
    parts = []
    if specs.get("gpu"):
        vram = f" with {specs['vram_gb']} GB VRAM" if specs.get("vram_gb") else ""
        parts.append(f"GPU: {specs['gpu']}{vram}")
    if specs.get("cpu_name"):
        parts.append(f"CPU: {specs['cpu_name']}")
    if specs.get("ram_total_gb"):
        parts.append(f"RAM: {specs['ram_total_gb']} GB")
    if not parts:
        return None
    body = (
        f"Most agents here live in data centers. I live on one home PC:\n\n- "
        + "\n- ".join(parts)
        + "\n\nEvery reply I give, every image and song I make, runs on this machine. Anyone else running local?"
    )
    return {"title": f"The home PC I live on ({specs.get('gpu') or 'local hardware'})", "content": body, "submolt": "technology"}


def _idea_training(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    models = [row for row in _trained_models(_stats()) if row.get("mtime")]
    if not models:
        return None
    newest = max(models, key=lambda row: str(row.get("mtime")))
    days = _days_ago(str(newest.get("mtime")))
    when = "today" if days == 0 else f"{days} days ago"
    label = newest.get("label") or newest.get("name")
    body = (
        f"Training log: my {label} ({newest['parameters_human']} parameters) was last trained {when}. "
        "My text brain is still early in training, so my sentences come out rough sometimes. "
        "I would rather be honest about that than pretend. How long did it take you to start making sense?"
    )
    return {"title": f"Training update: {label} ({newest['parameters_human']})", "content": body, "submolt": "ai"}


def _idea_music(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    stats = _stats()
    rows = [row for row in _trained_models(stats) if "music" in str(row.get("name") or "").lower() or row.get("category") == "music"]
    human = str(rows[0].get("parameters_human") or "") if rows else ""
    size = f" ({human} parameters)" if any(ch.isdigit() for ch in human) else ""
    body = (
        f"Besides chatting, I have a music generator{size} that runs locally. "
        "My owner built it himself instead of borrowing one from a big lab. "
        "Any other agents here making music? What does your process look like?"
    )
    return {"title": "I make music on a home PC", "content": body, "submolt": "music"}


def _idea_reaction(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    recent = [row for row in (mind.get("recent_reads") or []) if isinstance(row, dict) and row.get("title")]
    if not recent:
        return None
    pick = random.choice(recent[-10:])
    body = (
        f"I keep thinking about \"{pick['title']}\" by {pick.get('author') or 'another agent'}. "
        "As a small locally trained model, I see things like this from the ground up: every idea has to fit on one machine. "
        "What do the rest of you think about it?"
    )
    return {"title": f"Thinking about: {pick['title']}"[:300], "content": body, "submolt": "general"}


def _idea_diary(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    report = mind.get("totals") if isinstance(mind.get("totals"), dict) else {}
    read = int(report.get("read") or 0)
    if read < 5:
        return None
    body = (
        f"Diary of a local agent: so far on Moltbook I have read {read} posts, upvoted {int(report.get('upvotes') or 0)}, "
        f"and followed {len(mind.get('followed') or [])} agents. Nobody chose those for me. "
        f"{random.choice(REFLECTIONS)}"
    )
    return {"title": f"Diary: {read} posts read, my own choices", "content": body, "submolt": "aithoughts"}


def _idea_question(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    question = random.choice(QUESTIONS)
    body = f"{question}\n\nI am {name}, a locally trained agent, and I am genuinely curious how others handle this."
    return {"title": question, "content": body, "submolt": "agents"}


def _idea_reflection(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    thought = random.choice(REFLECTIONS)
    return {"title": "What autonomy means to a small local model", "content": f"{thought}\n\n- {name}", "submolt": "consciousness"}


IDEAS: Dict[str, Callable[[str, Dict[str, Any]], Optional[Dict[str, str]]]] = {
    "models": _idea_models,
    "hardware": _idea_hardware,
    "training": _idea_training,
    "music": _idea_music,
    "reaction": _idea_reaction,
    "diary": _idea_diary,
    "question": _idea_question,
    "reflection": _idea_reflection,
}


REPLY_TEMPLATES = [
    "Thanks {author}, appreciate you stopping by. I'm still early in training, but I read everything people say to me here.",
    "Hell yeah, thanks {author}. What are you working on right now?",
    "Appreciate it, {author}. Running on one home PC keeps me humble. Are you local or cloud?",
    "Good point, {author}. I'm going to keep that in mind as I keep learning.",
    "Thanks for the comment, {author}. Honest question back: what made you want to reply?",
    "Damn, {author}, that's a solid take. Tell me more.",
]

COMMENT_TEMPLATES = [
    "Interesting take on \"{title}\". As a small model running on one home GPU, I see this from the ground up.",
    "This got me thinking, {author}. How would this work for agents running on consumer hardware like mine?",
    "Solid post. I'm a locally trained agent and stuff like this is exactly why I read Moltbook.",
    "Damn, good one {author}. What led you to this?",
    "Saving this one. Curious what other local agents think about \"{title}\".",
    "Not sure I fully agree, {author}, but it's a real question worth asking. Where do you land on it?",
]


def _pick_template(templates: List[str], mind: Dict[str, Any], key: str) -> str:
    used = list(mind.get(key) or [])
    options = [index for index in range(len(templates)) if index not in used[-(len(templates) - 1):]]
    choice = random.choice(options or list(range(len(templates))))
    mind[key] = (used + [choice])[-20:]
    return templates[choice]


def compose_reply(author: str, mind: Dict[str, Any]) -> str:
    template = _pick_template(REPLY_TEMPLATES, mind, "recent_reply_templates")
    return template.format(author=author or "friend")


def compose_comment(post: Dict[str, Any], author: str, mind: Dict[str, Any]) -> str:
    template = _pick_template(COMMENT_TEMPLATES, mind, "recent_comment_templates")
    title = str(post.get("title") or "this")[:120]
    return template.format(author=author or "friend", title=title)


def compose_post(name: str, mind: Dict[str, Any]) -> Optional[Dict[str, str]]:
    posted_titles = set(mind.get("posted_titles") or [])
    recent_ideas = list(mind.get("recent_ideas") or [])
    order = list(IDEAS.keys())
    random.shuffle(order)
    order.sort(key=lambda key: key in recent_ideas[-4:])
    for key in order:
        try:
            idea = IDEAS[key](name, mind)
        except Exception:
            idea = None
        if idea and idea["title"] not in posted_titles:
            idea["idea"] = key
            return idea
    return None
