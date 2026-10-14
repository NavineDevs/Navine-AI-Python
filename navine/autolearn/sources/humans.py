from typing import Any, Dict, List, Optional

from navine.autolearn.sources import reddit

DEFAULT_HUMAN_SUBREDDITS = [
    "AskReddit",
    "NoStupidQuestions",
    "explainlikeimfive",
    "todayilearned",
    "LifeProTips",
    "relationship_advice",
    "offmychest",
    "unpopularopinion",
    "AmItheAsshole",
    "personalfinance",
    "worldnews",
    "news",
    "technology",
    "science",
    "Futurology",
]


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    subs = config.get("human_subreddits") or DEFAULT_HUMAN_SUBREDDITS
    per_sub = max(2, (max_items or 40) // max(1, len(subs)))
    items: List[Dict[str, Any]] = []
    for sub in subs:
        try:
            batch = reddit.fetch_reddit(str(sub).strip(), max_posts=per_sub)
            for entry in batch:
                entry["source"] = "humans"
                entry["source_id"] = f"humans:{sub}:{entry.get('source_id', '')}"
                items.append(entry)
        except Exception:
            continue
        if max_items and len(items) >= max_items:
            break
    return items[: max_items or len(items)]
