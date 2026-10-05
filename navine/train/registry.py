from typing import Any, Callable, Dict, List, Optional

from navine.train import chat, coding, creative, detective, games, general, hitboyx23, math, multimodal_text, nsfw, osint, thinking, unrestricted
from navine.voice.train import train_voice

TrainFn = Callable[..., None]

TRAINING_TYPES: Dict[str, Dict[str, Any]] = {
    "coding": {
        "description": "Python, JS, Rust, and other code generation",
        "data_format": "JSONL with language, prompt, code",
        "train_fn": coding.train,
        "supports_language": True,
    },
    "chat": {
        "description": "Conversational and instruction following",
        "data_format": "instruction_chat.txt blocks",
        "train_fn": chat.train,
        "supports_language": False,
    },
    "creative": {
        "description": "Stories, poems, and creative writing",
        "data_format": "plain text or JSONL",
        "train_fn": creative.train,
        "supports_language": False,
    },
    "math": {
        "description": "Math Q&A and step-by-step reasoning",
        "data_format": "JSONL question/answer",
        "train_fn": math.train,
        "supports_language": False,
    },
    "thinking": {
        "description": "Chain-of-thought and step-by-step analysis",
        "data_format": "instruction blocks in data/train/thinking/",
        "train_fn": thinking.train,
        "supports_language": False,
    },
    "detective": {
        "description": "Mystery puzzles, ciphers, and ARG-style investigation",
        "data_format": "instruction blocks and cipher JSONL in data/train/detective/",
        "train_fn": detective.train,
        "supports_language": False,
    },
    "osint": {
        "description": "Open-source intelligence, domain/username recon, and public records analysis",
        "data_format": "instruction blocks and source JSONL in data/train/osint/",
        "train_fn": osint.train,
        "supports_language": False,
    },
    "general": {
        "description": "Generic text corpus training",
        "data_format": "corpus txt",
        "train_fn": general.train,
        "supports_language": False,
    },
    "multimodal-text": {
        "description": "Captions for image and video alignment",
        "data_format": "JSONL caption pairs",
        "train_fn": multimodal_text.train,
        "supports_language": False,
    },
    "games": {
        "description": "Game strategy coaching and live-play observation dialogue",
        "data_format": "txt dialogue under data/train/games and chat gameplay files",
        "train_fn": games.train,
        "supports_language": False,
    },
    "nsfw": {
        "description": "Unrestricted adult and mature content training",
        "data_format": "instruction blocks in data/train/nsfw/",
        "train_fn": nsfw.train,
        "supports_language": False,
    },
    "unrestricted": {
        "description": "No-refusal direct responses for private local use",
        "data_format": "instruction blocks in data/train/unrestricted/",
        "train_fn": unrestricted.train,
        "supports_language": False,
    },
    "hitboyx23": {
        "description": "HitBoyXx23 AI full multimodal (all coding languages)",
        "data_format": "combined general, coding, thinking, detective data",
        "train_fn": hitboyx23.train,
        "supports_language": False,
    },
    "hitboyx23_python": {
        "description": "HitBoyXx23 AI full multimodal (Python only)",
        "data_format": "combined general, Python coding, thinking, detective data",
        "train_fn": hitboyx23.train_python,
        "supports_language": False,
    },
    "voice": {
        "description": "TTS voice calibration and custom voice registration",
        "data_format": "WAV samples under voices/ and outputs/voice/",
        "train_fn": train_voice,
        "supports_language": True,
    },
}

SEQUENTIAL_TYPES = [
    "general",
    "chat",
    "coding",
    "creative",
    "math",
    "thinking",
    "detective",
    "osint",
    "multimodal-text",
    "games",
    "nsfw",
    "unrestricted",
    "hitboyx23",
    "hitboyx23_python",
    "voice",
]

TYPE_ALIASES = {
    "uncensored": "nsfw",
    "adult": "nsfw",
    "no_refusal": "unrestricted",
    "norefusal": "unrestricted",
}


def canonical_training_type(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_")
    return TYPE_ALIASES.get(key, key)


def list_training_types() -> List[Dict[str, str]]:
    rows = []
    for name, meta in TRAINING_TYPES.items():
        rows.append(
            {
                "name": name,
                "description": meta["description"],
                "data_format": meta["data_format"],
            }
        )
    rows.append(
        {
            "name": "uncensored",
            "description": "Uncensored adult training (same corpus as NSFW)",
            "data_format": "instruction blocks in data/train/nsfw/",
        }
    )
    return rows


def train_type(
    name: str,
    language: Optional[str] = None,
    steps: Optional[int] = None,
) -> None:
    key = canonical_training_type(name)
    meta = TRAINING_TYPES.get(key)
    if not meta:
        raise ValueError(f"Unknown training type: {name}")
    fn: TrainFn = meta["train_fn"]
    if meta.get("supports_language"):
        fn(language=language, steps=steps)
    else:
        fn(steps=steps)


def train_all(language: Optional[str] = None, steps: Optional[int] = None) -> None:
    for name in SEQUENTIAL_TYPES:
        print(f"\n{'=' * 50}\nNavine AI - Python: starting {name} training\n{'=' * 50}")
        train_type(name, language=language, steps=steps)
