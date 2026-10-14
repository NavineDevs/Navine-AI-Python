from pathlib import Path
from typing import Any, Dict, List

from navine.image.quality import score_batch, score_image_path
from navine.utils.config import load_config
from navine.utils.paths import get_output_dir
from navine.utils.tier import resolve_checkpoint_dir

def _enterprise_image_prompts() -> List[str]:
    return [
        "photoreal portrait of a woman, studio lighting, sharp focus",
        "photoreal full body, natural skin texture, detailed anatomy",
        "hentai anime style character, detailed linework, vibrant colors",
    ]


def _enterprise_text_ckpt():
    cfg = load_config("text_enterprise")
    return resolve_checkpoint_dir("text", cfg) / "latest.pt"


def _enterprise_image_cfg():
    return load_config("image_enterprise")


def eval_text_enterprise() -> Dict[str, Any]:
    result: Dict[str, Any] = {"score": 0.0, "checks": []}
    try:
        from navine.text.chat import build_chat_prompt, clean_response
        from navine.text.infer import generate

        prompts = [
            "Write a Python function that merges two sorted lists.",
            "Explain quicksort in three sentences.",
            "Write a Python fibonacci function.",
        ]
        total = 0.0
        for user_prompt in prompts:
            prompt = build_chat_prompt(user_prompt, None, None, None, None, code_mode=("fibonacci" in user_prompt.lower() or "function" in user_prompt.lower()))
            out = generate(prompt, max_new_tokens=220, temperature=0.45)
            cleaned = clean_response(out, user_prompt, code_mode="def " in user_prompt.lower())
            check: Dict[str, Any] = {"prompt": user_prompt, "length": len(cleaned)}
            pts = 0.0
            lower = cleaned.lower()
            if len(cleaned.strip()) >= 40:
                pts += 0.25
            if "def " in cleaned or "return" in lower or "function" in lower:
                pts += 0.25
            if "sort" in lower or "merge" in lower or "quicksort" in lower or "fibonacci" in lower or "partition" in lower:
                pts += 0.25
            if "error" not in lower[:40] and "glossary" not in lower:
                pts += 0.25
            check["points"] = pts
            total += pts
            result["checks"].append(check)
        result["score"] = round(total / max(len(prompts), 1), 4)
        result["passes"] = result["score"] >= 0.55
    except Exception as exc:
        result["error"] = str(exc)
    return result


def eval_image_enterprise(prompts: List[str] | None = None) -> Dict[str, Any]:
    from navine.image.infer import generate

    cfg = _enterprise_image_cfg()
    ckpt = resolve_checkpoint_dir("image", cfg) / "latest.pt"
    ckpt_arg = str(ckpt) if ckpt.exists() else None
    out_dir = get_output_dir("image") / "enterprise_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, prompt in enumerate(prompts or _enterprise_image_prompts()):
        path = out_dir / f"eval_{i}.png"
        try:
            generate(
                prompt,
                output_path=str(path),
                enhance=True,
                auto_ingest=False,
                config_name="image_enterprise",
                checkpoint=ckpt_arg,
            )
            paths.append(path)
        except Exception:
            pass
    if not paths:
        return {"average_score": 0.0, "all_pass": False, "results": [], "error": "no outputs"}
    return score_batch(paths)


def eval_video_enterprise() -> Dict[str, Any]:
    result: Dict[str, Any] = {"passes": False, "frames_scored": 0}
    try:
        from navine.video.infer import generate as generate_video

        out_path = get_output_dir("video") / "enterprise_eval.mp4"
        generate_video("slow camera pan, cinematic lighting", output_path=str(out_path))
        result["output"] = str(out_path)
        result["exists"] = out_path.exists()
        if out_path.exists() and out_path.stat().st_size > 4096:
            result["passes"] = True
            result["size_bytes"] = out_path.stat().st_size
    except Exception as exc:
        result["error"] = str(exc)
    return result


def run_enterprise_eval(sample_prompts: Dict[str, str] | None = None) -> Dict[str, Any]:
    samples = sample_prompts or {}
    text_prompt = samples.get("text", "Write a Python hello world function.")
    image_prompt = samples.get("image", "photoreal portrait, sharp focus")
    return {
        "text": eval_text_enterprise(),
        "image": eval_image_enterprise([image_prompt] + _enterprise_image_prompts()[1:]),
        "video": eval_video_enterprise(),
        "text_custom": text_prompt,
        "image_custom": image_prompt,
    }


def score_checkpoint_image_raw(image_path: Path) -> Dict[str, Any]:
    return score_image_path(image_path)
