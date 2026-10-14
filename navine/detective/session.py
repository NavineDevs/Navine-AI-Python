from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def solve_mystery(
    prompt: str,
    output_dir: Optional[str] = None,
    seed: int = 0,
    include_image: bool = True,
    include_video: bool = True,
) -> Dict[str, Any]:
    from navine.text.chat import chat

    analysis = chat(prompt, model_profile="detective", use_search=True)
    scene_prompt = f"noir detective mystery scene inspired by: {prompt[:240]}"
    out_root = Path(output_dir) if output_dir else Path("outputs") / "detective"
    out_root.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "prompt": prompt,
        "analysis": analysis,
        "image": None,
        "video": None,
    }
    if include_image:
        from navine.image.infer import generate as generate_image

        image_path = generate_image(
            scene_prompt,
            output_path=str(out_root / "clue_scene.png"),
            seed=seed,
            model_profile="detective",
        )
        result["image"] = str(image_path)
    if include_video:
        from navine.video.infer import generate as generate_video

        video_path = generate_video(
            scene_prompt,
            output_path=str(out_root / "mystery_clip.mp4"),
            seed=seed,
            model_profile="detective",
        )
        result["video"] = str(video_path)
    manifest = out_root / "session.json"
    import json

    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest"] = str(manifest)
    return result
