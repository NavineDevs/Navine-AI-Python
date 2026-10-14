from pathlib import Path
from typing import List, Optional, Tuple

from navine.utils.paths import get_project_root


def extract_frames(video_path: Path, window: int = 8, stride: int = 4, max_clips: int = 200) -> List[List[Path]]:
    try:
        import cv2
    except ImportError:
        return []
    root = get_project_root()
    out_root = root / "data" / "video" / "frames" / video_path.stem
    out_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frames: List[Path] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        path = out_root / f"frame_{index:05d}.jpg"
        cv2.imwrite(str(path), frame)
        frames.append(path)
        index += 1
    capture.release()
    clips: List[List[Path]] = []
    for start in range(0, max(0, len(frames) - window + 1), stride):
        clip = frames[start : start + window]
        if len(clip) == window:
            clips.append(clip)
        if len(clips) >= max_clips:
            break
    return clips


def scan_video_sources(dirs: Optional[List[str]] = None) -> List[Path]:
    root = get_project_root()
    rel_dirs = dirs or ["data/video", "data/nsfw/local"]
    videos: List[Path] = []
    for rel in rel_dirs:
        base = root / rel
        if not base.exists():
            continue
        for ext in ("*.mp4", "*.webm", "*.mov", "*.gif"):
            videos.extend(base.rglob(ext))
    return sorted(set(videos))
