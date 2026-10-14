import shutil
import subprocess
from pathlib import Path
from typing import Optional


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def mux_audio_into_video(
    video_path: Path,
    audio_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    out = Path(output_path) if output_path else video_path.with_name(video_path.stem + "_audio" + video_path.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)

    if ffmpeg_available():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out.exists():
            return out
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-500:]}")

    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image

        reader = imageio.get_reader(str(video_path))
        fps = reader.get_meta_data().get("fps") or 12
        frames = [Image.fromarray(frame) for frame in reader]
        reader.close()
        if not frames:
            raise RuntimeError("No frames found in video")
        writer = imageio.get_writer(
            str(out),
            fps=int(fps),
            codec="libx264",
            audio_path=str(audio_path),
            macro_block_size=None,
        )
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
        writer.close()
        if out.exists():
            return out
    except Exception as exc:
        raise RuntimeError(
            "Could not mux audio into video. Install ffmpeg and add it to PATH, "
            f"or imageio[ffmpeg]. Detail: {exc}"
        ) from None

    raise RuntimeError("Audio mux produced no output file")


def narrate_video(
    video_path: Path,
    text: str,
    voice: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    from navine.assistant.voice import speak

    wav_path = Path(video_path).with_suffix(".narration.wav")
    speak(text, voice=voice, out_path=str(wav_path), play=False)
    return mux_audio_into_video(Path(video_path), wav_path, output_path=output_path)
