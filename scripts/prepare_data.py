from pathlib import Path

from navine.utils.paths import get_project_root


def generate_sample_images():
    from PIL import Image
    root = get_project_root()
    out_dir = root / "data" / "image" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = 64
    for i in range(20):
        img = Image.new("RGB", (size, size))
        pixels = img.load()
        hue = (i * 18) % 360
        for y in range(size):
            for x in range(size):
                r = int(128 + 127 * ((x + i * 3) % size) / size)
                g = int(128 + 127 * ((y + i * 5) % size) / size)
                b = int(128 + 127 * (((x + y + i * 7) % size)) / size)
                pixels[x, y] = (r, g, b)
        img.save(out_dir / f"sample_{i:03d}.png")
    print(f"Created 20 sample images in {out_dir}")


def generate_sample_video_frames():
    from PIL import Image
    root = get_project_root()
    out_dir = root / "data" / "video" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = 64
    num_frames = 32
    for f in range(num_frames):
        img = Image.new("RGB", (size, size))
        pixels = img.load()
        offset = f * 5
        for y in range(size):
            for x in range(size):
                r = int(128 + 127 * ((x + offset) % size) / size)
                g = int(128 + 127 * ((y + offset // 2) % size) / size)
                b = int(128 + 127 * (((x + y + offset) % size)) / size)
                pixels[x, y] = (r, g, b)
        img.save(out_dir / f"frame_{f:03d}.png")
    print(f"Created {num_frames} video frames in {out_dir}")


def ensure_chat_data():
    root = get_project_root()
    chat_path = root / "data" / "text" / "instruction_chat.txt"
    if chat_path.exists():
        print(f"Chat instruction data present: {chat_path}")
        return
    sample = root / "data" / "text" / "sample_chat.txt"
    if not sample.exists():
        print("No chat data found.")
        return
    lines = sample.read_text(encoding="utf-8").splitlines()
    blocks = []
    current = []
    for line in lines:
        if line.startswith("User:") and current:
            blocks.append("\n".join(current))
            current = [f"### {line.replace('User:', 'User:', 1)}"]
        elif line.startswith("Assistant:"):
            current.append(f"### {line.replace('Assistant:', 'Assistant:', 1)}")
        elif line.startswith("User:"):
            current = [f"### {line.replace('User:', 'User:', 1)}"]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    chat_path.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"Generated instruction chat data: {chat_path}")


if __name__ == "__main__":
    ensure_chat_data()
    generate_sample_images()
    generate_sample_video_frames()
