import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navine.inspect.image_inspector import inspect_image_model
from navine.inspect.text_inspector import inspect_text_model
from navine.device_manager import device_summary


def main() -> None:
    text = inspect_text_model(write_report=True)
    image = inspect_image_model(write_report=True)
    dev, summary = device_summary()
    report_path = Path(ROOT) / "logs" / "model_report.txt"
    extra = [
        "",
        "Device Summary",
        "=" * 40,
        f"device: {summary['device']}",
        f"mode: {summary['mode']}",
        f"dtype: {summary['dtype']}",
        f"cuda_available: {summary['hardware'].get('cuda_available')}",
        f"mps_available: {summary['hardware'].get('mps_available')}",
    ]
    if summary["hardware"].get("cuda_name"):
        extra.append(f"cuda_name: {summary['hardware']['cuda_name']}")
        extra.append(f"cuda_vram_gb: {summary['hardware'].get('cuda_vram_gb')}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(existing.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(f"Wrote combined report to {report_path}")
    print(f"Text OK: {text.get('ok')} | Image OK: {image.get('ok')} | Device: {dev}")


if __name__ == "__main__":
    main()
