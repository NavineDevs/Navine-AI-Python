from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "weights_manifest.json"
CHUNK = 8 * 1024 * 1024


def download(url: str, handle) -> None:
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(request) as response:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)


def restore(entry: dict, base_url: str, force: bool) -> None:
    target = ROOT / entry["path"]
    if target.exists() and target.stat().st_size == entry["size"] and not force:
        print(f"ok      {entry['path']}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    with partial.open("wb") as handle:
        for index, part in enumerate(entry["parts"], start=1):
            print(f"get     {entry['path']} part {index}/{len(entry['parts'])}")
            download(f"{base_url}/{part}", handle)
    if partial.stat().st_size != entry["size"]:
        partial.unlink()
        raise RuntimeError(f"Size mismatch for {entry['path']}")
    partial.replace(target)
    print(f"done    {entry['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights from the GitHub release.")
    parser.add_argument("--only", default="", help="Only restore paths containing this text")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()
    if not MANIFEST.exists():
        print("weights_manifest.json not found", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base_url = f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}"
    entries = [e for e in manifest["files"] if args.only in e["path"]]
    for entry in entries:
        restore(entry, base_url, args.force)
    print(f"Restored {len(entries)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
