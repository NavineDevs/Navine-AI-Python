from pathlib import Path
from typing import List

from navine.selfcode.io import get_project_root, is_searchable_dir

KEY_PACKAGES = (
    "text",
    "image",
    "video",
    "code",
    "search",
    "learn",
    "memory",
    "api",
    "auth",
    "train",
    "autolearn",
    "realtime",
    "selfcode",
)


def list_modules(max_depth: int = 2) -> List[str]:
    from navine.utils.brand import brand_package

    root = get_project_root()
    package_dir = root / brand_package()
    entries: List[str] = []
    if package_dir.is_dir():
        for name in sorted(package_dir.iterdir(), key=lambda p: p.name.lower()):
            if not name.is_dir():
                continue
            if not is_searchable_dir(name):
                continue
            rel = f"{brand_package()}/{name.name}"
            entries.append(rel)
            if max_depth >= 2 and name.name in KEY_PACKAGES:
                for child in sorted(name.iterdir(), key=lambda p: p.name.lower()):
                    if child.is_file() and child.suffix == ".py" and child.name != "__init__.py":
                        entries.append(f"{rel}/{child.name}")
                    elif child.is_dir() and is_searchable_dir(child):
                        if any(child.glob("*.py")):
                            entries.append(f"{rel}/{child.name}/")
    for top in ("web", "configs", "scripts", "app"):
        path = root / top
        if path.is_dir():
            entries.append(f"{top}/")
    return entries
