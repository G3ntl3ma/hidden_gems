from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "view" / "app.py").exists() and (candidate / "prisma" / "schema.prisma").exists():
            return candidate

    return Path(__file__).resolve().parents[1]


def ensure_project_root(start: Path) -> Path:
    root = find_project_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

