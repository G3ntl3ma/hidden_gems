from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from generated import Prisma  # pragma: no cover


def _load_generated_prisma() -> type["Prisma"]:
    """
    Import the generated Prisma client package.

    The generator outputs code to `prisma/generated/`, but the generated package
    name is still `prisma`. To avoid shadowing the PyPI `prisma` CLI module,
    we do not make the repo's `prisma/` directory a Python package.
    Instead, we temporarily add the generated output folder to `sys.path`.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    prisma_dir = repo_root / "prisma"
    # The generated client lives in `prisma/generated/` and is importable
    # as a top-level package named `generated` when `prisma/` is on sys.path.
    sys.path.insert(0, str(prisma_dir))

    from generated import Prisma  # type: ignore

    return Prisma


def get_db() -> "Prisma":
    """
    Returns a Prisma client instance.

    The client is generated from `prisma/schema.prisma`.
    """
    PrismaClient = _load_generated_prisma()
    return PrismaClient()


@contextmanager
def db_session() -> Iterator[Prisma]:
    db = get_db()
    db.connect()
    try:
        yield db
    finally:
        db.disconnect()

