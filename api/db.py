from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from prisma.generated import Prisma


def get_db() -> Prisma:
    """
    Returns a Prisma client instance.

    The client is generated from `prisma/schema.prisma`.
    """
    return Prisma()


@contextmanager
def db_session() -> Iterator[Prisma]:
    db = get_db()
    db.connect()
    try:
        yield db
    finally:
        db.disconnect()

