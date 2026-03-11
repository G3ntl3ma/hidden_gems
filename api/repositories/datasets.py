from __future__ import annotations

from typing import Iterable

import pandas as pd


def from_rows(rows: Iterable[dict]) -> pd.DataFrame:
    """Convert row dicts (from DB/API) into a DataFrame."""
    return pd.DataFrame(list(rows))

