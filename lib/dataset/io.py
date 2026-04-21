"""CSV loading utilities for the Pokemon dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


class DatasetLoader:
    """Lazily loads CSV tables and caches DataFrames."""

    def __init__(self, data_dir: str | Path = "csv") -> None:
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    def table(self, name: str) -> pd.DataFrame:
        """Return the table `name` loaded from `<data_dir>/<name>.csv`."""
        if name not in self._cache:
            file_path = self.data_dir / f"{name}.csv"
            self._cache[name] = pd.read_csv(file_path)
        return self._cache[name].copy()
