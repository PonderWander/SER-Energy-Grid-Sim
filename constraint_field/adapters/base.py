"""
constraint_field.adapters.base
==============================
Abstract base class for all data source adapters.

Every adapter must implement:
  - fetch(start, end, **kwargs) -> pd.DataFrame
  - name property

This enforces a uniform interface so adapters can be swapped behind
field construction without changing downstream code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Abstract data-source adapter."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Interface every subclass must satisfy
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this data source."""

    @abstractmethod
    def _fetch_raw(self, start: str, end: str, **kwargs) -> pd.DataFrame:
        """
        Download raw data for [start, end] and return a DataFrame with at
        minimum a DatetimeIndex and at least one value column.

        Parameters
        ----------
        start, end : str
            ISO 8601 date strings, e.g. "2023-01-01".
        """

    # ------------------------------------------------------------------
    # Public method: fetch with caching
    # ------------------------------------------------------------------

    def fetch(
        self,
        start: str,
        end: str,
        force_refresh: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return a DataFrame for [start, end].

        Results are cached to disk as Parquet.  Set force_refresh=True to
        bypass the cache and re-download.
        """
        cache_path = self._cache_path(start, end, kwargs)

        if not force_refresh and cache_path.exists():
            log.info("[%s] cache hit: %s", self.name, cache_path.name)
            return pd.read_parquet(cache_path)

        log.info("[%s] fetching %s → %s …", self.name, start, end)
        df = self._fetch_raw(start, end, **kwargs)
        df = self._validate(df)

        df.to_parquet(cache_path)
        log.info("[%s] cached to %s", self.name, cache_path.name)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_path(self, start: str, end: str, kwargs: dict) -> Path:
        """Deterministic cache filename based on adapter + params."""
        key = json.dumps({"adapter": self.name, "start": start, "end": end, **kwargs},
                         sort_keys=True)
        digest = hashlib.md5(key.encode()).hexdigest()[:10]
        return self.cache_dir / f"{self.name}_{start}_{end}_{digest}.parquet"

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        """Minimal sanity checks on fetched data."""
        if df.empty:
            raise ValueError("Adapter returned an empty DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        return df

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cache_dir={self.cache_dir})"
