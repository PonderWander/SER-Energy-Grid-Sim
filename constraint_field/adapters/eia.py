"""
constraint_field.adapters.eia
==============================
Adapter for EIA Open Data API v2.

Fetches:
  - Hourly electricity demand (D) by balancing authority
  - Hourly net generation (NG) by balancing authority  [optional]

API reference: https://www.eia.gov/opendata/
No API key required for the public endpoint used here.

Column contract
---------------
Returned DataFrame has a UTC DatetimeIndex and columns:
  demand_mwh    : hourly demand in MWh
  generation_mwh: hourly net generation in MWh  (if available)
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import pandas as pd
import requests

from .base import BaseAdapter

log = logging.getLogger(__name__)

EIA_V2_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


class EIAAdapter(BaseAdapter):
    """
    Pulls hourly demand and generation from EIA's Open Data v2 API.

    Parameters
    ----------
    region : str
        EIA balancing-authority abbreviation, e.g. "CISO" (CAISO),
        "MISO", "PJM", "ERCO" (ERCOT), "ISNE" (ISO-NE).
    cache_dir : str | Path
    api_key : str | None
        Optional EIA API key for higher rate limits.  None = anonymous.
    """

    def __init__(
        self,
        region: str = "CISO",
        cache_dir: str = "data/cache",
        api_key: str | None = None,
    ):
        super().__init__(cache_dir)
        self.region = region.upper()
        self.api_key = api_key

    @property
    def name(self) -> str:
        return f"eia_{self.region}"

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def _fetch_raw(self, start: str, end: str, **kwargs) -> pd.DataFrame:
        """Download demand + generation for [start, end]."""
        demand = self._fetch_series(start, end, series_type="D")
        gen    = self._fetch_series(start, end, series_type="NG")

        df = demand.rename(columns={"value": "demand_mwh"})
        if not gen.empty:
            df["generation_mwh"] = gen["value"].reindex(df.index)

        df.sort_index(inplace=True)
        return df

    def _fetch_series(
        self,
        start: str,
        end: str,
        series_type: Literal["D", "NG"] = "D",
    ) -> pd.DataFrame:
        """
        Fetch one series type from EIA v2 API.

        EIA returns paginated JSON; this method handles pagination
        automatically using offset until all rows are retrieved.
        """
        all_rows: list[dict] = []
        offset = 0
        page_size = 5000  # EIA max per request

        params: dict = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": self.region,
            "facets[type][]": series_type,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": page_size,
            "offset": offset,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        while True:
            params["offset"] = offset
            try:
                resp = requests.get(EIA_V2_BASE, params=params, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                log.warning("[EIA] request failed: %s", exc)
                break

            payload = resp.json()
            rows = payload.get("response", {}).get("data", [])
            if not rows:
                break

            all_rows.extend(rows)
            total = payload.get("response", {}).get("total", 0)
            offset += len(rows)
            if offset >= total:
                break
            time.sleep(0.3)   # polite pause between pages

        if not all_rows:
            log.warning("[EIA] no data returned for %s %s %s→%s",
                        self.region, series_type, start, end)
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["period"] = pd.to_datetime(df["period"], utc=True)
        df = df.set_index("period")[["value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.loc[~df.index.duplicated(keep="first")]
        return df

    # ------------------------------------------------------------------
    # Convenience: compute load factor
    # ------------------------------------------------------------------

    @staticmethod
    def load_factor(df: pd.DataFrame) -> pd.Series:
        """
        Demand / generation ratio as a crude utilisation proxy.
        Values > 1 imply net import; < 1 implies net export.
        """
        if "generation_mwh" not in df.columns or "demand_mwh" not in df.columns:
            raise ValueError("DataFrame must contain demand_mwh and generation_mwh.")
        return df["demand_mwh"] / df["generation_mwh"].replace(0, float("nan"))
