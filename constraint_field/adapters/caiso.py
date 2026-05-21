"""
constraint_field.adapters.caiso
================================
Adapter for CAISO OASIS public API.

Fetches:
  - Real-time interval LMP prices (PRC_INTVL_LMP)
  - System-level scheduling limit / inter-tie flows (ENE_SLRS)

OASIS API reference: http://www.caiso.com/market/Pages/OASIS/default.aspx
No authentication required.

Column contract
---------------
prices DataFrame  (UTC DatetimeIndex):
  lmp_total    : total LMP ($/MWh)
  lmp_energy   : energy component
  lmp_congestion: congestion component
  lmp_loss     : loss component
  node         : pricing node name

flows DataFrame  (UTC DatetimeIndex):
  net_flow_mw  : net scheduled interchange (positive = import)
  tie_name     : inter-tie identifier
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd
import requests

from .base import BaseAdapter

log = logging.getLogger(__name__)

OASIS_BASE = "http://oasis.caiso.com/oasisapi/SingleZip"


class CAISOAdapter(BaseAdapter):
    """
    Pulls price and interchange data from CAISO OASIS.

    Parameters
    ----------
    cache_dir : str
    node_filter : str | None
        Pricing node to extract from the LMP report.
        Default: "TH_NP15_GEN-APND" (NP15 trading hub).
    market_run_id : str
        "RTM" (real-time market) or "DAM" (day-ahead market).
    chunk_days : int
        OASIS has a 31-day query limit per request; we chunk automatically.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        node_filter: str = "TH_NP15_GEN-APND",
        market_run_id: str = "RTM",
        chunk_days: int = 7,
    ):
        super().__init__(cache_dir)
        self.node_filter = node_filter
        self.market_run_id = market_run_id
        self.chunk_days = chunk_days

    @property
    def name(self) -> str:
        return f"caiso_{self.market_run_id}_{self.node_filter[:10]}"

    # ------------------------------------------------------------------
    # Top-level fetch
    # ------------------------------------------------------------------

    def _fetch_raw(self, start: str, end: str, **kwargs) -> pd.DataFrame:
        """Fetch LMP prices and combine into a single DataFrame."""
        price_chunks = []
        for chunk_start, chunk_end in self._date_chunks(start, end):
            df = self._fetch_lmp_chunk(chunk_start, chunk_end)
            if not df.empty:
                price_chunks.append(df)
            time.sleep(1.0)   # OASIS rate limit courtesy pause

        if not price_chunks:
            raise RuntimeError(
                f"No CAISO LMP data retrieved for {start} → {end}. "
                "Check OASIS availability or try a different date range."
            )
        return pd.concat(price_chunks).sort_index()

    def fetch_flows(self, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Separately fetch inter-tie scheduling data.
        Returns DataFrame with net_flow_mw indexed by UTC timestamp.
        """
        cache_key = {"adapter": self.name + "_flows", "start": start, "end": end}
        import hashlib, json
        digest = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:10]
        cache_path = self.cache_dir / f"{self.name}_flows_{start}_{end}_{digest}.parquet"

        if not force_refresh and cache_path.exists():
            log.info("[CAISO flows] cache hit")
            return pd.read_parquet(cache_path)

        chunks = []
        for chunk_start, chunk_end in self._date_chunks(start, end):
            df = self._fetch_flow_chunk(chunk_start, chunk_end)
            if not df.empty:
                chunks.append(df)
            time.sleep(1.0)

        if not chunks:
            log.warning("[CAISO flows] no data returned; returning empty.")
            return pd.DataFrame()

        result = pd.concat(chunks).sort_index()
        result.to_parquet(cache_path)
        return result

    # ------------------------------------------------------------------
    # LMP fetch
    # ------------------------------------------------------------------

    def _fetch_lmp_chunk(self, start: str, end: str) -> pd.DataFrame:
        """Fetch one chunk of LMP data from OASIS."""
        params = {
            "queryname": "PRC_INTVL_LMP",
            "startdatetime": self._to_oasis_dt(start, hour=0),
            "enddatetime":   self._to_oasis_dt(end,   hour=23),
            "version": 1,
            "market_run_id": self.market_run_id,
            "node": self.node_filter,
            "resultformat": 6,   # CSV in ZIP
        }

        try:
            resp = requests.get(OASIS_BASE, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("[CAISO LMP] request failed for %s→%s: %s", start, end, exc)
            return pd.DataFrame()

        return self._parse_lmp_zip(resp.content, start, end)

    def _parse_lmp_zip(self, content: bytes, start: str, end: str) -> pd.DataFrame:
        """Extract and parse the CSV inside the OASIS ZIP response."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    log.warning("[CAISO LMP] ZIP contained no CSV for %s→%s", start, end)
                    return pd.DataFrame()
                raw = zf.read(csv_names[0])
        except zipfile.BadZipFile:
            log.warning("[CAISO LMP] bad ZIP response for %s→%s", start, end)
            return pd.DataFrame()

        df = pd.read_csv(io.BytesIO(raw))

        # OASIS column names vary by version; normalise
        df.columns = df.columns.str.strip().str.upper()

        # Expected columns: INTERVALSTARTTIME_GMT, MW, LMP_TYPE, NODE, ...
        ts_col = next((c for c in df.columns if "STARTTIME" in c), None)
        if ts_col is None:
            log.warning("[CAISO LMP] no timestamp column in CSV")
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index("timestamp")

        # Pivot LMP_TYPE into columns: LMP, MCC (congestion), MCE (loss)
        if "LMP_TYPE" in df.columns and "MW" in df.columns:
            df = df[["LMP_TYPE", "MW", "NODE"]].copy() if "NODE" in df.columns \
                else df[["LMP_TYPE", "MW"]].copy()
            df["MW"] = pd.to_numeric(df["MW"], errors="coerce")

            pivot = df.pivot_table(
                index="timestamp", columns="LMP_TYPE", values="MW", aggfunc="mean"
            )
            rename = {"LMP": "lmp_total", "MCC": "lmp_congestion", "MCE": "lmp_loss"}
            pivot.rename(columns={k: v for k, v in rename.items() if k in pivot.columns},
                         inplace=True)

            # derive energy component if possible
            if "lmp_total" in pivot.columns and "lmp_congestion" in pivot.columns \
                    and "lmp_loss" in pivot.columns:
                pivot["lmp_energy"] = (pivot["lmp_total"]
                                       - pivot["lmp_congestion"]
                                       - pivot["lmp_loss"])
            return pivot

        # Fallback: return whatever numeric columns exist
        num_cols = df.select_dtypes("number").columns.tolist()
        return df[num_cols]

    # ------------------------------------------------------------------
    # Flow / interchange fetch
    # ------------------------------------------------------------------

    def _fetch_flow_chunk(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch system-level energy scheduling limits (proxy for inter-tie flows).
        OASIS query: ENE_SLRS
        """
        params = {
            "queryname": "ENE_SLRS",
            "startdatetime": self._to_oasis_dt(start, hour=0),
            "enddatetime":   self._to_oasis_dt(end,   hour=23),
            "version": 1,
            "market_run_id": self.market_run_id,
            "resultformat": 6,
        }

        try:
            resp = requests.get(OASIS_BASE, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("[CAISO flows] request failed: %s", exc)
            return pd.DataFrame()

        return self._parse_flow_zip(resp.content)

    def _parse_flow_zip(self, content: bytes) -> pd.DataFrame:
        """Parse the scheduling limits ZIP."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return pd.DataFrame()
                raw = zf.read(csv_names[0])
        except zipfile.BadZipFile:
            return pd.DataFrame()

        df = pd.read_csv(io.BytesIO(raw))
        df.columns = df.columns.str.strip().str.upper()

        ts_col = next((c for c in df.columns if "STARTTIME" in c), None)
        if ts_col is None:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index("timestamp")

        # Return a simplified net flow proxy: sum of MW columns
        mw_cols = [c for c in df.columns if "MW" in c]
        if mw_cols:
            df["net_flow_mw"] = df[mw_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            return df[["net_flow_mw"]]
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_oasis_dt(date_str: str, hour: int = 0) -> str:
        """Convert 'YYYY-MM-DD' to OASIS datetime format 'YYYYMMDDTHH:00-0000'."""
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour)
        return dt.strftime("%Y%m%dT%H:00-0000")

    def _date_chunks(self, start: str, end: str) -> Iterator[tuple[str, str]]:
        """Split [start, end] into chunks of at most chunk_days days."""
        cur = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        while cur <= end_dt:
            chunk_end = min(cur + timedelta(days=self.chunk_days - 1), end_dt)
            yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            cur = chunk_end + timedelta(days=1)
