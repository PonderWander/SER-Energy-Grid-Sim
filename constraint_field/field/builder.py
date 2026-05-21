"""
constraint_field.field.builder
================================
Constructs the reduced static field from raw adapter outputs.

The two primary field variables:

  S (load pressure)
  -----------------
  Operationally: normalized demand / usage signal.
  S > 0  →  above-average demand pressure
  S < 0  →  below-average (slack) demand
  S is a dimensionless z-score unless minmax normalisation is chosen.

  R (constraint signal)
  ---------------------
  Operationally: normalized price / constraint-intensity signal.
  R > 0  →  elevated constraint pressure (expensive, scarce delivery)
  R < 0  →  slack constraint environment
  R is NOT treated as a market equilibrium output; it is read as a
  constraint-field intensity — the degree to which the system is
  pressing against its delivery limits.

Both variables are constructed to be comparable in scale so that
gradient and imbalance metrics are meaningful across the S–R plane.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NormMethod = Literal["minmax", "zscore", "rolling_zscore"]


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalise(
    series: pd.Series,
    method: NormMethod = "rolling_zscore",
    window: int = 168,
    clip_sigma: float = 3.0,
) -> pd.Series:
    """
    Normalise a raw series into a dimensionless field variable.

    Parameters
    ----------
    series : pd.Series
        Raw values (e.g. MWh demand, $/MWh LMP).
    method : str
        "minmax"         – scale to [0, 1] over full sample
        "zscore"         – full-sample z-score
        "rolling_zscore" – z-score against rolling window (removes trend)
    window : int
        Rolling window size in periods (only used for rolling_zscore).
    clip_sigma : float
        Clip at ±clip_sigma after normalisation (prevents spike dominance).

    Returns
    -------
    pd.Series
        Normalised values, same index as input.
    """
    s = series.copy().astype(float)

    if method == "minmax":
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.0, index=s.index, name=series.name)
        out = (s - lo) / (hi - lo)

    elif method == "zscore":
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            return pd.Series(0.0, index=s.index, name=series.name)
        out = (s - mu) / sigma

    elif method == "rolling_zscore":
        roll_mean = s.rolling(window=window, min_periods=max(1, window // 4)).mean()
        roll_std  = s.rolling(window=window, min_periods=max(1, window // 4)).std()
        roll_std  = roll_std.replace(0, np.nan)
        out = (s - roll_mean) / roll_std
        # Fill leading NaN from rolling with global zscore for warm-up
        global_z = (s - s.mean()) / (s.std() or 1.0)
        out = out.fillna(global_z)

    else:
        raise ValueError(f"Unknown normalisation method: '{method}'")

    return out.clip(-clip_sigma, clip_sigma).rename(series.name)


# ──────────────────────────────────────────────────────────────────────────────
# FieldBuilder
# ──────────────────────────────────────────────────────────────────────────────

class FieldBuilder:
    """
    Assembles S and R field panels from adapter DataFrames.

    Usage
    -----
    >>> builder = FieldBuilder(cfg["field"])
    >>> panel = builder.build(demand_df, price_df)
    >>> S_slice = panel["S"].loc["2023-02-14"]
    """

    def __init__(self, field_cfg: dict):
        self.cfg = field_cfg
        self.freq     = field_cfg.get("resample_freq", "1h")
        self.fill     = field_cfg.get("fill_method", "ffill")
        self.max_gap  = field_cfg.get("max_gap_hours", 3)

        # S config
        s_cfg = field_cfg.get("S", {})
        self.s_norm   = s_cfg.get("normalization", "rolling_zscore")
        self.s_window = s_cfg.get("rolling_window_hours", 168)
        self.s_clip   = s_cfg.get("clip_sigma", 3.0)

        # R config
        r_cfg = field_cfg.get("R", {})
        self.r_norm   = r_cfg.get("normalization", "rolling_zscore")
        self.r_window = r_cfg.get("rolling_window_hours", 168)
        self.r_clip   = r_cfg.get("clip_sigma", 3.0)
        self.r_raw_cap = r_cfg.get("raw_cap", 500.0)

    # ------------------------------------------------------------------
    # Primary build method
    # ------------------------------------------------------------------

    def build(
        self,
        demand_df: pd.DataFrame,
        price_df: pd.DataFrame,
        flows_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Construct the field panel [S, R] (and optionally raw signals).

        Parameters
        ----------
        demand_df : pd.DataFrame
            Must contain column 'demand_mwh'.
        price_df : pd.DataFrame
            Must contain column 'lmp_total' (or 'lmp_energy' as fallback).
        flows_df : pd.DataFrame | None
            Optional; if present, 'net_flow_mw' is attached for E inference.

        Returns
        -------
        pd.DataFrame
            Panel with columns: S, R, demand_mwh_raw, lmp_raw,
            [net_flow_mw_raw if provided], [lmp_congestion_raw if provided]
        """
        # ── 1. Extract raw signals ──────────────────────────────────────
        demand_raw = self._extract_demand(demand_df)
        price_raw  = self._extract_price(price_df)

        # ── 2. Align to common UTC hourly index ──────────────────────────
        demand_raw, price_raw = self._align(demand_raw, price_raw)

        # ── 3. Normalise → S, R ─────────────────────────────────────────
        S = normalise(demand_raw, self.s_norm, self.s_window, self.s_clip)
        S.name = "S"

        R = normalise(price_raw, self.r_norm, self.r_window, self.r_clip)
        R.name = "R"

        # ── 4. Assemble panel ────────────────────────────────────────────
        panel = pd.DataFrame({"S": S, "R": R,
                               "demand_mwh_raw": demand_raw,
                               "lmp_raw": price_raw})

        # Optional extras for E inference
        if flows_df is not None and "net_flow_mw" in flows_df.columns:
            flow_raw = (flows_df["net_flow_mw"]
                        .resample(self.freq).mean()
                        .reindex(panel.index)
                        .interpolate(limit=self.max_gap))
            panel["net_flow_mw_raw"] = flow_raw

        if "lmp_congestion" in price_df.columns:
            cong = (price_df["lmp_congestion"]
                    .resample(self.freq).mean()
                    .reindex(panel.index)
                    .interpolate(limit=self.max_gap))
            panel["lmp_congestion_raw"] = cong

        if "lmp_total" in price_df.columns and "lmp_congestion" in price_df.columns:
            panel["lmp_spread_raw"] = (
                price_df["lmp_total"] - price_df["lmp_congestion"]
            ).resample(self.freq).mean().reindex(panel.index).interpolate(limit=self.max_gap)

        log.info(
            "Field panel built: %d rows, columns=%s, missing S=%.1f%%, missing R=%.1f%%",
            len(panel),
            list(panel.columns),
            panel["S"].isna().mean() * 100,
            panel["R"].isna().mean() * 100,
        )
        return panel

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_demand(self, df: pd.DataFrame) -> pd.Series:
        if "demand_mwh" not in df.columns:
            raise KeyError("demand_df must contain 'demand_mwh' column.")
        s = df["demand_mwh"].copy().astype(float)
        s = s[s > 0]   # drop non-positive (physically implausible)
        return s

    def _extract_price(self, df: pd.DataFrame) -> pd.Series:
        if "lmp_total" in df.columns:
            col = "lmp_total"
        elif "lmp_energy" in df.columns:
            col = "lmp_energy"
        else:
            raise KeyError("price_df must contain 'lmp_total' or 'lmp_energy'.")
        s = df[col].copy().astype(float)
        s = s.clip(upper=self.r_raw_cap)
        return s

    def _align(
        self, demand: pd.Series, price: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Resample both series to common UTC hourly index, fill gaps."""
        demand = demand.resample(self.freq).mean()
        price  = price.resample(self.freq).mean()

        # Common index = union of both, then restrict to overlap
        common = demand.index.intersection(price.index)
        if len(common) == 0:
            raise ValueError(
                "Demand and price series have no overlapping timestamps. "
                "Check date ranges and timezone handling."
            )
        demand = demand.reindex(common)
        price  = price.reindex(common)

        # Fill gaps (forward-fill limited to max_gap)
        demand = demand.ffill(limit=self.max_gap).bfill(limit=1)
        price  = price.ffill(limit=self.max_gap).bfill(limit=1)

        # Report remaining NaN after fill
        for name, ser in [("demand", demand), ("price", price)]:
            n_nan = ser.isna().sum()
            if n_nan:
                log.warning("[align] %d NaNs remain in %s after fill", n_nan, name)

        return demand, price
