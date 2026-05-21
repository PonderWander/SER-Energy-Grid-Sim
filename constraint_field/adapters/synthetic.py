"""
constraint_field.adapters.synthetic
=====================================
Synthetic data adapter for unit-testing and offline demos.

Generates realistic-looking load and price series with:
  - daily seasonality (morning/evening demand peaks)
  - weekly seasonality (lower weekends)
  - correlated price spikes (heat events, constraint windows)
  - configurable inter-tie flow signal
  - optional shock events

Column contract (same as EIA + CAISO adapters)
-----------------------------------------------
demand_mwh, generation_mwh    (from fetch())
lmp_total, lmp_congestion,
lmp_energy, lmp_loss           (from fetch_prices())
net_flow_mw                    (from fetch_flows())
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseAdapter

log = logging.getLogger(__name__)

RNG = np.random.default_rng(seed=42)


class SyntheticAdapter(BaseAdapter):
    """
    Fully synthetic electricity market data adapter.

    Used for:
      - development and testing without network access
      - reproducible demonstration notebooks
      - unit tests for field construction and inference modules

    Parameters
    ----------
    peak_demand_mw : float
        Nominal system peak demand.
    base_price : float
        Base LMP level ($/MWh).
    congestion_prob : float
        Hourly probability that a congestion spike event begins.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        peak_demand_mw: float = 40_000.0,
        base_price: float = 45.0,
        congestion_prob: float = 0.02,
        seed: int = 42,
    ):
        super().__init__(cache_dir)
        self.peak_demand_mw = peak_demand_mw
        self.base_price = base_price
        self.congestion_prob = congestion_prob
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "synthetic"

    # ------------------------------------------------------------------
    # Demand / generation
    # ------------------------------------------------------------------

    def _fetch_raw(self, start: str, end: str, **kwargs) -> pd.DataFrame:
        """Generate synthetic demand and generation."""
        idx = self._make_index(start, end)
        n = len(idx)

        # Daily load shape: double-peak (morning ~9h, evening ~19h)
        hour = idx.hour.values
        daily_shape = (
            0.70
            + 0.15 * np.exp(-((hour - 9) ** 2) / 8)
            + 0.20 * np.exp(-((hour - 19) ** 2) / 6)
        )

        # Weekly shape: lower on weekends
        dow = idx.dayofweek.values
        weekly_factor = np.where(dow >= 5, 0.85, 1.0)

        # Long-term trend (slight growth over study period)
        trend = 1.0 + np.linspace(0, 0.02, n)

        # Noise
        noise = self.rng.normal(0, 0.025, n)

        demand = self.peak_demand_mw * daily_shape * weekly_factor * trend * (1 + noise)
        demand = np.clip(demand, 0, None)

        # Generation = demand ± small mismatch (import/export variation)
        flow_variation = self.rng.normal(0, 0.05, n) * self.peak_demand_mw
        generation = demand - flow_variation

        df = pd.DataFrame({
            "demand_mwh": demand,
            "generation_mwh": np.clip(generation, 0, None),
        }, index=idx)
        return df

    # ------------------------------------------------------------------
    # Prices (called separately)
    # ------------------------------------------------------------------

    def fetch_prices(self, start: str, end: str) -> pd.DataFrame:
        """
        Generate synthetic LMP prices correlated with demand.
        Returns DataFrame with: lmp_total, lmp_energy, lmp_congestion, lmp_loss
        """
        idx = self._make_index(start, end)
        n = len(idx)

        # Load demand signal for correlation
        demand_df = self._fetch_raw(start, end)
        load_norm = demand_df["demand_mwh"] / self.peak_demand_mw

        # Energy component: convex function of load (scarcity pricing)
        lmp_energy = self.base_price * (0.6 + 1.4 * load_norm.values ** 2)

        # Congestion events: random spikes
        congestion = np.zeros(n)
        in_congestion = False
        duration = 0
        for t in range(n):
            if not in_congestion:
                if self.rng.random() < self.congestion_prob:
                    in_congestion = True
                    duration = int(self.rng.integers(2, 8))  # 2-8 hour congestion window
            if in_congestion:
                magnitude = self.rng.uniform(10, 80)
                congestion[t] = magnitude
                duration -= 1
                if duration <= 0:
                    in_congestion = False

        # Loss component: small function of load
        lmp_loss = 0.5 * load_norm.values * self.rng.uniform(0.8, 1.2, n)

        lmp_total = lmp_energy + congestion + lmp_loss

        df = pd.DataFrame({
            "lmp_total": lmp_total,
            "lmp_energy": lmp_energy,
            "lmp_congestion": congestion,
            "lmp_loss": lmp_loss,
        }, index=idx)
        return df

    # ------------------------------------------------------------------
    # Flows
    # ------------------------------------------------------------------

    def fetch_flows(self, start: str, end: str) -> pd.DataFrame:
        """Generate synthetic net interchange flows."""
        idx = self._make_index(start, end)
        n = len(idx)

        demand_df = self._fetch_raw(start, end)
        imbalance = demand_df["demand_mwh"] - demand_df["generation_mwh"]

        # Net flow tracks imbalance with noise and smoothing
        noise = self.rng.normal(0, 500, n)
        net_flow = imbalance.values + noise

        # Apply exponential smoothing (real inter-ties have inertia)
        alpha = 0.3
        smoothed = np.zeros(n)
        smoothed[0] = net_flow[0]
        for t in range(1, n):
            smoothed[t] = alpha * net_flow[t] + (1 - alpha) * smoothed[t - 1]

        return pd.DataFrame({"net_flow_mw": smoothed}, index=idx)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _make_index(self, start: str, end: str) -> pd.DatetimeIndex:
        """Create hourly UTC DatetimeIndex for [start, end]."""
        return pd.date_range(
            start=pd.Timestamp(start, tz="UTC"),
            end=pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
            freq="1h",
        )
