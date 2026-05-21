"""
constraint_field.graph.node_signals
=====================================
Generate node-level S_t and R_t field vectors over the BA graph.

Data strategy
-------------
In the live prototype (with EIA-930 access), this module would:
  1. Pull hourly demand for each BA from EIA-930
  2. Pull hourly LMP / price proxy for each BA from ISO price APIs
  3. Normalise using rolling z-score per node

In this environment (EIA API blocked), we generate structurally
realistic synthetic signals with:
  - Geographically coherent spatial correlation (neighbouring BAs
    have correlated load patterns)
  - Temporal realism (diurnal + weekly seasonality per node)
  - Node-specific scale factors from documented peak_gw values
  - Correlated but differentiated price signals with occasional
    congestion events on specific corridors

All synthetic generation is clearly documented and can be swapped
for real API calls by replacing SyntheticNodeSignals with a
LiveNodeSignals class that implements the same interface.

Observability: SYNTHETIC (clearly labelled)
"""

from __future__ import annotations

import logging
from typing import Protocol

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Interface protocol — real and synthetic implementations share this
# ──────────────────────────────────────────────────────────────────────────────

class NodeSignalSource(Protocol):
    """
    Interface for node signal sources (real or synthetic).
    Must produce DataFrames with nodes as columns, DatetimeIndex as rows.
    """
    def demand(self, start: str, end: str) -> pd.DataFrame: ...
    def prices(self, start: str, end: str) -> pd.DataFrame: ...
    def flows(self, start: str, end: str) -> pd.DataFrame | None: ...


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic node signal generator
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticNodeSignals:
    """
    Generate spatially-correlated synthetic node signals.

    Spatial correlation model
    -------------------------
    Demand at node i has a common component (system-wide) plus a
    node-specific component. The correlation between nodes decays
    with geographic distance:

        rho_ij = base_rho * exp(-dist_ij / length_scale)

    where dist_ij is great-circle distance between BA centroids.

    Price spatial model
    -------------------
    Each node has a base price correlated with its load factor,
    plus congestion premium events that are correlated along edges:
    when a corridor is congested, the import-side node sees higher
    prices and the export side sees lower prices.

    Observability: SYNTHETIC
    """

    def __init__(
        self,
        G: nx.Graph,
        seed: int = 42,
        base_rho: float = 0.6,
        length_scale_km: float = 800.0,
        congestion_prob: float = 0.015,
    ):
        self.G               = G
        self.nodes           = list(G.nodes())
        self.n               = len(self.nodes)
        self.seed            = seed
        self.base_rho        = base_rho
        self.length_scale_km = length_scale_km
        self.congestion_prob = congestion_prob
        self.rng             = np.random.default_rng(seed)
        self._spatial_cov    = self._build_spatial_cov()

    def _build_spatial_cov(self) -> np.ndarray:
        """Build spatial correlation matrix from node coordinates."""
        lons = np.array([self.G.nodes[n]["lon"] for n in self.nodes])
        lats = np.array([self.G.nodes[n]["lat"] for n in self.nodes])

        # Approximate great-circle distances (km)
        R_earth = 6371.0
        C = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dlat = np.radians(lats[j] - lats[i])
                dlon = np.radians(lons[j] - lons[i])
                a = (np.sin(dlat/2)**2
                     + np.cos(np.radians(lats[i]))
                     * np.cos(np.radians(lats[j]))
                     * np.sin(dlon/2)**2)
                C[i, j] = 2 * R_earth * np.arcsin(np.sqrt(a))

        # Exponential correlation decay
        corr = self.base_rho * np.exp(-C / self.length_scale_km)
        np.fill_diagonal(corr, 1.0)

        # Make valid covariance matrix (positive definite)
        eigvals = np.linalg.eigvals(corr)
        if np.any(eigvals < 0):
            corr += (-eigvals.min() + 1e-6) * np.eye(self.n)

        return corr

    def _make_index(self, start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end,   tz="UTC") + pd.Timedelta(hours=23),
            freq="1h",
        )

    def demand(self, start: str, end: str) -> pd.DataFrame:
        """
        Generate spatially-correlated hourly demand for each node.

        Returns DataFrame: rows=timestamps, cols=node IDs, values in GW.
        Observability: SYNTHETIC
        """
        idx = self._make_index(start, end)
        T   = len(idx)
        peak_gw = np.array([
            self.G.nodes[n].get("peak_gw", 5.0) for n in self.nodes
        ])

        # Draw spatially correlated noise (multivariate normal)
        L = np.linalg.cholesky(self._spatial_cov)
        z = self.rng.standard_normal((T, self.n))
        z_corr = z @ L.T   # shape (T, n)

        # Diurnal + weekly shape (common, varies by node slightly)
        hour = idx.hour.values
        dow  = idx.dayofweek.values

        daily_shape = (
            0.65
            + 0.18 * np.exp(-((hour - 9)**2) / 8)
            + 0.22 * np.exp(-((hour - 19)**2) / 6)
        )[:, None]  # (T, 1)

        weekly = np.where(dow >= 5, 0.83, 1.0)[:, None]  # (T, 1)

        # Node-specific amplitude variation
        node_amp = 1.0 + 0.15 * self.rng.standard_normal(self.n)[None, :]

        demand_gw = (
            peak_gw[None, :]
            * daily_shape
            * weekly
            * node_amp
            * (1.0 + 0.04 * z_corr)
        )
        demand_gw = np.clip(demand_gw, 0, None)

        return pd.DataFrame(demand_gw, index=idx, columns=self.nodes)

    def prices(self, start: str, end: str) -> pd.DataFrame:
        """
        Generate spatially-correlated node-level LMP proxies.

        Price model:
          - Base energy component correlated with system load
          - Congestion events: spikes along specific edges, creating
            price differentials between connected nodes
          - Loss component: small, correlated with load

        Returns DataFrame: rows=timestamps, cols=node IDs, values in $/MWh.
        Observability: SYNTHETIC
        """
        idx      = self._make_index(start, end)
        T        = len(idx)
        dem_df   = self.demand(start, end)
        dem_norm = dem_df.divide(dem_df.max(), axis=1)  # normalize per node

        # Base LMP: convex in load
        base_price = 30.0
        lmp = base_price * (0.5 + 1.5 * dem_norm.values**2)

        # Spatially correlated noise in energy component
        L = np.linalg.cholesky(self._spatial_cov)
        z = self.rng.standard_normal((T, self.n)) @ L.T
        lmp += 3.0 * z

        # Congestion events: iterate over edges
        edges = list(self.G.edges())
        node_idx = {n: i for i, n in enumerate(self.nodes)}

        for t in range(T):
            for u, v in edges:
                if self.rng.random() < self.congestion_prob:
                    # Congestion on this corridor this hour
                    magnitude = self.rng.uniform(15, 60)
                    cap = self.G[u][v].get("capacity_gw", 1.0)
                    # Higher magnitude on lower-capacity lines
                    magnitude *= max(0.5, 1.0 / cap)
                    magnitude = min(magnitude, 80)
                    # Import side: higher price; export side: lower price
                    i, j = node_idx[u], node_idx[v]
                    lmp[t, i] += magnitude * 0.5
                    lmp[t, j] -= magnitude * 0.3

        lmp = np.clip(lmp, -50, 500)  # physical bounds
        return pd.DataFrame(lmp, index=idx, columns=self.nodes)

    def flows(self, start: str, end: str) -> pd.DataFrame | None:
        """
        Generate approximate net interchange flows on each edge.

        Returns DataFrame: rows=timestamps, cols="u_v" edge labels.
        Positive = net flow from u to v.
        Observability: SYNTHETIC
        """
        idx   = self._make_index(start, end)
        T     = len(idx)
        edges = list(self.G.edges())

        flow_data = {}
        for u, v in edges:
            cap = self.G[u][v].get("capacity_gw", 1.0)
            # Flow driven by price differential + noise
            noise = self.rng.normal(0, 0.1 * cap, T)
            base_flow = self.rng.normal(0, 0.3 * cap, T)
            # Smooth (transmission schedules have inertia)
            alpha = 0.4
            smooth = np.zeros(T)
            smooth[0] = base_flow[0]
            for t in range(1, T):
                smooth[t] = alpha * base_flow[t] + (1 - alpha) * smooth[t-1]
            flow_data[f"{u}_{v}"] = np.clip(smooth + noise, -cap, cap)

        return pd.DataFrame(flow_data, index=idx)


# ──────────────────────────────────────────────────────────────────────────────
# Field vector construction (S_t, R_t per node)
# ──────────────────────────────────────────────────────────────────────────────

def build_node_field(
    demand_df: pd.DataFrame,
    price_df:  pd.DataFrame,
    window:    int = 168,
    clip_sigma: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """
    Construct normalised S_t and R_t as node-indexed DataFrames.

    Each column is one node; each row is one timestep.
    Normalisation is rolling z-score per node independently.

    Returns
    -------
    dict with keys: "S", "R", "Phi", "Psi"
    All DataFrames share the same index and columns (node IDs).
    """
    def rolling_zscore(df: pd.DataFrame, window: int, clip: float) -> pd.DataFrame:
        roll_mean = df.rolling(window, min_periods=max(1, window//4)).mean()
        roll_std  = df.rolling(window, min_periods=max(1, window//4)).std()
        roll_std  = roll_std.replace(0, np.nan)
        z = (df - roll_mean) / roll_std
        # Fill warm-up period with global z-score per column
        global_z = (df - df.mean()) / df.std().replace(0, 1)
        z = z.fillna(global_z)
        return z.clip(-clip, clip)

    # Align to common index
    common = demand_df.index.intersection(price_df.index)
    D = demand_df.loc[common]
    P = price_df.loc[common].clip(upper=500)

    # Align columns (nodes)
    common_nodes = D.columns.intersection(P.columns)
    D = D[common_nodes]
    P = P[common_nodes]

    S   = rolling_zscore(D, window, clip_sigma)
    R   = rolling_zscore(P, window, clip_sigma)
    Phi = R - S
    Psi = np.sqrt(S**2 + R**2)

    log.info(
        "Node field built: %d nodes  %d timesteps\n"
        "  S: mean=%.3f  std=%.3f\n"
        "  R: mean=%.3f  std=%.3f\n"
        "  Phi: mean=%.3f  std=%.3f",
        len(common_nodes), len(common),
        S.values.mean(), S.values.std(),
        R.values.mean(), R.values.std(),
        Phi.values.mean(), Phi.values.std(),
    )

    return {"S": S, "R": R, "Phi": Phi, "Psi": Psi}
