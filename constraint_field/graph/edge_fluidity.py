"""
constraint_field.graph.edge_fluidity
======================================
Edge-level fluidity E_{ij,t} and weighted graph Laplacian L_t.

E_{ij,t} in [0,1] represents the effective transmissibility of
edge (i,j) at time t. It modulates the graph Laplacian:

    W_t = A .* E_t          (element-wise: adjacency scaled by fluidity)
    L_t = D_t - W_t         (weighted degree matrix - weighted adjacency)

Three candidate edge fluidity definitions
-----------------------------------------

E1_price_spread (edge)
  E_{ij} = f(|R_i - R_j|)   — inverse of LMP spread across the edge
  Rationale: large price differentials indicate binding transmission
  constraints → low fluidity. Converging prices → high fluidity.
  Observability: COMPUTED from node R field (SYNTHETIC in this env)

E2_flow_efficiency (edge)
  E_{ij} = tanh(beta * |flow_ij| / capacity_ij)
  Rationale: actual interchange relative to rated capacity is a
  direct measure of how effectively the corridor is being utilized.
  When utilisation is high, the line is physically at its limit
  → fluidity is low (or high, depending on direction convention).
  We use utilisation as a proxy for "how open" the corridor is:
  low utilisation (under-used) = mechanically available but not needed.
  We treat utilisation above a threshold as congestion → low E.
  Observability: SYNTHETIC flows; capacity from DOCUMENTED WECC ADS

E3_congestion_proxy (edge)
  E_{ij} = exp(-lambda * price_spread_norm)
  Same logic as scalar E3 but applied per edge.
  Uses calibrated scaling (SpreadCalibrator) per edge.
  Observability: COMPUTED from node prices (SYNTHETIC)
"""

from __future__ import annotations

import logging
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract edge pairs consistently
# ──────────────────────────────────────────────────────────────────────────────

def edge_keys(G: nx.Graph) -> list[tuple[str, str]]:
    """Return sorted list of (u, v) pairs for all graph edges."""
    return sorted(G.edges())


def edge_price_spread(
    G: nx.Graph,
    R: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute |R_i - R_j| for each edge at each timestep.

    Returns DataFrame: index=timestamps, columns="u_v" edge labels.
    """
    result = {}
    for u, v in edge_keys(G):
        if u in R.columns and v in R.columns:
            result[f"{u}_{v}"] = (R[u].astype(float) - R[v].astype(float)).abs()
    df = pd.DataFrame(result, index=R.index)
    # Ensure float64 throughout — pyarrow-backed DataFrames can carry string
    # dtype on columns when column names propagate into arithmetic.
    return df.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# E1: Inverse price-spread fluidity (edge-level)
# ──────────────────────────────────────────────────────────────────────────────

def E1_price_spread_edge(
    G:     nx.Graph,
    R:     pd.DataFrame,
    alpha: float = 2.0,
    smoothing: int = 3,
) -> pd.DataFrame:
    """
    E_{ij,t} = 1 / (1 + alpha * spread_norm_{ij,t})

    where spread_norm is |R_i - R_j| normalised by its 90th percentile.

    Observability: COMPUTED from synthetic R field.

    Returns
    -------
    pd.DataFrame: index=timestamps, columns="u_v", values in [0,1]
    """
    spreads = edge_price_spread(G, R)
    idx     = R.index
    result  = {}
    for col in spreads.columns:
        arr   = np.asarray(spreads[col], dtype=np.float64)
        scale = float(np.nanpercentile(arr, 90)) or 1.0
        norm  = np.clip(arr / scale, 0, 3)
        e     = 1.0 / (1.0 + alpha * norm)
        # Exponential weighted smoothing (manual, avoids pandas pyarrow issue)
        alpha_ew = 2.0 / (smoothing + 1)
        smoothed = np.zeros_like(e)
        smoothed[0] = e[0]
        for t in range(1, len(e)):
            smoothed[t] = alpha_ew * e[t] + (1 - alpha_ew) * smoothed[t-1]
        result[col] = pd.Series(smoothed, index=idx, dtype=np.float64)
    df = pd.DataFrame(result, index=idx).astype(np.float64)
    log.info("E1 edge fluidity: mean=%.3f  std=%.3f",
             float(df.values.mean()), float(df.values.std()))
    return df.clip(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# E2: Flow-efficiency fluidity (edge-level)
# ──────────────────────────────────────────────────────────────────────────────

def E2_flow_efficiency_edge(
    G:         nx.Graph,
    flows_df:  pd.DataFrame,
    beta:      float = 2.0,
    congestion_threshold: float = 0.75,
    smoothing: int = 3,
) -> pd.DataFrame:
    """
    Utilisation-based edge fluidity.

    utilisation_{ij,t} = |flow_{ij,t}| / capacity_{ij}

    Two regimes:
      utilisation < threshold : corridor is open → E = 1 - tanh(beta * util)
                                (more available capacity = higher fluidity)
      utilisation >= threshold: corridor is near-congested → E drops further

    This treats AVAILABLE capacity as fluidity, not utilisation itself.
    A completely idle line is fluid (E≈1); a saturated line is not (E≈0).

    Observability: flow=SYNTHETIC, capacity=DOCUMENTED.

    Returns
    -------
    pd.DataFrame: index=timestamps, columns="u_v", values in [0,1]
    """
    idx    = flows_df.index
    result = {}
    alpha_ew = 2.0 / (smoothing + 1)
    for u, v in edge_keys(G):
        col = f"{u}_{v}"
        if col not in flows_df.columns:
            result[col] = pd.Series(0.5, index=idx, dtype=np.float64)
            continue
        cap  = float(G[u][v].get("capacity_gw", 1.0))
        farr = np.asarray(flows_df[col], dtype=np.float64)
        util = np.clip(np.abs(farr) / cap, 0, 1)
        e    = 1.0 - np.tanh(beta * util)
        smoothed = np.zeros_like(e)
        smoothed[0] = e[0]
        for t in range(1, len(e)):
            smoothed[t] = alpha_ew * e[t] + (1 - alpha_ew) * smoothed[t-1]
        result[col] = pd.Series(smoothed, index=idx, dtype=np.float64)
    df = pd.DataFrame(result, index=idx).astype(np.float64)
    log.info("E2 flow-efficiency edge fluidity: mean=%.3f  std=%.3f",
             df.values.mean(), df.values.std())
    return df.clip(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# E3: Congestion proxy (calibrated exponential, edge-level)
# ──────────────────────────────────────────────────────────────────────────────

def E3_congestion_proxy_edge(
    G:      nx.Graph,
    R:      pd.DataFrame,
    lambda_: float = 1.0,
    smoothing: int = 3,
) -> pd.DataFrame:
    """
    E_{ij,t} = exp(-lambda * spread_norm_{ij,t})

    Uses per-edge calibration: scale = std of that edge's spread series.
    If std is near zero (always-flat spread), defaults to E=1.

    Observability: COMPUTED from synthetic R field.
    """
    spreads  = edge_price_spread(G, R)
    idx      = R.index
    result   = {}
    alpha_ew = 2.0 / (smoothing + 1)
    for col in spreads.columns:
        arr   = np.asarray(spreads[col], dtype=np.float64)
        scale = float(np.nanstd(arr))
        if scale < 1e-6:
            result[col] = pd.Series(1.0, index=idx, dtype=np.float64)
            continue
        norm = np.clip(arr / scale, 0, 3)
        e    = np.exp(-lambda_ * norm)
        smoothed = np.zeros_like(e)
        smoothed[0] = e[0]
        for t in range(1, len(e)):
            smoothed[t] = alpha_ew * e[t] + (1 - alpha_ew) * smoothed[t-1]
        result[col] = pd.Series(smoothed, index=idx, dtype=np.float64)
    df = pd.DataFrame(result, index=idx).astype(np.float64)
    log.info("E3 congestion-proxy edge fluidity: mean=%.3f  std=%.3f",
             float(df.values.mean()), float(df.values.std()))
    return df.clip(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Weighted adjacency and graph Laplacian
# ──────────────────────────────────────────────────────────────────────────────

def weighted_adjacency(
    G:    nx.Graph,
    E_df: pd.DataFrame,
    t:    int | pd.Timestamp,
    nodes: list[str] | None = None,
    use_capacity_base: bool = True,
) -> np.ndarray:
    """
    Construct weighted adjacency matrix W_t = A .* E_t at time t.

    W_{ij} = capacity_{ij} * E_{ij,t}   (if use_capacity_base=True)
           = E_{ij,t}                    (otherwise)

    Parameters
    ----------
    G      : graph with capacity_gw edge attributes
    E_df   : edge fluidity DataFrame (cols = "u_v" format)
    t      : timestep (integer iloc or Timestamp)
    nodes  : ordered list of node IDs (defines matrix row/col ordering)

    Returns
    -------
    np.ndarray  shape (n, n), symmetric
    """
    if nodes is None:
        nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    W = np.zeros((n, n))

    # Get E row at time t
    if isinstance(t, int):
        e_row = E_df.iloc[t]
    else:
        e_row = E_df.loc[t]

    for u, v in G.edges():
        if u not in node_idx or v not in node_idx:
            continue
        col = f"{u}_{v}"
        rev = f"{v}_{u}"
        e_val = float(e_row.get(col, e_row.get(rev, 0.5)))

        if use_capacity_base:
            cap = G[u][v].get("capacity_gw", 1.0)
            w   = cap * e_val
        else:
            w = e_val

        i, j = node_idx[u], node_idx[v]
        W[i, j] = W[j, i] = w

    return W


def graph_laplacian(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted graph Laplacian L = D - W.

    Returns
    -------
    L : np.ndarray  shape (n, n)
    D : np.ndarray  shape (n, n)  diagonal degree matrix
    """
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, D


def constant_laplacian(G: nx.Graph, nodes: list[str]) -> np.ndarray:
    """
    Baseline: graph Laplacian with constant unit weights (no E modulation).
    Used as the reduced-model comparison.
    """
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    W = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            cap = G[u][v].get("capacity_gw", 1.0)
            W[i, j] = W[j, i] = cap   # capacity-weighted but no fluidity modulation
    L, _ = graph_laplacian(W)
    return L
