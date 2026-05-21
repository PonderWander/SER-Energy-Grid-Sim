"""
experiments/cross_corridor/metrics.py
========================================
All routing diagnostics for the cross-corridor stress experiment.

Metrics A–G as specified in the experiment design document, plus
RMSE (for the E-vs-RMSE comparison) and a summary record builder.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .config import (
    SW_CORRIDOR, NW_CORRIDOR, CONNECTOR_NODES,
    COOCCURRENCE_THRESHOLDS, PATH_ACTIVATION_FRAC, LEAKAGE_MIN_FRAC,
)

log = logging.getLogger("cc_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_series(traj: np.ndarray, nodes: list[str], node: str) -> np.ndarray:
    """Extract Φ time-series for a single node."""
    if node not in nodes:
        return np.full(len(traj), np.nan)
    return traj[:, nodes.index(node)]


def _corridor_mean(traj: np.ndarray, nodes: list[str], corr_nodes: list[str]) -> np.ndarray:
    """Mean Φ across all nodes in a corridor, per timestep."""
    idx = [nodes.index(nd) for nd in corr_nodes if nd in nodes]
    if not idx:
        return np.zeros(len(traj))
    return traj[:, idx].mean(axis=1)


def _corridor_phi(traj: np.ndarray, nodes: list[str], corr: dict) -> np.ndarray:
    return _corridor_mean(traj, nodes, corr["nodes"])


# ─────────────────────────────────────────────────────────────────────────────
# A. Cross-corridor Φ correlation
# ─────────────────────────────────────────────────────────────────────────────

def metric_cross_corr(traj: np.ndarray, nodes: list[str]) -> dict:
    """
    A. Cross-corridor Φ correlation.

    Returns:
      sw_phi        : mean Φ across SW nodes (T,)
      nw_phi        : mean Φ across NW nodes (T,)
      corr_full     : Pearson r over full simulation
      corr_early    : r over first 20 steps (loading phase)
      corr_late     : r over last 20 steps (equilibration)
      lag_cc        : dict {lag: corr} for lags 0..10
      lag_peak      : lag at which |cross-corr| is maximised
    """
    sw_phi = _corridor_phi(traj, nodes, SW_CORRIDOR)
    nw_phi = _corridor_phi(traj, nodes, NW_CORRIDOR)
    T      = len(traj)
    half   = max(T // 4, 5)

    corr_full  = float(np.corrcoef(sw_phi, nw_phi)[0, 1])
    corr_early = float(np.corrcoef(sw_phi[:half], nw_phi[:half])[0, 1]) if half >= 3 else np.nan
    corr_late  = float(np.corrcoef(sw_phi[-half:], nw_phi[-half:])[0, 1]) if half >= 3 else np.nan

    # Lagged cross-correlation: SW leads NW (positive lag = SW leads)
    max_lag = min(10, T // 4)
    lag_cc  = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = sw_phi[:T-lag], nw_phi[lag:]
        else:
            a, b = sw_phi[-lag:], nw_phi[:T+lag]
        if len(a) >= 3:
            lag_cc[lag] = float(np.corrcoef(a, b)[0, 1])
        else:
            lag_cc[lag] = np.nan

    valid_lags = {k: v for k, v in lag_cc.items() if not np.isnan(v)}
    lag_peak   = max(valid_lags, key=lambda k: abs(valid_lags[k])) if valid_lags else 0

    return {
        "sw_phi": sw_phi,
        "nw_phi": nw_phi,
        "corr_full": corr_full,
        "corr_early": corr_early,
        "corr_late": corr_late,
        "lag_cc": lag_cc,
        "lag_peak": lag_peak,
    }


# ─────────────────────────────────────────────────────────────────────────────
# B. Stress co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

def metric_cooccurrence(traj: np.ndarray, nodes: list[str]) -> dict:
    """
    B. Stress co-occurrence: frequency with which high-Φ nodes in SW and NW
       activate (|Φ| > threshold) at the same timestep.

    Returns per-threshold:
      freq_sw    : fraction of steps where any SW node is above threshold
      freq_nw    : fraction of steps where any NW node is above threshold
      freq_joint : fraction of steps where both corridors have a node above threshold
      lift       : freq_joint / (freq_sw * freq_nw)  — correlation lift
    """
    T   = len(traj)
    out = {}
    sw_idx = [nodes.index(nd) for nd in SW_CORRIDOR["nodes"] if nd in nodes]
    nw_idx = [nodes.index(nd) for nd in NW_CORRIDOR["nodes"] if nd in nodes]

    for thresh in COOCCURRENCE_THRESHOLDS:
        sw_active = np.abs(traj[:, sw_idx]).max(axis=1) > thresh if sw_idx else np.zeros(T, dtype=bool)
        nw_active = np.abs(traj[:, nw_idx]).max(axis=1) > thresh if nw_idx else np.zeros(T, dtype=bool)
        freq_sw    = float(sw_active.mean())
        freq_nw    = float(nw_active.mean())
        freq_joint = float((sw_active & nw_active).mean())
        denom      = freq_sw * freq_nw
        lift       = float(freq_joint / denom) if denom > 1e-9 else np.nan
        out[f"thresh_{thresh}"] = {
            "freq_sw": freq_sw, "freq_nw": freq_nw,
            "freq_joint": freq_joint, "lift": lift,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# C. Leakage ratio
# ─────────────────────────────────────────────────────────────────────────────

def metric_leakage(
    traj: np.ndarray,
    nodes: list[str],
    Phi0: np.ndarray,
    flux_traj: dict[str, np.ndarray],
) -> dict:
    """
    C. Leakage ratio.

    Leakage = cumulative unsigned flux crossing from SW corridor into NW
              (or vice versa) through connector nodes, normalised by the
              total initial Φ energy in the source corridor.

    Cross-corridor flux is measured on edges that connect an SW-only node
    to an NW-only node (i.e., neither is a connector node).

    Because CISO, NEVP, LDWP, WALC are connectors, direct SW→NW transfer
    happens through these nodes. We measure the flux on all edges where
    one endpoint is exclusively SW and the other is exclusively NW or connector.
    """
    sw_only  = set(SW_CORRIDOR["nodes"]) - set(NW_CORRIDOR["nodes"])
    nw_only  = set(NW_CORRIDOR["nodes"]) - set(SW_CORRIDOR["nodes"])
    connectors = set(CONNECTOR_NODES)

    sw_idx   = [nodes.index(nd) for nd in sw_only  if nd in nodes]
    nw_idx   = [nodes.index(nd) for nd in nw_only  if nd in nodes]
    con_idx  = [nodes.index(nd) for nd in connectors if nd in nodes]

    # Initial Φ energy in each corridor
    energy_sw = float(np.abs(Phi0[sw_idx]).sum())  if sw_idx  else 1.0
    energy_nw = float(np.abs(Phi0[nw_idx]).sum())  if nw_idx  else 1.0
    energy_sw = max(energy_sw, 1e-9)
    energy_nw = max(energy_nw, 1e-9)

    # Identify cross-corridor edges: one end in SW-only, other in NW-only or connector
    cross_edges_sw_to_nw = []  # SW node → NW-only or connector
    cross_edges_nw_to_sw = []  # NW node → SW-only or connector
    for edge_key, flux_series in flux_traj.items():
        parts = edge_key.split("_")
        if len(parts) < 2:
            continue
        u, v = parts[0], "_".join(parts[1:])
        u_sw = u in sw_only; u_nw = u in nw_only; u_con = u in connectors
        v_sw = v in sw_only; v_nw = v in nw_only; v_con = v in connectors
        if (u_sw and (v_nw or v_con)) or ((u_nw or u_con) and v_sw):
            cross_edges_sw_to_nw.append(edge_key)
        if (u_nw and (v_sw or v_con)) or ((u_sw or u_con) and v_nw):
            cross_edges_nw_to_sw.append(edge_key)

    # Cumulative cross-corridor flux (sum over time and cross edges)
    def _cum_flux(edge_list):
        total = 0.0
        for ek in edge_list:
            if ek in flux_traj:
                total += float(flux_traj[ek].sum())
        return total

    def _peak_flux(edge_list):
        """Peak per-step flux across all cross edges (max over time)."""
        if not edge_list:
            return 0.0
        stacked = np.array([flux_traj[ek] for ek in edge_list if ek in flux_traj])
        return float(stacked.sum(axis=0).max()) if len(stacked) > 0 else 0.0

    cum_sw_to_nw  = _cum_flux(cross_edges_sw_to_nw)
    cum_nw_to_sw  = _cum_flux(cross_edges_nw_to_sw)
    peak_sw_to_nw = _peak_flux(cross_edges_sw_to_nw)
    peak_nw_to_sw = _peak_flux(cross_edges_nw_to_sw)

    # Leakage at connector nodes: peak |Φ| at each connector
    connector_peak = {}
    for nd in CONNECTOR_NODES:
        if nd in nodes:
            connector_peak[nd] = float(np.abs(traj[:, nodes.index(nd)]).max())

    return {
        # Peak-normalised: peak single-step cross-corridor flux / initial energy
        # Robust to step-count effects; comparable across conditions
        "leakage_sw_to_nw": peak_sw_to_nw / energy_sw,
        "leakage_nw_to_sw": peak_nw_to_sw / energy_nw,
        # Cumulative (for reference — inflated by long simulations)
        "leakage_cum_sw_to_nw": cum_sw_to_nw / energy_sw,
        "leakage_cum_nw_to_sw": cum_nw_to_sw / energy_nw,
        "cum_sw_to_nw":     cum_sw_to_nw,
        "cum_nw_to_sw":     cum_nw_to_sw,
        "peak_sw_to_nw":    peak_sw_to_nw,
        "peak_nw_to_sw":    peak_nw_to_sw,
        "energy_sw_init":   energy_sw,
        "energy_nw_init":   energy_nw,
        "connector_peak":   connector_peak,
        "n_cross_edges_sw_nw": len(cross_edges_sw_to_nw),
        "n_cross_edges_nw_sw": len(cross_edges_nw_to_sw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# D. Connector-node activation
# ─────────────────────────────────────────────────────────────────────────────

def metric_connector_activation(traj: np.ndarray, nodes: list[str]) -> dict:
    """
    D. Connector-node activation timing and peak Φ.

    For each connector node: peak |Φ|, step of peak, step of first exceedance
    above 10% of maximum connector activation.
    """
    out = {}
    all_peaks = []
    for nd in CONNECTOR_NODES:
        if nd in nodes:
            series  = np.abs(_node_series(traj, nodes, nd))
            peak_v  = float(series.max())
            peak_t  = int(series.argmax())
            all_peaks.append(peak_v)
            out[nd] = {"peak": peak_v, "peak_t": peak_t}

    # First activation: first node whose |Φ| exceeds 10% of the max peak
    if all_peaks:
        global_thresh = 0.10 * max(all_peaks)
        activation_order = []
        for nd in CONNECTOR_NODES:
            if nd not in out:
                continue
            series = np.abs(_node_series(traj, nodes, nd))
            first_t = next((t for t in range(len(series)) if series[t] >= global_thresh), None)
            out[nd]["first_activation_t"] = first_t
            if first_t is not None:
                activation_order.append((nd, first_t))
        activation_order.sort(key=lambda x: x[1])
        out["activation_order"] = [nd for nd, _ in activation_order]
    else:
        out["activation_order"] = []

    return out


# ─────────────────────────────────────────────────────────────────────────────
# E. Path activation order
# ─────────────────────────────────────────────────────────────────────────────

def metric_path_activation(
    traj: np.ndarray,
    nodes: list[str],
    flux_traj: dict[str, np.ndarray],
    E_df: pd.DataFrame,
    G,
) -> dict:
    """
    E. Path activation order.

    Rank edges by:
      1. cumulative flux over full simulation
      2. time of first significant flux (> 5% of peak edge flux)

    Compare observed activation order against E-based routing cost:
      routing_cost_ij = 1 / (E_ij_mean * cap_ij)  — lower E = higher cost

    Returns:
      edge_ranking_flux  : edges sorted by cumulative flux (desc)
      edge_ranking_time  : edges sorted by first activation time (asc)
      routing_cost_rank  : edges sorted by routing cost (asc = cheapest)
      rank_correlation   : Kendall τ between flux ranking and routing cost ranking
    """
    from .e_regimes import _canonical

    edge_flux_cum = {ek: float(v.sum()) for ek, v in flux_traj.items()}
    edge_flux_max = {ek: float(v.max())  for ek, v in flux_traj.items()}

    # First activation time per edge
    edge_first_t = {}
    for ek, v in flux_traj.items():
        max_flux   = edge_flux_max[ek]
        if max_flux < 1e-9:
            edge_first_t[ek] = len(v)
            continue
        thresh = 0.05 * max_flux
        first  = next((t for t in range(len(v)) if v[t] >= thresh), len(v))
        edge_first_t[ek] = first

    # Routing cost: 1 / (mean_E * cap)
    E_mean_row = E_df.mean()
    edge_cost  = {}
    for u, v in G.edges():
        col  = _canonical(u, v, E_mean_row.index)
        e_m  = float(E_mean_row.get(col, 0.5)) if col else 0.5
        cap  = G[u][v].get("capacity_gw", 1.0)
        ek   = f"{u}_{v}"
        edge_cost[ek] = 1.0 / max(e_m * cap, 1e-9)

    # Build rankings over edges that appear in all three dicts
    common = [ek for ek in flux_traj if ek in edge_cost and edge_flux_cum[ek] > 1e-9]
    if not common:
        return {"edge_ranking_flux": [], "edge_ranking_time": [],
                "routing_cost_rank": [], "rank_correlation": np.nan}

    rank_flux = sorted(common, key=lambda ek: -edge_flux_cum[ek])
    rank_time = sorted(common, key=lambda ek:  edge_first_t[ek])
    rank_cost = sorted(common, key=lambda ek:  edge_cost[ek])

    # Kendall τ: cumulative flux rank vs routing cost rank (lower cost → more flux)
    # For τ we want: cheaper edges should have higher flux → negative cost, positive flux
    cost_arr = np.array([edge_cost[ek]      for ek in common])
    flux_arr = np.array([edge_flux_cum[ek]  for ek in common])
    if len(common) >= 3:
        tau, pval = scipy_stats.kendalltau(-cost_arr, flux_arr)   # negative cost = higher E = expected more flux
    else:
        tau, pval = np.nan, np.nan

    return {
        "edge_ranking_flux":  rank_flux[:15],
        "edge_ranking_time":  rank_time[:15],
        "routing_cost_rank":  rank_cost[:15],
        "rank_correlation_tau":  float(tau) if not np.isnan(tau) else np.nan,
        "rank_correlation_pval": float(pval) if not np.isnan(pval) else np.nan,
        "edge_flux_cum":      edge_flux_cum,
        "edge_first_t":       edge_first_t,
        "edge_cost":          edge_cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# F. Spatial covariance decomposition
# ─────────────────────────────────────────────────────────────────────────────

def metric_spatial_covariance(traj: np.ndarray, nodes: list[str]) -> dict:
    """
    F. Spatial covariance of Φ.

    Returns:
      cov_matrix       : (n_nodes × n_nodes) covariance matrix
      sw_nw_block_mean : mean off-diagonal covariance between SW-only and NW-only nodes
      sw_block_mean    : mean within-SW covariance
      nw_block_mean    : mean within-NW covariance
      cross_corridor_ratio : sw_nw_block_mean / sqrt(sw_block_mean * nw_block_mean)
                             (normalised cross-corridor interaction)
    """
    sw_only  = [nd for nd in SW_CORRIDOR["nodes"] if nd in nodes and nd not in NW_CORRIDOR["nodes"]]
    nw_only  = [nd for nd in NW_CORRIDOR["nodes"] if nd in nodes and nd not in SW_CORRIDOR["nodes"]]
    sw_idx   = [nodes.index(nd) for nd in sw_only]
    nw_idx   = [nodes.index(nd) for nd in nw_only]

    cov = np.cov(traj.T)   # (n_nodes, n_nodes)

    def _block_mean(row_idx, col_idx, excl_diag=False):
        if not row_idx or not col_idx:
            return np.nan
        vals = []
        for i in row_idx:
            for j in col_idx:
                if excl_diag and i == j:
                    continue
                vals.append(cov[i, j])
        return float(np.mean(vals)) if vals else np.nan

    sw_block  = _block_mean(sw_idx, sw_idx, excl_diag=True)
    nw_block  = _block_mean(nw_idx, nw_idx, excl_diag=True)
    sw_nw_block = _block_mean(sw_idx, nw_idx, excl_diag=False)

    denom = np.sqrt(max(sw_block, 0) * max(nw_block, 0))
    cross_ratio = float(sw_nw_block / denom) if denom > 1e-9 else np.nan

    return {
        "cov_matrix":           cov,
        "sw_block_mean":        sw_block,
        "nw_block_mean":        nw_block,
        "sw_nw_block_mean":     sw_nw_block,
        "cross_corridor_ratio": cross_ratio,
        "sw_only_nodes":        sw_only,
        "nw_only_nodes":        nw_only,
    }


# ─────────────────────────────────────────────────────────────────────────────
# G. Moran's I (spatial autocorrelation support metric)
# ─────────────────────────────────────────────────────────────────────────────

def metric_moran(traj: np.ndarray, nodes: list[str], G) -> dict:
    """
    G. Moran's I of the Φ field at each timestep.

    Uses row-normalised adjacency (topology only, not E-weighted) so that
    differences across E regimes reflect routing, not the autocorrelation
    statistic itself.
    """
    n      = len(nodes)
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    A      = np.zeros((n, n))
    for u, v in G.edges():
        if u in n_idx and v in n_idx:
            A[n_idx[u], n_idx[v]] = A[n_idx[v], n_idx[u]] = 1.0
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    W = A / row_sum   # row-normalised adjacency (topology only)

    def _moran(vals):
        z  = vals - vals.mean()
        W0 = W.sum()
        if W0 < 1e-9 or z.std() < 1e-9:
            return 0.0
        return float(len(z) * (z @ W @ z) / (W0 * (z @ z)))

    mi_series = np.array([_moran(traj[t]) for t in range(len(traj))])
    return {
        "moran_series": mi_series,
        "moran_mean":   float(mi_series.mean()),
        "moran_std":    float(mi_series.std()),
        "moran_peak":   float(mi_series.max()),
        "moran_peak_t": int(mi_series.argmax()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RMSE (for comparison only)
# ─────────────────────────────────────────────────────────────────────────────

def metric_rmse(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
) -> float:
    """RMSE between two Φ trajectories (e.g., calibrated_E1 vs uniform)."""
    return float(np.sqrt(np.mean((traj_a - traj_b) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Summary record builder
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_record(
    regime:         str,
    loading:        float,
    loading_variant: str,
    traj:           np.ndarray,
    Phi0:           np.ndarray,
    flux_traj:      dict,
    E_df:           pd.DataFrame,
    G,
    nodes:          list[str],
    traj_uniform:   Optional[np.ndarray] = None,
) -> dict:
    """Compute all metrics and return a flat dict suitable for a DataFrame row."""
    cc   = metric_cross_corr(traj, nodes)
    coo  = metric_cooccurrence(traj, nodes)
    leak = metric_leakage(traj, nodes, Phi0, flux_traj)
    conn = metric_connector_activation(traj, nodes)
    path = metric_path_activation(traj, nodes, flux_traj, E_df, G)
    cov  = metric_spatial_covariance(traj, nodes)
    mi   = metric_moran(traj, nodes, G)

    rmse_vs_uniform = metric_rmse(traj, traj_uniform) if traj_uniform is not None else np.nan

    rec = {
        "regime":           regime,
        "loading":          loading,
        "loading_variant":  loading_variant,
        # A
        "corr_sw_nw_full":  cc["corr_full"],
        "corr_sw_nw_early": cc["corr_early"],
        "corr_sw_nw_late":  cc["corr_late"],
        "lag_peak":         cc["lag_peak"],
        # B
        "cooc_freq_sw_1s":  coo.get("thresh_1.0", {}).get("freq_sw",    np.nan),
        "cooc_freq_nw_1s":  coo.get("thresh_1.0", {}).get("freq_nw",    np.nan),
        "cooc_joint_1s":    coo.get("thresh_1.0", {}).get("freq_joint", np.nan),
        "cooc_lift_1s":     coo.get("thresh_1.0", {}).get("lift",       np.nan),
        "cooc_joint_2s":    coo.get("thresh_2.0", {}).get("freq_joint", np.nan),
        "cooc_lift_2s":     coo.get("thresh_2.0", {}).get("lift",       np.nan),
        # C
        "leakage_sw_to_nw": leak["leakage_sw_to_nw"],
        "leakage_nw_to_sw": leak["leakage_nw_to_sw"],
        # D
        "connector_first":  conn["activation_order"][0] if conn["activation_order"] else None,
        **{f"conn_peak_{nd}": conn.get(nd, {}).get("peak", np.nan) for nd in CONNECTOR_NODES},
        **{f"conn_t_{nd}":    conn.get(nd, {}).get("first_activation_t", np.nan) for nd in CONNECTOR_NODES},
        # E
        "path_rank_tau":    path["rank_correlation_tau"],
        "path_rank_pval":   path["rank_correlation_pval"],
        "top_flux_edge":    path["edge_ranking_flux"][0]  if path["edge_ranking_flux"] else None,
        "first_active_edge":path["edge_ranking_time"][0]  if path["edge_ranking_time"] else None,
        # F
        "cov_sw_block":     cov["sw_block_mean"],
        "cov_nw_block":     cov["nw_block_mean"],
        "cov_sw_nw_cross":  cov["sw_nw_block_mean"],
        "cross_ratio":      cov["cross_corridor_ratio"],
        # G
        "moran_mean":       mi["moran_mean"],
        "moran_peak":       mi["moran_peak"],
        # RMSE
        "rmse_vs_uniform":  rmse_vs_uniform,
    }
    return rec
