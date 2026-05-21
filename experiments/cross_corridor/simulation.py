"""
experiments/cross_corridor/simulation.py
==========================================
Initialisation and simulation engine for the cross-corridor loading experiment.

Key difference from single-corridor scripts:
- Φ0 is initialised on *both* SW and NW corridors simultaneously.
- Loading variants control the relative amplitude on each spine.
- State-dependent E is recomputed each step from current Φ.
- The simulation records per-step edge flux for path-activation analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit

from constraint_field.graph.propagation import GraphPropagator, PropagationConfig
from .config import (
    SW_CORRIDOR, NW_CORRIDOR, CONNECTOR_NODES,
    ETA, GAMMA, STEPS, SEED, SD_ALPHA, SD_MID,
)
from .e_regimes import (
    build_state_dependent, _canonical,
)

log = logging.getLogger("cc_simulation")


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_dual_corridor_phi(
    G,
    nodes: list[str],
    sw_loading: float,
    nw_loading: float,
) -> np.ndarray:
    """
    Initialise Φ as monotone gradients along both corridors simultaneously.

    SW spine: CISO → WALC → AZPS → WACM
      Φ_CISO = +sw_loading, Φ_WALC = +sw/3, Φ_AZPS = -sw/3, Φ_WACM = -sw

    NW spine: CISO → PACW → BPAT
      Φ_PACW = +nw_loading/2 (CISO already set by SW), Φ_BPAT = -nw_loading/2
      (CISO is shared; its value is dominated by SW loading)

    Connector nodes (NEVP, LDWP, WALC) are set to their SW value where applicable;
    others start at zero and evolve freely.
    """
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    Phi0  = np.zeros(len(nodes))

    # SW monotone gradient
    sw_path = SW_CORRIDOR["path"]   # ["CISO", "WALC", "AZPS", "WACM"]
    L_sw    = len(sw_path)
    for pos, nd in enumerate(sw_path):
        if nd in n_idx:
            frac            = pos / max(L_sw - 1, 1)
            Phi0[n_idx[nd]] = sw_loading * (1.0 - 2.0 * frac)

    # NW monotone gradient — applied additively to PACW and BPAT
    # CISO already holds the SW value; we add the NW contribution on top
    nw_path = NW_CORRIDOR["path"]   # ["CISO", "PACW", "BPAT"]
    # Skip CISO (pos 0): already set above; set PACW and BPAT relative to NW loading
    # PACW  = +nw_loading/2
    # BPAT  = -nw_loading/2
    nw_offsets = {
        nw_path[1]: +nw_loading * 0.5,    # PACW
        nw_path[2]: -nw_loading * 0.5,    # BPAT
    }
    for nd, offset in nw_offsets.items():
        if nd in n_idx:
            Phi0[n_idx[nd]] += offset

    # IPCO and NWMT get a small NW-aligned gradient
    nw_tail = {"IPCO": -nw_loading * 0.3, "NWMT": -nw_loading * 0.4}
    for nd, val in nw_tail.items():
        if nd in n_idx:
            Phi0[n_idx[nd]] = val

    return Phi0


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_weighted_laplacian(
    G,
    nodes: list[str],
    e_dict: dict[str, float],
) -> np.ndarray:
    """Build row-normalised Laplacian from edge-E dict {col: value}."""
    n     = len(nodes)
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    W     = np.zeros((n, n))
    for u, v in G.edges():
        col = _canonical(u, v, e_dict)
        e_val = e_dict.get(col, 0.5) if col else 0.5
        cap   = G[u][v].get("capacity_gw", 1.0)
        i, j  = n_idx[u], n_idx[v]
        W[i, j] = W[j, i] = cap * e_val
    d     = W.sum(axis=1)
    d_inv = np.where(d > 1e-9, 1.0 / d, 0.0)
    return np.diag(d_inv) @ (np.diag(d) - W)


def _build_static_laplacian(
    G,
    nodes: list[str],
    E_row: pd.Series,
) -> np.ndarray:
    """Build row-normalised Laplacian from a static E row (pd.Series)."""
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    n     = len(nodes)
    W     = np.zeros((n, n))
    for u, v in G.edges():
        col   = _canonical(u, v, E_row.index)
        e_val = float(E_row.get(col, 0.5)) if col else 0.5
        cap   = G[u][v].get("capacity_gw", 1.0)
        i, j  = n_idx[u], n_idx[v]
        W[i, j] = W[j, i] = cap * e_val
    d     = W.sum(axis=1)
    d_inv = np.where(d > 1e-9, 1.0 / d, 0.0)
    return np.diag(d_inv) @ (np.diag(d) - W)


# ─────────────────────────────────────────────────────────────────────────────
# Per-step edge flux
# ─────────────────────────────────────────────────────────────────────────────

def _edge_fluxes(G, nodes: list[str], phi_t: np.ndarray, e_row_or_dict) -> dict[str, float]:
    """
    Compute unsigned E-weighted Φ flux on every edge at a single timestep.
    Flux_ij = E_ij * cap_ij * |Φ_i − Φ_j|
    """
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    result = {}
    for u, v in G.edges():
        col = _canonical(u, v, e_row_or_dict if hasattr(e_row_or_dict, '__contains__') else e_row_or_dict.index)
        if col is None:
            continue
        if isinstance(e_row_or_dict, dict):
            e_val = e_row_or_dict.get(col, 0.5)
        else:
            e_val = float(e_row_or_dict.get(col, 0.5))
        cap  = G[u][v].get("capacity_gw", 1.0)
        dphi = abs(float(phi_t[n_idx[u]]) - float(phi_t[n_idx[v]])) if u in n_idx and v in n_idx else 0.0
        result[f"{u}_{v}"] = e_val * cap * dphi
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    G,
    nodes: list[str],
    Phi0: np.ndarray,
    E_df: pd.DataFrame,
    regime: str,
    eta:   float = ETA,
    gamma: float = GAMMA,
    steps: int   = STEPS,
    seed:  int   = SEED,
) -> dict:
    """
    Run diffusion simulation from Phi0 under specified E regime.

    Parameters
    ----------
    regime : one of "uniform" | "calibrated_E1" | "state_dep" | "shuffled" | "inverted"
        For "state_dep", E_df is used as the base E1 (recomputed each step).
        For all others, E_df is used directly (static or pre-built variant).

    Returns
    -------
    dict with keys:
        traj       : np.ndarray (steps+1, n_nodes)   Φ trajectory
        flux_traj  : dict {edge_label: np.ndarray (steps+1,)}  per-step flux
        L_traj     : np.ndarray (steps+1, n, n)  Laplacian at each step (memory: stored sparsely as diag + off-diag only if needed)
    """
    n     = len(nodes)
    traj  = np.zeros((steps + 1, n))
    traj[0] = Phi0.copy()

    # Edge list for flux recording
    edges    = list(G.edges())
    edge_keys = [f"{u}_{v}" for u, v in edges]
    flux_arr  = np.zeros((steps + 1, len(edges)))

    # Static E: use median row of E_df (representative single-row) for non-state-dep regimes
    E_median_row = E_df.median()

    # For state_dep, use E_df row 0 as the base E1 (calibrated)
    E_base_row = E_df.iloc[0]

    # Record initial flux
    if regime == "state_dep":
        e_init = build_state_dependent(E_base_row, G, nodes, traj[0])
        flux_arr[0] = [_edge_fluxes(G, nodes, traj[0], e_init).get(ek, 0) for ek in edge_keys]
    else:
        flux_arr[0] = [_edge_fluxes(G, nodes, traj[0], E_median_row).get(ek, 0) for ek in edge_keys]

    for t in range(1, steps + 1):
        phi_t = traj[t - 1].copy()

        if regime == "state_dep":
            e_dict = build_state_dependent(E_base_row, G, nodes, phi_t)
            L_cur  = _build_weighted_laplacian(G, nodes, e_dict)
            flux_arr[t] = [_edge_fluxes(G, nodes, phi_t, e_dict).get(ek, 0) for ek in edge_keys]
        else:
            # Use a row from the regime-specific E_df at this timestep
            # (clamp to available rows)
            row_idx = min(t, len(E_df) - 1)
            E_row   = E_df.iloc[row_idx]
            L_cur   = _build_static_laplacian(G, nodes, E_row)
            flux_arr[t] = [_edge_fluxes(G, nodes, phi_t, E_row).get(ek, 0) for ek in edge_keys]

        traj[t] = phi_t - eta * (L_cur @ phi_t) - gamma * phi_t

    # Convert flux_arr to dict of per-edge series
    flux_traj = {ek: flux_arr[:, ei] for ei, ek in enumerate(edge_keys)}

    return {"traj": traj, "flux_traj": flux_traj}
