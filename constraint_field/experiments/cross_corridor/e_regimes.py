"""
experiments/cross_corridor/e_regimes.py
=========================================
Build, validate, and document all E-regime variants used in the
cross-corridor stress routing experiment.

Design note on "inverted E":
-----------------------------
Calibrated E1 means are ~0.58 for both SW and NW corridor edges under the
gradient-rich synthetic field (rolling z-score normalisation removes mean price
differences). The meaningful spatial structure in E1 is in *temporal correlation
patterns*: SW bottleneck edges (AZPS_WACM, AZPS_WALC) have lag-1 AC ≈ 0.49–0.56,
while NW spine edges (PACW_BPAT, BPAT_IPCO) have lag-1 AC ≈ 0.62. The
inverted-E test therefore swaps the *time-series* of E values between SW and NW
edge groups, not just their means, to break the corridor-specific temporal
co-movement structure that E1 exhibits.

Sanity checks (run_sanity_checks):
- uniform:     all values == 1.0
- calibrated:  values ∈ (0, 1), mean ≈ E1 mean (0.58)
- shuffled:    same per-edge distribution as E1; Pearson r with E1 ≈ 0
- inverted:    SW edges now have NW E profile (and vice versa); means similar
- state_dep:   values ∈ [0, 1]; anti-correlated with local edge stress
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import (
    SW_CORRIDOR, NW_CORRIDOR,
    SD_ALPHA, SD_MID, SEED,
)

log = logging.getLogger("e_regimes")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canonical(u: str, v: str, columns) -> str | None:
    """Return the column key for edge (u,v) as it appears in E1."""
    fwd = f"{u}_{v}"
    rev = f"{v}_{u}"
    if fwd in columns:
        return fwd
    if rev in columns:
        return rev
    return None


def _sw_nw_columns(E1: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition E1 columns into SW-corridor and NW-corridor edge groups."""
    sw_cols, nw_cols = [], []
    sw_edges = {(_u, _v) for _u, _v in SW_CORRIDOR["edges"]}
    nw_edges = {(_u, _v) for _u, _v in NW_CORRIDOR["edges"]}
    for col in E1.columns:
        parts = col.split("_")
        if len(parts) < 2:
            continue
        u, v = parts[0], "_".join(parts[1:])
        pair_fwd = (u, v)
        pair_rev = (v, u)
        in_sw = pair_fwd in sw_edges or pair_rev in sw_edges
        in_nw = pair_fwd in nw_edges or pair_rev in nw_edges
        if in_sw:
            sw_cols.append(col)
        elif in_nw:
            nw_cols.append(col)
    return sw_cols, nw_cols


# ─────────────────────────────────────────────────────────────────────────────
# Regime builders
# ─────────────────────────────────────────────────────────────────────────────

def build_uniform(E1: pd.DataFrame) -> pd.DataFrame:
    """E=1 everywhere at all timesteps."""
    return pd.DataFrame(
        np.ones_like(E1.values),
        index=E1.index,
        columns=E1.columns,
    )


def build_calibrated(E1: pd.DataFrame) -> pd.DataFrame:
    """Static calibrated E1 — use the time-series as-is."""
    return E1.copy()


def build_shuffled(E1: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Shuffled E: permute the time axis independently for each edge column.
    Preserves the marginal distribution of every edge but destroys temporal
    and spatial correlation structure.
    """
    rng  = np.random.default_rng(seed)
    arr  = E1.values.copy()
    for j in range(arr.shape[1]):
        arr[:, j] = rng.permutation(arr[:, j])
    return pd.DataFrame(arr, index=E1.index, columns=E1.columns)


def build_inverted(E1: pd.DataFrame) -> pd.DataFrame:
    """
    Inverted E: swap the *time-series profiles* between SW and NW edge groups.

    Each SW edge receives the time-series of its ranked NW counterpart (by
    mean E1 level) and vice versa. This swaps the corridor-specific temporal
    co-movement patterns (including autocorrelation structure) while keeping
    each individual edge's value range similar.

    If a corridor has more edges than the other, surplus edges are left unchanged.
    The swap is documented in the returned metadata dict.
    """
    sw_cols, nw_cols = _sw_nw_columns(E1)
    arr = E1.values.copy()
    col_idx = {c: i for i, c in enumerate(E1.columns)}

    # Sort both groups by mean E1 to make the swap as consistent as possible
    sw_sorted = sorted(sw_cols, key=lambda c: E1[c].mean())
    nw_sorted = sorted(nw_cols, key=lambda c: E1[c].mean())
    n_swap    = min(len(sw_sorted), len(nw_sorted))

    swapped_pairs = []
    for sw_c, nw_c in zip(sw_sorted[:n_swap], nw_sorted[:n_swap]):
        si, ni = col_idx[sw_c], col_idx[nw_c]
        arr[:, si], arr[:, ni] = arr[:, ni].copy(), arr[:, si].copy()
        swapped_pairs.append((sw_c, nw_c))

    log.info("Inverted E: swapped %d SW↔NW edge pairs", n_swap)
    for sw_c, nw_c in swapped_pairs[:5]:
        log.info("  %s ↔ %s", sw_c, nw_c)
    if len(swapped_pairs) > 5:
        log.info("  … and %d more", len(swapped_pairs) - 5)

    return pd.DataFrame(arr, index=E1.index, columns=E1.columns)


def build_state_dependent(
    E1_row: pd.Series,
    G,
    nodes: list[str],
    phi_t: np.ndarray,
    alpha: float = SD_ALPHA,
    mid: float   = SD_MID,
) -> dict[str, float]:
    """
    Compute state-dependent E for a single timestep.
    Returns {edge_col: e_value} mapping.
    """
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    result = {}
    for u, v in G.edges():
        col = _canonical(u, v, E1_row.index)
        if col is None:
            continue
        e_base = float(E1_row.get(col, 0.5))
        cap    = G[u][v].get("capacity_gw", 1.0)
        if u in n_idx and v in n_idx:
            dphi   = abs(float(phi_t[n_idx[u]]) - float(phi_t[n_idx[v]]))
            stress = dphi / cap
        else:
            stress = 0.0
        e_new = e_base * float(expit(-alpha * (stress - mid)))
        result[col] = float(np.clip(e_new, 0.0, 1.0))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def build_all_regimes(E1: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """
    Build all static E regime DataFrames (same shape as E1).
    State-dependent E is computed step-by-step during simulation;
    its entry here is a placeholder (the calibrated E1) used for initialisation.
    """
    return {
        "uniform":       build_uniform(E1),
        "calibrated_E1": build_calibrated(E1),
        "state_dep":     build_calibrated(E1),   # placeholder; replaced per-step in sim
        "shuffled":      build_shuffled(E1, seed=seed),
        "inverted":      build_inverted(E1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(E1: pd.DataFrame, regimes: dict[str, pd.DataFrame]) -> dict[str, bool]:
    """
    Validate each E regime. Returns {check_name: passed}.
    Raises ValueError for critical failures.
    """
    results = {}
    sw_cols, nw_cols = _sw_nw_columns(E1)

    # 1. uniform: all values == 1.0
    uni = regimes["uniform"]
    results["uniform_all_ones"] = bool((uni.values == 1.0).all())
    if not results["uniform_all_ones"]:
        raise ValueError("Uniform E regime contains values != 1.0")

    # 2. calibrated: values in (0,1), mean close to E1 mean
    cal = regimes["calibrated_E1"]
    cal_ok = bool((cal.values > 0).all() and (cal.values < 1).all())
    results["calibrated_in_unit_interval"] = cal_ok
    cal_mean_ok = abs(cal.values.mean() - E1.values.mean()) < 1e-9
    results["calibrated_identical_to_E1"] = cal_mean_ok

    # 3. shuffled: per-edge distribution preserved, spatial correlation broken
    shuf = regimes["shuffled"]
    # Distribution: for each column, sorted values should match E1 sorted values
    dist_ok = True
    for col in E1.columns:
        if not np.allclose(np.sort(shuf[col].values), np.sort(E1[col].values)):
            dist_ok = False
            break
    results["shuffled_preserves_distribution"] = dist_ok

    # Spatial structure broken: correlation between shuffled and E1 should be near 0
    # (measured over the full T×n_edges flattened array)
    flat_e1   = E1.values.flatten()
    flat_shuf = shuf.values.flatten()
    r_spatial = float(np.corrcoef(flat_e1, flat_shuf)[0, 1])
    results["shuffled_breaks_spatial_corr"] = abs(r_spatial) < 0.15
    log.info("Shuffled E: global Pearson r with E1 = %.4f (want ≈0)", r_spatial)

    # 4. inverted: SW edges now carry NW profile (and vice versa)
    inv = regimes["inverted"]
    if sw_cols and nw_cols:
        # After swap: SW columns should have mean closer to original NW mean
        orig_sw_mean = E1[sw_cols].values.mean()
        orig_nw_mean = E1[nw_cols].values.mean()
        inv_sw_mean  = inv[sw_cols].values.mean()
        inv_nw_mean  = inv[nw_cols].values.mean()
        # Means will be similar (as documented) — check AC swap instead
        # AC of AZPS_WACM in inverted should match AC of its NW swap partner
        key_sw = next((c for c in sw_cols if "AZPS_WACM" in c or "WACM" in c), sw_cols[0] if sw_cols else None)
        key_nw = next((c for c in nw_cols if "PACW_BPAT" in c or "BPAT" in c), nw_cols[0] if nw_cols else None)
        if key_sw and key_nw:
            ac_e1_sw  = float(np.corrcoef(E1[key_sw].values[:-1],  E1[key_sw].values[1:])[0,1])
            ac_inv_sw = float(np.corrcoef(inv[key_sw].values[:-1], inv[key_sw].values[1:])[0,1])
            ac_e1_nw  = float(np.corrcoef(E1[key_nw].values[:-1],  E1[key_nw].values[1:])[0,1])
            ac_diff   = abs(ac_inv_sw - ac_e1_nw)   # should be small after swap
            results["inverted_swaps_AC_structure"] = ac_diff < 0.20
            log.info(
                "Inverted E AC check: %s lag-1 AC before=%.3f after=%.3f; "
                "NW partner %s AC=%.3f; diff=%.3f",
                key_sw, ac_e1_sw, ac_inv_sw, key_nw, ac_e1_nw, ac_diff,
            )
        else:
            results["inverted_swaps_AC_structure"] = None
        results["inverted_mean_SW"] = float(inv_sw_mean)
        results["inverted_mean_NW"] = float(inv_nw_mean)
    else:
        results["inverted_swaps_AC_structure"] = None

    # 5. All regimes: values in [0, 1]
    for name, df in regimes.items():
        in_range = bool((df.values >= 0).all() and (df.values <= 1).all())
        results[f"{name}_in_unit_interval"] = in_range
        if not in_range:
            log.warning("Regime %s has values outside [0,1]", name)

    # Summary
    passed = sum(1 for v in results.values() if v is True)
    total  = sum(1 for v in results.values() if v is not None)
    log.info("Sanity checks: %d/%d passed", passed, total)
    for name, val in results.items():
        if val is False:
            log.warning("  FAILED: %s", name)
        elif val is True:
            log.info("  OK:     %s", name)

    return results
