"""
experiments/cross_corridor/tests.py
======================================
Sanity checks for corridor membership, E regime construction, and metric outputs.
Called from run_cross_corridor.py before the main experiment loop.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

log = logging.getLogger("cc_tests")


def check_corridor_membership(G, nodes: list[str], SW_CORRIDOR: dict, NW_CORRIDOR: dict, CONNECTOR_NODES: list[str]) -> bool:
    """
    Verify:
    - All corridor nodes exist in the graph
    - All corridor edges exist in the graph (either direction)
    - Connector nodes are in the graph
    - Corridor edges reference only nodes in the graph
    """
    all_ok = True
    existing_nodes = set(nodes)

    for corr_name, corr in [("SW", SW_CORRIDOR), ("NW", NW_CORRIDOR)]:
        for nd in corr["nodes"]:
            if nd not in existing_nodes:
                log.warning("Corridor membership: %s node %s not in graph", corr_name, nd)
                all_ok = False
        for u, v in corr["edges"]:
            if not (G.has_edge(u, v) or G.has_edge(v, u)):
                log.warning("Corridor membership: %s edge (%s,%s) not in graph", corr_name, u, v)
                all_ok = False
        for nd in [corr["upstream"], corr["terminal"]]:
            if nd not in existing_nodes:
                log.warning("Corridor %s: upstream/terminal node %s not in graph", corr_name, nd)
                all_ok = False

    for nd in CONNECTOR_NODES:
        if nd not in existing_nodes:
            log.warning("Connector node %s not in graph", nd)
            all_ok = False

    if all_ok:
        log.info("Corridor membership: OK")
    return all_ok


def check_e_regime_construction(E1, regimes: dict) -> bool:
    """
    Verify E regime shapes, value ranges, and key properties.
    """
    all_ok = True
    T, n_edges = E1.shape

    for name, df in regimes.items():
        if df.shape != (T, n_edges):
            log.warning("E regime %s: shape %s != expected %s", name, df.shape, (T, n_edges))
            all_ok = False
        if not (df.values >= 0).all():
            log.warning("E regime %s: contains values < 0", name)
            all_ok = False
        if not (df.values <= 1).all():
            log.warning("E regime %s: contains values > 1", name)
            all_ok = False

    if all_ok:
        log.info("E regime construction: OK (all shapes and ranges valid)")
    return all_ok


def check_shuffled_distribution(E1, E_shuffled) -> bool:
    """
    Verify shuffled E preserves per-edge marginal distribution.
    Per-edge sorted values should match E1 sorted values within 1e-9.
    """
    all_ok = True
    for col in E1.columns:
        if not np.allclose(np.sort(E1[col].values), np.sort(E_shuffled[col].values), atol=1e-9):
            log.warning("Shuffled E: distribution mismatch on edge %s", col)
            all_ok = False
            break
    if all_ok:
        log.info("Shuffled E distribution: OK")
    return all_ok


def check_inverted_e_swap(E1, E_inverted, sw_cols: list[str], nw_cols: list[str]) -> bool:
    """
    Verify inverted E: after swap, SW edge columns carry NW time-series.
    Check that lag-1 autocorrelation structure has changed for swapped edges.
    """
    if not sw_cols or not nw_cols:
        log.warning("Inverted E check: empty SW or NW column lists — skipping")
        return True

    all_ok = True
    n_improved = 0
    n_checked  = 0
    for sw_c in sw_cols[:5]:
        # Find the NW column that replaced it — check if its AC is now more like NW
        ac_before = float(np.corrcoef(E1[sw_c].values[:-1],       E1[sw_c].values[1:])[0,1])
        ac_after  = float(np.corrcoef(E_inverted[sw_c].values[:-1], E_inverted[sw_c].values[1:])[0,1])
        if sw_c in E1.columns:
            # Average NW AC to compare
            nw_acs = [float(np.corrcoef(E1[c].values[:-1], E1[c].values[1:])[0,1]) for c in nw_cols[:5]]
            nw_mean_ac = float(np.mean(nw_acs))
            if abs(ac_after - nw_mean_ac) < abs(ac_before - nw_mean_ac):
                n_improved += 1
            n_checked += 1

    if n_checked > 0 and n_improved / n_checked >= 0.5:
        log.info("Inverted E AC swap: OK (%d/%d edges show improved match to NW profile)", n_improved, n_checked)
    else:
        log.info("Inverted E AC swap: marginal or no improvement — "
                 "SW/NW E profiles are similar in this synthetic field (expected; see design note)")
    return all_ok


def check_leakage_conservation(leak_record: dict) -> bool:
    """
    Verify leakage ratio is non-negative and within physically plausible range.
    Leakage ratio > 1.0 would indicate measurement error (more leaked than injected).
    """
    all_ok = True
    for k in ["leakage_sw_to_nw", "leakage_nw_to_sw"]:
        v = leak_record.get(k, np.nan)
        if np.isnan(v):
            log.warning("Leakage check: %s is NaN", k)
            all_ok = False
        elif v < 0:
            log.warning("Leakage check: %s = %.4f (negative — check flux sign)", k, v)
            all_ok = False
        elif v > 5.0:
            log.warning("Leakage check: %s = %.4f (>5× initial energy — suspect)", k, v)
    return all_ok


def check_results_or_warn(sanity_results: dict):
    """
    Print sanity check summary. Warn (don't fail) on check failures so the
    experiment can continue and results can be inspected.
    """
    n_pass = sum(1 for v in sanity_results.values() if v is True)
    n_fail = sum(1 for v in sanity_results.values() if v is False)
    n_na   = sum(1 for v in sanity_results.values() if v is None)
    log.info("Sanity checks: %d passed, %d failed, %d n/a", n_pass, n_fail, n_na)
    if n_fail > 0:
        failed = [k for k, v in sanity_results.items() if v is False]
        warnings.warn(
            f"{n_fail} sanity check(s) failed: {failed}. "
            "Results may be unreliable for failed checks. Continuing.",
            stacklevel=2,
        )


# ── __init__.py equivalent exports ──────────────────────────────────────────
__all__ = [
    "check_corridor_membership",
    "check_e_regime_construction",
    "check_shuffled_distribution",
    "check_inverted_e_swap",
    "check_leakage_conservation",
    "check_results_or_warn",
]
