"""
scripts/run_cross_corridor.py
================================
Cross-corridor stress routing experiment.

Tests whether the spatial structure of E acts as a routing-cost matrix
for cross-corridor Φ transfer, independent of aggregate RMSE.

Core hypothesis:
  Pre-existing E gradients select which corridors absorb overflow during
  network-wide loading. Calibrated E1 should show different leakage ratios,
  connector activation order, and covariance structure from uniform or
  shuffled E — even when all regimes produce similar aggregate RMSE.

Usage:
  PYTHONPATH=. python scripts/run_cross_corridor.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from constraint_field.graph.network import build_graph, node_order
from constraint_field.graph.edge_fluidity import E1_price_spread_edge
from scripts.run_spatial_experiments import build_gradient_rich_field

from experiments.cross_corridor.config import (
    SW_CORRIDOR, NW_CORRIDOR, CONNECTOR_NODES,
    LOADING_SIGMA, LOADING_VARIANTS, E_REGIMES,
    ETA, GAMMA, STEPS, SEED, OUT_DAT,
)
from experiments.cross_corridor.e_regimes import (
    build_all_regimes, run_sanity_checks,
)
from experiments.cross_corridor.simulation import (
    init_dual_corridor_phi, simulate,
)
from experiments.cross_corridor.metrics import (
    metric_cross_corr, metric_cooccurrence, metric_leakage,
    metric_connector_activation, metric_path_activation,
    metric_spatial_covariance, metric_moran, metric_rmse,
    build_summary_record,
)
from experiments.cross_corridor.visualize import generate_all_figures
from experiments.cross_corridor import tests as sanity_tests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_cross_corridor")


def main():
    # ── 1. Build graph, field, calibrated E1 ─────────────────────────────────
    log.info("Building graph and gradient-rich field (seed=%d, T=1440) …", SEED)
    G      = build_graph()
    nodes  = node_order(G)
    field  = build_gradient_rich_field(G, nodes, seed=SEED, T=1440)
    E1     = E1_price_spread_edge(G, field["R"])
    bg_std = float(field["Phi"].values.std(axis=1).mean())

    log.info("  σ=%.4f  η=%.2f  γ=%.3f  steps=%d", bg_std, ETA, GAMMA, STEPS)
    log.info("  SW corridor: %s", SW_CORRIDOR["nodes"])
    log.info("  NW corridor: %s", NW_CORRIDOR["nodes"])
    log.info("  Connectors:  %s", CONNECTOR_NODES)

    # ── 2. Build E regimes and run sanity checks ──────────────────────────────
    log.info("Building E regimes …")
    regimes = build_all_regimes(E1, seed=SEED)
    log.info("Running sanity checks …")
    check_results = run_sanity_checks(E1, regimes)
    sanity_tests.check_results_or_warn(check_results)

    # ── 3. Run all conditions ─────────────────────────────────────────────────
    n_total = len(E_REGIMES) * len(LOADING_SIGMA) * len(LOADING_VARIANTS)
    log.info("Running %d conditions (%d regimes × %d loadings × %d variants) …",
             n_total, len(E_REGIMES), len(LOADING_SIGMA), len(LOADING_VARIANTS))

    all_records = []
    all_results = {}   # {(regime, loading, variant): {traj, metrics, ...}}

    # Pre-compute uniform traj for each (loading, variant) for RMSE comparison
    uniform_trajs = {}

    for loading in LOADING_SIGMA:
        for var_name, (sw_frac, nw_frac) in LOADING_VARIANTS.items():
            sw_load = loading * sw_frac * bg_std
            nw_load = loading * nw_frac * bg_std
            Phi0    = init_dual_corridor_phi(G, nodes, sw_load, nw_load)

            # Uniform first (for RMSE baseline)
            res_uni = simulate(G, nodes, Phi0, regimes["uniform"], "uniform",
                               eta=ETA, gamma=GAMMA, steps=STEPS, seed=SEED)
            uniform_trajs[(loading, var_name)] = res_uni["traj"]

            for regime_name in E_REGIMES:
                E_regime_df = regimes[regime_name]
                res = simulate(G, nodes, Phi0, E_regime_df, regime_name,
                               eta=ETA, gamma=GAMMA, steps=STEPS, seed=SEED)

                traj      = res["traj"]
                flux_traj = res["flux_traj"]

                # Compute all metrics
                cc   = metric_cross_corr(traj, nodes)
                coo  = metric_cooccurrence(traj, nodes)
                leak = metric_leakage(traj, nodes, Phi0, flux_traj)
                conn = metric_connector_activation(traj, nodes)
                path = metric_path_activation(traj, nodes, flux_traj, E_regime_df, G)
                cov  = metric_spatial_covariance(traj, nodes)
                mi   = metric_moran(traj, nodes, G)
                rmse = metric_rmse(traj, uniform_trajs[(loading, var_name)])

                all_results[(regime_name, loading, var_name)] = {
                    "traj":      traj,
                    "flux_traj": flux_traj,
                    "Phi0":      Phi0,
                    "metrics":   {
                        "cc": cc, "coo": coo, "leak": leak,
                        "conn": conn, "path": path,
                        "cov": cov, "mi": mi,
                    },
                }

                rec = build_summary_record(
                    regime=regime_name, loading=loading,
                    loading_variant=var_name, traj=traj,
                    Phi0=Phi0, flux_traj=flux_traj,
                    E_df=E_regime_df, G=G, nodes=nodes,
                    traj_uniform=uniform_trajs[(loading, var_name)],
                )
                all_records.append(rec)

                log.info(
                    "  %-14s  %-10s  %3.1fσ  "
                    "sw→nw=%.3f  nw→sw=%.3f  cross_cov=%.3f  "
                    "MI=%.3f  RMSE_vs_uni=%.4f  conn_first=%s",
                    regime_name, var_name, loading,
                    leak["leakage_sw_to_nw"], leak["leakage_nw_to_sw"],
                    cov["cross_corridor_ratio"],
                    mi["moran_mean"], rmse,
                    conn.get("activation_order", ["?"])[0] if conn.get("activation_order") else "—",
                )

    # ── 4. Collate and save results ───────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_DAT / "cross_corridor.csv", index=False)
    log.info("Saved cross_corridor.csv (%d rows)", len(df))

    # ── 5. Generate figures ───────────────────────────────────────────────────
    log.info("Generating figures …")
    generate_all_figures(df, all_results, nodes, G)

    # ── 6. Write report ───────────────────────────────────────────────────────
    log.info("Writing report …")
    write_report(df, all_results, nodes, bg_std)

    log.info("Complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, results: dict, nodes: list[str], bg_std: float):
    """
    Answers all nine questions from the experiment design spec.
    """
    sep   = "=" * 72
    lines = [
        sep,
        "  CROSS-CORRIDOR STRESS ROUTING EXPERIMENT — FINDINGS",
        sep, "",
        f"  Background σ = {bg_std:.4f}  |  η={ETA}  γ={GAMMA}  steps={STEPS}",
        f"  E regimes:  {list(E_REGIMES.keys())}",
        f"  Loading:    {LOADING_SIGMA} × σ",
        f"  Variants:   {list(LOADING_VARIANTS.keys())}",
        "",
    ]

    # Helper
    def _regime_val(regime, variant, col, agg="mean"):
        sub = df[(df["regime"] == regime) & (df["loading_variant"] == variant)]
        if sub.empty or col not in sub.columns:
            return np.nan
        return float(sub[col].agg(agg))

    sym = "symmetric"
    regimes = list(E_REGIMES.keys())

    # ── Q1: Did E affect aggregate RMSE? ──────────────────────────────────────
    lines += ["─" * 72, "Q1: Did E affect aggregate RMSE?", "─" * 72, ""]
    lines.append(f"  {'Regime':<20} {'RMSE vs uniform (mean over loading × variant)':>46}")
    lines.append("  " + "-" * 50)
    for r in regimes:
        if r == "uniform":
            lines.append(f"  {r:<20}   0.0000  (baseline)")
            continue
        rmse_m = _regime_val(r, sym, "rmse_vs_uniform", "mean")
        lines.append(f"  {r:<20}   {rmse_m:.5f}")
    lines += [
        "",
        "  Interpretation: RMSE vs uniform captures the mean-field difference",
        "  across all nodes and timesteps. If all values are <0.01, E does not",
        "  materially reduce prediction error relative to uniform transmissibility.",
        "",
    ]

    # ── Q2: Did E affect routing? ─────────────────────────────────────────────
    lines += ["─" * 72, "Q2: Did E affect routing?", "─" * 72, ""]
    lines.append(f"  {'Regime':<20} {'leak_SW→NW':>12} {'leak_NW→SW':>12} {'cross_cov_ratio':>16} {'Moran_I':>9}")
    lines.append("  " + "-"*68)
    uni_sw = _regime_val("uniform", sym, "leakage_sw_to_nw"); uni_cr = _regime_val("uniform", sym, "cross_ratio")
    for rg in regimes:
        leak_sw = _regime_val(rg, sym, "leakage_sw_to_nw")
        leak_nw = _regime_val(rg, sym, "leakage_nw_to_sw")
        cross_r = _regime_val(rg, sym, "cross_ratio")
        moran_v = _regime_val(rg, sym, "moran_mean")
        lines.append(f"  {rg:<20} {leak_sw:12.4f} {leak_nw:12.4f} {cross_r:16.4f} {moran_v:9.4f}")
    lines += [
        "",
        "  Note: leakage = peak single-step cross-corridor flux / initial energy.",
        "  Higher leakage = more Φ escaping its corridor of origin per step.",
        "",
    ]

    # ── Q3: Static E1 vs uniform ──────────────────────────────────────────────
    lines += ["─" * 72, "Q3: Did static calibrated E1 differ from E=1?", "─" * 72, ""]
    high = 2.5
    for metric, label in [
        ("leakage_sw_to_nw", "SW→NW leakage at 2.5σ"),
        ("cross_ratio",      "Cross-corridor cov ratio at 2.5σ"),
        ("moran_mean",       "Moran's I at 2.5σ"),
        ("conn_first",       "First connector activation"),
    ]:
        for regime in ["uniform", "calibrated_E1"]:
            sub = df[(df["regime"]==regime) & (df["loading"]==high) & (df["loading_variant"]==sym)]
            if not sub.empty and metric in sub.columns:
                val = sub.iloc[0][metric]
                lines.append(f"  {label:40s} {regime}: {val}")
    lines += [""]

    # ── Q4: State-dependent E ─────────────────────────────────────────────────
    lines += ["─" * 72, "Q4: Did state-dependent E suppress or redirect leakage?", "─" * 72, ""]
    for r in ["calibrated_E1", "state_dep"]:
        high_sub  = df[(df["regime"]==r) & (df["loading_variant"]==sym)].sort_values("loading")
        leak_low  = high_sub[high_sub["loading"] <= 1.0]["leakage_sw_to_nw"].mean()
        leak_high = high_sub[high_sub["loading"] >= 2.0]["leakage_sw_to_nw"].mean()
        lines.append(f"  {r:<20}: leak@low_load={leak_low:.4f}  leak@high_load={leak_high:.4f}"
                     f"  {'(suppressed under load)' if leak_high < leak_low else '(amplified under load)'}")
    lines += [""]

    # ── Q5: Shuffled E ────────────────────────────────────────────────────────
    lines += ["─" * 72, "Q5: Did shuffled E destroy the corridor-specific routing pattern?", "─" * 72, ""]
    for r in ["calibrated_E1", "shuffled"]:
        tau_m = _regime_val(r, sym, "path_rank_tau")
        cross = _regime_val(r, sym, "cross_ratio")
        lines.append(f"  {r:<20}: path_rank_tau={tau_m:.3f}  cross_ratio={cross:.4f}")
    lines += [
        "  Shuffled E breaks temporal correlation but preserves marginal distribution.",
        "  If routing metrics differ from calibrated E1, spatial E structure matters.",
        "",
    ]

    # ── Q6: Inverted E ────────────────────────────────────────────────────────
    lines += ["─" * 72, "Q6: Did inverted E reverse the preferred leakage pathway?", "─" * 72, ""]
    for r in ["calibrated_E1", "inverted"]:
        sw_nw = _regime_val(r, sym, "leakage_sw_to_nw")
        nw_sw = _regime_val(r, sym, "leakage_nw_to_sw")
        lines.append(f"  {r:<20}: SW→NW={sw_nw:.4f}  NW→SW={nw_sw:.4f}"
                     f"  ratio={sw_nw/max(nw_sw,1e-9):.2f}")
    lines += [
        "  Inverted E swaps the temporal correlation structure of SW and NW edges.",
        "  A reversal in the preferred leakage direction (SW→NW vs NW→SW) would",
        "  confirm that E's temporal structure, not just its mean level, drives routing.",
        "",
    ]

    # ── Q7: Connector nodes ───────────────────────────────────────────────────
    lines += ["─" * 72, "Q7: Which connector nodes mediated cross-corridor transfer?", "─" * 72, ""]
    for r in ["calibrated_E1", "state_dep", "shuffled"]:
        sub = df[(df["regime"]==r) & (df["loading"]==high) & (df["loading_variant"]==sym)]
        if not sub.empty:
            first = sub.iloc[0].get("connector_first", "?")
            peaks = {nd: sub.iloc[0].get(f"conn_peak_{nd}", np.nan) for nd in CONNECTOR_NODES}
            pk_str = "  ".join(f"{nd}={v:.3f}" for nd, v in peaks.items())
            lines.append(f"  {r:<20}: first={first}  peaks: {pk_str}")
    lines += [""]

    # ── Q8: E as prediction-error or routing-cost variable? ───────────────────
    lines += ["─" * 72,
              "Q8: Is E better interpreted as a prediction-error variable",
              "    or as a routing-cost / transmissibility field?",
              "─" * 72, ""]

    # Compare RMSE range vs routing metric range
    rmse_range  = df[df["regime"] != "uniform"]["rmse_vs_uniform"].agg(["min","max","std"])
    leak_range  = df["leakage_sw_to_nw"].agg(["min","max","std"])
    cross_range = df["cross_ratio"].agg(["min","max","std"])

    lines += [
        f"  RMSE variation (non-uniform regimes): min={rmse_range['min']:.5f}  "
        f"max={rmse_range['max']:.5f}  std={rmse_range['std']:.5f}",
        f"  Leakage SW→NW variation:             min={leak_range['min']:.4f}  "
        f"max={leak_range['max']:.4f}  std={leak_range['std']:.4f}",
        f"  Cross-corridor cov ratio variation:  min={cross_range['min']:.4f}  "
        f"max={cross_range['max']:.4f}  std={cross_range['std']:.4f}",
        "",
    ]

    rmse_spread  = rmse_range["max"] - rmse_range["min"]
    leak_spread  = leak_range["max"] - leak_range["min"]
    route_signal = leak_spread > rmse_spread * 10

    if route_signal:
        lines += [
            "  FINDING: Routing metrics show substantially greater variation across",
            "  E regimes than RMSE. E has limited effect on aggregate prediction error",
            "  but materially changes the spatial allocation of Φ under constrained",
            "  loading. The experiment supports interpreting E as a routing-cost field",
            "  rather than a prediction-error variable.",
        ]
    else:
        lines += [
            "  FINDING: Routing metric variation is modest relative to RMSE variation.",
            "  Further investigation with stronger E spatial gradients (e.g., real",
            "  nodal LMP data with genuine cross-corridor price differences) is needed.",
        ]

    lines += ["", sep]

    report = "\n".join(lines)
    report_path = OUT_DAT / "cross_corridor_report.txt"
    report_path.write_text(report)
    print("\n" + report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
