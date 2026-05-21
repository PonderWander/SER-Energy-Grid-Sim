"""
scripts/run_graph_analysis.py
==============================
Graph-based constraint-field analysis over the Western Interconnect
coarse BA network (14 nodes, 28 edges).

Pipeline
--------
1.  Build graph topology (Option A: coarse real network)
2.  Generate spatially-correlated synthetic node signals
3.  Compute node-level S, R, Phi field vectors
4.  Compute three edge fluidity candidates (E1, E2, E3)
5.  Run reduced (constant L) and upgraded (dynamic E) propagation
6.  Compare propagation shape, persistence, and bottlenecks
7.  Produce spatial figures, propagation comparison, animation, dashboard

Data observability summary
--------------------------
  Graph topology   : DOCUMENTED (EIA interchange + WECC public maps)
  Node demand      : SYNTHETIC (spatially-correlated; structure from EIA)
  Node prices      : SYNTHETIC (congestion events on edges)
  Edge flows       : SYNTHETIC (capacity-constrained)
  Edge E values    : COMPUTED from synthetic signals
  Propagation      : MODEL (graph Laplacian diffusion)

Usage
-----
  python scripts/run_graph_analysis.py
  python scripts/run_graph_analysis.py --steps 96 --eta 0.12 --e-candidate E2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constraint_field import load_config
from constraint_field.graph.network import build_graph, graph_summary, node_order
from constraint_field.graph.node_signals import SyntheticNodeSignals, build_node_field
from constraint_field.graph.edge_fluidity import (
    E1_price_spread_edge,
    E2_flow_efficiency_edge,
    E3_congestion_proxy_edge,
    constant_laplacian,
)
from constraint_field.graph.propagation import (
    GraphPropagator,
    PropagationConfig,
    propagation_metrics,
    bottleneck_analysis,
)
from constraint_field.graph.visualize_graph import (
    plot_field_snapshot,
    plot_propagation_comparison,
    plot_bottleneck_map,
    make_phi_animation,
    plot_graph_dashboard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graph_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph constraint-field analysis")
    p.add_argument("--start",       default="2023-01-01")
    p.add_argument("--end",         default="2023-03-31")
    p.add_argument("--steps",       type=int, default=72,
                   help="Simulation horizon (hours)")
    p.add_argument("--eta",         type=float, default=0.10,
                   help="Diffusion transmissibility coefficient")
    p.add_argument("--gamma",       type=float, default=0.05,
                   help="Damping / mean-reversion rate")
    p.add_argument("--e-candidate", default="E1",
                   choices=["E1", "E2", "E3"],
                   help="Which edge E to use in upgraded propagation")
    p.add_argument("--anim-frames", type=int, default=72,
                   help="Number of frames in Phi animation")
    p.add_argument("--out-dir",     default="outputs")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def build_metrics_table(
    reduced_metrics:  dict,
    upgraded_metrics: dict,
    e_candidate:      str,
) -> pd.DataFrame:
    """Combine scalar metrics into a comparison DataFrame."""
    rows = []
    for label, m in [("Reduced (constant L)", reduced_metrics),
                     (f"Upgraded ({e_candidate})", upgraded_metrics)]:
        rows.append({
            "model":           label,
            "rmse":            m.get("rmse", np.nan),
            "max_resid":       m.get("max_resid", np.nan),
            "persistence":     m.get("persistence", np.nan),
            "spatial_std_mean":m.get("spatial_std_mean", np.nan),
            "clustering_coeff":m.get("clustering_coeff", np.nan),
        })
    return pd.DataFrame(rows).set_index("model")


def print_report(
    G,
    field:           dict,
    E_all:           dict,
    reduced_metrics: dict,
    upgraded_metrics:dict,
    bottleneck_df:   pd.DataFrame,
    e_candidate:     str,
    args:            argparse.Namespace,
) -> str:
    """Build and return the text report."""
    sep   = "=" * 68
    nodes = node_order(G)
    Phi   = field["Phi"]

    lines = [
        sep,
        "  GRAPH CONSTRAINT-FIELD ANALYSIS — FINDINGS REPORT",
        sep,
        "",
        "NETWORK SUMMARY",
        f"  Nodes : {G.number_of_nodes()} Western Interconnect BAs",
        f"  Edges : {G.number_of_edges()} interchange corridors",
        f"  Period: {args.start} → {args.end}",
        f"  Topology observability: DOCUMENTED (EIA + WECC public maps)",
        f"  Signal observability  : SYNTHETIC (spatially-correlated)",
        "",
        "─" * 68,
        "1. FIELD STATISTICS (node-level Phi = R - S)",
        "─" * 68,
        "",
    ]

    phi_stats = pd.DataFrame({
        "mean":  Phi.mean(),
        "std":   Phi.std(),
        "p75":   Phi.quantile(0.75),
        "max":   Phi.max(),
        "min":   Phi.min(),
    }).round(3)
    lines.append(phi_stats.to_string())
    lines.append("")

    lines += [
        "─" * 68,
        "2. EDGE FLUIDITY CANDIDATES",
        "─" * 68,
        "",
    ]
    for name, E_df in E_all.items():
        arr = E_df.values
        lines.append(
            f"  {name}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
            f"min={arr.min():.3f}  max={arr.max():.3f}"
        )
    lines.append("")

    lines += [
        "─" * 68,
        "3. PROPAGATION COMPARISON",
        "─" * 68,
        "",
        f"  Operator : graph Laplacian diffusion",
        f"  η (eta)  : {args.eta}  (transmissibility)",
        f"  γ (gamma): {args.gamma}  (damping)",
        f"  Steps    : {args.steps} hours",
        "",
    ]

    for label, m in [("Reduced (constant L, E=1)", reduced_metrics),
                     (f"Upgraded ({e_candidate})", upgraded_metrics)]:
        lines.append(
            f"  {label}:\n"
            f"    RMSE={m.get('rmse',np.nan):.4f}  "
            f"max_resid={m.get('max_resid',np.nan):.4f}  "
            f"persistence={m.get('persistence',np.nan):.4f}  "
            f"spatial_std={m.get('spatial_std_mean',np.nan):.4f}"
        )
    lines.append("")

    lines += [
        "─" * 68,
        "4. BOTTLENECK EDGES (lowest mean E)",
        "─" * 68,
        "",
    ]
    if not bottleneck_df.empty:
        lines.append(
            bottleneck_df[["edge","mean_E","min_E","std_E"]].head(8)
            .to_string(index=False, float_format="%.3f")
        )
    lines.append("")

    # Interpretation
    r_rmse = reduced_metrics.get("rmse", np.nan)
    u_rmse = upgraded_metrics.get("rmse", np.nan)
    gain   = r_rmse - u_rmse

    lines += [
        "─" * 68,
        "5. INTERPRETATION",
        "─" * 68,
        "",
        f"  RMSE improvement from E-weighting: {gain:+.4f} "
        f"({'E adds value' if gain > 0 else 'no improvement'})",
        "",
        "  Propagation differences:",
        f"    Reduced (constant weights): treats all corridors as equally",
        f"    fluid — Phi diffuses uniformly across the graph.",
        f"    Upgraded ({e_candidate}): low-E edges impede propagation —",
        f"    Phi pools near bottlenecks, creating spatial clustering.",
        "",
        "  Data observability notes:",
        "    - Node prices for non-CAISO BAs are APPROXIMATED with",
        "      regional offsets + distance-decay correlation.",
        "    - Edge flows are SYNTHETIC (capacity-constrained noise).",
        "    - Bottleneck rankings should be interpreted as structural",
        "      tendencies of the model, not confirmed physical limits.",
        "",
        sep,
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg  = load_config()

    out_fig = Path(args.out_dir) / "figures"
    out_dat = Path(args.out_dir) / "data"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_dat.mkdir(parents=True, exist_ok=True)
    DPI = cfg["output"]["dpi"]

    log.info("=" * 60)
    log.info("  Graph Constraint-Field Analysis")
    log.info("  Western Interconnect — %d nodes", 14)
    log.info("=" * 60)

    # ── 1. Build graph ────────────────────────────────────────────────────
    log.info("[1/8] Building graph …")
    G     = build_graph()
    nodes = node_order(G)
    log.info("\n%s", graph_summary(G))

    # ── 2. Synthetic node signals ─────────────────────────────────────────
    log.info("[2/8] Generating node signals (%s → %s) …", args.start, args.end)
    synth  = SyntheticNodeSignals(G, seed=args.seed, congestion_prob=0.018)
    dem_df = synth.demand(args.start, args.end)
    pri_df = synth.prices(args.start, args.end)
    flow_df= synth.flows(args.start, args.end)

    log.info("  demand: %s  price: %s  flows: %s",
             dem_df.shape, pri_df.shape, flow_df.shape)

    # ── 3. Node field ─────────────────────────────────────────────────────
    log.info("[3/8] Building node-level S, R, Phi …")
    field = build_node_field(dem_df, pri_df, window=168, clip_sigma=3.0)
    S, R, Phi, Psi = field["S"], field["R"], field["Phi"], field["Psi"]

    # Save panels
    Phi.to_csv(out_dat / "graph_phi.csv")
    S.to_csv(out_dat / "graph_s.csv")
    R.to_csv(out_dat / "graph_r.csv")
    log.info("  Phi shape: %s  mean=%.3f  std=%.3f",
             Phi.shape, Phi.values.mean(), Phi.values.std())

    # ── 4. Edge fluidity ──────────────────────────────────────────────────
    log.info("[4/8] Computing edge fluidity (E1, E2, E3) …")
    E1 = E1_price_spread_edge(G, R)
    E2 = E2_flow_efficiency_edge(G, flow_df)
    E3 = E3_congestion_proxy_edge(G, R)

    E_all = {"E1 (price-spread)": E1, "E2 (flow-efficiency)": E2,
             "E3 (congestion-proxy)": E3}
    for name, E_df in E_all.items():
        log.info("  %s: mean=%.3f std=%.3f min=%.3f max=%.3f",
                 name, E_df.values.mean(), E_df.values.std(),
                 E_df.values.min(), E_df.values.max())

    # Select active E
    E_active = {"E1": E1, "E2": E2, "E3": E3}[args.e_candidate]

    # ── 5. Propagation ────────────────────────────────────────────────────
    log.info("[5/8] Running propagation (reduced + upgraded) …")
    start_idx = len(Phi) // 3   # skip warm-up

    cfg_base = dict(eta=args.eta, gamma=args.gamma, steps=args.steps,
                    noise_std=0.005, seed=args.seed)

    # Reduced: constant graph Laplacian (E = 1 everywhere)
    cfg_reduced = PropagationConfig(**cfg_base, use_E=False)
    prop_r      = GraphPropagator(G, nodes, cfg_reduced)
    traj_r      = prop_r.run(Phi, E_active, start_t=start_idx)
    # Upgraded: dynamic E-weighted Laplacian
    cfg_upgraded = PropagationConfig(**cfg_base, use_E=True)
    prop_u       = GraphPropagator(G, nodes, cfg_upgraded)
    traj_u       = prop_u.run(Phi, E_active, start_t=start_idx)

    # Joint metrics comparison
    metrics_full = propagation_metrics(traj_r, traj_u, nodes)
    met_r = metrics_full[metrics_full["model"] == "reduced"]
    met_u = metrics_full[metrics_full["model"] == "upgraded"]

    def _overall(df):
        row = df[df["node"] == "OVERALL"]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    mr, mu = _overall(met_r), _overall(met_u)
    log.info("  Reduced : RMSE=%.4f  MAE=%.4f  max_resid=%.4f",
             mr.get("rmse",float("nan")), mr.get("mae",float("nan")), mr.get("max_resid",float("nan")))
    log.info("  Upgraded: RMSE=%.4f  MAE=%.4f  max_resid=%.4f",
             mu.get("rmse",float("nan")), mu.get("mae",float("nan")), mu.get("max_resid",float("nan")))

    # Save trajectories
    traj_r.to_csv(out_dat / "graph_traj_reduced.csv")
    traj_u.to_csv(out_dat / "graph_traj_upgraded.csv")

    # Bottleneck analysis
    bn_df = bottleneck_analysis(G, E_active, nodes)
    bn_df.to_csv(out_dat / "graph_bottlenecks.csv", index=False)
    log.info("  Top 3 bottlenecks: %s",
             bn_df.head(3)["edge"].tolist())

    # For report: simple two-row comparison table
    metrics_df = build_metrics_table(mr, mu, args.e_candidate)
    metrics_df.to_csv(out_dat / "graph_metrics.csv")
    # Full node-level metrics for dashboard
    metrics_full.to_csv(out_dat / "graph_metrics_full.csv", index=False)

    # ── 6. Figures ────────────────────────────────────────────────────────
    log.info("[6/8] Generating spatial figures …")

    # Pick three representative time slices
    T      = len(Phi)
    t_peak = int(Phi.abs().sum(axis=1).idxmax() == Phi.index) \
             if False else Phi.abs().sum(axis=1).values.argmax()
    t_slices = [T//4, T//2, t_peak]

    for i, t_snap in enumerate(t_slices):
        fig = plot_field_snapshot(
            G, field, E_active,
            t=t_snap,
            title=f"Field Snapshot — {str(Phi.index[t_snap])[:13]}",
        )
        fig.savefig(out_fig / f"21_field_snapshot_{i+1}.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    log.info("  Saved 21_field_snapshot_1/2/3.png")

    # ── 7. Propagation comparison figure ─────────────────────────────────
    log.info("[7/8] Propagation comparison figure …")

    # Show 4 geographically spread nodes
    show_nodes = ["CISO", "BPAT", "AZPS", "PSCO"]
    show_nodes = [n for n in show_nodes if n in nodes]

    fig = plot_propagation_comparison(
        traj_r, traj_u, nodes=show_nodes,
        n_show=len(show_nodes),
        title=f"Graph Propagation: Reduced vs Upgraded ({args.e_candidate})",
    )
    fig.savefig(out_fig / "22_propagation_comparison.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 22_propagation_comparison.png")

    # Bottleneck map
    fig = plot_bottleneck_map(G, bn_df, field, t=t_peak)
    fig.savefig(out_fig / "23_bottleneck_map.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 23_bottleneck_map.png")

    # Full dashboard
    fig = plot_graph_dashboard(
        G, field, E_active, traj_r, traj_u, bn_df, metrics_full,
        nodes=nodes, snapshot_t=t_peak,
    )
    fig.savefig(out_fig / "24_graph_dashboard.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 24_graph_dashboard.png")

    # ── 8. Animation ──────────────────────────────────────────────────────
    log.info("[8/8] Building Phi animation (%d frames) …", args.anim_frames)
    try:
        anim = make_phi_animation(
            G, field, E_active,
            n_frames=args.anim_frames,
            interval=120,
        )
        anim_path = out_fig / "25_phi_animation.gif"
        anim.save(str(anim_path), writer="pillow", fps=8)
        log.info("  Saved 25_phi_animation.gif")
    except Exception as exc:
        log.warning("  Animation failed: %s — skipping GIF", exc)

    # ── Report ────────────────────────────────────────────────────────────
    report = print_report(
        G, field, E_all,
        mr, mu, bn_df,
        args.e_candidate, args,
    )
    report_path = out_dat / "graph_report.txt"
    report_path.write_text(report)
    print("\n" + report)
    log.info("✓  Graph analysis complete.  Outputs in %s/", args.out_dir)


if __name__ == "__main__":
    main()
