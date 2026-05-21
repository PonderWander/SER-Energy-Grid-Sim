"""
scripts/run_graph_model.py
============================
End-to-end graph constraint field model.

Produces:
  - WECC coarse BA graph with spatially-correlated synthetic signals
  - Three edge fluidity candidates (E1, E2, E3)
  - Graph Laplacian propagation: reduced vs upgraded comparison
  - Phi animation (GIF)
  - Full dashboard figure
  - Bottleneck analysis table
  - Observability documentation

Usage
-----
  python scripts/run_graph_model.py
  python scripts/run_graph_model.py --snapshot 72 --steps 96
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
from constraint_field.graph import (
    build_wecc_graph,
    SyntheticNodeSignals,
    build_node_field,
    E1_price_spread_edge,
    E2_flow_efficiency_edge,
    E3_congestion_proxy_edge,
    GraphPropagator,
    PropagationConfig,
    propagation_metrics,
    bottleneck_analysis,
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
log = logging.getLogger("graph_model")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="config/default.yaml")
    p.add_argument("--start",        default="2023-01-01")
    p.add_argument("--end",          default="2023-03-31")
    p.add_argument("--snapshot",     type=int, default=48,
                   help="Timestep index for spatial snapshots")
    p.add_argument("--steps",        type=int, default=72,
                   help="Propagation simulation horizon (hours)")
    p.add_argument("--eta",          type=float, default=0.04)
    p.add_argument("--gamma",        type=float, default=0.10)
    p.add_argument("--E-candidate",  default="E1",
                   choices=["E1","E2","E3"])
    p.add_argument("--n-anim-frames",type=int, default=96)
    p.add_argument("--out-dir",      default="outputs")
    return p.parse_args()


def observability_report(G) -> str:
    """Generate a structured observability documentation string."""
    lines = [
        "=" * 64,
        "  GRAPH MODEL OBSERVABILITY DOCUMENTATION",
        "=" * 64,
        "",
        "TOPOLOGY (Option A: WECC Coarse BA Graph)",
        "-" * 40,
        f"  Nodes : {G.number_of_nodes()} balancing authorities",
        f"  Edges : {G.number_of_edges()} interchange paths",
        "",
        "  Node observability:",
        "    ALL nodes: DOCUMENTED",
        "    Source: EIA-930 balancing authority list",
        "    Coordinates: approximate centroids from NERC BA boundary map",
        "",
        "  Edge observability:",
    ]
    for u, v, d in G.edges(data=True):
        lines.append(
            f"    {u:6s}—{v:6s}: {d['observability']:12s}  "
            f"cap≈{d['capacity_gw']:.1f}GW  [{d['corridor']}]"
        )
    lines += [
        "",
        "SIGNALS",
        "-" * 40,
        "  Demand S_t: SYNTHETIC",
        "    Model: spatially-correlated multivariate normal with",
        "           diurnal/weekly seasonality, node-specific amplitude",
        "           from EIA-860 documented peak_gw values.",
        "    In live deployment: replace with EIA-930 hourly BA demand.",
        "",
        "  Price R_t: SYNTHETIC",
        "    Model: load-correlated base price + corridor-specific",
        "           congestion events on documented interchange paths.",
        "    In live deployment: replace with ISO LMP APIs per BA.",
        "",
        "EDGE FLUIDITY",
        "-" * 40,
        "  E1 (price-spread inverse): COMPUTED from synthetic R field",
        "  E2 (flow efficiency):      COMPUTED from synthetic flows + ",
        "                             DOCUMENTED capacity from WECC ADS",
        "  E3 (congestion proxy):     COMPUTED from synthetic R field",
        "",
        "  All E candidates: SYNTHETIC in this environment.",
        "  In live deployment: E2 flows from EIA-930 interchange;",
        "  E1/E3 from ISO price APIs.",
        "",
        "PROPAGATION",
        "-" * 40,
        "  Graph Laplacian diffusion — mathematical model only.",
        "  Does not reconstruct physical AC power flow.",
        "  Represents administrative-level constraint pressure",
        "  propagation, not electrical physics.",
        "=" * 64,
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    out_fig = Path(args.out_dir) / "figures"
    out_dat = Path(args.out_dir) / "data"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_dat.mkdir(parents=True, exist_ok=True)
    DPI = cfg["output"]["dpi"]

    log.info("=" * 60)
    log.info("  Graph Constraint Field Model — WECC Coarse BA Graph")
    log.info("=" * 60)

    # ── Build graph ───────────────────────────────────────────────────────
    log.info("[1/7] Building WECC BA graph …")
    G     = build_wecc_graph()
    nodes = list(G.nodes())
    log.info("  %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    obs_report = observability_report(G)
    (out_dat / "graph_observability.txt").write_text(obs_report)
    print("\n" + obs_report)

    # ── Generate node signals ─────────────────────────────────────────────
    log.info("[2/7] Generating node signals (SYNTHETIC) …")
    source    = SyntheticNodeSignals(G, seed=42, congestion_prob=0.018)
    demand_df = source.demand(args.start, args.end)
    price_df  = source.prices(args.start, args.end)
    flows_df  = source.flows( args.start, args.end)

    log.info("  demand_df: %d rows × %d nodes", *demand_df.shape)
    log.info("  price_df : %d rows × %d nodes", *price_df.shape)
    log.info("  flows_df : %d rows × %d edges", *flows_df.shape)

    # ── Build node field vectors ──────────────────────────────────────────
    log.info("[3/7] Building node field vectors S, R, Phi …")
    field = build_node_field(demand_df, price_df)
    Phi   = field["Phi"]
    field["Phi"].to_csv(out_dat / "node_phi.csv")
    field["S"].to_csv(  out_dat / "node_S.csv")
    field["R"].to_csv(  out_dat / "node_R.csv")

    # ── Edge fluidity candidates ──────────────────────────────────────────
    log.info("[4/7] Computing edge fluidity candidates …")
    E1 = E1_price_spread_edge(G, field["R"], alpha=2.0)
    E2 = E2_flow_efficiency_edge(G, flows_df, beta=2.0)
    E3 = E3_congestion_proxy_edge(G, field["R"], lambda_=1.0)

    # Select active E candidate
    E_map = {"E1": E1, "E2": E2, "E3": E3}
    E_active = E_map[args.E_candidate]
    log.info("  Active E candidate: %s  mean=%.3f  std=%.3f",
             args.E_candidate, E_active.values.mean(), E_active.values.std())

    # Save all E candidates
    E1.to_csv(out_dat / "edge_E1.csv")
    E2.to_csv(out_dat / "edge_E2.csv")
    E3.to_csv(out_dat / "edge_E3.csv")

    # Bottleneck analysis
    bt_df = bottleneck_analysis(G, E_active, nodes)
    bt_df.to_csv(out_dat / "bottlenecks.csv", index=False)
    log.info("  Top 3 bottleneck edges:\n%s",
             bt_df[["edge","corridor","mean_E","frac_lt_03"]].head(3).to_string())

    # ── Graph propagation ─────────────────────────────────────────────────
    log.info("[5/7] Running graph propagation (reduced vs upgraded) …")
    cfg_prop = PropagationConfig(
        eta=args.eta, gamma=args.gamma,
        noise_std=0.01, steps=args.steps, seed=42,
    )
    propagator = GraphPropagator(G, nodes, cfg_prop)

    # Inject a shock at CISO node at t=24
    shock = {
        "node": "CISO" if "CISO" in nodes else nodes[0],
        "t_start": 24,
        "duration": 3,
        "magnitude": 1.5,
    }
    start_t = len(Phi) // 3

    both     = propagator.compare(Phi, E_active, start_t=start_t, shock=shock)
    reduced  = both["reduced"]
    upgraded = both["upgraded"]

    metrics = propagation_metrics(reduced, upgraded, nodes)
    metrics.to_csv(out_dat / "propagation_metrics.csv", index=False)
    overall = metrics[metrics["node"] == "OVERALL"]
    log.info("  Propagation RMSE:\n%s", overall[["model","rmse","mae"]].to_string())

    reduced.to_csv( out_dat / "prop_reduced.csv")
    upgraded.to_csv(out_dat / "prop_upgraded.csv")

    # ── Figures ───────────────────────────────────────────────────────────
    log.info("[6/7] Generating figures …")

    fig = plot_field_snapshot(G, field, E_active, t=args.snapshot,
                              title=f"WECC BA Graph — Phi & E snapshot (t={args.snapshot})")
    fig.savefig(out_fig / "21_field_snapshot.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 21_field_snapshot.png")

    fig = plot_propagation_comparison(reduced, upgraded, nodes, n_show=4,
                                      title=f"Propagation — {args.E_candidate} edge fluidity")
    fig.savefig(out_fig / "22_propagation_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 22_propagation_comparison.png")

    fig = plot_bottleneck_map(G, bt_df, field, t=args.snapshot)
    fig.savefig(out_fig / "23_bottleneck_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 23_bottleneck_map.png")

    fig = plot_graph_dashboard(
        G, field, E_active, reduced, upgraded, bt_df, metrics, nodes,
        snapshot_t=args.snapshot,
        title=f"Graph Constraint Field Dashboard — WECC ({args.E_candidate})",
    )
    fig.savefig(out_fig / "24_graph_dashboard.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 24_graph_dashboard.png")

    # Animation
    log.info("  Generating Phi animation (%d frames) …", args.n_anim_frames)
    try:
        anim = make_phi_animation(G, field, E_active,
                                   n_frames=args.n_anim_frames, interval=120)
        anim.save(str(out_fig / "25_phi_animation.gif"),
                  writer="pillow", fps=8)
        log.info("  Saved 25_phi_animation.gif")
    except Exception as exc:
        log.warning("  Animation failed: %s  (pillow may not be installed)", exc)

    # ── Summary report ────────────────────────────────────────────────────
    log.info("[7/7] Writing summary …")
    rmse_r = float(overall[overall["model"]=="reduced"]["rmse"].iloc[0])
    rmse_u = float(overall[overall["model"]=="upgraded"]["rmse"].iloc[0])
    delta  = rmse_r - rmse_u

    summary = f"""
GRAPH MODEL SUMMARY
{'='*50}
Graph:    WECC Coarse BA — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges
Signals:  SYNTHETIC (spatially-correlated, documented peak GW)
Period:   {args.start} → {args.end}
Steps:    {args.steps}h  |  E candidate: {args.E_candidate}

PROPAGATION
  Reduced  (constant L) RMSE = {rmse_r:.4f}
  Upgraded (dynamic E)  RMSE = {rmse_u:.4f}
  Delta RMSE            = {delta:+.4f} ({'upgraded better' if delta>0 else 'no improvement'})

TOP BOTTLENECK EDGES (lowest mean E)
{bt_df[['edge','corridor','mean_E','frac_lt_03']].head(5).to_string(index=False)}

OBSERVABILITY SUMMARY
  Node topology:  DOCUMENTED (EIA-930 BA list)
  Edge topology:  {sum(1 for _,_,d in G.edges(data=True) if d['observability']=='DOCUMENTED')} DOCUMENTED, {sum(1 for _,_,d in G.edges(data=True) if d['observability']=='APPROXIMATED')} APPROXIMATED
  Signals S,R,E:  SYNTHETIC
  Capacity GW:    DOCUMENTED (WECC ADS approximate ATC)
"""
    print(summary)
    (out_dat / "graph_summary.txt").write_text(summary)
    log.info("✓ Graph model complete. Outputs in %s/", args.out_dir)


if __name__ == "__main__":
    main()
