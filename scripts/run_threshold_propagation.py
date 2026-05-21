"""
scripts/run_threshold_propagation.py
======================================
Determine whether hop-ordered propagation appears consistently once
local |Phi| exceeds a threshold relative to background spatial variance,
and whether the same corridors/nodes repeatedly mediate transmission
across source nodes and regimes.

Usage:  python scripts/run_threshold_propagation.py
"""

from __future__ import annotations
import logging, sys
from pathlib import Path
from itertools import product as iproduct
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from constraint_field.graph.network import build_graph, node_order
from constraint_field.graph.edge_fluidity import E1_price_spread_edge
from constraint_field.graph.propagation import GraphPropagator, PropagationConfig
from scripts.run_spatial_experiments import (
    build_gradient_rich_field, adjacency_W, node_distances, geo_distances,
    moran_i, ETA, GAMMA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("threshold_prop")

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)
DPI = 150

SHOCK_NODES  = ["CISO", "AZPS", "PACW", "NEVP", "BPAT", "PSCO"]
THRESHOLDS   = [1.0, 1.5, 2.0, 2.5]
SHOCK_STEPS  = 36
START_T      = 500


def run_impulse(G, nodes, Phi, E1, shock_node, shock_mag, eta=ETA, gamma=GAMMA, steps=SHOCK_STEPS, start_t=START_T):
    shock = {"node": shock_node, "t_start": 1, "magnitude": shock_mag, "duration": 1}
    def _run(use_shock):
        cfg  = PropagationConfig(eta=eta, gamma=gamma, steps=steps, use_E=True, noise_std=0.0, seed=42)
        prop = GraphPropagator(G, nodes, cfg)
        return prop.run(Phi, E1, start_t=start_t, shock=shock if use_shock else None)
    t_shock = _run(True)
    t_base  = _run(False)
    sim_cols = [f"{nd}_sim" for nd in nodes if f"{nd}_sim" in t_shock.columns]
    s_arr = t_shock[sim_cols].values.astype(float)
    b_arr = t_base[sim_cols].values.astype(float)
    return s_arr - b_arr


def is_hop_ordered(impulse, nodes, hop_D, source_node, min_nodes=4):
    src_i = nodes.index(source_node)
    hops  = np.array([int(hop_D[src_i, j]) for j in range(len(nodes))])
    pmags = np.abs(impulse).max(axis=0)
    ptimes = np.argmax(np.abs(impulse), axis=0).astype(float)
    mask = (hops > 0) & (pmags > 0.01)
    if mask.sum() < min_nodes:
        return np.nan, np.nan, False
    tau, pval = scipy_stats.kendalltau(hops[mask], ptimes[mask])
    return float(tau), float(pval), bool(tau > 0 and pval < 0.05)


def transmission_path(impulse, nodes, hop_D, source_node, thresh_frac=0.15):
    src_i    = nodes.index(source_node)
    src_peak = float(np.abs(impulse[:, src_i]).max())
    if src_peak < 1e-6:
        return []
    cutoff = thresh_frac * src_peak
    active = []
    for i, nd in enumerate(nodes):
        if nd == source_node:
            continue
        pmag = float(np.abs(impulse[:, i]).max())
        pt   = int(np.argmax(np.abs(impulse[:, i])))
        if pmag >= cutoff:
            active.append({"node": nd, "hop": int(hop_D[src_i, i]),
                           "peak_mag": pmag, "peak_t": pt, "rel_mag": pmag / src_peak})
    return sorted(active, key=lambda x: (x["hop"], x["peak_t"]))


def run_threshold_experiment(G, nodes, field, E1, hop_D):
    Phi           = field["Phi"]
    phi_arr       = Phi.values
    spatial_std_t = phi_arr.std(axis=1)
    bg_std        = float(spatial_std_t.mean())
    consistency_records = []
    all_paths           = []
    node_med  = {nd: 0 for nd in nodes}
    edge_med  = {f"{u}_{v}": 0 for u, v in G.edges()}
    total_cond = 0

    for src_node in SHOCK_NODES:
        if src_node not in nodes:
            continue
        for thresh_mult in THRESHOLDS:
            shock_mag = thresh_mult * bg_std
            total_cond += 1
            log.info("  src=%-6s  thresh=%.1f×  mag=%.3f", src_node, thresh_mult, shock_mag)
            impulse = run_impulse(G, nodes, Phi, E1, src_node, shock_mag)
            tau, pval, ordered = is_hop_ordered(impulse, nodes, hop_D, src_node)
            path = transmission_path(impulse, nodes, hop_D, src_node, thresh_frac=0.12)
            consistency_records.append({
                "source": src_node, "threshold": thresh_mult, "shock_mag": shock_mag,
                "kendall_tau": tau, "pvalue": pval, "hop_ordered": ordered,
                "n_active": len(path), "src_degree": G.degree(src_node),
            })
            active_nodes = {p["node"] for p in path}
            for nd in active_nodes:
                node_med[nd] += 1
            for p in path:
                tgt = p["node"]
                try:
                    sp = nx.shortest_path(G, src_node, tgt)
                    for k in range(len(sp) - 1):
                        u, v = sp[k], sp[k+1]
                        key  = f"{u}_{v}"
                        rkey = f"{v}_{u}"
                        if key in edge_med:
                            edge_med[key] += 1
                        elif rkey in edge_med:
                            edge_med[rkey] += 1
                except nx.NetworkXNoPath:
                    pass
            for p in path:
                all_paths.append({"source": src_node, "threshold": thresh_mult, **p})

    consistency_df = pd.DataFrame(consistency_records)
    consistency_df.to_csv(OUT_DAT / "T1_consistency.csv", index=False)
    mediation_df = pd.DataFrame([
        {"node": nd, "med_count": cnt, "med_frac": cnt / total_cond, "degree": G.degree(nd)}
        for nd, cnt in node_med.items()
    ]).sort_values("med_frac", ascending=False)
    mediation_df.to_csv(OUT_DAT / "T2_node_mediation.csv", index=False)
    edge_df = pd.DataFrame([
        {"edge": e, "u": e.split("_")[0], "v": e.split("_")[1],
         "med_count": cnt, "med_frac": cnt / total_cond,
         "capacity_gw": G[e.split("_")[0]][e.split("_")[1]].get("capacity_gw", 1.0)
                        if G.has_edge(e.split("_")[0], e.split("_")[1]) else np.nan}
        for e, cnt in edge_med.items() if cnt > 0
    ]).sort_values("med_frac", ascending=False)
    edge_df.to_csv(OUT_DAT / "T3_edge_mediation.csv", index=False)
    paths_df = pd.DataFrame(all_paths)
    paths_df.to_csv(OUT_DAT / "T4_paths.csv", index=False)
    return consistency_df, mediation_df, edge_df, paths_df, total_cond


def plot_consistency_matrix(consistency_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title, cmap, vcenter in [
        (axes[0], "kendall_tau", "Kendall τ (hop ordering)", "RdYlGn", 0),
        (axes[1], "pvalue",      "p-value (ordering test)",  "RdYlGn_r", None),
    ]:
        pv = consistency_df.pivot_table(index="source", columns="threshold", values=col, aggfunc="mean")
        if vcenter is not None:
            vmax = max(abs(pv.values.max()), abs(pv.values.min()), 0.1)
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=vcenter, vmax=vmax)
        else:
            norm = mcolors.Normalize(0, 0.1)
        im = ax.imshow(pv.values, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([f"{t:.1f}×" for t in pv.columns])
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index)
        ax.set_xlabel("Threshold (× bg_std)")
        ax.set_ylabel("Shock source node")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        for i, src in enumerate(pv.index):
            for j, thr in enumerate(pv.columns):
                row = consistency_df[(consistency_df["source"]==src) & (consistency_df["threshold"]==thr)]
                if row.empty:
                    continue
                tau_val = float(row["kendall_tau"].iloc[0])
                ordered = bool(row["hop_ordered"].iloc[0])
                pv_val  = float(row["pvalue"].iloc[0]) if not pd.isna(row["pvalue"].iloc[0]) else 1.0
                txt = f"{tau_val:.2f}" + ("★" if ordered else "") if col == "kendall_tau" else f"{pv_val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black" if abs(tau_val) < 0.7 else "white")
    fig.suptitle("Consistency of Hop-Ordered Propagation\n★ = significant (τ>0, p<0.05)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "T1_consistency_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved T1_consistency_matrix.png")


def plot_mediation_maps(G, nodes, mediation_df, edge_df, total_cond):
    pos       = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}
    node_freq = {row["node"]: row["med_frac"] for _, row in mediation_df.iterrows()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    freq_vals  = np.array([node_freq.get(nd, 0) for nd in nodes])
    node_sizes = 200 + 800 * freq_vals
    node_cols  = [plt.cm.YlOrRd(f) for f in freq_vals]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_cols, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    edge_freq_map = {}
    for _, row in edge_df.iterrows():
        u, v = row["u"], row["v"]
        edge_freq_map[(u, v)] = row["med_frac"]
        edge_freq_map[(v, u)] = row["med_frac"]
    e_cmap = plt.cm.Blues
    for u, v in G.edges():
        f    = edge_freq_map.get((u, v), 0)
        col  = e_cmap(0.2 + 0.8 * f)
        width = 0.5 + 5 * f
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=[col], width=width, alpha=0.85)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Node mediation freq", shrink=0.7)
    ax.set_title("(A) Node & edge mediation frequency")
    ax.axis("off")
    ax2 = axes[1]
    mdf = mediation_df.sort_values("med_frac", ascending=True)
    colors = [plt.cm.YlOrRd(f) for f in mdf["med_frac"]]
    bars = ax2.barh(mdf["node"], mdf["med_frac"], color=colors, alpha=0.9)
    ax2.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.4)
    for bar, val in zip(bars, mdf["med_frac"]):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("Mediation frequency")
    ax2.set_title("(B) Node mediation frequency")
    ax3 = axes[2]
    if not edge_df.empty:
        edf = edge_df.head(15).sort_values("med_frac", ascending=True)
        e_colors = [plt.cm.Blues(0.3 + 0.7 * f) for f in edf["med_frac"]]
        e_bars = ax3.barh(edf["edge"], edf["med_frac"], color=e_colors, alpha=0.9)
        ax3.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.4)
        for bar, val in zip(e_bars, edf["med_frac"]):
            ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=8)
        ax3.set_xlabel("Mediation frequency")
        ax3.set_title("(C) Edge (corridor) mediation frequency (top 15)")
    fig.suptitle(f"Transmission Mediation Maps  (total conditions={total_cond})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "T2_mediation_maps.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved T2_mediation_maps.png")


def plot_threshold_profiles(consistency_df, paths_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    thresholds = sorted(consistency_df["threshold"].unique())
    t_colors   = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    ax = axes[0, 0]
    frac_ordered = consistency_df.groupby("threshold")["hop_ordered"].mean().reindex(thresholds)
    ax.bar([f"{t:.1f}×" for t in thresholds], frac_ordered.values, color=t_colors, alpha=0.85)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
    for i, (t, v) in enumerate(zip(thresholds, frac_ordered.values)):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("Shock magnitude (× bg_std)")
    ax.set_ylabel("Fraction of sources with hop-ordered propagation")
    ax.set_title("(A) Hop-ordering consistency by threshold")
    ax.set_ylim(0, 1.1)
    ax = axes[0, 1]
    mean_tau = consistency_df.groupby("threshold")["kendall_tau"].mean()
    std_tau  = consistency_df.groupby("threshold")["kendall_tau"].std()
    ax.bar([f"{t:.1f}×" for t in thresholds], mean_tau.reindex(thresholds).values,
           yerr=std_tau.reindex(thresholds).values, color=t_colors, alpha=0.85, capsize=4)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Shock magnitude (× bg_std)")
    ax.set_ylabel("Mean Kendall τ ± std")
    ax.set_title("(B) Mean hop-ordering strength by threshold")
    ax = axes[1, 0]
    mean_active = consistency_df.groupby("threshold")["n_active"].mean()
    ax.bar([f"{t:.1f}×" for t in thresholds], mean_active.reindex(thresholds).values, color=t_colors, alpha=0.85)
    ax.set_xlabel("Shock magnitude (× bg_std)")
    ax.set_ylabel("Mean number of active carrier nodes")
    ax.set_title("(C) Propagation reach by threshold")
    ax = axes[1, 1]
    if not paths_df.empty:
        for thresh, tcolor in zip(thresholds, t_colors):
            sub = paths_df[paths_df["threshold"] == thresh]
            if sub.empty:
                continue
            by_hop = sub.groupby("hop")["peak_t"].agg(["mean", "std"])
            ax.errorbar(by_hop.index, by_hop["mean"], yerr=by_hop["std"],
                        color=tcolor, marker="o", ms=5, lw=1.2, label=f"{thresh:.1f}×", capsize=3)
    ax.set_xlabel("Hop distance from source")
    ax.set_ylabel("Peak arrival time (steps)")
    ax.set_title("(D) Arrival time by hop distance")
    ax.legend(fontsize=8, title="Threshold")
    fig.suptitle("Threshold-Conditioned Propagation Profiles", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "T3_threshold_profiles.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved T3_threshold_profiles.png")


def plot_source_comparison(consistency_df, paths_df, hop_D, nodes):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    sources = SHOCK_NODES
    source_colors = plt.cm.tab10(np.linspace(0, 0.6, len(sources)))
    ax = axes[0, 0]
    for src, col in zip(sources, source_colors):
        sub = consistency_df[consistency_df["source"] == src]
        if sub.empty:
            continue
        ax.plot(sub["threshold"], sub["kendall_tau"], color=col, marker="o", ms=5, lw=1.2, label=src)
        for _, row in sub.iterrows():
            if row["hop_ordered"]:
                ax.scatter(row["threshold"], row["kendall_tau"], s=80, color=col, zorder=5, marker="*")
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("Threshold (× bg_std)")
    ax.set_ylabel("Kendall τ")
    ax.set_title("(A) τ by source and threshold\n(★ = significant)")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[0, 1]
    frac = consistency_df.groupby("source")["hop_ordered"].mean().reindex(sources)
    ax.bar(sources, frac.values, color=source_colors[:len(sources)], alpha=0.85)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Source node")
    ax.set_ylabel("Fraction of thresholds with ordered propagation")
    ax.set_title("(B) Consistency by source node")
    for i, (src, v) in enumerate(zip(sources, frac.values)):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax = axes[0, 2]
    deg_frac = consistency_df.groupby("source").agg({"hop_ordered": "mean", "src_degree": "first"}).reset_index()
    ax.scatter(deg_frac["src_degree"], deg_frac["hop_ordered"], s=100,
               c=source_colors[:len(deg_frac)], alpha=0.85, zorder=5)
    for _, row in deg_frac.iterrows():
        ax.annotate(row["source"], (row["src_degree"], row["hop_ordered"]),
                    fontsize=8, ha="left", xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Source node degree")
    ax.set_ylabel("Fraction of conditions with ordered prop.")
    ax.set_title("(C) Node degree vs propagation consistency")
    for ax, src, col in zip(axes[1, :], sources[:3], source_colors[:3]):
        sub = paths_df[(paths_df["source"] == src) & (paths_df["threshold"] == 1.5)]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data\n{src} at 1.5×", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{src}")
            continue
        by_hop = sub.groupby("hop").agg(peak_t_mean=("peak_t","mean"), peak_t_std=("peak_t","std"), rel_mag_mean=("rel_mag","mean"))
        ax2 = ax.twinx()
        ax.errorbar(by_hop.index, by_hop["peak_t_mean"], yerr=by_hop["peak_t_std"].fillna(0),
                    color=col, marker="o", ms=5, lw=1.4, label="arrival time", capsize=3)
        ax2.bar(by_hop.index, by_hop["rel_mag_mean"], alpha=0.3, color=col, label="rel magnitude")
        ax.set_xlabel("Hop distance")
        ax.set_ylabel("Peak arrival time (steps)")
        ax2.set_ylabel("Relative magnitude")
        ax.set_title(f"(D{sources.index(src)+1}) {src}  1.5× threshold")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Source-Node Comparison of Propagation Consistency", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "T4_source_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved T4_source_comparison.png")


def write_report(consistency_df, mediation_df, edge_df, paths_df, total_cond):
    sep = "=" * 68
    lines = [sep, "  THRESHOLD-CONDITIONED PROPAGATION — FINDINGS", sep, ""]
    by_thresh = consistency_df.groupby("threshold").agg(
        frac_ordered=("hop_ordered","mean"), mean_tau=("kendall_tau","mean"),
        std_tau=("kendall_tau","std"), n_active=("n_active","mean"))
    for thresh, row in by_thresh.iterrows():
        sig = "★" if row["frac_ordered"] >= 0.5 else " "
        lines.append(f"  {thresh:.1f}× bg_std {sig}: ordered in {row['frac_ordered']:.0%}  τ={row['mean_tau']:.3f}±{row['std_tau']:.3f}  active={row['n_active']:.1f}")
    lines += ["", "Top node mediators:"]
    for _, row in mediation_df.head(8).iterrows():
        flag = "  ← persistent" if row["med_frac"] > 0.5 else ""
        lines.append(f"  {row['node']:6s}  {row['med_frac']:.2f}  (degree={row['degree']}){flag}")
    lines += ["", sep]
    report = "\n".join(lines)
    (OUT_DAT / "threshold_report.txt").write_text(report)
    print("\n" + report)
    return report


def main():
    log.info("Building graph and gradient-rich field …")
    G     = build_graph()
    nodes = node_order(G)
    field = build_gradient_rich_field(G, nodes, seed=42, T=1440)
    E1    = E1_price_spread_edge(G, field["R"])
    W_adj = adjacency_W(G, nodes)
    hop_D = node_distances(G, nodes)
    geo_D = geo_distances(G, nodes)
    bg_std = float(field["Phi"].values.std(axis=1).mean())
    log.info("  Background spatial std: %.4f", bg_std)
    log.info("  Shock nodes: %s", SHOCK_NODES)
    log.info("  Thresholds:  %s × bg_std", THRESHOLDS)
    consistency_df, mediation_df, edge_df, paths_df, total_cond = \
        run_threshold_experiment(G, nodes, field, E1, hop_D)
    plot_consistency_matrix(consistency_df)
    plot_mediation_maps(G, nodes, mediation_df, edge_df, total_cond)
    plot_threshold_profiles(consistency_df, paths_df)
    plot_source_comparison(consistency_df, paths_df, hop_D, nodes)
    write_report(consistency_df, mediation_df, edge_df, paths_df, total_cond)
    log.info("Complete.")


if __name__ == "__main__":
    main()
