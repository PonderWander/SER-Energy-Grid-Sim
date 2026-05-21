"""
experiments/cross_corridor/visualize.py
==========================================
Six required figures for the cross-corridor routing experiment.

Figure 1: Loading sweep vs leakage ratio, faceted by E regime
Figure 2: Cross-corridor Φ correlation over time per loading level
Figure 3: Connector-node activation timing heatmap
Figure 4: Path activation order / early edge flux ranking
Figure 5: Spatial covariance matrix with corridor blocks annotated
Figure 6: Comparison of routing outcomes: real vs shuffled vs inverted vs uniform E
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd

from .config import (
    SW_CORRIDOR, NW_CORRIDOR, CONNECTOR_NODES,
    E_REGIMES, LOADING_SIGMA, LOADING_VARIANTS,
    OUT_FIG, DPI,
)

log = logging.getLogger("cc_viz")

REGIME_COLORS = {
    "uniform":       "#9E9E9E",
    "calibrated_E1": "#2196F3",
    "state_dep":     "#4CAF50",
    "shuffled":      "#FF9800",
    "inverted":      "#9C27B0",
}
REGIME_LINES = {
    "uniform": "--", "calibrated_E1": "-",
    "state_dep": "-.", "shuffled": ":", "inverted": "-",
}
REGIME_LABELS = {k: v for k, v in E_REGIMES.items()}

VARIANT_MARKERS = {
    "symmetric": "o", "SW_heavy": "s", "NW_heavy": "^", "diffuse": "D"
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Leakage ratio vs loading, faceted by E regime
# ─────────────────────────────────────────────────────────────────────────────

def fig1_leakage(df: pd.DataFrame):
    """
    Loading sweep vs leakage ratio, one subplot per E regime,
    lines coloured by loading variant.
    """
    regimes  = [r for r in E_REGIMES if r in df["regime"].unique()]
    variants = [v for v in LOADING_VARIANTS if v in df["loading_variant"].unique()]
    n_reg    = len(regimes)
    fig, axes = plt.subplots(1, n_reg, figsize=(5 * n_reg, 5), sharey=True)
    if n_reg == 1:
        axes = [axes]

    var_colors = plt.cm.tab10(np.linspace(0, 0.5, len(variants)))

    for ax, regime in zip(axes, regimes):
        for var, vc in zip(variants, var_colors):
            sub = df[(df["regime"] == regime) & (df["loading_variant"] == var)
                     ].sort_values("loading")
            if sub.empty:
                continue
            ax.plot(sub["loading"], sub["leakage_sw_to_nw"],
                    color=vc, ls="-", lw=1.6, marker=VARIANT_MARKERS.get(var,"o"), ms=5,
                    label=f"{var} SW→NW")
            ax.plot(sub["loading"], sub["leakage_nw_to_sw"],
                    color=vc, ls="--", lw=1.2, marker=VARIANT_MARKERS.get(var,"o"), ms=4,
                    alpha=0.7, label=f"{var} NW→SW")

        ax.set_xlabel("Loading (× σ)")
        ax.set_ylabel("Leakage ratio" if regime == regimes[0] else "")
        ax.set_title(REGIME_LABELS.get(regime, regime), fontsize=8)
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        if regime == regimes[0]:
            ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Figure 1: Cross-Corridor Leakage Ratio vs Loading\n"
                 "(solid = SW→NW, dashed = NW→SW)", fontsize=11)
    fig.tight_layout()
    fpath = OUT_FIG / "CC_fig1_leakage.png"
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("→ Saved %s", fpath)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Cross-corridor Φ correlation over time
# ─────────────────────────────────────────────────────────────────────────────

def fig2_cross_corr_time(
    results: dict,   # {(regime, loading, variant): sim_result}
    target_variant: str = "symmetric",
):
    """
    Cross-corridor mean-Φ time-series per loading level, one subplot per regime.
    Separate panels for SW and NW mean Φ, overlaid across loading levels.
    """
    regimes  = list(E_REGIMES.keys())
    loadings = LOADING_SIGMA
    n_reg    = len([r for r in regimes if any((r, l, target_variant) in results for l in loadings)])
    if n_reg == 0:
        log.warning("fig2: no results for variant %s", target_variant)
        return

    active_regimes = [r for r in regimes if any((r, l, target_variant) in results for l in loadings)]
    fig, axes = plt.subplots(2, len(active_regimes), figsize=(5 * len(active_regimes), 8))
    if len(active_regimes) == 1:
        axes = axes.reshape(2, 1)

    load_cmap = plt.cm.plasma
    load_norms = mcolors.Normalize(min(loadings), max(loadings))

    for col, regime in enumerate(active_regimes):
        ax_sw = axes[0, col]
        ax_nw = axes[1, col]
        for loading in loadings:
            key = (regime, loading, target_variant)
            if key not in results:
                continue
            res   = results[key]
            col_c = load_cmap(load_norms(loading))
            sw_phi = res["metrics"]["cc"]["sw_phi"]
            nw_phi = res["metrics"]["cc"]["nw_phi"]
            T      = len(sw_phi)
            ax_sw.plot(range(T), sw_phi, color=col_c, lw=1.2, alpha=0.85,
                       label=f"{loading:.1f}σ")
            ax_nw.plot(range(T), nw_phi, color=col_c, lw=1.2, alpha=0.85)

        ax_sw.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        ax_nw.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        ax_sw.set_title(REGIME_LABELS.get(regime, regime)[:22], fontsize=8)
        ax_sw.set_ylabel("Mean Φ (SW)" if col == 0 else "")
        ax_nw.set_ylabel("Mean Φ (NW)" if col == 0 else "")
        ax_nw.set_xlabel("Step")
        if col == 0:
            ax_sw.legend(fontsize=6, title="Load", ncol=2)

    sm = plt.cm.ScalarMappable(cmap=load_cmap, norm=load_norms)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:, -1], label="Loading (× σ)", shrink=0.6)
    fig.suptitle(f"Figure 2: SW and NW Mean Φ Over Time  ({target_variant} loading)",
                 fontsize=11)
    fig.tight_layout()
    fpath = OUT_FIG / "CC_fig2_crosscorr_time.png"
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("→ Saved %s", fpath)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Connector-node activation timing heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig3_connector_heatmap(df: pd.DataFrame):
    """
    Heatmap of connector-node first-activation time.
    Rows = (regime × variant), Columns = loading level.
    Separate heatmaps for each connector node.
    """
    conn_nodes  = CONNECTOR_NODES
    regimes     = [r for r in E_REGIMES if r in df["regime"].unique()]
    variants    = [v for v in LOADING_VARIANTS if v in df["loading_variant"].unique()]
    loadings    = sorted(df["loading"].unique())
    row_labels  = [f"{r[:12]}\n{v[:8]}" for r in regimes for v in variants]
    n_rows      = len(regimes) * len(variants)

    fig, axes = plt.subplots(1, len(conn_nodes), figsize=(4 * len(conn_nodes), max(4, n_rows * 0.5 + 2)))
    if len(conn_nodes) == 1:
        axes = [axes]

    for ax, nd in zip(axes, conn_nodes):
        col_key = f"conn_t_{nd}"
        if col_key not in df.columns:
            ax.set_title(f"{nd} (no data)")
            continue

        mat = np.full((n_rows, len(loadings)), np.nan)
        for ri, regime in enumerate(regimes):
            for vi, variant in enumerate(variants):
                row_i = ri * len(variants) + vi
                for ci, load in enumerate(loadings):
                    sub = df[(df["regime"] == regime) & (df["loading_variant"] == variant)
                             & (df["loading"] == load)]
                    if not sub.empty:
                        val = sub.iloc[0][col_key]
                        mat[row_i, ci] = val if not pd.isna(val) else np.nan

        vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1
        im   = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r",
                          vmin=0, vmax=vmax)
        ax.set_xticks(range(len(loadings)))
        ax.set_xticklabels([f"{l:.1f}σ" for l in loadings], fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_title(f"{nd}\n(first activation step)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Step")

        # Annotate cells
        for ri in range(n_rows):
            for ci in range(len(loadings)):
                val = mat[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f"{int(val)}", ha="center", va="center",
                            fontsize=6, color="black" if val < vmax * 0.6 else "white")

    fig.suptitle("Figure 3: Connector-Node First-Activation Time Heatmap\n"
                 "(lower = activates earlier)", fontsize=11)
    fig.tight_layout()
    fpath = OUT_FIG / "CC_fig3_connector_heatmap.png"
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("→ Saved %s", fpath)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Path activation order / edge flux ranking
# ─────────────────────────────────────────────────────────────────────────────

def fig4_path_activation(
    results: dict,
    loading: float = 2.5,
    variant: str = "symmetric",
):
    """
    Bar charts of top-ranked edges by cumulative flux under each E regime,
    plus a Kendall τ comparison showing how well E-based routing cost predicts
    observed flux ranking.
    """
    active_regimes = [r for r in E_REGIMES if (r, loading, variant) in results]
    if not active_regimes:
        log.warning("fig4: no results at loading=%.1f variant=%s", loading, variant)
        return

    n_reg = len(active_regimes)
    fig, axes = plt.subplots(2, n_reg, figsize=(5 * n_reg, 10))
    if n_reg == 1:
        axes = axes.reshape(2, 1)

    for col, regime in enumerate(active_regimes):
        res  = results[(regime, loading, variant)]
        path = res["metrics"]["path"]
        col_c = REGIME_COLORS.get(regime, "#607D8B")

        # Top edges by cumulative flux
        ax = axes[0, col]
        top_edges = path["edge_ranking_flux"][:10]
        cum_flux  = [path["edge_flux_cum"].get(ek, 0) for ek in top_edges]
        ax.barh(range(len(top_edges)), cum_flux[::-1],
                color=col_c, alpha=0.85)
        ax.set_yticks(range(len(top_edges)))
        ax.set_yticklabels([e.replace("_", "→") for e in top_edges[::-1]], fontsize=7)
        ax.set_xlabel("Cumulative flux")
        ax.set_title(f"{REGIME_LABELS.get(regime, regime)[:20]}\nTop edges by flux", fontsize=8)

        # Early activation: first-activated vs routing cost
        ax2 = axes[1, col]
        early_edges = path["edge_ranking_time"][:10]
        first_ts    = [path["edge_first_t"].get(ek, 999) for ek in early_edges]
        costs       = [path["edge_cost"].get(ek, 1) for ek in early_edges]
        sc = ax2.scatter(costs[::-1], first_ts[::-1],
                         c=range(len(early_edges)), cmap="viridis",
                         s=80, alpha=0.85, zorder=5)
        for i, ek in enumerate(early_edges[::-1]):
            ax2.annotate(ek.replace("_","→"), (costs[len(early_edges)-1-i],
                          first_ts[len(early_edges)-1-i]),
                         fontsize=6, ha="left", xytext=(3,2), textcoords="offset points")
        tau  = path.get("rank_correlation_tau", np.nan)
        pval = path.get("rank_correlation_pval", np.nan)
        tau_str = f"τ={tau:.3f} p={pval:.3f}" if not np.isnan(tau) else "τ=n/a"
        ax2.set_xlabel("Routing cost (1/E·cap)")
        ax2.set_ylabel("First activation step")
        ax2.set_title(f"Cost vs activation time\n{tau_str}", fontsize=8)

    fig.suptitle(f"Figure 4: Path Activation Order  (loading={loading:.1f}σ, {variant})\n"
                 "Top: flux ranking | Bottom: routing cost vs activation timing", fontsize=11)
    fig.tight_layout()
    fpath = OUT_FIG / "CC_fig4_path_activation.png"
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("→ Saved %s", fpath)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Spatial covariance matrix
# ─────────────────────────────────────────────────────────────────────────────

def fig5_covariance(
    results: dict,
    nodes: list[str],
    loading: float = 2.5,
    variant: str = "symmetric",
):
    """
    Spatial covariance matrix of Φ with SW/NW corridor blocks annotated,
    one panel per E regime.
    """
    active = [r for r in E_REGIMES if (r, loading, variant) in results]
    if not active:
        log.warning("fig5: no results at loading=%.1f variant=%s", loading, variant)
        return

    n_reg = len(active)
    fig, axes = plt.subplots(1, n_reg, figsize=(5 * n_reg, 5))
    if n_reg == 1:
        axes = [axes]

    sw_nodes = [nd for nd in SW_CORRIDOR["nodes"] if nd in nodes]
    nw_nodes = [nd for nd in NW_CORRIDOR["nodes"] if nd in nodes and nd not in SW_CORRIDOR["nodes"]]
    order    = sw_nodes + nw_nodes + [nd for nd in nodes if nd not in sw_nodes and nd not in nw_nodes]
    order    = [nd for nd in order if nd in nodes]   # safety
    order_idx = [nodes.index(nd) for nd in order]

    for ax, regime in zip(axes, active):
        res = results[(regime, loading, variant)]
        cov = res["metrics"]["cov"]["cov_matrix"]
        cov_reord = cov[np.ix_(order_idx, order_idx)]
        vmax = max(abs(cov_reord).max(), 1e-6)
        im   = ax.imshow(cov_reord, aspect="auto", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=6, rotation=90)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Annotate corridor blocks
        n_sw = len(sw_nodes)
        n_nw = len(nw_nodes)
        ax.axvline(n_sw - 0.5, color="k", lw=1.2, ls="--", alpha=0.6)
        ax.axhline(n_sw - 0.5, color="k", lw=1.2, ls="--", alpha=0.6)
        ax.axvline(n_sw + n_nw - 0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax.axhline(n_sw + n_nw - 0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax.text(n_sw / 2, -1.2, "SW", ha="center", va="bottom", fontsize=8, color="#F44336")
        ax.text(n_sw + n_nw / 2, -1.2, "NW", ha="center", va="bottom", fontsize=8, color="#2196F3")

        cross = res["metrics"]["cov"]["cross_corridor_ratio"]
        ax.set_title(f"{REGIME_LABELS.get(regime, regime)[:22]}\n"
                     f"cross-corridor ratio={cross:.3f}", fontsize=8)

    fig.suptitle(f"Figure 5: Spatial Covariance Matrix  (loading={loading:.1f}σ, {variant})\n"
                 "(nodes reordered: SW | NW | other; dashed line = block boundary)", fontsize=11)
    fig.tight_layout()
    fpath = OUT_FIG / "CC_fig5_covariance.png"
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("→ Saved %s", fpath)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Routing comparison — real vs shuffled vs inverted vs uniform
# ─────────────────────────────────────────────────────────────────────────────

def fig6_routing_comparison(df: pd.DataFrame, results: dict, nodes: list[str], G):
    """
    Four-panel comparison of key routing metrics across E regimes.

    Panel A: Leakage ratio vs loading (all regimes, symmetric variant)
    Panel B: Cross-corridor covariance ratio vs loading
    Panel C: Moran's I vs loading
    Panel D: Top-flux edge identity by regime at high loading (network map)
    """
    regimes  = [r for r in E_REGIMES if r in df["regime"].unique()]
    high_load = 2.5
    sym_var   = "symmetric"

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # A: Leakage ratio
    ax = fig.add_subplot(gs[0, 0])
    for regime in regimes:
        sub = df[(df["regime"] == regime) & (df["loading_variant"] == sym_var)
                 ].sort_values("loading")
        if sub.empty:
            continue
        col_c = REGIME_COLORS.get(regime, "#607D8B")
        ax.plot(sub["loading"], sub["leakage_sw_to_nw"],
                color=col_c, ls=REGIME_LINES.get(regime, "-"), lw=1.6, marker="o", ms=5,
                label=REGIME_LABELS.get(regime, regime)[:18])
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("SW→NW leakage ratio")
    ax.set_title("(A) Leakage ratio: SW→NW\n(symmetric loading)")
    ax.legend(fontsize=7); ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)

    # B: Cross-corridor covariance ratio
    ax = fig.add_subplot(gs[0, 1])
    for regime in regimes:
        sub = df[(df["regime"] == regime) & (df["loading_variant"] == sym_var)
                 ].sort_values("loading")
        if sub.empty:
            continue
        col_c = REGIME_COLORS.get(regime, "#607D8B")
        ax.plot(sub["loading"], sub["cross_ratio"],
                color=col_c, ls=REGIME_LINES.get(regime, "-"), lw=1.6, marker="s", ms=5,
                label=REGIME_LABELS.get(regime, regime)[:18])
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Cross-corridor covariance ratio")
    ax.set_title("(B) SW↔NW covariance ratio\n(higher = more correlated corridors)")
    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3); ax.legend(fontsize=7)

    # C: Moran's I
    ax = fig.add_subplot(gs[1, 0])
    for regime in regimes:
        sub = df[(df["regime"] == regime) & (df["loading_variant"] == sym_var)
                 ].sort_values("loading")
        if sub.empty:
            continue
        col_c = REGIME_COLORS.get(regime, "#607D8B")
        ax.plot(sub["loading"], sub["moran_mean"],
                color=col_c, ls=REGIME_LINES.get(regime, "-"), lw=1.6, marker="^", ms=5,
                label=REGIME_LABELS.get(regime, regime)[:18])
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Moran's I (mean)")
    ax.set_title("(C) Spatial autocorrelation (Moran's I)\nvs loading")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4); ax.legend(fontsize=7)

    # D: Network map — edge flux at high loading, calibrated_E1 vs shuffled
    ax = fig.add_subplot(gs[1, 1])
    pos = {nd: (G.nodes[nd]["lon"], G.nodes[nd]["lat"]) for nd in G.nodes()}

    for regime, alpha_val in [("calibrated_E1", 0.9), ("shuffled", 0.5)]:
        key = (regime, high_load, sym_var)
        if key not in results:
            continue
        res      = results[key]
        traj     = res["traj"]
        phi_snap = traj[min(20, len(traj)-1)]
        n        = len(nodes)
        n_idx    = {nd: i for i, nd in enumerate(nodes)}
        max_phi  = max(abs(phi_snap).max(), 0.5)

        path_res  = res["metrics"]["path"]
        edge_flux = path_res.get("edge_flux_cum", {})
        max_flux  = max(edge_flux.values(), default=1e-9)

        for u, v in G.edges():
            fwd = f"{u}_{v}"; rev = f"{v}_{u}"
            flux = edge_flux.get(fwd, edge_flux.get(rev, 0))
            lw   = 0.4 + 4.0 * (flux / max(max_flux, 1e-9))
            col  = REGIME_COLORS.get(regime, "#607D8B")
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=col, lw=lw, alpha=alpha_val * 0.8, zorder=3)

    # Node markers for calibrated_E1
    key_cal = ("calibrated_E1", high_load, sym_var)
    if key_cal in results:
        phi_snap = results[key_cal]["traj"][min(20, len(results[key_cal]["traj"])-1)]
        for nd in nodes:
            ni  = nodes.index(nd)
            val = phi_snap[ni]
            col = "#F44336" if val > 0 else "#2196F3"
            ax.scatter(pos[nd][0], pos[nd][1], s=60 + 300*abs(val),
                       c=col, alpha=0.8, zorder=5)
        for nd in nodes:
            ax.annotate(nd, pos[nd], fontsize=6, ha="center",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"(D) Edge flux at {high_load:.1f}σ symmetric\n"
                 "blue=calibrated E1 (thick=high flux), orange=shuffled")

    fig.suptitle("Figure 6: Routing Outcomes — Real vs Shuffled vs Inverted vs Uniform E",
                 fontsize=12)
    fig.savefig(OUT_FIG / "CC_fig6_routing_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("→ Saved CC_fig6_routing_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Master caller
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(df: pd.DataFrame, results: dict, nodes: list[str], G):
    log.info("Generating figures 1–6 …")
    fig1_leakage(df)
    fig2_cross_corr_time(results)
    fig3_connector_heatmap(df)
    fig4_path_activation(results)
    fig5_covariance(results, nodes)
    fig6_routing_comparison(df, results, nodes, G)
    log.info("All figures complete.")
