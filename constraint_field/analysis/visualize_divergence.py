"""
constraint_field.analysis.visualize_divergence
================================================
Visualisation functions for the divergence analysis layer.

Produces:
  1. Time-series panel: divergence metrics vs instability
  2. Lead/lag correlation plots
  3. Regime-stratified divergence distributions
  4. Bounded-price diagnostic scatter/density plots
  5. Model comparison bar chart
  6. Full summary dashboard
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PALETTE = {
    "R":           "#F44336",
    "D1":          "#9C27B0",
    "D2":          "#2196F3",
    "D3":          "#FF9800",
    "D4":          "#009688",
    "D5_pos":      "#E91E63",
    "D5_neg":      "#3F51B5",
    "instability": "#212121",
    "BP1":         "#FF5722",
}
CLUSTER_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Divergence time-series panel
# ──────────────────────────────────────────────────────────────────────────────

def plot_divergence_timeseries(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    figsize: tuple = (16, 14),
    title: str = "Divergence Metrics vs Instability",
) -> plt.Figure:
    """
    Multi-panel time-series showing each divergence metric alongside
    the instability index.
    """
    if metrics is None:
        metrics = [c for c in ["D1","D2","D3","D4","D5_pos","D5_neg","BP2"]
                   if c in df.columns]

    sub = df.loc[start:end] if (start or end) else df
    inst = sub["instability_index"]
    inst_thresh = inst.quantile(0.85)

    n = len(metrics)
    fig, axes = plt.subplots(n + 1, 1, figsize=figsize, sharex=True)

    # Top: instability reference
    ax0 = axes[0]
    ax0.plot(sub.index, inst, color=PALETTE["instability"], lw=1.0, label="Instability")
    ax0.axhline(inst_thresh, color="red", lw=0.7, ls="--",
                label=f"85th pct ({inst_thresh:.2f})")
    ax0.fill_between(sub.index, inst_thresh, inst,
                     where=(inst > inst_thresh), alpha=0.25, color="red")
    ax0.set_ylabel("Instability")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.set_title(title)

    # Each metric
    for ax, metric in zip(axes[1:], metrics):
        col = PALETTE.get(metric, "#607D8B")
        series = sub[metric]
        ax.plot(sub.index, series, color=col, lw=0.85, alpha=0.85, label=metric)
        ax.fill_between(sub.index, 0, series, alpha=0.15, color=col)
        # Overlay instability as faint reference
        ax2 = ax.twinx()
        ax2.plot(sub.index, inst, color=PALETTE["instability"],
                 lw=0.5, alpha=0.25, ls="--")
        ax2.set_ylabel("inst.", fontsize=6, color="gray")
        ax2.tick_params(labelsize=5)
        corr = series.corr(inst)
        ax.set_ylabel(metric, fontsize=8)
        ax.legend([f"{metric}  (ρ={corr:.3f})"], fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (UTC)")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Lead/lag correlation
# ──────────────────────────────────────────────────────────────────────────────

def plot_lead_lag(
    lead_lag_dict: dict[str, pd.DataFrame],
    figsize: tuple = (12, 5),
    title: str = "Lead/Lag Correlation with Instability Index",
) -> plt.Figure:
    """
    Plot lead/lag correlation curves for multiple divergence metrics.
    Positive lag = metric LEADS instability.
    """
    colors = ["#9C27B0", "#2196F3", "#FF9800", "#F44336", "#4CAF50"]
    fig, ax = plt.subplots(figsize=figsize)

    for (metric, ll_df), col in zip(lead_lag_dict.items(), colors):
        ax.plot(ll_df["lag_hours"], ll_df["correlation"],
                color=col, lw=1.4, label=metric, marker="o", ms=2.5)
        # Shade significant lags (rough |r| > 0.1 as visual guide)
        sig = ll_df[ll_df["pvalue"] < 0.05]
        ax.scatter(sig["lag_hours"], sig["correlation"],
                   color=col, s=18, zorder=5, alpha=0.7)

    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color="k", lw=0.8, ls="-", alpha=0.3)
    ax.fill_betweenx([-1, 1], 0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 24,
                     alpha=0.05, color="green", label="Metric leads →")
    ax.set_xlabel("Lag (hours)  [positive = metric measured earlier]")
    ax.set_ylabel("Correlation with instability index")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Regime-stratified distributions
# ──────────────────────────────────────────────────────────────────────────────

def plot_regime_divergence(
    df: pd.DataFrame,
    regime_summary: pd.DataFrame,
    metric: str = "D2",
    figsize: tuple = (14, 8),
    title: str = "Divergence by Regime",
) -> plt.Figure:
    """
    Box plots and scatter of divergence metric by S-R cluster,
    coloured by mean instability.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    clusters = sorted(df["cluster"].unique())
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]

    # Box plot of divergence by cluster
    ax = axes[0]
    data = [df.loc[df["cluster"] == c, metric].dropna().values for c in clusters]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "lw": 1.5})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    labels = []
    for c in clusters:
        if c in regime_summary.index:
            lbl = regime_summary.loc[c, "regime_label"]
        else:
            lbl = f"C{c}"
        labels.append(f"C{c}\n({lbl})")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"(A) {metric} distribution by regime")

    # Bar chart: mean instability by cluster
    ax2 = axes[1]
    if not regime_summary.empty:
        clusters_sorted = regime_summary.index.tolist()
        mean_inst = regime_summary["mean_instability"].values
        frac_hi   = regime_summary["frac_high_instability"].values
        frac_bp1  = regime_summary["frac_BP1"].values

        x = np.arange(len(clusters_sorted))
        w = 0.28
        ax2.bar(x - w, mean_inst, w, label="Mean instability",   color="#F44336", alpha=0.8)
        ax2.bar(x,     frac_hi,   w, label="Frac high-instability", color="#FF9800", alpha=0.8)
        ax2.bar(x + w, frac_bp1,  w, label="Frac BP1 (bounded-price)", color="#9C27B0", alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [f"C{c}\n{regime_summary.loc[c,'regime_label']}" for c in clusters_sorted],
            fontsize=8
        )
        ax2.set_ylabel("Value")
        ax2.set_title("(B) Instability & bounded-price incidence by regime")
        ax2.legend(fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Bounded-price scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_bounded_price(
    df: pd.DataFrame,
    phi_thresh: float | None = None,
    r_mod_thresh: float | None = None,
    figsize: tuple = (14, 6),
    title: str = "Bounded-Price Diagnostic",
) -> plt.Figure:
    """
    Two-panel scatter revealing the bounded-price signature:
      Left:  |R| vs instability, coloured by |Phi|
      Right: |Phi| vs instability, coloured by |R| — annotating
             the quadrant where R is moderate but Phi is high
    """
    if phi_thresh is None:
        phi_thresh = df["D1"].quantile(0.75)
    if r_mod_thresh is None:
        r_mod_thresh = df["R"].abs().quantile(0.50)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    d1  = df["D1"].values
    r   = df["R"].abs().values
    inst = df["instability_index"].values

    # Left: R vs instability coloured by |Phi|
    ax = axes[0]
    sc = ax.scatter(r, inst, c=d1, cmap="YlOrRd", s=5, alpha=0.5,
                    vmin=0, vmax=np.percentile(d1, 95))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("|Phi| magnitude", fontsize=8)
    ax.axvline(r_mod_thresh, color="blue", lw=0.9, ls="--",
               label=f"|R| moderate threshold ({r_mod_thresh:.2f})")
    ax.set_xlabel("|R| (constraint signal magnitude)")
    ax.set_ylabel("Instability index")
    ax.set_title("(A) R vs instability\n(colour = |Phi|)")
    ax.legend(fontsize=7)

    # Right: |Phi| vs instability coloured by |R|
    ax2 = axes[1]
    sc2 = ax2.scatter(d1, inst, c=r, cmap="Blues", s=5, alpha=0.5,
                      vmin=0, vmax=np.percentile(r, 95))
    cb2 = fig.colorbar(sc2, ax=ax2)
    cb2.set_label("|R| magnitude", fontsize=8)

    # Shade the bounded-price quadrant: high |Phi|, moderate |R|
    ax2.axvline(phi_thresh, color="purple", lw=0.9, ls="--",
                label=f"|Phi| threshold ({phi_thresh:.2f})")
    ax2.set_xlabel("|Phi| = |R - S| (imbalance magnitude)")
    ax2.set_ylabel("Instability index")
    ax2.set_title("(B) |Phi| vs instability\n(colour = |R|, left region = bounded-price)")
    ax2.legend(fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    comparison_table: pd.DataFrame,
    figsize: tuple = (12, 5),
    title: str = "Model Comparison: R alone vs Divergence Metrics",
) -> plt.Figure:
    """
    Grouped bar chart of R², AUC, and F1 for all models.
    """
    tbl = comparison_table.copy()
    metrics = ["train_R²", "test_R²", "AUC", "F1"]
    metrics = [m for m in metrics if m in tbl.columns]

    x    = np.arange(len(tbl))
    w    = 0.8 / len(metrics)
    cols = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (metric, col) in enumerate(zip(metrics, cols)):
        vals = tbl[metric].fillna(0).values
        bars = ax.bar(x + i * w - 0.4 + w/2, vals, w,
                      label=metric, color=col, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tbl.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, min(1.05, tbl[metrics].max().max() * 1.25))
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Full divergence dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_divergence_dashboard(
    df: pd.DataFrame,
    model_results: dict,
    regime_summary: pd.DataFrame,
    bp_diag: dict,
    figsize: tuple = (18, 20),
    title: str = "Field Divergence Analysis — Full Dashboard",
) -> plt.Figure:
    """
    Comprehensive 5-row summary dashboard.
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.52, wspace=0.35)
    inst = df["instability_index"]
    inst_thresh = inst.quantile(0.85)

    # ── Row 0: S, R, Phi, instability time-series ────────────────────────
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(df.index, df["S"],   color="#2196F3", lw=0.8, alpha=0.7, label="S")
    ax_ts.plot(df.index, df["R"],   color="#F44336", lw=0.8, alpha=0.7, label="R")
    ax_ts.plot(df.index, df["Phi"], color="#9C27B0", lw=0.9, alpha=0.8, label="Φ")
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(df.index, inst, color="#212121", lw=0.6, alpha=0.5, ls="--")
    ax_ts2.axhline(inst_thresh, color="red", lw=0.5, ls=":", alpha=0.4)
    ax_ts2.set_ylabel("Instability", fontsize=7, color="gray")
    ax_ts2.tick_params(labelsize=6)
    ax_ts.set_title("(A) S, R, Phi and instability over time")
    ax_ts.set_ylabel("Normalised field values")
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)

    # ── Row 1: D1, D2 time-series + D5 asymmetry ────────────────────────
    ax_d1 = fig.add_subplot(gs[1, :2])
    for metric, col in [("D1", PALETTE["D1"]), ("D2", PALETTE["D2"])]:
        if metric in df.columns:
            ax_d1.plot(df.index, df[metric], color=col, lw=0.85,
                       alpha=0.8, label=metric)
    ax_d1_r = ax_d1.twinx()
    ax_d1_r.plot(df.index, inst, color="black", lw=0.5, alpha=0.2, ls="--")
    ax_d1_r.tick_params(labelsize=5)
    ax_d1.set_title("(B) D1 = |Phi|  and  D2 = |Phi|/(1+|R|)")
    ax_d1.set_ylabel("Metric value")
    ax_d1.legend(fontsize=7)

    ax_d5 = fig.add_subplot(gs[1, 2:])
    if "D5_pos" in df.columns and "D5_neg" in df.columns:
        ax_d5.fill_between(df.index, 0,  df["D5_pos"],
                           alpha=0.45, color=PALETTE["D5_pos"], label="D5+ (R>S)")
        ax_d5.fill_between(df.index, 0, -df["D5_neg"],
                           alpha=0.45, color=PALETTE["D5_neg"], label="D5− (S>R)")
    ax_d5.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_d5.set_title("(C) Signed asymmetry: D5+ vs D5−")
    ax_d5.set_ylabel("Asymmetry")
    ax_d5.legend(fontsize=7)

    # ── Row 2: Scatter grids — |Phi| vs inst, R vs inst ──────────────────
    ax_sc1 = fig.add_subplot(gs[2, :2])
    sc1 = ax_sc1.scatter(df["D1"], inst,
                         c=df["R"].abs(), cmap="RdYlBu_r", s=4, alpha=0.4,
                         vmin=0, vmax=df["R"].abs().quantile(0.95))
    cb1 = fig.colorbar(sc1, ax=ax_sc1, shrink=0.8)
    cb1.set_label("|R|", fontsize=7)
    ax_sc1.set_xlabel("|Phi|")
    ax_sc1.set_ylabel("Instability")
    ax_sc1.set_title("(D) |Phi| vs instability (colour=|R|)")

    ax_sc2 = fig.add_subplot(gs[2, 2:])
    sc2 = ax_sc2.scatter(df["R"].abs(), inst,
                         c=df["D1"], cmap="YlOrRd", s=4, alpha=0.4,
                         vmin=0, vmax=df["D1"].quantile(0.95))
    cb2 = fig.colorbar(sc2, ax=ax_sc2, shrink=0.8)
    cb2.set_label("|Phi|", fontsize=7)
    ax_sc2.set_xlabel("|R|")
    ax_sc2.set_ylabel("Instability")
    ax_sc2.set_title("(E) |R| vs instability (colour=|Phi|)")

    # ── Row 3: Regime bars + lead-lag ────────────────────────────────────
    ax_reg = fig.add_subplot(gs[3, :2])
    if not regime_summary.empty:
        clusters = regime_summary.index.tolist()
        x = np.arange(len(clusters))
        w = 0.35
        ax_reg.bar(x - w/2, regime_summary["mean_instability"], w,
                   color="#F44336", alpha=0.8, label="Mean instability")
        ax_reg.bar(x + w/2, regime_summary["frac_BP1"], w,
                   color="#9C27B0", alpha=0.8, label="Frac bounded-price (BP1)")
        ax_reg.set_xticks(x)
        ax_reg.set_xticklabels(
            [f"C{c}\n{regime_summary.loc[c,'regime_label']}"
             for c in clusters], fontsize=7
        )
        ax_reg.set_title("(F) Mean instability & bounded-price by regime")
        ax_reg.legend(fontsize=7)

    ax_ll = fig.add_subplot(gs[3, 2:])
    ll_data = model_results.get("lead_lag", {})
    colors_ll = ["#9C27B0", "#2196F3", "#FF9800"]
    for (metric, ll_df), col in zip(ll_data.items(), colors_ll):
        ax_ll.plot(ll_df["lag_hours"], ll_df["correlation"],
                   color=col, lw=1.2, label=metric, marker="o", ms=2)
    ax_ll.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_ll.axvline(0, color="k", lw=0.7, ls="-", alpha=0.25)
    ax_ll.set_xlabel("Lag hours (positive = metric leads)")
    ax_ll.set_ylabel("Correlation")
    ax_ll.set_title("(G) Lead/lag correlation with instability")
    ax_ll.legend(fontsize=7)

    # ── Row 4: Model comparison table ────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[4, :])
    ax_tbl.axis("off")
    tbl = model_results.get("comparison_table", pd.DataFrame())
    if not tbl.empty:
        def fmt(x):
            if isinstance(x, float):
                return f"{x:.4f}" if not np.isnan(x) else "—"
            return str(x)
        cell_text = [[fmt(v) for v in row] for row in tbl.reset_index().values]
        col_labels = ["Model"] + tbl.columns.tolist()
        t = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                         loc="center", cellLoc="center")
        t.scale(1, 1.6)
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        # Highlight best test R²
        best_r2_row = tbl["test_R²"].fillna(-1).argmax() + 1
        for j in range(len(col_labels)):
            t[best_r2_row, j].set_facecolor("#d4edda")
    ax_tbl.set_title("(H) Model comparison: R alone vs divergence metrics",
                     fontsize=9, pad=12)

    fig.suptitle(title, fontsize=13, y=1.005)
    return fig
