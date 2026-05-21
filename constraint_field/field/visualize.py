"""
constraint_field.field.visualize
==================================
Visualisation functions for the static field layer (S, R, Φ, Ψ, clusters).

All functions accept a panel DataFrame and return a matplotlib Figure,
so they can be saved, embedded in notebooks, or displayed interactively.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Colour palette consistent across plots
PALETTE = {
    "S":       "#2196F3",   # blue
    "R":       "#F44336",   # red
    "Phi":     "#9C27B0",   # purple  (imbalance)
    "Psi":     "#FF9800",   # orange  (intensity)
    "E":       "#4CAF50",   # green   (fluidity)
    "neutral": "#607D8B",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Time-series overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_field_timeseries(
    panel: pd.DataFrame,
    title: str = "Static Constraint Field – S and R",
    start: str | None = None,
    end: str | None = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot S and R time-series with shaded imbalance band Φ = R − S.

    Parameters
    ----------
    panel : pd.DataFrame  with columns S, R (and optionally Phi)
    start, end : str | None  – optional slice for zoom
    """
    df = panel.copy()
    if start or end:
        df = df.loc[start:end]
    if "Phi" not in df.columns:
        df["Phi"] = df["R"] - df["S"]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.plot(df.index, df["S"], color=PALETTE["S"],  lw=1.2, label="S (load pressure)")
    ax.plot(df.index, df["R"], color=PALETTE["R"],  lw=1.2, label="R (constraint signal)")
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.fill_between(df.index, df["S"], df["R"],
                    where=(df["R"] > df["S"]),
                    alpha=0.15, color=PALETTE["R"], label="Positive Φ (over-constraint)")
    ax.fill_between(df.index, df["S"], df["R"],
                    where=(df["R"] <= df["S"]),
                    alpha=0.10, color=PALETTE["S"], label="Negative Φ (fluidity)")
    ax.set_ylabel("Normalised field value")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title)

    ax2 = axes[1]
    ax2.plot(df.index, df["Phi"], color=PALETTE["Phi"], lw=1.0, label="Φ = R − S")
    ax2.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax2.fill_between(df.index, 0, df["Phi"],
                     where=(df["Phi"] > 0), alpha=0.3, color=PALETTE["R"])
    ax2.fill_between(df.index, 0, df["Phi"],
                     where=(df["Phi"] <= 0), alpha=0.2, color=PALETTE["S"])
    ax2.set_ylabel("Φ (imbalance)")
    ax2.set_xlabel("Time (UTC)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Phase-space portrait (S–R plane)
# ──────────────────────────────────────────────────────────────────────────────

def plot_phase_portrait(
    panel: pd.DataFrame,
    color_by: str = "time",   # "time" | "cluster" | "Psi" | "instability_index"
    title: str = "S–R Phase Portrait",
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    Scatter plot in the S–R plane.

    Points can be coloured by time (trajectory), cluster label,
    field intensity Ψ, or instability index.
    """
    df = panel.dropna(subset=["S", "R"]).copy()

    fig, ax = plt.subplots(figsize=figsize)

    if color_by == "time":
        c = np.linspace(0, 1, len(df))
        sc = ax.scatter(df["S"], df["R"], c=c, cmap="plasma", s=6, alpha=0.6)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Time (normalised)")

    elif color_by == "cluster" and "cluster" in df.columns:
        clusters = df["cluster"].unique()
        cmap = plt.cm.get_cmap("tab10", len(clusters))
        for i, cl in enumerate(sorted(clusters)):
            mask = df["cluster"] == cl
            ax.scatter(df.loc[mask, "S"], df.loc[mask, "R"],
                       s=8, alpha=0.6, color=cmap(i), label=f"Cluster {cl}")
        ax.legend(fontsize=8)

    elif color_by in df.columns:
        sc = ax.scatter(df["S"], df["R"], c=df[color_by],
                        cmap="YlOrRd", s=6, alpha=0.6)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(color_by)

    else:
        ax.scatter(df["S"], df["R"], s=6, alpha=0.4, color=PALETTE["neutral"])

    # Quadrant lines
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.4)

    # Quadrant labels
    ax.text(0.95, 0.95, "Demand+Constraint\nstress",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="gray")
    ax.text(0.05, 0.95, "Supply\nconstraint",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            color="gray")
    ax.text(0.05, 0.05, "Slack",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            color="gray")
    ax.text(0.95, 0.05, "Demand\ngrowth",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="gray")

    ax.set_xlabel("S (load pressure)")
    ax.set_ylabel("R (constraint signal)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Field gradient heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_gradient_heatmap(
    panel: pd.DataFrame,
    figsize: tuple = (14, 5),
    title: str = "Field Gradients",
) -> plt.Figure:
    """
    2D heatmap of ∂S/∂t and ∂R/∂t over time, organised by
    hour-of-day × day (revealing diurnal structure).
    """
    df = panel.copy()
    for col in ["dS_dt", "dR_dt"]:
        if col not in df.columns:
            df[col] = df["S"].diff() if "S" in col else df["R"].diff()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, col, label, cmap in zip(
        axes,
        ["dS_dt", "dR_dt"],
        ["∂S/∂t", "∂R/∂t"],
        ["RdBu_r", "PuOr_r"],
    ):
        df_plot = df[[col]].copy()
        df_plot["hour"] = df_plot.index.hour
        df_plot["date"] = df_plot.index.date

        pivot = df_plot.pivot_table(index="hour", columns="date", values=col, aggfunc="mean")
        vmax = np.nanpercentile(np.abs(pivot.values), 95)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax, vmax=vmax,
            origin="upper",
        )
        ax.set_xlabel("Day index")
        ax.set_ylabel("Hour of day")
        ax.set_title(f"{label}")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Instability / stress timeline
# ──────────────────────────────────────────────────────────────────────────────

def plot_instability(
    panel: pd.DataFrame,
    figsize: tuple = (14, 4),
    title: str = "Field Instability Index",
) -> plt.Figure:
    """
    Plot rolling instability index with high-stress threshold shading.
    """
    df = panel.copy()
    if "instability_index" not in df.columns:
        from .analysis import rolling_instability
        df = df.join(rolling_instability(df))

    threshold = df["instability_index"].quantile(0.85)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df["instability_index"], color=PALETTE["Phi"], lw=1.0,
            label="Instability index")
    ax.axhline(threshold, color="red", lw=0.8, ls="--",
               label=f"85th percentile ({threshold:.2f})")
    ax.fill_between(df.index, threshold, df["instability_index"],
                    where=(df["instability_index"] > threshold),
                    alpha=0.3, color="red", label="High-stress periods")
    ax.set_ylabel("Instability |Φ| rolling 75th-pct")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Combined static dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_static_dashboard(
    panel: pd.DataFrame,
    title: str = "Static Constraint Field – Dashboard",
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    4-panel summary dashboard for the static field layer.
    """
    from .analysis import compute_field_indicators, rolling_instability

    df = panel.copy()
    if "Phi" not in df.columns:
        df = compute_field_indicators(df)
    if "instability_index" not in df.columns:
        df = df.join(rolling_instability(df))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Panel A: S and R time-series ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(df.index, df["S"], color=PALETTE["S"], lw=1.1, label="S")
    ax_a.plot(df.index, df["R"], color=PALETTE["R"], lw=1.1, label="R")
    ax_a.fill_between(df.index, df["S"], df["R"],
                      where=(df["R"] > df["S"]), alpha=0.15, color=PALETTE["R"])
    ax_a.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_a.set_title("(A) Field Variables S and R over Time")
    ax_a.set_ylabel("Normalised value")
    ax_a.legend(fontsize=8)

    # ── Panel B: Phase portrait ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    if "cluster" in df.columns:
        clusters = df["cluster"].unique()
        cmap = plt.cm.get_cmap("tab10", max(len(clusters), 1))
        for i, cl in enumerate(sorted(clusters)):
            mask = df["cluster"] == cl
            ax_b.scatter(df.loc[mask, "S"], df.loc[mask, "R"],
                         s=5, alpha=0.5, color=cmap(i), label=f"C{cl}")
        ax_b.legend(fontsize=7, markerscale=2)
    else:
        c = np.linspace(0, 1, len(df))
        ax_b.scatter(df["S"], df["R"], c=c, cmap="plasma", s=5, alpha=0.5)
    ax_b.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_b.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_b.set_xlabel("S"); ax_b.set_ylabel("R")
    ax_b.set_title("(B) S–R Phase Portrait")

    # ── Panel C: Imbalance Φ ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(df.index, df["Phi"], color=PALETTE["Phi"], lw=0.8)
    ax_c.fill_between(df.index, 0, df["Phi"],
                      where=(df["Phi"] > 0), alpha=0.3, color=PALETTE["R"])
    ax_c.fill_between(df.index, 0, df["Phi"],
                      where=(df["Phi"] <= 0), alpha=0.2, color=PALETTE["S"])
    ax_c.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_c.set_title("(C) Field Imbalance Φ = R − S")
    ax_c.set_ylabel("Φ")

    # ── Panel D: Instability ────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, :])
    inst = df["instability_index"]
    threshold = inst.quantile(0.85)
    ax_d.plot(df.index, inst, color=PALETTE["Phi"], lw=0.9, label="Instability")
    ax_d.axhline(threshold, color="red", lw=0.7, ls="--",
                 label=f"85th pct = {threshold:.2f}")
    ax_d.fill_between(df.index, threshold, inst,
                      where=(inst > threshold), alpha=0.3, color="red")
    ax_d.set_title("(D) Rolling Instability Index")
    ax_d.set_ylabel("|Φ| 75th pct (24h)")
    ax_d.set_xlabel("Time (UTC)")
    ax_d.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.01)
    return fig
