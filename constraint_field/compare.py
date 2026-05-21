"""
constraint_field.compare
=========================
Comparative analysis framework: reduced static field (S, R only)
vs. dynamic upgraded field (S, R, E).

This module provides:
  1. Quantitative metrics for comparing propagation behaviour
  2. Side-by-side visualisations
  3. A summary report function

Comparison dimensions
---------------------
- Tracking accuracy (RMSE of simulated vs observed S)
- Shock propagation speed and decay
- Congestion persistence (time above instability threshold)
- Recovery timing after high-stress events
- Residual autocorrelation (unexplained structure)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

log = logging.getLogger(__name__)

PALETTE = {
    "reduced":  "#607D8B",  # grey-blue
    "upgraded": "#4CAF50",  # green
    "observed": "#212121",  # near-black
    "E":        "#FF9800",  # orange
    "shock":    "#F44336",  # red
}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_comparison_metrics(
    reduced: pd.DataFrame,
    upgraded: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute side-by-side quantitative metrics for reduced vs upgraded runs.

    Parameters
    ----------
    reduced  : result dict from Simulator.run(use_E=False)
    upgraded : result dict from Simulator.run(use_E=True)

    Returns
    -------
    pd.DataFrame with one row per model variant.
    """

    def _metrics(df: pd.DataFrame, label: str) -> dict:
        resid = df.get("residual", df.get("residual_noE", pd.Series(dtype=float)))
        S_sim = df.get("S_sim",   df.get("S_sim_noE",   pd.Series(dtype=float)))
        S_obs = df["S_obs"]

        rmse = np.sqrt((resid**2).mean())
        mae  = resid.abs().mean()

        # Autocorrelation of residuals at lag 1 (unexplained serial structure)
        r_ac1, _ = stats.pearsonr(resid.iloc[1:], resid.iloc[:-1])

        # Fraction of time |residual| > 1σ of S_obs
        sigma_S = S_obs.std()
        frac_large_resid = (resid.abs() > sigma_S).mean()

        return {
            "model":            label,
            "rmse":             rmse,
            "mae":              mae,
            "residual_ac1":     r_ac1,
            "frac_large_resid": frac_large_resid,
            "max_residual":     resid.abs().max(),
            "n_obs":            len(df),
        }

    rows = [
        _metrics(reduced,  "reduced (no E)"),
        _metrics(upgraded, "upgraded (with E)"),
    ]
    return pd.DataFrame(rows).set_index("model")


def shock_recovery_analysis(
    reduced: pd.DataFrame,
    upgraded: pd.DataFrame,
    shock_t: int,
    recovery_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Compare recovery timing after a shock event.

    "Recovery" = |S_sim − S_obs| drops below recovery_threshold for 3+ steps.

    Parameters
    ----------
    shock_t : int
        Time index of shock onset (relative to simulation start).
    recovery_threshold : float
        Residual magnitude below which recovery is declared.

    Returns
    -------
    pd.DataFrame with recovery statistics.
    """
    records = []
    for label, df in [("reduced", reduced), ("upgraded", upgraded)]:
        col = "residual" if "residual" in df.columns else "residual_noE"
        resid = df[col].abs().values

        # Find recovery point after shock
        post_shock = resid[shock_t:]
        recovery_t = None
        for i in range(len(post_shock) - 2):
            if all(post_shock[i:i+3] < recovery_threshold):
                recovery_t = shock_t + i
                break

        records.append({
            "model":            label,
            "peak_residual":    resid[shock_t:shock_t+12].max(),
            "recovery_hours":   recovery_t - shock_t if recovery_t else np.nan,
            "mean_post_resid":  post_shock[:24].mean(),
        })

    return pd.DataFrame(records).set_index("model")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────

def plot_simulation_comparison(
    reduced: pd.DataFrame,
    upgraded: pd.DataFrame,
    E_series: pd.Series | None = None,
    title: str = "Reduced vs Upgraded Simulation",
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Side-by-side plot comparing reduced and upgraded simulation trajectories.

    Panels:
      A – S_sim trajectories vs S_obs
      B – Residuals for both models
      C – E series (if provided)
      D – Residual distributions
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.30)

    idx = reduced.index

    # ── A: Trajectories ─────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(idx, reduced["S_obs"], color=PALETTE["observed"],
              lw=1.4, label="S observed", zorder=5)

    S_sim_reduced  = reduced.get("S_sim",    reduced.get("S_sim_noE"))
    S_sim_upgraded = upgraded.get("S_sim",   upgraded.get("S_sim_noE"))

    ax_a.plot(idx, S_sim_reduced,  color=PALETTE["reduced"],
              lw=1.0, ls="--", alpha=0.8, label="S_sim reduced (no E)")
    ax_a.plot(idx, S_sim_upgraded, color=PALETTE["upgraded"],
              lw=1.0, ls="-",  alpha=0.8, label="S_sim upgraded (with E)")

    # Shade shock period if present
    shock_mask = reduced["shock_forcing"] != 0
    if shock_mask.any():
        for region in _mask_regions(shock_mask):
            ax_a.axvspan(idx[region[0]], idx[region[1]],
                         alpha=0.2, color=PALETTE["shock"])
        ax_a.plot([], [], color=PALETTE["shock"], alpha=0.4,
                  linewidth=8, label="Shock active")

    ax_a.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax_a.set_title(f"(A) {title}")
    ax_a.set_ylabel("S (normalised)")
    ax_a.legend(fontsize=8, loc="upper right")

    # ── B: Residuals ────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, :])
    resid_red = reduced.get("residual", reduced.get("residual_noE"))
    resid_upg = upgraded["residual"]

    ax_b.plot(idx, resid_red, color=PALETTE["reduced"],  lw=0.9, alpha=0.8,
              label="Residual – reduced")
    ax_b.plot(idx, resid_upg, color=PALETTE["upgraded"], lw=0.9, alpha=0.8,
              label="Residual – upgraded")
    ax_b.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax_b.set_title("(B) Residuals (S_sim − S_obs)")
    ax_b.set_ylabel("Residual")
    ax_b.legend(fontsize=8)

    # ── C: E series ─────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[2, :])
    if E_series is not None:
        E_slice = E_series.reindex(idx)
        ax_c.plot(idx, E_slice, color=PALETTE["E"], lw=1.0, label="E (fluidity)")
        ax_c.fill_between(idx, 0, E_slice, alpha=0.25, color=PALETTE["E"])
        ax_c.set_ylim(0, 1)
        ax_c.set_title("(C) Inferred E – Delivery Fluidity")
        ax_c.set_ylabel("E ∈ [0,1]")
        ax_c.legend(fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "E not provided", ha="center", va="center",
                  transform=ax_c.transAxes, color="gray")
        ax_c.set_title("(C) E not provided")

    # ── D: Residual distributions ────────────────────────────────────────
    ax_d1 = fig.add_subplot(gs[3, 0])
    ax_d2 = fig.add_subplot(gs[3, 1])

    for ax, resid, label, col in [
        (ax_d1, resid_red, "Reduced",  PALETTE["reduced"]),
        (ax_d2, resid_upg, "Upgraded", PALETTE["upgraded"]),
    ]:
        ax.hist(resid.dropna(), bins=40, color=col, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"(D) Residual dist – {label}\n"
                     f"RMSE={np.sqrt((resid**2).mean()):.3f}  "
                     f"MAE={resid.abs().mean():.3f}")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")

    fig.tight_layout()
    return fig


def plot_E_candidates(
    E_df: pd.DataFrame,
    panel: pd.DataFrame,
    figsize: tuple = (14, 8),
    title: str = "E Candidate Comparison",
) -> plt.Figure:
    """
    Compare all E candidates side-by-side against S and R.

    Parameters
    ----------
    E_df : pd.DataFrame
        Output of infer_all_E() — columns E1, E2, E3, E_composite.
    panel : pd.DataFrame
        Field panel containing S and R.
    """
    e_cols = [c for c in E_df.columns]
    n_e = len(e_cols)

    fig, axes = plt.subplots(n_e + 2, 1, figsize=figsize, sharex=True)

    # S and R
    ax = axes[0]
    ax.plot(panel.index, panel["S"], color="#2196F3", lw=1.0, label="S")
    ax.plot(panel.index, panel["R"], color="#F44336", lw=1.0, label="R")
    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)
    ax.set_ylabel("S, R")
    ax.set_title(title)
    ax.legend(fontsize=7)

    # E candidates
    colors = ["#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    for i, col in enumerate(e_cols):
        ax = axes[i + 1]
        ax.plot(E_df.index, E_df[col], color=colors[i % len(colors)], lw=0.9)
        ax.fill_between(E_df.index, 0, E_df[col], alpha=0.2,
                        color=colors[i % len(colors)])
        ax.set_ylim(0, 1)
        ax.set_ylabel(col, fontsize=8)
        ax.axhline(0.5, color="k", lw=0.4, ls="--", alpha=0.3)

    # Correlation table in last panel
    ax_last = axes[-1]
    corr = E_df.corr().round(2)
    ax_last.axis("off")
    tbl = ax_last.table(
        cellText=corr.values,
        rowLabels=corr.index,
        colLabels=corr.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1, 1.5)
    ax_last.set_title("E candidate correlation matrix", pad=2, fontsize=9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_report(
    metrics: pd.DataFrame,
    panel_stress: pd.DataFrame | None = None,
) -> None:
    """Print a formatted comparison report to stdout."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  CONSTRAINT FIELD MODEL – COMPARISON REPORT")
    print(f"{sep}\n")

    print("Propagation accuracy metrics")
    print("-" * 40)
    print(metrics.to_string())

    if panel_stress is not None:
        print("\n\nStatic field stress summary")
        print("-" * 40)
        print(panel_stress.to_string())

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _mask_regions(mask: pd.Series) -> list[tuple[int, int]]:
    """Convert a boolean mask to a list of (start, end) index pairs."""
    regions = []
    in_region = False
    start = 0
    vals = mask.values
    for i, v in enumerate(vals):
        if v and not in_region:
            start = i
            in_region = True
        elif not v and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, len(vals) - 1))
    return regions
