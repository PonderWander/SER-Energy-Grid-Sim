"""
constraint_field.analysis.visualize_decontaminated
====================================================
Visualisations for the decontaminated lead analysis.

Plots
-----
1. Point-in-time target time-series (I1–I4) vs original instability
2. Clean lead/lag correlation curves per target
3. Event-study plot: divergence shock → future I1 path
4. Time-of-day structure: raw vs residualised series
5. Bounded-price test: raw and residualised, side by side
6. Regime-stratified summary bar chart
7. Full decontamination dashboard
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PALETTE = {
    "D1":      "#9C27B0",
    "D2":      "#2196F3",
    "BP1":     "#FF5722",
    "I1":      "#212121",
    "I2":      "#607D8B",
    "I3":      "#F44336",
    "I4":      "#FF9800",
    "event":   "#F44336",
    "control": "#607D8B",
    "resid":   "#4CAF50",
}
CLUSTER_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]


def plot_pit_targets(
    df: pd.DataFrame,
    figsize: tuple = (16, 10),
    title: str = "Point-in-Time Instability Targets vs Original Rolling Index",
) -> plt.Figure:
    """Compare I1–I4 and original instability_index over time."""
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    series_def = [
        ("instability_index", "Original (24h rolling 75th pct |Phi|)", "#9E9E9E", True),
        ("I1", "I1 = instantaneous |Phi|",                 PALETTE["I1"], False),
        ("I2", "I2 = 3h rolling mean |Phi|",               PALETTE["I2"], False),
        ("I3", "I3 = binary spike (|Phi| > p90)",          PALETTE["I3"], True),
        ("I4", "I4 = binary spike (Psi > p90)",            PALETTE["I4"], True),
    ]
    for ax, (col, label, color, is_binary) in zip(axes, series_def):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"{col} not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        s = df[col]
        if is_binary and s.nunique() <= 2:
            ax.fill_between(df.index, 0, s, alpha=0.5, color=color, step="post")
        else:
            ax.plot(df.index, s, color=color, lw=0.9, alpha=0.85)
        ax.set_ylabel(col, fontsize=8)
        ax.set_title(label, fontsize=8, pad=2)
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)

    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_clean_lead_lag(
    lead_lag_results: dict[str, dict[str, pd.DataFrame]],
    figsize: tuple = (16, 10),
    title: str = "Clean Lead/Lag: Divergence Metrics → Point-in-Time Targets",
) -> plt.Figure:
    """
    Grid of lead/lag plots: rows = predictors, cols = targets.

    lead_lag_results: {predictor: {target: ll_df}}
    """
    predictors = list(lead_lag_results.keys())
    targets    = list(next(iter(lead_lag_results.values())).keys())
    nrow, ncol = len(predictors), len(targets)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                              sharey=False, sharex=True)
    if nrow == 1:
        axes = [axes]
    if ncol == 1:
        axes = [[ax] for ax in axes]

    target_colors = [PALETTE.get(t, "#607D8B") for t in targets]

    for i, pred in enumerate(predictors):
        for j, tgt in enumerate(targets):
            ax   = axes[i][j]
            col  = PALETTE.get(pred, "#333333")
            ll   = lead_lag_results[pred].get(tgt, pd.DataFrame())

            if ll.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
            else:
                ax.plot(ll["lag"], ll["corr"], color=col, lw=1.4,
                        marker="o", ms=3, label=f"{pred}→{tgt}")
                # Shade significant lags
                sig = ll[ll["pvalue"] < 0.05]
                if not sig.empty:
                    ax.scatter(sig["lag"], sig["corr"],
                               color=col, s=22, zorder=5, alpha=0.9,
                               edgecolors="black", linewidths=0.4)
                # Best lead annotation
                pos_leads = ll[ll["lag"] >= 1]
                if not pos_leads.empty:
                    best_row = pos_leads.loc[pos_leads["corr"].abs().idxmax()]
                    ax.axvline(best_row["lag"], color=col, lw=0.7, ls=":",
                               alpha=0.6)
                    ax.text(best_row["lag"], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
                            f"  lag={int(best_row['lag'])}h\n  r={best_row['corr']:.2f}",
                            fontsize=6, color=col, va="top")

            ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.35)
            ax.axvline(0, color="k", lw=0.6, ls="-",  alpha=0.20)
            ax.set_title(f"{pred} → {tgt}", fontsize=8, pad=2)
            if j == 0:
                ax.set_ylabel("Correlation", fontsize=7)
            if i == nrow - 1:
                ax.set_xlabel("Lead hours", fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_event_study(
    event_results: dict[str, dict],
    figsize: tuple = (14, 6),
    title: str = "Event Study: Divergence Shock → Future Instability Path",
) -> plt.Figure:
    """
    For each event definition, plot the average I1 path
    from t-12 to t+24, with ±1 SE shading and control comparison.
    """
    n_events = len(event_results)
    fig, axes = plt.subplots(1, n_events, figsize=figsize, sharey=True)
    if n_events == 1:
        axes = [axes]

    for ax, (event_name, res) in zip(axes, event_results.items()):
        if not res or "event_path" not in res:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(event_name)
            continue

        idx     = res["event_path"].index
        ev_mean = res["event_path"].values
        ev_se   = res["event_se"].values
        ct_mean = res["control_path"].values
        ct_se   = res["control_se"].values

        ax.plot(idx, ev_mean, color=PALETTE["event"], lw=1.6,
                label=f"Event (n={res['n_events']})")
        ax.fill_between(idx, ev_mean - ev_se, ev_mean + ev_se,
                        alpha=0.25, color=PALETTE["event"])

        ax.plot(idx, ct_mean, color=PALETTE["control"], lw=1.2, ls="--",
                label=f"Control (n={res['n_controls']})")
        ax.fill_between(idx, ct_mean - ct_se, ct_mean + ct_se,
                        alpha=0.15, color=PALETTE["control"])

        ax.axvline(0, color="black", lw=1.0, ls="-", alpha=0.5, label="Event t=0")
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        ax.set_xlabel("Hours relative to event")
        ax.set_title(f"{event_name}\n(target={res['target_col']})", fontsize=8)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean I1 (instantaneous |Phi|)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_tod_residualisation(
    df: pd.DataFrame,
    col: str = "D1",
    figsize: tuple = (14, 8),
    title: str = "Time-of-Day Structure: Raw vs Residualised",
) -> plt.Figure:
    """
    Show the diurnal pattern in a divergence metric and its residual,
    to establish how much ToD structure was removed.
    """
    resid_col = f"{col}_resid"
    if resid_col not in df.columns:
        log.warning("No residualised column '%s'; skipping ToD plot.", resid_col)
        return plt.figure()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    df2 = df.copy()
    df2["hour"] = df2.index.hour

    # Hourly mean: raw
    ax = axes[0, 0]
    hourly_raw = df2.groupby("hour")[col].mean()
    ax.bar(hourly_raw.index, hourly_raw.values, color=PALETTE["D1"], alpha=0.8)
    ax.set_title(f"(A) Hourly mean of {col} (raw)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean value")

    # Hourly mean: residual
    ax = axes[0, 1]
    hourly_res = df2.groupby("hour")[resid_col].mean()
    ax.bar(hourly_res.index, hourly_res.values, color=PALETTE["resid"], alpha=0.8)
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_title(f"(B) Hourly mean of {col} (residualised)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean residual")

    # Time-series comparison
    ax = axes[1, 0]
    sample = df2.iloc[:7 * 24]   # first week
    ax.plot(sample.index, sample[col], color=PALETTE["D1"], lw=0.9,
            alpha=0.8, label=f"{col} raw")
    ax.set_title(f"(C) {col} raw — first week")
    ax.set_ylabel(col)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(sample.index, sample[resid_col], color=PALETTE["resid"], lw=0.9,
            alpha=0.8, label=f"{col} residualised")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_title(f"(D) {col} residualised — first week")
    ax.set_ylabel(f"{col} residual")
    ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_bp_raw_vs_resid(
    bp_raw:   dict,
    bp_resid: dict,
    figsize: tuple = (12, 5),
    title: str = "Bounded-Price Test: Raw vs Time-of-Day Residualised",
) -> plt.Figure:
    """
    Side-by-side bar chart showing segment means for raw vs residualised test.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, bp, label in [(axes[0], bp_raw, "Raw"), (axes[1], bp_resid, "Residualised")]:
        segs = bp.get("segments", {})
        names = list(segs.keys())
        means = [segs[k]["mean"] for k in names if "mean" in segs[k]]
        valid_names = [k for k in names if "mean" in segs[k]]

        colors = [PALETTE["event"] if "high_phi_mod" in k else PALETTE["control"]
                  for k in valid_names]
        bars = ax.bar(range(len(valid_names)), means, color=colors, alpha=0.8)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(max(means, default=0)) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(len(valid_names)))
        ax.set_xticklabels(
            [s.replace("_", "\n") for s in valid_names],
            fontsize=7, rotation=10, ha="right"
        )
        pval = bp.get("mw_pvalue", np.nan)
        cd   = bp.get("cohens_d", np.nan)
        ax.set_title(
            f"({label})\nMann-Whitney p={pval:.4f}  d={cd:.3f}\n"
            f"{'★ significant' if bp.get('significant_05') else '✗ not significant'}",
            fontsize=9
        )
        ax.set_ylabel("Mean I1 (instantaneous |Phi|)")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_regime_stratified(
    regime_df: pd.DataFrame,
    figsize: tuple = (12, 5),
    title: str = "Regime-Stratified: Bounded-Price Effect and Best Lead",
) -> plt.Figure:
    """
    Bar chart of bounded-price Cohen's d and best lead lag per cluster.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    clusters = regime_df.index.tolist()
    x = np.arange(len(clusters))
    labels = [
        f"C{c}\n{regime_df.loc[c, 'regime_label']}"
        for c in clusters
    ]

    # Cohen's d by cluster
    ax = axes[0]
    d_vals = regime_df["bp_cohens_d"].values
    colors = [PALETTE["event"] if v > 0 and regime_df.loc[c, "bp_sig"]
              else PALETTE["control"]
              for c, v in zip(clusters, d_vals)]
    bars = ax.bar(x, d_vals, color=colors, alpha=0.85)
    for bar, sig in zip(bars, regime_df["bp_sig"].values):
        if sig:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, "★", ha="center", fontsize=11)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cohen's d (bounded-price effect)")
    ax.set_title("(A) Bounded-price effect size by regime\n(★ = p<0.05)")

    # Best lead D1 and D2
    ax2 = axes[1]
    w = 0.35
    for i, (pred, col) in enumerate([("D1", PALETTE["D1"]), ("D2", PALETTE["D2"])]):
        lead_col = f"best_lead_{pred}"
        if lead_col in regime_df.columns:
            ax2.bar(x + i * w - w/2, regime_df[lead_col].fillna(0),
                    w, color=col, alpha=0.8, label=f"Best lead {pred}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Best lead (hours ahead)")
    ax2.set_title("(B) Best clean lead lag by regime")
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_decontamination_dashboard(
    df: pd.DataFrame,
    lead_lag_results: dict,
    event_results: dict,
    bp_raw: dict,
    bp_resid: dict,
    regime_df: pd.DataFrame,
    figsize: tuple = (18, 22),
    title: str = "Decontaminated Lead Analysis — Full Dashboard",
) -> plt.Figure:
    """
    6-row summary dashboard.
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(6, 4, figure=fig, hspace=0.58, wspace=0.38)

    # ── Row 0: I1 and I2 vs original instability ────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(df.index, df["instability_index"], color="#9E9E9E",
             lw=0.8, alpha=0.7, label="Original (24h rolling)")
    ax0.plot(df.index, df["I1"], color=PALETTE["I1"],
             lw=0.9, alpha=0.8, label="I1 (instantaneous)")
    ax0.plot(df.index, df["I2"], color=PALETTE["I2"],
             lw=0.9, alpha=0.7, ls="--", label="I2 (3h rolling)")
    ax0_r = ax0.twinx()
    ax0_r.fill_between(df.index, 0, df["I3"],
                       alpha=0.15, color=PALETTE["I3"], step="post", label="I3 event")
    ax0_r.set_ylabel("I3 spike", fontsize=7, color=PALETTE["I3"])
    ax0_r.set_ylim(0, 3)
    ax0_r.tick_params(labelsize=6)
    ax0.set_title("(A) Point-in-time targets vs original rolling instability")
    ax0.set_ylabel("Instability value")
    ax0.legend(fontsize=7, loc="upper right")

    # ── Row 1: Clean lead/lag D1→I1 and D2→I1 ───────────────────────────
    ll_targets = ["I1", "I2", "I3"]
    pred_list  = [p for p in ["D1", "D2"] if p in lead_lag_results]
    tgt_colors = [PALETTE["I1"], PALETTE["I2"], PALETTE["I3"]]

    for pi, pred in enumerate(pred_list):
        ax_ll = fig.add_subplot(gs[1, pi * 2: pi * 2 + 2])
        for tgt, tcol in zip(ll_targets, tgt_colors):
            ll = lead_lag_results.get(pred, {}).get(tgt, pd.DataFrame())
            if not ll.empty:
                ax_ll.plot(ll["lag"], ll["corr"], color=tcol,
                           lw=1.2, marker="o", ms=2.5, alpha=0.85, label=tgt)
                sig = ll[ll["pvalue"] < 0.05]
                ax_ll.scatter(sig["lag"], sig["corr"], color=tcol, s=20,
                              zorder=5, alpha=0.9)
        ax_ll.axhline(0, color="k", lw=0.5, ls="--", alpha=0.35)
        ax_ll.axvline(0, color="k", lw=0.6, alpha=0.20)
        ax_ll.set_title(f"(B{pi+1}) {pred} → target lead/lag (clean)")
        ax_ll.set_xlabel("Lead hours (positive = predictor leads)")
        ax_ll.set_ylabel("Correlation")
        ax_ll.legend(fontsize=7)

    # ── Row 2: Event study ────────────────────────────────────────────────
    ev_list = list(event_results.items())[:3]
    for ei, (ev_name, res) in enumerate(ev_list):
        ax_ev = fig.add_subplot(gs[2, ei + (4 - len(ev_list)) // 2])
        if res and "event_path" in res:
            idx = res["event_path"].index
            ax_ev.plot(idx, res["event_path"].values,
                       color=PALETTE["event"], lw=1.4,
                       label=f"Event (n={res['n_events']})")
            ax_ev.fill_between(
                idx,
                res["event_path"].values - res["event_se"].values,
                res["event_path"].values + res["event_se"].values,
                alpha=0.2, color=PALETTE["event"],
            )
            ax_ev.plot(idx, res["control_path"].values,
                       color=PALETTE["control"], lw=1.1, ls="--",
                       label=f"Control (n={res['n_controls']})")
            ax_ev.axvline(0, color="k", lw=0.8, ls="-", alpha=0.4)
        ax_ev.set_title(f"(C) Event: {ev_name}", fontsize=8)
        ax_ev.set_xlabel("Relative hours")
        ax_ev.set_ylabel("Mean I1")
        ax_ev.legend(fontsize=6)

    # ── Row 3: ToD residualisation ────────────────────────────────────────
    ax_tod = fig.add_subplot(gs[3, :2])
    df2 = df.copy()
    df2["hour"] = df2.index.hour
    for col, lbl, col_color in [
        ("D1",       "D1 raw",    PALETTE["D1"]),
        ("D1_resid", "D1 resid",  PALETTE["resid"]),
    ]:
        if col in df2.columns:
            hmean = df2.groupby("hour")[col].mean()
            ax_tod.plot(hmean.index, hmean.values, color=col_color,
                        lw=1.4, marker="o", ms=4, label=lbl)
    ax_tod.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)
    ax_tod.set_xlabel("Hour of day")
    ax_tod.set_ylabel("Mean value")
    ax_tod.set_title("(D) Diurnal pattern: D1 raw vs residualised")
    ax_tod.legend(fontsize=8)
    ax_tod.set_xticks(range(0, 24, 3))

    ax_tod2 = fig.add_subplot(gs[3, 2:])
    for col, lbl, col_color in [
        ("I1",       "I1 raw",    PALETTE["I1"]),
        ("I1_resid", "I1 resid",  PALETTE["resid"]),
    ]:
        if col in df2.columns:
            hmean = df2.groupby("hour")[col].mean()
            ax_tod2.plot(hmean.index, hmean.values, color=col_color,
                         lw=1.4, marker="o", ms=4, label=lbl)
    ax_tod2.set_title("(E) Diurnal pattern: I1 raw vs residualised")
    ax_tod2.set_xlabel("Hour of day")
    ax_tod2.legend(fontsize=8)
    ax_tod2.set_xticks(range(0, 24, 3))

    # ── Row 4: BP test raw vs resid ──────────────────────────────────────
    for bi, (bp, lbl, ax_pos) in enumerate([
        (bp_raw,   "Raw",          gs[4, :2]),
        (bp_resid, "Residualised", gs[4, 2:]),
    ]):
        ax_bp = fig.add_subplot(ax_pos)
        segs  = bp.get("segments", {})
        valid = [(k, v) for k, v in segs.items() if "mean" in v]
        names = [k for k, _ in valid]
        means = [v["mean"] for _, v in valid]
        colors = [PALETTE["event"] if "high_phi_mod" in k else PALETTE["control"]
                  for k in names]
        ax_bp.bar(range(len(names)), means, color=colors, alpha=0.8)
        ax_bp.set_xticks(range(len(names)))
        ax_bp.set_xticklabels(
            [n.replace("_", "\n") for n in names], fontsize=6, rotation=8
        )
        pv = bp.get("mw_pvalue", np.nan)
        cd = bp.get("cohens_d", np.nan)
        ax_bp.set_title(
            f"(F{bi+1}) BP test ({lbl})\np={pv:.4f}  d={cd:.3f}  "
            f"{'★ sig' if bp.get('significant_05') else '✗ not sig'}",
            fontsize=8,
        )
        ax_bp.set_ylabel("Mean I1")
        ax_bp.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)

    # ── Row 5: Regime-stratified ──────────────────────────────────────────
    if not regime_df.empty:
        ax_reg = fig.add_subplot(gs[5, :2])
        clusters = regime_df.index.tolist()
        xr = np.arange(len(clusters))
        d_vals = regime_df["bp_cohens_d"].fillna(0).values
        r_colors = [PALETTE["event"] if (d > 0 and regime_df.loc[c, "bp_sig"])
                    else PALETTE["control"]
                    for c, d in zip(clusters, d_vals)]
        ax_reg.bar(xr, d_vals, color=r_colors, alpha=0.85)
        for xi, (c, sig) in enumerate(zip(clusters, regime_df["bp_sig"].values)):
            if sig:
                ax_reg.text(xi, d_vals[xi] + 0.01, "★", ha="center", fontsize=11)
        ax_reg.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax_reg.set_xticks(xr)
        ax_reg.set_xticklabels(
            [f"C{c}\n{regime_df.loc[c,'regime_label']}" for c in clusters],
            fontsize=7
        )
        ax_reg.set_ylabel("Cohen's d")
        ax_reg.set_title("(G) BP effect size by regime (★ p<0.05)")

        ax_lead = fig.add_subplot(gs[5, 2:])
        w = 0.35
        for pi, (pred, pcol) in enumerate([("D1", PALETTE["D1"]), ("D2", PALETTE["D2"])]):
            col = f"best_lead_{pred}"
            if col in regime_df.columns:
                ax_lead.bar(xr + pi * w - w/2,
                            regime_df[col].fillna(0), w,
                            color=pcol, alpha=0.8, label=f"Lead {pred}")
        ax_lead.set_xticks(xr)
        ax_lead.set_xticklabels(
            [f"C{c}\n{regime_df.loc[c,'regime_label']}" for c in clusters],
            fontsize=7
        )
        ax_lead.set_ylabel("Best lead lag (hours)")
        ax_lead.set_title("(H) Best clean lead by regime")
        ax_lead.legend(fontsize=7)

    fig.suptitle(title, fontsize=13, y=1.005)
    return fig
