"""
constraint_field.analysis.regimes
====================================
Regime-stratified divergence analysis.

Tests whether divergence metrics behave differently across the S-R
field clusters identified in static analysis, and whether high-
divergence states are concentrated in particular regimes.

Also implements bounded-price diagnostics:
  - does instability rise when Phi is elevated but R is bounded?
  - is there a detectable R ceiling effect?
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


def regime_divergence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster summary of divergence metrics and instability.

    Returns DataFrame indexed by cluster with columns:
      n, frac_total, mean_S, mean_R, mean_Phi, mean_D1, mean_D2,
      mean_D3, mean_instability, frac_high_instability,
      frac_BP1, mean_D5_pos, mean_D5_neg
    """
    inst_thresh = df["instability_index"].quantile(0.85)

    def cluster_stats(g):
        row = {
            "n":                   len(g),
            "frac_total":          len(g) / len(df),
            "mean_S":              g["S"].mean(),
            "mean_R":              g["R"].mean(),
            "mean_Phi":            g["Phi"].mean(),
            "mean_D1":             g["D1"].mean() if "D1" in g else np.nan,
            "mean_D2":             g["D2"].mean() if "D2" in g else np.nan,
            "mean_D3":             g["D3"].mean() if "D3" in g else np.nan,
            "mean_D4":             g["D4"].mean() if "D4" in g else np.nan,
            "mean_instability":    g["instability_index"].mean(),
            "frac_high_instability": (g["instability_index"] > inst_thresh).mean(),
            "frac_BP1":            g["BP1"].mean() if "BP1" in g else np.nan,
            "mean_D5_pos":         g["D5_pos"].mean() if "D5_pos" in g else np.nan,
            "mean_D5_neg":         g["D5_neg"].mean() if "D5_neg" in g else np.nan,
        }
        return pd.Series(row)

    summary = df.groupby("cluster").apply(cluster_stats)

    # Add regime label based on mean S and R quadrant
    def quadrant_label(row):
        s_high = row["mean_S"] > 0
        r_high = row["mean_R"] > 0
        if s_high and r_high:
            return "demand+constraint"
        elif not s_high and r_high:
            return "supply-constraint"
        elif s_high and not r_high:
            return "demand-growth"
        else:
            return "slack"

    summary["regime_label"] = summary.apply(quadrant_label, axis=1)
    return summary.sort_values("mean_instability", ascending=False)


def bounded_price_diagnostics(df: pd.DataFrame) -> dict:
    """
    Quantify evidence for price as a bounded / partial signal.

    Tests
    -----
    1. Contingency: when |Phi| is high, is instability elevated
       even when R is moderate (bounded)?
    2. Correlation comparison: corr(R, instability) vs
       corr(D2, instability) — D2 is specifically designed to
       capture the bounded-price case.
    3. High-Phi / low-R periods: characterise these windows.
    4. R ceiling analysis: does instability cluster near R maxima?

    Returns
    -------
    dict with structured diagnostic results
    """
    phi_thresh = df["D1"].quantile(0.75)   # high imbalance
    r_mod      = df["R"].abs().quantile(0.50)  # moderate price (below median)
    inst_thresh = df["instability_index"].quantile(0.85)

    # Segment into four quadrants of (|Phi|, |R|)
    high_phi = df["D1"] > phi_thresh
    mod_r    = df["R"].abs() < r_mod
    high_r   = df["R"].abs() >= r_mod

    segments = {
        "high_phi_low_R":  df[high_phi & mod_r],
        "high_phi_high_R": df[high_phi & high_r],
        "low_phi_low_R":   df[~high_phi & mod_r],
        "low_phi_high_R":  df[~high_phi & high_r],
    }

    segment_stats = {}
    for name, seg in segments.items():
        if len(seg) == 0:
            continue
        segment_stats[name] = {
            "n":                    len(seg),
            "frac_total":           len(seg) / len(df),
            "mean_instability":     seg["instability_index"].mean(),
            "frac_high_instability": (seg["instability_index"] > inst_thresh).mean(),
            "mean_R":               seg["R"].mean(),
            "mean_D1":              seg["D1"].mean(),
            "mean_D2":              seg["D2"].mean() if "D2" in seg else np.nan,
        }

    # Test: is instability in high_phi_low_R significantly higher than
    # in low_phi_low_R? (both have moderate R, so R alone can't explain)
    group_a = segments.get("high_phi_low_R", pd.DataFrame())
    group_b = segments.get("low_phi_low_R",  pd.DataFrame())
    mannwhitney = None
    if len(group_a) >= 5 and len(group_b) >= 5:
        stat, pval = scipy_stats.mannwhitneyu(
            group_a["instability_index"],
            group_b["instability_index"],
            alternative="greater"
        )
        mannwhitney = {
            "test":      "Mann-Whitney U (one-sided: high_phi_low_R > low_phi_low_R)",
            "statistic": float(stat),
            "pvalue":    float(pval),
            "significant_at_05": pval < 0.05,
            "interpretation": (
                "Divergence adds signal beyond R when p < 0.05: "
                "high imbalance periods have higher instability "
                "even at similar moderate price levels."
            ),
        }

    # Correlation comparison
    corrs = {}
    for col in ["R", "D1", "D2", "D3", "D4", "D5_pos", "D5_neg", "BP2"]:
        if col in df.columns:
            c = df[col].corr(df["instability_index"])
            corrs[col] = round(float(c), 4)

    # R ceiling analysis: find hours where R is in top 5%
    r_ceiling = df["R"].quantile(0.95)
    near_ceiling = df[df["R"] >= r_ceiling]

    return {
        "segment_stats":     segment_stats,
        "mannwhitney_test":  mannwhitney,
        "correlations_with_instability": corrs,
        "r_ceiling_value":   float(r_ceiling),
        "near_ceiling_n":    len(near_ceiling),
        "near_ceiling_mean_instability": float(near_ceiling["instability_index"].mean())
            if len(near_ceiling) else np.nan,
        "near_ceiling_mean_phi": float(near_ceiling["Phi"].mean())
            if len(near_ceiling) else np.nan,
        "near_ceiling_phi_std": float(near_ceiling["Phi"].std())
            if len(near_ceiling) else np.nan,
        "phi_thresholds":    {"phi_high": float(phi_thresh), "r_mod": float(r_mod)},
    }
