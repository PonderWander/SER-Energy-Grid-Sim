"""
constraint_field.analysis.divergence
======================================
Divergence metrics quantifying the gap between field pressure and
price expression.

Core hypothesis
---------------
If price (R) is a bounded or partial signal of underlying field pressure,
then the divergence between S and R — measured via Phi and its derivatives —
should contain information about instability that R alone does not carry.

All metrics are derived from the existing field variables S, R, Phi, Psi
and are fully transparent (no black-box transformations).

Metric definitions
------------------
D1  = |Phi|                            raw imbalance magnitude
D2  = |Phi| / (1 + |R|)               imbalance scaled by constraint signal
                                       (high D2 when imbalance is large
                                        but price is moderate — possible
                                        price-bounding signature)
D3  = |Phi| / (1 + Psi)               imbalance as fraction of total field
                                       intensity (relative divergence)
D4  = rolling mean of |Phi|            persistence of imbalance over window
D5+ = max(Phi, 0)  /  max(-Phi, 0)    signed asymmetry components:
                                       D5_pos: R exceeding S (over-constraint)
                                       D5_neg: S exceeding R (under-expression)

Bounded-price diagnostics
--------------------------
BP1 = high |Phi| ∩ moderate R          price-bounding indicator
BP2 = persistent |Phi| with bounded R  rolling signal
BP3 = R ceiling clustering             repeated visits to near-max R
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Default configurable window for rolling metrics
DEFAULT_WINDOW = 24   # hours


# ──────────────────────────────────────────────────────────────────────────────
# Individual divergence metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_divergence_metrics(
    panel: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    r_moderate_threshold: float = 0.75,   # |R| below this = "moderate price"
    phi_high_threshold: float = 0.5,      # |Phi| above this = "high imbalance"
) -> pd.DataFrame:
    """
    Compute all divergence metrics and append to panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain S, R, Phi, Psi (output of run_static_analysis).
    window : int
        Rolling window for D4 persistence metric (hours).
    r_moderate_threshold : float
        |R| below this quantile is considered "moderate price" for BP1.
    phi_high_threshold : float
        |Phi| above this quantile is considered "high imbalance" for BP1.

    Returns
    -------
    pd.DataFrame
        Extended panel with columns D1–D5_neg, BP1–BP3, and their
        summary statistics attached as metadata in .attrs.
    """
    df = panel.copy()

    S   = df["S"]
    R   = df["R"]
    Phi = df["Phi"]       # = R - S
    Psi = df["Psi"]       # = sqrt(S^2 + R^2)

    # ── D1: Raw imbalance magnitude ──────────────────────────────────────
    df["D1"] = Phi.abs()

    # ── D2: Imbalance / (1 + |R|)  — price-bounded imbalance ────────────
    # High D2: large divergence even though price is moderate
    # This is the key metric for the bounded-price hypothesis
    df["D2"] = Phi.abs() / (1.0 + R.abs())

    # ── D3: Imbalance / (1 + Psi)  — relative divergence ────────────────
    # Normalises by total field intensity; captures imbalance as fraction
    # of system's overall stress magnitude
    df["D3"] = Phi.abs() / (1.0 + Psi)

    # ── D4: Rolling persistence of |Phi| ─────────────────────────────────
    # Sustained imbalance is more structurally significant than spikes;
    # this captures how long the system has been in a divergent state
    df["D4"] = (
        Phi.abs()
        .rolling(window=window, min_periods=max(1, window // 4))
        .mean()
        .rename("D4")
    )

    # ── D5: Signed asymmetry components ──────────────────────────────────
    # D5_pos: R > S  (constraint signal exceeds load pressure)
    #         → price is amplifying beyond underlying demand
    #         → potential over-constraint or supply-scarcity signal
    # D5_neg: S > R  (load pressure exceeds constraint signal)
    #         → price is NOT keeping up with underlying demand pressure
    #         → this is the bounded-price signature
    df["D5_pos"] = Phi.clip(lower=0)        # max(Phi,  0) = max(R-S, 0)
    df["D5_neg"] = (-Phi).clip(lower=0)     # max(-Phi, 0) = max(S-R, 0)

    # ── Bounded-price diagnostics ─────────────────────────────────────────

    # Compute adaptive thresholds from empirical distribution
    r_mod_thresh   = R.abs().quantile(r_moderate_threshold)
    phi_high_thresh = Phi.abs().quantile(phi_high_threshold)

    # BP1: Binary indicator — high imbalance AND moderate price
    # This is the direct test of whether price is bounding
    df["BP1"] = (
        (df["D1"] > phi_high_thresh) & (R.abs() < r_mod_thresh)
    ).astype(int)

    # BP2: Rolling fraction of recent hours that were BP1
    # Captures persistence of bounded-price state
    df["BP2"] = (
        df["BP1"]
        .rolling(window=window, min_periods=1)
        .mean()
        .rename("BP2")
    )

    # BP3: R ceiling proximity — how close is R to its rolling 95th percentile?
    # Repeated visits to near-ceiling R while Phi varies = price ceiling evidence
    R_ceiling = R.rolling(window=window * 7, min_periods=window).quantile(0.95)
    df["BP3"] = (R / (R_ceiling.abs() + 1e-6)).clip(-1, 1)

    # ── Store threshold metadata ──────────────────────────────────────────
    df.attrs["divergence_window"]      = window
    df.attrs["r_moderate_threshold"]   = float(r_mod_thresh)
    df.attrs["phi_high_threshold"]     = float(phi_high_thresh)
    df.attrs["divergence_metric_cols"] = [
        "D1", "D2", "D3", "D4", "D5_pos", "D5_neg", "BP1", "BP2", "BP3"
    ]

    log.info(
        "Divergence metrics computed: window=%dh  phi_threshold=%.3f  r_threshold=%.3f\n"
        "  BP1 (high imbalance + moderate R): %.1f%% of hours",
        window, phi_high_thresh, r_mod_thresh, df["BP1"].mean() * 100,
    )

    return df


def divergence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for all divergence metrics.

    Returns a DataFrame indexed by metric name with columns:
    mean, std, median, p75, p90, max, corr_with_instability.
    """
    metric_cols = df.attrs.get(
        "divergence_metric_cols",
        [c for c in df.columns if c.startswith(("D", "BP"))]
    )
    instability = df["instability_index"]

    rows = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        corr = s.corr(instability)
        rows.append({
            "metric":   col,
            "mean":     s.mean(),
            "std":      s.std(),
            "median":   s.median(),
            "p75":      s.quantile(0.75),
            "p90":      s.quantile(0.90),
            "max":      s.max(),
            "corr_instability": corr,
        })

    return pd.DataFrame(rows).set_index("metric")
