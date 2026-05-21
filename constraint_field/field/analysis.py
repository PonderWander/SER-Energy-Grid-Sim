"""
constraint_field.field.analysis
=================================
Static field analysis on the S–R panel.

Implements:
  1. Temporal gradients of S and R
  2. Local imbalance / instability indicator  Φ = R − S
  3. Constraint intensity metric  Ψ = |R| + |S|
  4. Clustering / hotspot detection over time slices
  5. Summary stress metrics

Design note
-----------
All analysis is performed on the temporal axis (time as the "spatial"
dimension in the reduced single-node formulation).

For multi-node / multi-region extensions, the same functions generalise
to spatial gradients over the network graph — see the network extension
stub at the bottom of this file.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_gradients(
    panel: pd.DataFrame,
    method: Literal["central", "forward"] = "central",
) -> pd.DataFrame:
    """
    Compute temporal gradients of S and R.

    ∂S/∂t and ∂R/∂t (per-hour changes in normalised units).

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns 'S' and 'R'.
    method : str
        "central"  – centred finite difference (better interior accuracy)
        "forward"  – forward difference (causal; for real-time use)

    Returns
    -------
    pd.DataFrame
        Adds columns: dS_dt, dR_dt, d2S_dt2, d2R_dt2
    """
    out = panel.copy()

    for var in ["S", "R"]:
        if method == "central":
            grad = np.gradient(panel[var].fillna(0).values)
            grad2 = np.gradient(grad)
        else:  # forward
            grad = panel[var].diff().values
            grad[0] = grad[1]
            grad2 = np.diff(grad, prepend=grad[0])

        out[f"d{var}_dt"]  = grad
        out[f"d2{var}_dt2"] = grad2

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Imbalance and constraint intensity
# ──────────────────────────────────────────────────────────────────────────────

def compute_field_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived scalar field indicators.

    Φ  (imbalance)          = R − S
        Positive Φ: price/constraint signal exceeds load pressure
        → potential over-constraint, transmission limit binding
        Negative Φ: load pressure exceeds price signal
        → delivery is relatively fluid; demand not fully priced

    Ψ  (field intensity)    = √(S² + R²)
        Magnitude of the field vector — overall system stress level.
        High Ψ: system is far from a slack, low-signal state.

    Θ  (field angle, deg)   = arctan2(R, S)
        Orientation in the S–R plane.
        0–90°:  high S and high R  (demand-driven constraint)
        90–180°: high R, low S     (supply-side constraint / scarcity)
        180–270°: low S, low R     (slack system)
        270–360°: high S, low R    (demand growth, unconstrained)

    Parameters
    ----------
    panel : pd.DataFrame  with columns S, R

    Returns
    -------
    pd.DataFrame  with added columns Phi, Psi, Theta
    """
    out = panel.copy()
    S, R = panel["S"].fillna(0), panel["R"].fillna(0)

    out["Phi"]   = R - S                                    # imbalance
    out["Psi"]   = np.sqrt(S**2 + R**2)                    # intensity
    out["Theta"] = np.degrees(np.arctan2(R.values, S.values)) % 360  # angle

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Smoothed instability
# ──────────────────────────────────────────────────────────────────────────────

def rolling_instability(
    panel: pd.DataFrame,
    window: int = 24,
    percentile: float = 75.0,
) -> pd.Series:
    """
    Local instability index: rolling high-percentile of |Φ|.

    Periods where the imbalance between constraint signal and load
    pressure is persistently large indicate structural stress.

    Returns
    -------
    pd.Series  (same index as panel)
    """
    if "Phi" not in panel.columns:
        panel = compute_field_indicators(panel)

    abs_phi = panel["Phi"].abs()
    return (abs_phi
            .rolling(window=window, min_periods=1)
            .quantile(percentile / 100.0)
            .rename("instability_index"))


# ──────────────────────────────────────────────────────────────────────────────
# Clustering / hotspot detection
# ──────────────────────────────────────────────────────────────────────────────

def cluster_field_states(
    panel: pd.DataFrame,
    method: Literal["kmeans", "dbscan"] = "kmeans",
    n_clusters: int = 4,
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 5,
    features: list[str] | None = None,
) -> pd.Series:
    """
    Cluster observations in (S, R) field space to identify
    qualitative system regimes.

    Typical clusters for a 4-cluster solution:
      0: Slack low-demand / low-price
      1: Normal operation
      2: Demand-driven stress (high S, moderate R)
      3: Constraint-dominated stress (high R, any S)

    Parameters
    ----------
    panel : pd.DataFrame
    method : "kmeans" | "dbscan"
    n_clusters : int  (kmeans only)
    features : list[str] | None
        Columns to cluster on.  Defaults to ["S", "R"].

    Returns
    -------
    pd.Series  with integer cluster labels, named "cluster"
    """
    if features is None:
        features = ["S", "R"]
        if "Phi" in panel.columns:
            features.append("Phi")

    X = panel[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = model.fit_predict(X_scaled)

    elif method == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(X_scaled)

    else:
        raise ValueError(f"Unknown clustering method: '{method}'")

    return pd.Series(labels, index=panel.index, name="cluster")


# ──────────────────────────────────────────────────────────────────────────────
# Summary stress metrics
# ──────────────────────────────────────────────────────────────────────────────

def summary_stress(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary stress metrics for the full panel or a time slice.

    Returns a single-row DataFrame suitable for comparison across
    regions, time periods, or model configurations.

    Metrics
    -------
    mean_S, std_S         : load pressure statistics
    mean_R, std_R         : constraint signal statistics
    mean_Phi              : mean imbalance (R − S)
    mean_Psi              : mean field intensity
    frac_high_constraint  : fraction of hours with R > 1 (above +1σ)
    frac_dual_stress      : fraction with both S > 0.5 and R > 0.5
    max_instability       : peak rolling instability index
    """
    if "Phi" not in panel.columns:
        panel = compute_field_indicators(panel)
    if "instability_index" not in panel.columns:
        panel = panel.join(rolling_instability(panel))

    S, R = panel["S"], panel["R"]

    return pd.DataFrame([{
        "mean_S":              S.mean(),
        "std_S":               S.std(),
        "mean_R":              R.mean(),
        "std_R":               R.std(),
        "mean_Phi":            panel["Phi"].mean(),
        "mean_Psi":            panel["Psi"].mean(),
        "frac_high_constraint": (R > 1.0).mean(),
        "frac_dual_stress":    ((S > 0.5) & (R > 0.5)).mean(),
        "max_instability":     panel["instability_index"].max(),
        "n_obs":               len(panel),
    }])


# ──────────────────────────────────────────────────────────────────────────────
# Full analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_static_analysis(
    panel: pd.DataFrame,
    analysis_cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Run the full static field analysis pipeline in one call.

    Adds columns:
      dS_dt, dR_dt, d2S_dt2, d2R_dt2,
      Phi, Psi, Theta,
      instability_index,
      cluster

    Parameters
    ----------
    panel : pd.DataFrame  (output of FieldBuilder.build())
    analysis_cfg : dict   (from config yaml, optional)

    Returns
    -------
    pd.DataFrame  enriched panel
    """
    cfg = analysis_cfg or {}
    grad_method  = cfg.get("gradient_method", "central")
    clust_method = cfg.get("clustering", {}).get("method", "kmeans")
    n_clusters   = cfg.get("clustering", {}).get("n_clusters", 4)
    dbscan_eps   = cfg.get("clustering", {}).get("dbscan_eps", 0.3)
    dbscan_min   = cfg.get("clustering", {}).get("dbscan_min_samples", 5)

    log.info("Running static field analysis …")
    panel = compute_gradients(panel, method=grad_method)
    panel = compute_field_indicators(panel)
    panel = panel.join(rolling_instability(panel))
    panel["cluster"] = cluster_field_states(
        panel, method=clust_method,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min,
    )
    log.info("Static analysis complete.")
    return panel
