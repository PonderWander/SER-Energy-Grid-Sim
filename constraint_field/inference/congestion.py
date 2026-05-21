"""
constraint_field.inference.congestion
========================================
E₁ – Congestion Inverse

Interpretation
--------------
E₁ captures delivery fluidity as the inverse of constraint intensity.
When the system is heavily congested (large congestion component in LMP,
or large |Φ| imbalance), effective delivery is impeded → E₁ falls.
When congestion is low, delivery is fluent → E₁ is near 1.

Formula
-------
If direct congestion component is available (lmp_congestion_raw):

  congestion_norm = lmp_congestion_raw / (|R_scale| + ε)
  raw_E₁ = 1 / (1 + α × congestion_norm)

If congestion component is not available, use field imbalance as proxy:

  congestion_proxy = |R − S|  = |Φ|
  congestion_norm = (congestion_proxy − min) / (max − min)
  raw_E₁ = 1 / (1 + α × congestion_norm)

α (alpha) controls sensitivity:
  - large α → E₁ responds sharply to even moderate congestion
  - small α → E₁ is more permissive

Final E₁ is smoothed and normalised to [0, 1].

Conceptual grounding
---------------------
This is analogous to resistance in a flow circuit:
higher congestion ≡ higher resistance ≡ lower fluidity.
E₁ is the conductance (inverse resistance) of the delivery network
at each time step, inferred from observable constraint signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseEInference, smooth_series, normalise_to_unit


class CongestionInverseE(BaseEInference):
    """
    E₁: delivery fluidity inferred from the inverse of congestion intensity.

    Parameters (from config yaml, key: E1_congestion_inverse)
    ----------------------------------------------------------
    alpha : float
        Sensitivity parameter.  Default 2.0.
        Higher → E₁ drops sharply with moderate congestion.
    smoothing_hours : int
        EWM smoothing window.  Default 3 hours.
    """

    @property
    def name(self) -> str:
        return "E1_congestion_inverse"

    def infer(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Infer E₁.

        Uses 'lmp_congestion_raw' if present; falls back to |Φ|.
        """
        alpha          = self.cfg.get("alpha", 2.0)
        smoothing_hours = self.cfg.get("smoothing_hours", 3)

        # ── Choose congestion signal ──────────────────────────────────────
        if "lmp_congestion_raw" in panel.columns:
            # Use absolute congestion component; cap negatives at 0
            cong_raw = panel["lmp_congestion_raw"].clip(lower=0)
            # Normalise by rolling 95th percentile to make it scale-invariant
            scale = (cong_raw
                     .rolling(168, min_periods=24)
                     .quantile(0.95)
                     .fillna(cong_raw.quantile(0.95) or 1.0))
            cong_norm = (cong_raw / (scale + 1e-6)).clip(0, 1)
            source = "lmp_congestion_raw"
        else:
            # Fallback: imbalance magnitude |Φ| = |R − S|
            if "Phi" in panel.columns:
                phi_abs = panel["Phi"].abs()
            else:
                phi_abs = (panel["R"] - panel["S"]).abs()
            cong_norm = normalise_to_unit(phi_abs)
            source = "|Phi| proxy"

        # ── Apply inverse transform ───────────────────────────────────────
        # E₁ = 1 / (1 + α × congestion_norm)
        # → range: 1/(1+α) when fully congested, 1.0 when zero congestion
        raw_E = 1.0 / (1.0 + alpha * cong_norm)

        # Normalise to full [0,1] so all E candidates are comparable
        E = normalise_to_unit(raw_E)

        # Smooth (delivery adjustments have inertia)
        E = smooth_series(E, smoothing_hours)

        E.name = self.name
        return E.clip(0, 1)

    def decompose(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame showing the intermediate quantities for inspection.
        Useful for debugging and interpretation.
        """
        E = self.infer(panel)
        if "lmp_congestion_raw" in panel.columns:
            cong = panel["lmp_congestion_raw"].clip(lower=0)
        else:
            cong = (panel["R"] - panel["S"]).abs()
        return pd.DataFrame({
            "congestion_signal": cong,
            "E1_raw": 1.0 / (1.0 + self.cfg.get("alpha", 2.0)
                              * normalise_to_unit(cong)),
            "E1_smoothed": E,
        })
