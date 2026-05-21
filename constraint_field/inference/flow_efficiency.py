"""
constraint_field.inference.flow_efficiency
============================================
E₂ – Regional Flow Efficiency

Interpretation
--------------
E₂ measures how efficiently the network is moving energy relative to
the pressure imbalance that demands such movement.

A system with high net interchange (imports meeting demand pressure)
has high delivery fluidity → E₂ is high.

A system with large demand pressure S but low actual net flows has
impeded delivery → E₂ is low.

Formula
-------
If net_flow_mw_raw is available:

  flow_pressure_ratio = |net_flow_mw| / (|S_pressure_mw| + ε)
  S_pressure_mw = S × demand_capacity_proxy

  raw_E₂ = tanh(β × flow_pressure_ratio)

If net_flow_mw is not available, we use the load factor
(demand / generation) as a proxy for how much the grid is
relying on external delivery:

  lf = demand_mwh / generation_mwh
  # lf > 1 → net importer; lf < 1 → net exporter
  flow_proxy = |lf - 1|  ×  sign(demand − generation)
  raw_E₂ = tanh(β × normalise(|flow_proxy|))

β (beta) controls saturation speed:
  - large β → E₂ saturates quickly; moderate flows = high fluidity
  - small β → E₂ remains sensitive to large flows

Conceptual grounding
---------------------
This candidate treats the grid like a hydraulic network.
Flow efficiency = actual throughput / required throughput.
When the grid delivers what is demanded of it efficiently, E₂ ≈ 1.
When the grid is constrained and cannot meet the implied flow
requirement, E₂ drops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseEInference, smooth_series, normalise_to_unit


class FlowEfficiencyE(BaseEInference):
    """
    E₂: delivery fluidity from net interchange efficiency.

    Parameters (config key: E2_flow_efficiency)
    -------------------------------------------
    beta : float
        Saturation parameter for tanh transform.  Default 1.5.
    epsilon : float
        Floor for denominator (avoid divide-by-zero).  Default 0.01.
    smoothing_hours : int
        EWM window.  Default 3.
    """

    @property
    def name(self) -> str:
        return "E2_flow_efficiency"

    def infer(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """Infer E₂ from flow / pressure ratio."""
        beta            = self.cfg.get("beta", 1.5)
        epsilon         = self.cfg.get("epsilon", 0.01)
        smoothing_hours = self.cfg.get("smoothing_hours", 3)

        # ── Choose flow signal ────────────────────────────────────────────
        if "net_flow_mw_raw" in panel.columns:
            # Actual interchange data available
            flow_abs = panel["net_flow_mw_raw"].abs()

            # Demand in MW terms (raw column if available)
            if "demand_mwh_raw" in panel.columns:
                demand_mw = panel["demand_mwh_raw"].abs()
            else:
                # Approximate from S signal scaled to unit range
                S_norm = normalise_to_unit(panel["S"].abs())
                demand_mw = S_norm * 1.0  # stays normalised

            flow_norm = normalise_to_unit(flow_abs)
            demand_norm = normalise_to_unit(demand_mw)

            ratio = flow_norm / (demand_norm + epsilon)
            source = "net_flow_mw_raw"

        elif "demand_mwh_raw" in panel.columns and "generation_mwh" in panel.columns:
            # Load factor proxy
            lf = (panel["demand_mwh_raw"] /
                  panel["generation_mwh"].replace(0, np.nan))
            # |lf - 1| captures how much external delivery is needed
            flow_proxy = (lf - 1.0).abs()
            ratio = normalise_to_unit(flow_proxy)
            source = "load_factor_proxy"

        else:
            # Last resort: use S itself as a proxy for delivery demand
            # (higher demand pressure implies higher flow requirement)
            ratio = normalise_to_unit(panel["S"].abs())
            source = "S_proxy"

        # ── tanh transform ────────────────────────────────────────────────
        # tanh saturates at 1; maps ratio ∈ [0,∞) → E₂ ∈ [0,1)
        raw_E = np.tanh(beta * ratio)

        E = pd.Series(raw_E, index=panel.index, name=self.name)

        # Smooth
        E = smooth_series(E, smoothing_hours)

        return E.clip(0, 1)

    def decompose(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return intermediate quantities for inspection."""
        E = self.infer(panel)
        if "net_flow_mw_raw" in panel.columns:
            flow_signal = panel["net_flow_mw_raw"].abs()
        elif "demand_mwh_raw" in panel.columns:
            flow_signal = (panel["demand_mwh_raw"] /
                           panel.get("generation_mwh", panel["demand_mwh_raw"])
                           .replace(0, np.nan) - 1).abs()
        else:
            flow_signal = panel["S"].abs()

        return pd.DataFrame({
            "flow_signal": flow_signal,
            "E2_smoothed": E,
        })
