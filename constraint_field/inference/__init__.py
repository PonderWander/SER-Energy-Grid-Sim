"""
constraint_field.inference
==========================
E (delivery fluidity) inference layer.

Three candidate constructions of the inferred corollary variable E:

  E1 – CongestionInverseE     congestion_inverse
  E2 – FlowEfficiencyE        flow_efficiency
  E3 – PriceSpreadE           price_spread

Plus a composite weighted average of all three.

Use get_E_inferrer() for config-driven instantiation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base            import BaseEInference, smooth_series, normalise_to_unit
from .congestion      import CongestionInverseE
from .flow_efficiency import FlowEfficiencyE
from .price_spread    import PriceSpreadE
from .calibration     import (
    SpreadCalibrator, CalibrationResult, SpreadStats,
    EDistributionQuality, TRANSFORMS, candidate_scales,
    transform_exponential, transform_rational,
    transform_minmax_inv, transform_logistic,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Composite E
# ──────────────────────────────────────────────────────────────────────────────

class CompositeE(BaseEInference):
    """
    Weighted average of E1, E2, E3.

    Each candidate is normalised to [0,1] before combining so that
    differences in spread / scale do not dominate the composite.

    Parameters (config key: composite)
    ------------------------------------
    weights : list[float]  [w1, w2, w3]  — sum need not equal 1 (auto-normalised)
    """

    @property
    def name(self) -> str:
        return "E_composite"

    def infer(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        weights = self.cfg.get("weights", [0.4, 0.3, 0.3])
        w = np.array(weights[:3])
        w = w / w.sum()  # normalise

        candidates = {
            "E1": CongestionInverseE(self.cfg).infer(panel, **kwargs),
            "E2": FlowEfficiencyE(self.cfg).infer(panel, **kwargs),
            "E3": PriceSpreadE(self.cfg).infer(panel, **kwargs),
        }

        composite = sum(w[i] * candidates[k] for i, k in enumerate(candidates))
        composite = pd.Series(composite, index=panel.index, name=self.name)
        return composite.clip(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_E_inferrer(inference_cfg: dict) -> BaseEInference:
    """
    Return the active E inference candidate from config.

    Parameters
    ----------
    inference_cfg : dict
        The 'inference' block from the YAML config.

    Returns
    -------
    BaseEInference instance ready to call .infer(panel)
    """
    active = inference_cfg.get("active", "E1")

    mapping = {
        "E1": (CongestionInverseE, "E1_congestion_inverse"),
        "E2": (FlowEfficiencyE,    "E2_flow_efficiency"),
        "E3": (PriceSpreadE,       "E3_price_spread"),
        "composite": (CompositeE,  "composite"),
    }

    if active not in mapping:
        raise ValueError(f"Unknown E candidate '{active}'. Options: {list(mapping)}")

    cls, cfg_key = mapping[active]
    candidate_cfg = inference_cfg.get(cfg_key, {})
    return cls(candidate_cfg)


def infer_all_E(panel: pd.DataFrame, inference_cfg: dict) -> pd.DataFrame:
    """
    Convenience: run all three E candidates + composite on a panel.

    Returns
    -------
    pd.DataFrame with columns: E1, E2, E3, E_composite
    """
    candidates = {
        "E1": CongestionInverseE(inference_cfg.get("E1_congestion_inverse", {})),
        "E2": FlowEfficiencyE(   inference_cfg.get("E2_flow_efficiency", {})),
        "E3": PriceSpreadE(      inference_cfg.get("E3_price_spread", {})),
    }
    results = {}
    for label, inferrer in candidates.items():
        try:
            results[label] = inferrer.infer(panel)
        except Exception as exc:
            log.warning("[%s] inference failed: %s", label, exc)
            results[label] = pd.Series(np.nan, index=panel.index)

    # Composite using equal weights as default comparison
    valid = [v for v in results.values() if not v.isna().all()]
    if valid:
        composite = sum(valid) / len(valid)
        composite.name = "E_composite"
        results["E_composite"] = composite

    return pd.DataFrame(results)


__all__ = [
    "BaseEInference",
    "CongestionInverseE",
    "FlowEfficiencyE",
    "PriceSpreadE",
    "CompositeE",
    "get_E_inferrer",
    "infer_all_E",
]
