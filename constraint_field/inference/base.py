"""
constraint_field.inference.base
================================
Abstract base for all E (delivery fluidity) inference candidates.

E is NOT a directly observed variable.  It is an inferred corollary
that upgrades the reduced static field (S, R) into a dynamic form
capable of modelling propagation, transmissibility, and dissipation.

Operational interpretation of E
--------------------------------
E ∈ [0, 1]  (after normalisation)

  E ≈ 1  →  high delivery fluidity:
             the system can transfer energy freely; constraints are not binding;
             shocks propagate readily and dissipate quickly.

  E ≈ 0  →  low delivery fluidity:
             the system is congested / constrained; effective transmission
             is impeded; shocks are absorbed locally and persist longer.

E is used in the dynamic layer as a transmissibility modulator:
  - scales the diffusion/propagation coefficient
  - modulates damping rate
  - can trigger threshold nonlinearities

All E candidates must:
  1. Return a pd.Series in [0, 1] on the same DatetimeIndex as the panel.
  2. Expose a `name` property identifying the candidate.
  3. Accept the field panel + optional raw auxiliary signals.
  4. Be fully inspectable — no black-box transformations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def smooth_series(series: pd.Series, hours: int = 3) -> pd.Series:
    """Exponential weighted smoothing over `hours` periods."""
    if hours <= 1:
        return series
    return series.ewm(span=hours, adjust=False).mean()


def normalise_to_unit(series: pd.Series, clip: bool = True) -> pd.Series:
    """Min-max normalise to [0, 1], handling edge cases."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index, name=series.name)
    out = (series - lo) / (hi - lo)
    if clip:
        out = out.clip(0, 1)
    return out.rename(series.name)


class BaseEInference(ABC):
    """Abstract base for E inference candidates."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'E1_congestion_inverse'."""

    @abstractmethod
    def infer(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Infer E from the field panel and optional auxiliary data.

        Parameters
        ----------
        panel : pd.DataFrame
            Must contain S and R.  May also contain net_flow_mw_raw,
            lmp_congestion_raw, lmp_spread_raw depending on candidate.
        **kwargs
            Additional named signals (e.g. flows_df, weather_df).

        Returns
        -------
        pd.Series
            E values in [0, 1], same DatetimeIndex as panel.
            Named with self.name.
        """

    def __call__(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """Allow candidate to be called like a function."""
        E = self.infer(panel, **kwargs)
        assert E.between(0, 1).all() or E.isna().any(), \
            f"{self.name}: E values outside [0,1] detected."
        log.info("[%s] E inferred: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                 self.name, E.mean(), E.std(), E.min(), E.max())
        return E
