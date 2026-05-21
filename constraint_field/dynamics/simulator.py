"""
constraint_field.dynamics.simulator
=====================================
Discrete-time propagation simulator for the dynamic field layer.

Runs configurable propagation experiments over the observed S, R series
using an inferred E as the transmissibility modulator.

Two modes
---------
1. Reduced (use_E=False):  propagation without E — S and R only.
   This serves as the baseline for comparison.

2. Upgraded (use_E=True):  E modulates the propagation operator.
   This is the dynamic corollary extension.

The Simulator accepts a field panel + E series and produces a
trajectory DataFrame containing:
  - S_sim:  simulated S path
  - R_obs:  observed R (used as exogenous input)
  - E_sim:  E values used (if use_E=True)
  - residual: S_sim − S_obs  (tracking error vs. observed)

Shock injection
---------------
The simulator supports injecting a step or pulse shock at a chosen
time index, enabling comparison of:
  - shock propagation WITH E  (dynamic upgraded)
  - shock propagation WITHOUT E (reduced)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .operators import get_operator

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shock specification
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Shock:
    """
    Represents an external perturbation injected into the simulation.

    Attributes
    ----------
    t_start : int
        Time index (relative to simulation start) at which shock begins.
    duration : int
        Number of time steps shock is applied (1 = impulse).
    magnitude : float
        Size of the shock (added directly to ΔS).
    label : str
        Human-readable label for plots.
    """
    t_start: int = 24
    duration: int = 1
    magnitude: float = 1.0
    label: str = "shock"

    def is_active(self, t: int) -> bool:
        return self.t_start <= t < self.t_start + self.duration

    def forcing(self, t: int) -> float:
        return self.magnitude if self.is_active(t) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────

class Simulator:
    """
    Discrete-time field propagation simulator.

    Parameters
    ----------
    dynamics_cfg : dict
        The 'dynamics' block from YAML config.
    seed : int
        Random seed for noise reproducibility.
    """

    def __init__(self, dynamics_cfg: dict, seed: int = 42):
        self.cfg   = dynamics_cfg
        self.seed  = seed
        self.dt    = dynamics_cfg.get("dt", 1)          # hours per step
        self.steps = dynamics_cfg.get("steps", 72)      # simulation horizon
        self.use_E = dynamics_cfg.get("use_E", True)

        # Optional modifiers
        self.threshold_cfg = dynamics_cfg.get("threshold", {})
        self.memory_cfg    = dynamics_cfg.get("memory", {})

    # ------------------------------------------------------------------
    # Main simulation entry point
    # ------------------------------------------------------------------

    def run(
        self,
        panel: pd.DataFrame,
        E_series: pd.Series | None = None,
        start_idx: int = 0,
        shocks: list[Shock] | None = None,
        operator_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Run a simulation starting at `start_idx` in the panel.

        Parameters
        ----------
        panel : pd.DataFrame
            Must contain columns S and R.
        E_series : pd.Series | None
            Inferred E values (required if use_E=True).
        start_idx : int
            Integer position in panel to begin simulation.
        shocks : list[Shock] | None
            Optional shock events to inject.
        operator_name : str | None
            Override the operator in config (for sweep experiments).

        Returns
        -------
        pd.DataFrame
            Simulation trajectory with columns:
            S_sim, S_obs, R_obs, E_used, residual, shock_forcing
        """
        shocks = shocks or []

        # ── Override operator if specified ────────────────────────────────
        cfg = dict(self.cfg)
        if operator_name:
            cfg["operator"] = operator_name

        rng = np.random.default_rng(self.seed)
        operator = get_operator(cfg, rng=rng)

        # ── Slice relevant portion of panel ───────────────────────────────
        end_idx = min(start_idx + self.steps, len(panel))
        panel_slice = panel.iloc[start_idx:end_idx].copy()
        n = len(panel_slice)

        S_obs = panel_slice["S"].values
        R_obs = panel_slice["R"].values

        if E_series is not None:
            E_vals = E_series.reindex(panel_slice.index).fillna(0.5).values
        else:
            E_vals = np.ones(n) * 0.5   # neutral if not supplied

        S_mean = float(panel["S"].mean())

        # ── Initial conditions ────────────────────────────────────────────
        S_sim       = np.zeros(n)
        shock_force = np.zeros(n)

        S_sim[0] = S_obs[0]   # initialise from observed state
        S_prev   = S_sim[0]

        # Memory term accumulator
        use_memory = self.memory_cfg.get("enabled", False)
        mem_alpha  = self.memory_cfg.get("alpha", 0.2)
        S_memory   = S_sim[0]

        # ── Forward simulation ────────────────────────────────────────────
        for t in range(1, n):
            S_cur = S_sim[t - 1]
            R_cur = R_obs[t]
            E_cur = E_vals[t] if self.use_E else 0.5  # 0.5 = neutral E

            # Base increment from operator
            delta_S = operator.step(
                S=S_cur,
                S_prev=S_prev,
                R=R_cur,
                E=E_cur,
                S_mean=S_mean,
                t=t,
                dt=self.dt,
            )

            # Shock injection
            for shock in shocks:
                sf = shock.forcing(t)
                delta_S    += sf
                shock_force[t] = sf

            # Optional threshold nonlinearity
            if self.threshold_cfg.get("enabled", False):
                level   = self.threshold_cfg.get("level", 1.5)
                extra_d = self.threshold_cfg.get("extra_damping", 0.3)
                if abs(S_cur) > level:
                    delta_S -= extra_d * S_cur * self.dt

            # Optional memory term
            if use_memory:
                S_memory = (1 - mem_alpha) * S_memory + mem_alpha * S_cur
                delta_S  += mem_alpha * (S_memory - S_cur) * self.dt

            # Update
            S_sim[t] = S_cur + delta_S
            S_prev   = S_cur

        # ── Assemble results DataFrame ─────────────────────────────────────
        result = pd.DataFrame({
            "S_sim":         S_sim,
            "S_obs":         S_obs,
            "R_obs":         R_obs,
            "E_used":        E_vals if self.use_E else np.full(n, np.nan),
            "residual":      S_sim - S_obs,
            "shock_forcing": shock_force,
        }, index=panel_slice.index)

        log.info(
            "Simulation complete [use_E=%s, operator=%s]: "
            "n=%d, RMSE=%.4f, max|residual|=%.4f",
            self.use_E, cfg.get("operator", "?"),
            n,
            np.sqrt(np.mean(result["residual"]**2)),
            result["residual"].abs().max(),
        )
        return result

    # ------------------------------------------------------------------
    # Comparative experiment: run with and without E
    # ------------------------------------------------------------------

    def compare(
        self,
        panel: pd.DataFrame,
        E_series: pd.Series,
        start_idx: int = 0,
        shocks: list[Shock] | None = None,
        operator_name: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Run both reduced (no E) and upgraded (with E) simulations.

        Returns
        -------
        dict with keys "reduced" and "upgraded"
        """
        # Reduced: use_E = False
        orig_use_E = self.use_E
        self.use_E = False
        reduced = self.run(panel, E_series, start_idx, shocks, operator_name)
        reduced.columns = [f"{c}_noE" if c not in ("S_obs", "R_obs", "shock_forcing")
                           else c for c in reduced.columns]

        # Upgraded: use_E = True
        self.use_E = True
        upgraded = self.run(panel, E_series, start_idx, shocks, operator_name)

        self.use_E = orig_use_E  # restore

        return {"reduced": reduced, "upgraded": upgraded}

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        panel: pd.DataFrame,
        E_series: pd.Series,
        param_grid: dict[str, list],
        start_idx: int = 0,
    ) -> pd.DataFrame:
        """
        Sweep over parameter combinations and collect summary metrics.

        Parameters
        ----------
        param_grid : dict
            Keys are nested config paths (e.g., "diffusion.gamma"),
            values are lists of values to try.

        Returns
        -------
        pd.DataFrame  summary table (one row per parameter combination)
        """
        import itertools

        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        records = []

        for combo in itertools.product(*values):
            # Apply parameter overrides to a copy of config
            cfg_copy = {k: v for k, v in self.cfg.items()}
            params = dict(zip(keys, combo))
            for key_path, val in params.items():
                parts = key_path.split(".")
                d = cfg_copy
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = val

            sim = Simulator(cfg_copy, seed=self.seed)
            try:
                result = sim.run(panel, E_series, start_idx)
                rmse = np.sqrt((result["residual"]**2).mean())
                max_resid = result["residual"].abs().max()
                records.append({**params, "rmse": rmse, "max_residual": max_resid})
            except Exception as exc:
                log.warning("Sweep failed for params %s: %s", params, exc)
                records.append({**params, "rmse": np.nan, "max_residual": np.nan})

        return pd.DataFrame(records)
