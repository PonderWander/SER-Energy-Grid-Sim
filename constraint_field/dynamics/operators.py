"""
constraint_field.dynamics.operators
=====================================
Configurable propagation operators for the dynamic field layer.

These operators define how the field state (S, R) evolves over time
when E modulates transmissibility.

Three operator families
-----------------------
1. Diffusion
   Models field relaxation toward a mean, with E-modulated spreading.
   Analogy: heat equation with variable conductivity.

   ΔS_t = −γ·(S_t − S̄) + η·E_t·∇²S_t + ε_noise

   - γ (gamma): mean-reversion / damping rate
   - η (eta):   transmissibility weight; E scales this coefficient
   - ∇²S:       discrete Laplacian (second temporal derivative proxy)

2. Gradient Flow
   Models field motion driven by constraint signal gradients.
   Analogy: pressure-driven flow following ∇R.

   ΔS_t = −μ·∇R_t + κ·E_t·(S̄ − S_t)

   - μ (mu):    sensitivity to constraint gradient
   - κ (kappa): E-modulated restoration force

3. Damped Wave
   Second-order dynamics: inertia + damping + restoring force.
   Analogy: damped harmonic oscillator coupled to R field.

   S̈_t ≈ −2ζω·Ṡ_t − ω²·(S_t − R_t) + E_t · forcing_t

   - ω (omega): natural frequency
   - ζ (zeta):  damping ratio (< 1 = underdamped, > 1 = overdamped)

Design notes
------------
- All operators take the current state vector and return ΔS (increment).
- E is treated as an external scalar field modulating the propagation.
- Operators are stateless (no internal memory); state is managed by Simulator.
- Optional threshold and memory terms are applied as wrappers.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Protocol (interface)
# ──────────────────────────────────────────────────────────────────────────────

class PropagationOperator(Protocol):
    """Interface contract for all propagation operators."""

    def step(
        self,
        S: float,
        S_prev: float,
        R: float,
        E: float,
        S_mean: float,
        t: int,
        dt: float,
    ) -> float:
        """
        Compute the increment ΔS for one time step.

        Parameters
        ----------
        S       : current S value
        S_prev  : previous S value (for velocity estimate)
        R       : current R value
        E       : current E value (delivery fluidity, [0,1])
        S_mean  : long-run mean of S (used as attractor)
        t       : current time index
        dt      : time step size (hours)

        Returns
        -------
        float   : ΔS increment to apply
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# 1. Diffusion operator
# ──────────────────────────────────────────────────────────────────────────────

class DiffusionOperator:
    """
    ΔS = −γ·(S − S̄) + η·E·∇²S + ε_noise

    The E-modulated term η·E·∇²S models how fluidity enables or
    suppresses the propagation of deviations from the mean.

    When E=0 (fully congested), propagation vanishes and the system
    can only mean-revert via the γ term.

    When E=1 (fully fluid), maximum propagation occurs — deviations
    spread through the network readily.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.gamma     = cfg.get("gamma", 0.1)
        self.eta       = cfg.get("eta", 0.05)
        self.noise_std = cfg.get("noise_std", 0.01)
        self.rng       = rng or np.random.default_rng(0)

    def step(
        self,
        S: float,
        S_prev: float,
        R: float,
        E: float,
        S_mean: float,
        t: int,
        dt: float,
    ) -> float:
        # Mean reversion term
        mean_rev = -self.gamma * (S - S_mean)

        # Discrete Laplacian proxy: ∇²S ≈ (S_prev − 2S + S_future)
        # Since S_future is unknown, use backward difference: (S_prev − S)
        laplacian_proxy = S_prev - S

        # E-modulated propagation
        propagation = self.eta * E * laplacian_proxy

        # Stochastic noise (optional; set noise_std=0 to suppress)
        noise = self.rng.normal(0, self.noise_std)

        return (mean_rev + propagation + noise) * dt

    @property
    def name(self) -> str:
        return "diffusion"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Gradient flow operator
# ──────────────────────────────────────────────────────────────────────────────

class GradientFlowOperator:
    """
    ΔS = −μ·∇R + κ·E·(S̄ − S)

    Models load/usage pressure being driven by constraint gradients.
    ∇R = R_t − R_{t-1}  (discrete temporal gradient of constraint signal)

    Interpretation:
      - A rising R (tightening constraint) pushes S downward (demand response)
        or upward (supply scarcity raises apparent usage pressure).
      - E modulates how strongly the system can restore toward equilibrium.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.mu    = cfg.get("mu", 0.08)
        self.kappa = cfg.get("kappa", 0.05)
        self.rng   = rng or np.random.default_rng(0)
        self._R_prev: float | None = None

    def step(
        self,
        S: float,
        S_prev: float,
        R: float,
        E: float,
        S_mean: float,
        t: int,
        dt: float,
    ) -> float:
        # Constraint gradient
        R_prev = self._R_prev if self._R_prev is not None else R
        grad_R = R - R_prev
        self._R_prev = R

        # Gradient-driven term (constraint pushes field)
        gradient_term = -self.mu * grad_R

        # E-modulated restoration
        restoration = self.kappa * E * (S_mean - S)

        return (gradient_term + restoration) * dt

    def reset(self):
        """Reset internal state for new simulation run."""
        self._R_prev = None

    @property
    def name(self) -> str:
        return "gradient_flow"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Damped wave operator
# ──────────────────────────────────────────────────────────────────────────────

class DampedWaveOperator:
    """
    Second-order dynamics (damped harmonic oscillator):

      S̈ ≈ −2ζω·Ṡ − ω²·(S − R) + E·forcing

    Discretised (Störmer–Verlet-like, explicit Euler for simplicity):

      velocity_t = S_t − S_{t-1}
      acceleration = −2ζω·velocity − ω²·(S − R) + E·forcing
      ΔS = velocity + acceleration·dt

    This operator captures oscillatory dynamics: a system driven away
    from its constraint equilibrium will oscillate and decay.

    E modulates the forcing term — high fluidity allows the system to
    respond to external forcing; low fluidity suppresses it.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.omega   = cfg.get("omega", 0.3)
        self.zeta    = cfg.get("zeta", 0.5)
        self.forcing = cfg.get("forcing_amplitude", 0.1)
        self.rng     = rng or np.random.default_rng(0)

    def step(
        self,
        S: float,
        S_prev: float,
        R: float,
        E: float,
        S_mean: float,
        t: int,
        dt: float,
    ) -> float:
        velocity     = S - S_prev
        damping      = -2 * self.zeta * self.omega * velocity
        restoring    = -self.omega**2 * (S - R)
        forcing      = E * self.forcing * np.sin(2 * np.pi * t / 24)  # diurnal forcing

        acceleration = damping + restoring + forcing
        delta_S      = (velocity + acceleration * dt) * dt

        return delta_S

    @property
    def name(self) -> str:
        return "damped_wave"


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_operator(dynamics_cfg: dict, rng: np.random.Generator | None = None):
    """
    Instantiate the propagation operator from config.

    Parameters
    ----------
    dynamics_cfg : dict
        The 'dynamics' block from the YAML config.
    rng : np.random.Generator | None

    Returns
    -------
    An operator with a .step() method.
    """
    op_name = dynamics_cfg.get("operator", "diffusion")
    op_cfg  = dynamics_cfg.get(op_name, {})

    registry = {
        "diffusion":     DiffusionOperator,
        "gradient_flow": GradientFlowOperator,
        "damped_wave":   DampedWaveOperator,
    }

    if op_name not in registry:
        raise ValueError(
            f"Unknown operator '{op_name}'.  Available: {list(registry)}"
        )

    return registry[op_name](op_cfg, rng=rng)
