"""constraint_field.dynamics – propagation operators and simulator."""

from .operators import (
    DiffusionOperator,
    GradientFlowOperator,
    DampedWaveOperator,
    get_operator,
)
from .simulator import Simulator, Shock

__all__ = [
    "DiffusionOperator", "GradientFlowOperator", "DampedWaveOperator",
    "get_operator",
    "Simulator", "Shock",
]
