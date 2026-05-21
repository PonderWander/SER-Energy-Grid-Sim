"""constraint_field.field – static field construction and analysis."""

from .builder  import FieldBuilder, normalise
from .analysis import (
    run_static_analysis,
    compute_gradients,
    compute_field_indicators,
    rolling_instability,
    cluster_field_states,
    summary_stress,
)
from .visualize import (
    plot_field_timeseries,
    plot_phase_portrait,
    plot_gradient_heatmap,
    plot_instability,
    plot_static_dashboard,
)

__all__ = [
    "FieldBuilder", "normalise",
    "run_static_analysis",
    "compute_gradients", "compute_field_indicators",
    "rolling_instability", "cluster_field_states", "summary_stress",
    "plot_field_timeseries", "plot_phase_portrait",
    "plot_gradient_heatmap", "plot_instability", "plot_static_dashboard",
]
