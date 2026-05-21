"""constraint_field.analysis — divergence metrics, predictive models, regime analysis."""

from .divergence import compute_divergence_metrics, divergence_summary
from .models import run_model_comparison, lead_lag_correlation, fit_ols, fit_logistic
from .regimes import regime_divergence_summary, bounded_price_diagnostics
from .visualize_divergence import (
    plot_divergence_timeseries,
    plot_lead_lag,
    plot_regime_divergence,
    plot_bounded_price,
    plot_model_comparison,
    plot_divergence_dashboard,
)

__all__ = [
    "compute_divergence_metrics", "divergence_summary",
    "run_model_comparison", "lead_lag_correlation", "fit_ols", "fit_logistic",
    "regime_divergence_summary", "bounded_price_diagnostics",
    "plot_divergence_timeseries", "plot_lead_lag", "plot_regime_divergence",
    "plot_bounded_price", "plot_model_comparison", "plot_divergence_dashboard",
]
