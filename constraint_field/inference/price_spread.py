"""
constraint_field.inference.price_spread
=========================================
E₃ – Price-Spread Adjusted Transmissibility (calibrated)

This module replaces the previous naive fixed-scale implementation with
a data-calibrated approach that:

  1. Computes full summary statistics of the spread distribution
  2. Tests multiple robust scale candidates (median, IQR, std, p75, …)
  3. Tests four transform families (exponential, rational, minmax_inv, logistic)
  4. Selects the combination that best preserves ranking without collapsing
  5. Makes the chosen transform and scale configurable

Background
----------
Persistent, unresolved price spreads imply the market cannot equate
marginal costs across the network — a symptom of binding transmission
constraints impeding delivery fluidity.

Key insight from diagnosis
--------------------------
The spread signal is typically one of two structural types:

Type A – Zero-inflated (e.g. lmp_congestion):
  ~90% of hours have congestion = 0; active events are $10–80.
  Naive scales built from full-distribution quantiles are 0 or near-0,
  causing exp(-spread/~0) = exp(-inf) → complete collapse to 0.
  Remedy: use nz_median or nz_p75 (nonzero subset statistics).

Type B – Always-positive shifted (e.g. lmp_total - lmp_energy):
  The spread is always large ($40–$90), making exp(-spread/p75) collapse
  to near-zero because spread >> scale for almost all observations.
  Remedy: use the spread's own IQR, std, or range-based scale; or use
  minmax_inv / rational which are less sensitive to absolute scale.

The SpreadCalibrator automatically detects and handles both types.

Config reference
----------------
inference:
  E3_price_spread:
    transform: "auto"       # auto | exponential | rational | minmax_inv | logistic
    scale: "auto"           # auto | std | iqr | mad | p75 | p90 | nz_median | nz_p75 | ...
    logistic_midpoint: null  # null = use spread median
    logistic_slope: null     # null = use chosen scale value
    smoothing_hours: 3
    spread_window_hours: 24
    verbose_calibration: true
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseEInference, smooth_series, normalise_to_unit
from .calibration import (
    SpreadCalibrator,
    CalibrationResult,
    TRANSFORMS,
    candidate_scales,
    SpreadStats,
)

log = logging.getLogger(__name__)


class PriceSpreadE(BaseEInference):
    """
    E₃: delivery fluidity from price-spread-adjusted transmissibility.

    Uses data-calibrated scale selection to avoid distribution collapse.

    Parameters (config key: E3_price_spread)
    ----------------------------------------
    transform : str
        "auto"        – select best transform automatically
        "exponential" – E = exp(-spread / scale)
        "rational"    – E = 1 / (1 + spread / scale)
        "minmax_inv"  – E = 1 - normalised_spread  [scale-independent]
        "logistic"    – E = 1 / (1 + exp((spread - midpoint) / slope))
    scale : str
        "auto"        – select best scale automatically
        any key from candidate_scales(): "std","iqr","mad","p75","p90",
        "p95","mean","median","nz_median","nz_mean","nz_std","nz_p75","half_max"
    logistic_midpoint : float | None
        Sigmoid midpoint. None = spread median.
    logistic_slope : float | None
        Sigmoid slope. None = chosen scale value.
    smoothing_hours : int
        EWM smoothing window. Default 3.
    spread_window_hours : int
        Rolling range window when no direct spread signal is available.
    verbose_calibration : bool
        Log full calibration report. Default True.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Cache calibration result so decompose() can reuse it
        self._calibration: CalibrationResult | None = None

    @property
    def name(self) -> str:
        return "E3_price_spread"

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def infer(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """Infer E₃ with calibrated spread transform."""
        smoothing = self.cfg.get("smoothing_hours", 3)
        verbose   = self.cfg.get("verbose_calibration", True)

        # 1. Extract spread signal
        spread, spread_name = self._extract_spread(panel)

        # 2. Run calibration
        cal = self._run_calibration(spread, spread_name, verbose)
        self._calibration = cal

        # 3. Apply selected (or overridden) transform
        transform_choice = self.cfg.get("transform", "auto")
        scale_choice     = self.cfg.get("scale", "auto")

        if transform_choice == "auto" and scale_choice == "auto":
            E = cal.best_E.copy()
        else:
            E = self._apply_override(spread, cal, transform_choice, scale_choice)

        E = E.rename(self.name)
        E = smooth_series(E, smoothing)
        return E.clip(0, 1)

    # ------------------------------------------------------------------
    # Calibration runner
    # ------------------------------------------------------------------

    def _run_calibration(
        self,
        spread: pd.Series,
        name: str,
        verbose: bool,
    ) -> CalibrationResult:
        calibrator = SpreadCalibrator(
            transforms=["exponential", "rational", "minmax_inv", "logistic"],
        )
        result = calibrator.calibrate(spread, name=name, verbose=verbose)
        if verbose:
            result.print_report()
        return result

    # ------------------------------------------------------------------
    # Manual transform override
    # ------------------------------------------------------------------

    def _apply_override(
        self,
        spread: pd.Series,
        cal: CalibrationResult,
        transform_choice: str,
        scale_choice: str,
    ) -> pd.Series:
        spread_c = spread.clip(lower=0)

        # Resolve scale value
        if scale_choice == "auto":
            scale_val = cal.best.scale_value
        else:
            scales = candidate_scales(cal.stats)
            if scale_choice not in scales:
                log.warning(
                    "[E3] scale '%s' not in candidates %s; using auto",
                    scale_choice, list(scales),
                )
                scale_val = cal.best.scale_value
            else:
                scale_val = scales[scale_choice]

        t_name = cal.best.transform if transform_choice == "auto" else transform_choice

        if t_name == "logistic":
            mid   = self.cfg.get("logistic_midpoint", None)
            slope = self.cfg.get("logistic_slope", scale_val)
            return TRANSFORMS["logistic"](spread_c, scale=scale_val,
                                          midpoint=mid, slope=slope)
        elif t_name == "minmax_inv":
            return TRANSFORMS["minmax_inv"](spread_c)
        else:
            fn = TRANSFORMS.get(t_name, TRANSFORMS["rational"])
            return fn(spread_c, scale=scale_val)

    # ------------------------------------------------------------------
    # Spread extraction
    # ------------------------------------------------------------------

    def _extract_spread(self, panel: pd.DataFrame) -> tuple[pd.Series, str]:
        """
        Choose the best available spread signal.

        Priority order:
          1. lmp_spread_raw  (explicit hub-node spread)
          2. lmp_congestion_raw  (congestion component — zero-inflated)
          3. Rolling range of R  (temporal variability proxy)
        """
        window = self.cfg.get("spread_window_hours", 24)

        if "lmp_spread_raw" in panel.columns:
            spread = panel["lmp_spread_raw"].abs()
            name   = "lmp_spread_raw"
        elif "lmp_congestion_raw" in panel.columns:
            spread = panel["lmp_congestion_raw"].abs()
            name   = "lmp_congestion_raw"
        else:
            R        = panel["R"]
            roll_max = R.rolling(window=window, min_periods=1).max()
            roll_min = R.rolling(window=window, min_periods=1).min()
            spread   = (roll_max - roll_min).clip(lower=0)
            name     = f"R_rolling_range_w{window}"

        return spread, name

    # ------------------------------------------------------------------
    # Decompose for inspection
    # ------------------------------------------------------------------

    def decompose(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Return all intermediate quantities for full inspection.

        Columns:
          spread_raw, E3_exponential, E3_rational, E3_minmax_inv,
          E3_logistic, E3_final, R
        """
        spread, name = self._extract_spread(panel)

        if self._calibration is None:
            cal = self._run_calibration(spread, name, verbose=False)
            self._calibration = cal
        else:
            cal = self._calibration

        spread_c   = spread.clip(lower=0)
        best_scale = cal.best.scale_value if cal.best.scale_value > 0 else 1.0

        out = pd.DataFrame({"spread_raw": spread_c}, index=panel.index)

        for t_name in ["exponential", "rational", "minmax_inv", "logistic"]:
            try:
                if t_name == "minmax_inv":
                    E_t = TRANSFORMS[t_name](spread_c)
                else:
                    E_t = TRANSFORMS[t_name](spread_c, scale=best_scale)
                out[f"E3_{t_name}"] = E_t.values
            except Exception as exc:
                log.warning("decompose: %s failed: %s", t_name, exc)

        out["E3_final"] = self.infer(panel).values
        out["R"]        = panel["R"].values
        return out

    # ------------------------------------------------------------------
    # Diagnostic visualisation
    # ------------------------------------------------------------------

    def plot_calibration(
        self,
        panel: pd.DataFrame,
        figsize: tuple = (16, 14),
    ):
        """
        Full calibration diagnostic figure.

        Row 0: raw spread histogram  |  spread time-series
        Row 1: E histogram per transform (using best calibrated scale)
        Row 2: E time-series for all transforms
        Row 3: quality score summary table
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        spread, name = self._extract_spread(panel)
        spread_c = spread.clip(lower=0)

        if self._calibration is None:
            cal = self._run_calibration(spread, name, verbose=False)
            self._calibration = cal
        else:
            cal = self._calibration

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.58, wspace=0.38)

        colors = {
            "exponential": "#F44336",
            "rational":    "#2196F3",
            "minmax_inv":  "#FF9800",
            "logistic":    "#9C27B0",
        }
        best_scale = cal.best.scale_value if cal.best.scale_value > 0 else 1.0
        best_t     = cal.best.transform

        # ── Row 0: Spread histogram + time-series ─────────────────────────
        ax_sh = fig.add_subplot(gs[0, :2])
        vals = spread_c.dropna()
        ax_sh.hist(vals, bins=60, color="#607D8B", alpha=0.8, edgecolor="white",
                   log=(len(vals) > 100 and vals.max() > 10 * (vals.median() + 1e-9)))

        ref_lines = [
            (cal.stats.median,    "median",    "#F44336"),
            (cal.stats.p75,       "p75",       "#FF9800"),
            (cal.stats.p90,       "p90",       "#9C27B0"),
        ]
        if not np.isnan(cal.stats.nz_median) and cal.stats.nz_median > 0:
            ref_lines.append((cal.stats.nz_median, "nz_median", "#4CAF50"))

        for val, lbl, col in ref_lines:
            if val > 0:
                ax_sh.axvline(val, color=col, lw=1.2, ls="--",
                              label=f"{lbl}={val:.2f}")
        ax_sh.axvline(best_scale, color="black", lw=1.8, ls="-",
                      label=f"★ scale ({cal.best.scale_name}={best_scale:.2f})")
        ax_sh.set_xlabel(name)
        ax_sh.set_ylabel("Count")
        ax_sh.set_title(
            f"(A) Raw spread: {name}\n"
            f"n={cal.stats.n}  nonzero={cal.stats.frac_nonzero:.1%}  "
            f"std={cal.stats.std:.2f}  IQR={cal.stats.iqr:.2f}"
        )
        ax_sh.legend(fontsize=6.5, loc="upper right")

        ax_st = fig.add_subplot(gs[0, 2:])
        ax_st.plot(panel.index, spread_c, color="#607D8B", lw=0.7, alpha=0.7)
        ax_st.axhline(best_scale, color="black", lw=0.9, ls="--",
                      alpha=0.7, label=f"scale={best_scale:.2f}")
        ax_st.set_title("(B) Spread time-series")
        ax_st.set_ylabel(name)
        ax_st.legend(fontsize=7)

        # ── Row 1: E histograms per transform ─────────────────────────────
        transforms = ["exponential", "rational", "minmax_inv", "logistic"]
        for i, t_name in enumerate(transforms):
            ax = fig.add_subplot(gs[1, i])
            try:
                if t_name == "minmax_inv":
                    E_t = TRANSFORMS[t_name](spread_c)
                else:
                    E_t = TRANSFORMS[t_name](spread_c, scale=best_scale)

                ax.hist(E_t.dropna(), bins=40, color=colors[t_name],
                        alpha=0.85, edgecolor="white", range=(0, 1))

                flt05 = (E_t < 0.05).mean()
                fgt95 = (E_t > 0.95).mean()
                title = (
                    f"(C{i+1}) {t_name}\n"
                    f"μ={E_t.mean():.2f} σ={E_t.std():.2f}\n"
                    f"<5%={flt05:.1%}  >95%={fgt95:.1%}"
                )
                if t_name == best_t:
                    ax.set_facecolor("#f0fff4")
                    title += "\n★ SELECTED"
                ax.set_title(title, fontsize=7.5)
                ax.set_xlabel("E value")
                ax.set_ylabel("Count")
                ax.set_xlim(0, 1)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="red")
                ax.set_title(f"(C{i+1}) {t_name}\nFailed", fontsize=8)

        # ── Row 2: E time-series all transforms ───────────────────────────
        ax_et = fig.add_subplot(gs[2, :])
        for t_name in transforms:
            try:
                if t_name == "minmax_inv":
                    E_t = TRANSFORMS[t_name](spread_c)
                else:
                    E_t = TRANSFORMS[t_name](spread_c, scale=best_scale)
                is_best = (t_name == best_t)
                ax_et.plot(
                    panel.index, E_t,
                    color=colors[t_name],
                    lw=1.8 if is_best else 0.8,
                    alpha=1.0 if is_best else 0.55,
                    ls="-" if is_best else "--",
                    label=f"{t_name}" + (" ★" if is_best else ""),
                )
            except Exception:
                pass
        ax_et.axhline(0.5, color="k", lw=0.5, ls=":", alpha=0.3)
        ax_et.set_ylim(-0.05, 1.05)
        ax_et.set_ylabel("E₃ value")
        ax_et.set_title(
            f"(D) E₃ all transforms — scale: {cal.best.scale_name}={best_scale:.3f}"
        )
        ax_et.legend(fontsize=8, loc="upper right", ncol=2)

        # ── Row 3: Quality table ──────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[3, :])
        ax_tbl.axis("off")
        tbl = cal.quality_table[
            ["transform","scale_name","scale_value","mean","std",
             "frac_lt_05","frac_gt_95","score","acceptable"]
        ].head(16)

        def fmt(x):
            if isinstance(x, bool):
                return "✓" if x else "✗"
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)

        cell_text = [[fmt(v) for v in row] for row in tbl.values]
        row_colors = [
            ["#d4edda" if tbl.iloc[i]["acceptable"] else "#fff0f0"]
            * len(tbl.columns)
            for i in range(len(tbl))
        ]
        t = ax_tbl.table(
            cellText=cell_text,
            colLabels=tbl.columns.tolist(),
            rowColours=[rc[0] for rc in row_colors],
            loc="center",
            cellLoc="center",
        )
        t.scale(1, 1.25)
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        ax_tbl.set_title(
            f"(E) Top-16 calibration results  "
            f"[★ {best_t} / {cal.best.scale_name}={best_scale:.3f}  "
            f"score={cal.best.score():.4f}]",
            fontsize=9, pad=10,
        )

        fig.suptitle(
            f"E₃ Spread Calibration Diagnostics — {name}",
            fontsize=12, y=1.015,
        )
        return fig
