"""
constraint_field.inference.calibration
========================================
Data-calibrated scale parameter selection for spread-based E transforms.

Problem diagnosis
-----------------
The naive approach of using panel["R"].abs().quantile(p) as R_scale for
the price-spread E transform fails in the common case where the spread
signal is:
  1. Zero-inflated  (e.g. congestion = 0 most hours; spikes rarely)
  2. Shifted / always-positive  (e.g. lmp_spread = lmp_total - lmp_energy
     which is always ~$45-$80 because it includes the energy base)
  3. Heavy-tailed  (rare large congestion events dominate scale choices)

In these cases a single naive percentile of the raw signal produces:
  - exp(-spread / 0) = undefined / NaN
  - exp(-spread / very_large) ≈ exp(0) ≈ 1  (collapsed to 1)
  - exp(-spread / very_small) ≈ exp(-∞) ≈ 0  (collapsed to 0)

Calibration strategy
--------------------
1. Compute full summary statistics of the raw spread
2. Identify the structural type (zero-inflated, always-positive, mixed)
3. Select scale candidates from ROBUST measures of the ACTIVE part of
   the distribution (nonzero, or interquartile range, or MAD)
4. Evaluate the resulting E distribution for each candidate
5. Select the scale that produces the best-spread E distribution
   (minimizes |frac>0.95 - frac<0.05|, maximizes E std, avoids collapse)

Four transform families
-----------------------
All share the same interface: transform(spread, scale) -> E in [0,1]

  exponential:   E = exp(-spread / scale)
  rational:      E = 1 / (1 + spread / scale)
  minmax_inv:    E = 1 - (spread - min) / (max - min)   [scale unused]
  logistic:      E = 1 / (1 + exp((spread - midpoint) / slope))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TransformName = Literal["exponential", "rational", "minmax_inv", "logistic"]


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpreadStats:
    """Full characterisation of a spread series."""
    n:            int
    n_nonzero:    int
    frac_nonzero: float
    min:          float
    max:          float
    mean:         float
    std:          float
    median:       float
    mad:          float        # median absolute deviation
    iqr:          float        # interquartile range
    p25:          float
    p75:          float
    p90:          float
    p95:          float
    # Statistics restricted to the nonzero / active subset
    nz_median:    float        # median of spread[spread > 0]
    nz_mean:      float
    nz_std:       float
    nz_p75:       float

    @classmethod
    def compute(cls, spread: pd.Series) -> "SpreadStats":
        s      = spread.dropna()
        s_pos  = s[s > 0]
        return cls(
            n            = len(s),
            n_nonzero    = len(s_pos),
            frac_nonzero = len(s_pos) / max(len(s), 1),
            min          = float(s.min()),
            max          = float(s.max()),
            mean         = float(s.mean()),
            std          = float(s.std()),
            median       = float(s.median()),
            mad          = float((s - s.median()).abs().median()),
            iqr          = float(s.quantile(0.75) - s.quantile(0.25)),
            p25          = float(s.quantile(0.25)),
            p75          = float(s.quantile(0.75)),
            p90          = float(s.quantile(0.90)),
            p95          = float(s.quantile(0.95)),
            nz_median    = float(s_pos.median()) if len(s_pos) else float("nan"),
            nz_mean      = float(s_pos.mean())   if len(s_pos) else float("nan"),
            nz_std       = float(s_pos.std())    if len(s_pos) else float("nan"),
            nz_p75       = float(s_pos.quantile(0.75)) if len(s_pos) else float("nan"),
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def log_summary(self, name: str = "spread") -> None:
        log.info(
            "[calibration] %s summary: n=%d  nonzero=%.1f%%  "
            "min=%.3f  max=%.3f  mean=%.3f  std=%.3f\n"
            "                    median=%.3f  MAD=%.3f  IQR=%.3f  "
            "p75=%.3f  p90=%.3f  p95=%.3f\n"
            "                    nz_median=%.3f  nz_p75=%.3f",
            name, self.n, self.frac_nonzero * 100,
            self.min, self.max, self.mean, self.std,
            self.median, self.mad, self.iqr,
            self.p75, self.p90, self.p95,
            self.nz_median, self.nz_p75,
        )


# ──────────────────────────────────────────────────────────────────────────────
# E distribution quality metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EDistributionQuality:
    """Quality metrics for an inferred E distribution."""
    scale_name:   str
    scale_value:  float
    transform:    str
    mean:         float
    std:          float
    min:          float
    max:          float
    frac_lt_05:   float    # fraction < 0.05 (collapsed low)
    frac_gt_95:   float    # fraction > 0.95 (collapsed high)
    frac_lt_10:   float
    frac_gt_90:   float
    collapse_score: float  # 0 = perfect spread, 1 = fully collapsed

    @classmethod
    def compute(
        cls,
        E: pd.Series,
        scale_name: str,
        scale_value: float,
        transform: str,
    ) -> "EDistributionQuality":
        e = E.dropna()
        frac_lt_05 = float((e < 0.05).mean())
        frac_gt_95 = float((e > 0.95).mean())
        frac_lt_10 = float((e < 0.10).mean())
        frac_gt_90 = float((e > 0.90).mean())
        # Collapse score: 0 = well-spread, 1 = all in tails
        collapse = max(frac_lt_05 + frac_gt_95, 0.0)
        return cls(
            scale_name    = scale_name,
            scale_value   = scale_value,
            transform     = transform,
            mean          = float(e.mean()),
            std           = float(e.std()),
            min           = float(e.min()),
            max           = float(e.max()),
            frac_lt_05    = frac_lt_05,
            frac_gt_95    = frac_gt_95,
            frac_lt_10    = frac_lt_10,
            frac_gt_90    = frac_gt_90,
            collapse_score= collapse,
        )

    def is_acceptable(
        self,
        max_collapse: float = 0.70,
        min_std: float = 0.01,
    ) -> bool:
        """Return True if E is not collapsed and has meaningful variation."""
        return (
            self.collapse_score < max_collapse
            and self.std > min_std
            and self.frac_lt_05 < 0.85
            and self.frac_gt_95 < 0.85
        )

    def score(self) -> float:
        """
        Scalar quality score (higher = better).
        Rewards high std, penalises collapse into tails.
        """
        return self.std - 2.0 * self.collapse_score


# ──────────────────────────────────────────────────────────────────────────────
# Transform functions
# ──────────────────────────────────────────────────────────────────────────────

def transform_exponential(spread: pd.Series, scale: float) -> pd.Series:
    """
    E = exp(-spread / scale)

    Range: (0, 1] when spread ≥ 0.
    At spread=0      → E = 1.0  (zero congestion = full fluidity)
    At spread=scale  → E = 1/e ≈ 0.368
    At spread=3×scale → E ≈ 0.05

    Sensitive to scale choice.  Requires scale >> 0.
    """
    if scale <= 0:
        raise ValueError(f"exponential transform requires scale > 0, got {scale}")
    return np.exp(-spread / scale).clip(0, 1).rename("E")


def transform_rational(spread: pd.Series, scale: float) -> pd.Series:
    """
    E = 1 / (1 + spread / scale)  =  scale / (scale + spread)

    Range: (0, 1].
    At spread=0     → E = 1.0
    At spread=scale → E = 0.5   (half-life point at the scale)
    At spread=9×sc  → E = 0.1

    Heavier tail than exponential — less sensitive to scale choice.
    Preferred when spread has rare large spikes.
    """
    if scale <= 0:
        raise ValueError(f"rational transform requires scale > 0, got {scale}")
    return (scale / (scale + spread.clip(lower=0))).clip(0, 1).rename("E")


def transform_minmax_inv(spread: pd.Series, scale: float = 1.0) -> pd.Series:
    """
    E = 1 - (spread - spread.min()) / (spread.max() - spread.min())

    Range: [0, 1].  Does not require scale parameter (scale arg ignored).
    Preserves linear ranking perfectly.
    Sensitive to outliers (max/min are extremes).

    Use as a baseline / sanity check against other transforms.
    """
    lo, hi = spread.min(), spread.max()
    if hi == lo:
        return pd.Series(0.5, index=spread.index, name="E")
    return (1.0 - (spread - lo) / (hi - lo)).clip(0, 1).rename("E")


def transform_logistic(
    spread: pd.Series,
    scale: float = 1.0,
    midpoint: float | None = None,
    slope: float | None = None,
) -> pd.Series:
    """
    E = 1 / (1 + exp((spread - midpoint) / slope))

    Sigmoid centered at `midpoint` with steepness `slope`.
    At spread = midpoint → E = 0.5
    Smaller slope → steeper transition.

    Parameters
    ----------
    scale   : used as slope if slope is None  (convenience)
    midpoint: default = median of spread (center the sigmoid at the median)
    slope   : default = scale
    """
    if midpoint is None:
        midpoint = float(spread.median())
    if slope is None:
        slope = scale
    if slope <= 0:
        slope = 1.0
    arg = (spread - midpoint) / slope
    return (1.0 / (1.0 + np.exp(arg))).clip(0, 1).rename("E")


# Registry
TRANSFORMS: dict[str, Callable] = {
    "exponential": transform_exponential,
    "rational":    transform_rational,
    "minmax_inv":  transform_minmax_inv,
    "logistic":    transform_logistic,
}


# ──────────────────────────────────────────────────────────────────────────────
# Scale candidates
# ──────────────────────────────────────────────────────────────────────────────

def candidate_scales(stats: SpreadStats) -> dict[str, float]:
    """
    Return a dict of scale candidate names → values derived from
    robust statistics of the spread distribution.

    Prefers measures from the ACTIVE (nonzero) part of the distribution
    for zero-inflated signals.

    All candidates are filtered to be > 0 before returning.
    """
    candidates: dict[str, float] = {}

    # Global spread measures
    if stats.std > 0:
        candidates["std"]     = stats.std
    if stats.iqr > 0:
        candidates["iqr"]     = stats.iqr
    if stats.mad > 0:
        candidates["mad"]     = stats.mad
    if stats.p75 > 0:
        candidates["p75"]     = stats.p75
    if stats.p90 > 0:
        candidates["p90"]     = stats.p90
    if stats.p95 > 0:
        candidates["p95"]     = stats.p95
    if stats.mean > 0:
        candidates["mean"]    = stats.mean
    if stats.median > 0:
        candidates["median"]  = stats.median

    # Nonzero-subset measures (important for zero-inflated signals like congestion)
    if not np.isnan(stats.nz_median) and stats.nz_median > 0:
        candidates["nz_median"] = stats.nz_median
    if not np.isnan(stats.nz_mean) and stats.nz_mean > 0:
        candidates["nz_mean"]   = stats.nz_mean
    if not np.isnan(stats.nz_std) and stats.nz_std > 0:
        candidates["nz_std"]    = stats.nz_std
    if not np.isnan(stats.nz_p75) and stats.nz_p75 > 0:
        candidates["nz_p75"]    = stats.nz_p75

    # Half-max: scale at which exp(-1) = 0.368  →  set scale = median of nonzero
    # or fall back to range / 4
    half_max = stats.nz_median if (not np.isnan(stats.nz_median) and stats.nz_median > 0) \
               else (stats.max - stats.min) / 4.0
    if half_max > 0:
        candidates["half_max"] = half_max

    return {k: float(v) for k, v in candidates.items() if v > 0}


# ──────────────────────────────────────────────────────────────────────────────
# Calibration engine
# ──────────────────────────────────────────────────────────────────────────────

class SpreadCalibrator:
    """
    Evaluates all (transform × scale) combinations and selects the best.

    Usage
    -----
    >>> cal = SpreadCalibrator(transforms=["exponential","rational","logistic"])
    >>> result = cal.calibrate(spread_series, name="lmp_congestion")
    >>> best_E = result.best_E
    >>> result.print_report()
    """

    def __init__(
        self,
        transforms: list[TransformName] | None = None,
        max_collapse: float = 0.70,
        min_std: float = 0.01,
    ):
        self.transforms   = transforms or ["exponential", "rational", "minmax_inv", "logistic"]
        self.max_collapse = max_collapse
        self.min_std      = min_std

    def calibrate(
        self,
        spread: pd.Series,
        name: str = "spread",
        verbose: bool = True,
    ) -> "CalibrationResult":
        """
        Run full calibration pipeline.

        Returns CalibrationResult with .best_E, .quality_table, .stats.
        """
        # 1. Compute summary statistics
        stats = SpreadStats.compute(spread)
        if verbose:
            stats.log_summary(name)

        # 2. Get candidate scales
        scales = candidate_scales(stats)
        if not scales:
            log.warning("[calibration] no valid scales found for '%s'; using std=1.0", name)
            scales = {"fallback": 1.0}

        if verbose:
            log.info("[calibration] %d scale candidates: %s",
                     len(scales),
                     {k: f"{v:.3f}" for k, v in scales.items()})

        # 3. Evaluate all (transform × scale) combinations
        records: list[EDistributionQuality] = []
        E_store: dict[tuple, pd.Series] = {}

        spread_clipped = spread.clip(lower=0)

        for t_name in self.transforms:
            fn = TRANSFORMS[t_name]
            if t_name == "minmax_inv":
                # scale-independent: evaluate once
                E = fn(spread_clipped)
                q = EDistributionQuality.compute(E, "n/a", 0.0, t_name)
                records.append(q)
                E_store[(t_name, "n/a")] = E
            elif t_name == "logistic":
                # logistic uses slope derived from each scale and midpoint=median
                for s_name, s_val in scales.items():
                    E = fn(spread_clipped, scale=s_val)
                    q = EDistributionQuality.compute(E, s_name, s_val, t_name)
                    records.append(q)
                    E_store[(t_name, s_name)] = E
            else:
                for s_name, s_val in scales.items():
                    E = fn(spread_clipped, scale=s_val)
                    q = EDistributionQuality.compute(E, s_name, s_val, t_name)
                    records.append(q)
                    E_store[(t_name, s_name)] = E

        # 4. Rank by score
        records.sort(key=lambda r: -r.score())

        # 5. Select best
        acceptable = [r for r in records if r.is_acceptable(self.max_collapse, self.min_std)]
        best_record = acceptable[0] if acceptable else records[0]

        best_key = (best_record.transform, best_record.scale_name)
        best_E   = E_store[best_key].rename(name + "_E")

        if verbose:
            log.info(
                "[calibration] BEST → transform=%s  scale=%s (%.4f)  "
                "mean=%.3f  std=%.3f  frac<0.05=%.3f  frac>0.95=%.3f  score=%.4f",
                best_record.transform, best_record.scale_name, best_record.scale_value,
                best_record.mean, best_record.std,
                best_record.frac_lt_05, best_record.frac_gt_95, best_record.score(),
            )

        return CalibrationResult(
            name      = name,
            stats     = stats,
            scales    = scales,
            records   = records,
            E_store   = E_store,
            best      = best_record,
            best_E    = best_E,
        )


@dataclass
class CalibrationResult:
    """Holds all calibration outputs for a single spread series."""
    name:    str
    stats:   SpreadStats
    scales:  dict[str, float]
    records: list[EDistributionQuality]
    E_store: dict[tuple, pd.Series]
    best:    EDistributionQuality
    best_E:  pd.Series

    @property
    def quality_table(self) -> pd.DataFrame:
        """All (transform × scale) evaluations as a sortable DataFrame."""
        return pd.DataFrame([
            {
                "transform":     r.transform,
                "scale_name":    r.scale_name,
                "scale_value":   r.scale_value,
                "mean":          r.mean,
                "std":           r.std,
                "frac_lt_05":    r.frac_lt_05,
                "frac_gt_95":    r.frac_gt_95,
                "collapse_score":r.collapse_score,
                "score":         r.score(),
                "acceptable":    r.is_acceptable(),
            }
            for r in self.records
        ]).sort_values("score", ascending=False).reset_index(drop=True)

    def get_E(self, transform: str, scale_name: str) -> pd.Series:
        """Retrieve a specific E series by transform and scale name."""
        key = (transform, scale_name)
        if key not in self.E_store:
            raise KeyError(f"No E stored for {key}.  Available: {list(self.E_store)}")
        return self.E_store[key]

    def print_report(self) -> None:
        """Print a human-readable calibration report."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  SPREAD CALIBRATION REPORT – {self.name}")
        print(f"{sep}")
        print(f"\nSpread statistics:")
        print(f"  n={self.stats.n}  nonzero={self.stats.frac_nonzero:.1%}")
        print(f"  min={self.stats.min:.4f}  max={self.stats.max:.4f}")
        print(f"  mean={self.stats.mean:.4f}  std={self.stats.std:.4f}")
        print(f"  median={self.stats.median:.4f}  MAD={self.stats.mad:.4f}  IQR={self.stats.iqr:.4f}")
        print(f"  p75={self.stats.p75:.4f}  p90={self.stats.p90:.4f}  p95={self.stats.p95:.4f}")
        if self.stats.frac_nonzero < 1.0:
            print(f"  [nonzero subset] median={self.stats.nz_median:.4f}  "
                  f"mean={self.stats.nz_mean:.4f}  std={self.stats.nz_std:.4f}  "
                  f"p75={self.stats.nz_p75:.4f}")
        print(f"\nTop-5 calibration results (by score):")
        tbl = self.quality_table
        cols = ["transform","scale_name","scale_value","mean","std",
                "frac_lt_05","frac_gt_95","score","acceptable"]
        print(tbl[cols].head(5).to_string(index=False, float_format="%.4f"))
        print(f"\nSELECTED: transform={self.best.transform}  "
              f"scale={self.best.scale_name} ({self.best.scale_value:.4f})")
        print(f"  E: mean={self.best.mean:.4f}  std={self.best.std:.4f}  "
              f"min={self.best.min:.4f}  max={self.best.max:.4f}")
        print(f"  frac<0.05={self.best.frac_lt_05:.3f}  "
              f"frac>0.95={self.best.frac_gt_95:.3f}")
        print(f"{sep}\n")
