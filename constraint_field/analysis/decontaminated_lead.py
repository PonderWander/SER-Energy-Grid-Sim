"""
constraint_field.analysis.decontaminated_lead
===============================================
Clean lead/lag analysis free of rolling-window overlap contamination.

The problem with the original lead analysis
-------------------------------------------
Both D4 (24h rolling mean of |Phi|) and the instability index
(24h rolling 75th-percentile of |Phi|) are rolling summaries of the
same underlying series. Their apparent R² of 0.76 and 20-hour lead
are mechanical: any value of |Phi| at time t is included in both the
D4 window and the instability window for multiple future hours.

Decontamination strategy
------------------------
1. Use point-in-time instability targets that share no window with any
   predictor:
     I1 = |Phi_t|                        (instantaneous, no window)
     I2 = 3h rolling mean of |Phi|       (short, non-overlapping for lags>3)
     I3 = binary: |Phi_t| > p90          (threshold event, point-in-time)
     I4 = binary: Psi_t > p90            (orthogonal stress measure)

2. For lead-lag tests, enforce a *gap* between predictor window end
   and target window start equal to the lag, so no shared observations
   can exist.

3. Residualise both predictor and target on hour-of-day and
   day-of-week dummies before testing, removing regular timing
   structure that could create spurious leads.

4. Event-study: define divergence shocks, then trace the *future*
   instability path (point-in-time I1) forward in time, compared to
   matched control periods with similar time-of-day.

5. Regime-stratified tests: repeat within each cluster so that
   cross-regime composition effects don't inflate the signal.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Point-in-time instability targets
# ──────────────────────────────────────────────────────────────────────────────

def build_pit_targets(
    df: pd.DataFrame,
    short_window: int = 3,
    threshold_pct: float = 90.0,
) -> pd.DataFrame:
    """
    Build point-in-time instability targets with no 24h rolling window.

    I1: |Phi_t|                    — instantaneous, completely clean
    I2: rolling mean |Phi|, w=3    — short smoothing only
    I3: 1{|Phi_t| > p90}          — binary spike indicator
    I4: 1{Psi_t > p90}            — orthogonal field-intensity spike

    Parameters
    ----------
    df : pd.DataFrame  with Phi, Psi already computed
    short_window : int  window for I2 (must be << 24 to avoid overlap)
    threshold_pct : float  percentile for I3/I4 thresholds

    Returns
    -------
    df with I1, I2, I3, I4 added
    """
    assert short_window < 12, "short_window must be well below 24h to avoid overlap"

    phi_abs = df["Phi"].abs()
    psi     = df["Psi"]

    df = df.copy()
    df["I1"] = phi_abs                                         # instantaneous
    df["I2"] = phi_abs.rolling(short_window, min_periods=1).mean()  # 3h smooth
    df["I3"] = (phi_abs > phi_abs.quantile(threshold_pct / 100)).astype(int)
    df["I4"] = (psi     > psi.quantile(threshold_pct / 100)).astype(int)

    log.info(
        "PIT targets built: I3 threshold=%.3f (%.0fth pct)  "
        "I4 threshold=%.3f  I3 base rate=%.1f%%  I4 base rate=%.1f%%",
        phi_abs.quantile(threshold_pct / 100), threshold_pct,
        psi.quantile(threshold_pct / 100),
        df["I3"].mean() * 100, df["I4"].mean() * 100,
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Decontaminated lead/lag correlation
# ──────────────────────────────────────────────────────────────────────────────

def clean_lead_lag(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    max_lead: int = 24,
    min_gap: int = 0,
    method: Literal["pearson", "spearman", "pointbiserial"] = "pearson",
) -> pd.DataFrame:
    """
    Compute lead correlations between predictor at t and target at t+lag,
    where lag >= min_gap ensures no window overlap.

    For point-in-time targets (I1, I3, I4) min_gap should be 0.
    For I2 (3h window), min_gap should be >= 3 to avoid overlap.

    Parameters
    ----------
    predictor : str  column name (D1, D2, etc.)
    target    : str  column name (I1, I2, I3, I4)
    max_lead  : int  maximum lead in hours
    min_gap   : int  minimum lag before testing begins (contamination gap)
    method    : correlation method

    Returns
    -------
    pd.DataFrame with columns: lag, corr, pvalue, n, method
    """
    x = df[predictor].values
    y = df[target].values
    n = len(x)

    records = []
    for lag in range(min_gap, max_lead + 1):
        # predictor at t, target at t+lag
        xi = x[:n - lag] if lag > 0 else x
        yi = y[lag:]     if lag > 0 else y
        assert len(xi) == len(yi)

        valid = ~(np.isnan(xi) | np.isnan(yi))
        xi, yi = xi[valid], yi[valid]
        if len(xi) < 30:
            continue

        if method == "spearman":
            r, p = scipy_stats.spearmanr(xi, yi)
        elif method == "pointbiserial":
            r, p = scipy_stats.pointbiserialr(yi.astype(int), xi)
        else:
            r, p = scipy_stats.pearsonr(xi, yi)

        records.append({
            "lag":    lag,
            "corr":   float(r),
            "pvalue": float(p),
            "n":      int(len(xi)),
            "method": method,
        })

    return pd.DataFrame(records)


def best_lead(ll_df: pd.DataFrame, min_lag: int = 1) -> dict:
    """
    Return the lag with the maximum |correlation| at lag >= min_lag.
    """
    sub = ll_df[ll_df["lag"] >= min_lag].copy()
    if sub.empty:
        return {}
    idx = sub["corr"].abs().idxmax()
    row = sub.loc[idx]
    return {
        "best_lag":  int(row["lag"]),
        "corr":      float(row["corr"]),
        "pvalue":    float(row["pvalue"]),
        "n":         int(row["n"]),
        "sig_05":    row["pvalue"] < 0.05,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Event-study analysis
# ──────────────────────────────────────────────────────────────────────────────

def event_study(
    df: pd.DataFrame,
    event_col: str,
    target_col: str = "I1",
    window_before: int = 12,
    window_after:  int = 24,
    min_gap_between_events: int = 12,
    n_control_draws: int = 3,
    seed: int = 42,
) -> dict:
    """
    Average instability path around divergence shock events.

    For each event (top-decile D1, top-decile D2, or BP1 hour):
      - extract target from t-window_before to t+window_after
      - average across events

    Control: randomly sampled non-event hours matched by hour-of-day,
    with the same path extracted and averaged.

    Returns
    -------
    dict with:
      event_path   : pd.Series  mean I1 path (index = relative hour)
      control_path : pd.Series  mean I1 path for controls
      n_events     : int
      n_controls   : int
      event_times  : list of timestamps
    """
    rng = np.random.default_rng(seed)

    # Identify event times (deduplicated with minimum gap)
    event_mask = df[event_col].astype(bool)
    event_idx  = df.index[event_mask].tolist()

    # Deduplicate: keep events separated by at least min_gap_between_events
    deduped = []
    last_t  = None
    for t in event_idx:
        if last_t is None or (t - last_t).total_seconds() / 3600 >= min_gap_between_events:
            deduped.append(t)
            last_t = t
    event_times = deduped

    # Collect event paths
    event_paths = []
    valid_events = []
    for t in event_times:
        try:
            loc     = df.index.get_loc(t)
            lo      = loc - window_before
            hi      = loc + window_after + 1
            if lo < 0 or hi > len(df):
                continue
            path = df[target_col].iloc[lo:hi].values
            if len(path) == window_before + window_after + 1:
                event_paths.append(path)
                valid_events.append(t)
        except Exception:
            continue

    if not event_paths:
        log.warning("event_study: no valid event windows for '%s'", event_col)
        return {}

    event_arr  = np.array(event_paths)
    event_mean = np.nanmean(event_arr, axis=0)
    event_se   = np.nanstd(event_arr, axis=0) / np.sqrt(len(event_paths))

    # Control: sample non-event hours matched on hour-of-day
    non_event_idx = df.index[~event_mask].tolist()
    event_hours   = set(t.hour for t in valid_events)
    candidates    = [t for t in non_event_idx if t.hour in event_hours]

    control_paths = []
    attempts      = 0
    max_attempts  = len(candidates)
    needed        = len(valid_events) * n_control_draws

    drawn = rng.choice(len(candidates),
                       size=min(needed * 2, len(candidates)),
                       replace=False)
    for idx_c in drawn:
        t = candidates[idx_c]
        try:
            loc = df.index.get_loc(t)
            lo  = loc - window_before
            hi  = loc + window_after + 1
            if lo < 0 or hi > len(df):
                continue
            path = df[target_col].iloc[lo:hi].values
            if len(path) == window_before + window_after + 1:
                control_paths.append(path)
        except Exception:
            continue
        if len(control_paths) >= needed:
            break

    control_arr  = np.array(control_paths) if control_paths else np.zeros_like(event_arr)
    control_mean = np.nanmean(control_arr, axis=0)
    control_se   = np.nanstd(control_arr, axis=0) / np.sqrt(max(len(control_paths), 1))

    rel_hours = np.arange(-window_before, window_after + 1)

    return {
        "event_path":    pd.Series(event_mean, index=rel_hours),
        "event_se":      pd.Series(event_se,   index=rel_hours),
        "control_path":  pd.Series(control_mean, index=rel_hours),
        "control_se":    pd.Series(control_se,   index=rel_hours),
        "n_events":      len(valid_events),
        "n_controls":    len(control_paths),
        "event_col":     event_col,
        "target_col":    target_col,
        "event_times":   valid_events,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Time-of-day residualisation
# ──────────────────────────────────────────────────────────────────────────────

def residualise_on_tod(
    df: pd.DataFrame,
    columns: list[str],
    dummies: list[str] = ("hour", "dayofweek"),
) -> pd.DataFrame:
    """
    Remove regular time-of-day and day-of-week structure from columns
    by OLS projection onto dummies, returning residuals.

    This ensures that any remaining correlation between predictors and
    targets is not explainable by shared diurnal / weekly patterns.

    Parameters
    ----------
    columns : list of column names to residualise
    dummies : which time features to use as controls

    Returns
    -------
    df with new columns: {col}_resid for each col in columns
    """
    tod = pd.DataFrame(index=df.index)
    if "hour" in dummies:
        tod = tod.join(pd.get_dummies(df.index.hour, prefix="h", drop_first=True))
    if "dayofweek" in dummies:
        tod = tod.join(pd.get_dummies(df.index.dayofweek, prefix="dow", drop_first=True))

    X = tod.values.astype(float)
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])

    out = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        y = df[col].fillna(df[col].mean()).values
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
        except np.linalg.LinAlgError:
            resid = y - y.mean()
        out[f"{col}_resid"] = resid

    log.info(
        "Residualised %d columns on %s dummies",
        len(columns), list(dummies),
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. Bounded-price test (clean, point-in-time)
# ──────────────────────────────────────────────────────────────────────────────

def bounded_price_test_clean(
    df: pd.DataFrame,
    phi_pct: float = 75.0,
    r_pct:   float = 50.0,
    target:  str   = "I1",
    use_resid: bool = False,
) -> dict:
    """
    Mann-Whitney test of bounded-price effect using point-in-time target.

    Optionally use residualised columns to control for time-of-day.

    Parameters
    ----------
    phi_pct : percentile threshold for 'high imbalance'
    r_pct   : percentile threshold for 'moderate price'
    target  : 'I1' (instantaneous) or 'I1_resid' (after ToD removal)
    use_resid : if True, use D1_resid and R_resid for segmentation

    Returns
    -------
    dict with segment stats, Mann-Whitney result, and effect size
    """
    d1_col = "D1_resid" if use_resid and "D1_resid" in df.columns else "D1"
    r_col  = "R_resid"  if use_resid and "R_resid"  in df.columns else "R"
    t_col  = f"{target}_resid" if use_resid and f"{target}_resid" in df.columns else target

    phi_thresh = df[d1_col].quantile(phi_pct / 100)
    r_thresh   = df[r_col].abs().quantile(r_pct / 100)

    high_phi = df[d1_col] > phi_thresh
    mod_r    = df[r_col].abs() < r_thresh

    grp_A = df.loc[high_phi & mod_r,   t_col].dropna()   # bounded-price
    grp_B = df.loc[~high_phi & mod_r,  t_col].dropna()   # low-phi, same mod-R
    grp_C = df.loc[high_phi & ~mod_r,  t_col].dropna()   # high phi, high R
    grp_D = df.loc[~high_phi & ~mod_r, t_col].dropna()   # baseline

    def _seg_stats(g, label):
        if len(g) == 0:
            return {"label": label, "n": 0}
        return {
            "label":  label,
            "n":      len(g),
            "mean":   float(g.mean()),
            "median": float(g.median()),
            "std":    float(g.std()),
        }

    segs = {
        "high_phi_mod_R":  _seg_stats(grp_A, "high |Phi|, mod R  [bounded-price]"),
        "low_phi_mod_R":   _seg_stats(grp_B, "low  |Phi|, mod R  [control]"),
        "high_phi_high_R": _seg_stats(grp_C, "high |Phi|, high R [expressed]"),
        "low_phi_high_R":  _seg_stats(grp_D, "low  |Phi|, high R [baseline]"),
    }

    # Primary test: A vs B — same moderate-R context, different imbalance
    mw_stat, mw_p = (np.nan, np.nan)
    cohens_d      = np.nan
    if len(grp_A) >= 10 and len(grp_B) >= 10:
        mw_stat, mw_p = scipy_stats.mannwhitneyu(
            grp_A, grp_B, alternative="greater"
        )
        # Effect size: rank-biserial correlation
        n1, n2     = len(grp_A), len(grp_B)
        rb         = 1 - 2 * mw_stat / (n1 * n2)
        # Cohen's d (approximate)
        pool_std   = np.sqrt((grp_A.std()**2 + grp_B.std()**2) / 2)
        cohens_d   = (grp_A.mean() - grp_B.mean()) / pool_std if pool_std > 0 else np.nan

    return {
        "segments":          segs,
        "mw_statistic":      float(mw_stat),
        "mw_pvalue":         float(mw_p),
        "rank_biserial":     float(rb) if not np.isnan(mw_stat) else np.nan,
        "cohens_d":          float(cohens_d),
        "significant_05":    bool(mw_p < 0.05) if not np.isnan(mw_p) else False,
        "target_used":       t_col,
        "d1_col_used":       d1_col,
        "thresholds":        {"phi": float(phi_thresh), "r_mod": float(r_thresh)},
        "use_resid":         use_resid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Regime-stratified tests
# ──────────────────────────────────────────────────────────────────────────────

def regime_stratified_tests(
    df: pd.DataFrame,
    predictors: list[str],
    target:     str = "I1",
    max_lead:   int = 12,
) -> pd.DataFrame:
    """
    Run bounded-price test and best lead within each cluster separately.

    Returns a DataFrame indexed by cluster with columns:
      n, bp_pvalue, bp_cohens_d, best_lead_D1, best_lead_corr_D1,
      best_lead_D2, best_lead_corr_D2, regime_label
    """
    rows = []
    for cluster in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster].copy()
        n   = len(sub)

        # Bounded-price test within cluster
        bp = bounded_price_test_clean(sub, target=target)

        row = {
            "cluster":      cluster,
            "n":            n,
            "bp_pvalue":    bp["mw_pvalue"],
            "bp_cohens_d":  bp["cohens_d"],
            "bp_sig":       bp["significant_05"],
        }

        # Best lead per predictor within cluster
        for pred in predictors:
            if pred not in sub.columns:
                continue
            if n < 50:
                row[f"best_lead_{pred}"] = np.nan
                row[f"lead_corr_{pred}"] = np.nan
                continue
            ll = clean_lead_lag(sub, pred, target, max_lead=max_lead)
            bl = best_lead(ll, min_lag=1)
            row[f"best_lead_{pred}"] = bl.get("best_lag",   np.nan)
            row[f"lead_corr_{pred}"] = bl.get("corr",       np.nan)
            row[f"lead_p_{pred}"]    = bl.get("pvalue",     np.nan)

        # Regime label
        mean_S = sub["S"].mean()
        mean_R = sub["R"].mean()
        if mean_S > 0 and mean_R > 0:
            row["regime_label"] = "demand+constraint"
        elif mean_S <= 0 and mean_R > 0:
            row["regime_label"] = "supply-constraint"
        elif mean_S > 0 and mean_R <= 0:
            row["regime_label"] = "demand-growth"
        else:
            row["regime_label"] = "slack"

        rows.append(row)

    return pd.DataFrame(rows).set_index("cluster")
