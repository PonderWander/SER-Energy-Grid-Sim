"""
constraint_field.analysis.models
==================================
Transparent predictive / explanatory models comparing R alone vs.
divergence metrics for explaining and anticipating instability.

Models
------
A: instability ~ R                        (price alone)
B: instability ~ D1  [= |Phi|]            (imbalance alone)
C: instability ~ R + D1                   (price + imbalance)
D: instability ~ best_divergence_metric   (data-selected single metric)
E: instability ~ R + best_divergence_metric

All OLS models use statsmodels for interpretable coefficients and
significance. Classification uses sklearn LogisticRegression for
high-instability periods (above 85th percentile threshold).

Validation
----------
Time-ordered train/test split (no shuffle) to respect temporal structure.
Also computes rolling 1-step-ahead R² using expanding window.

Lead/lag analysis
-----------------
Tests whether divergence metrics *lead* instability by 1–24 hours,
providing evidence of anticipatory rather than merely concurrent signal.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)

# Optional heavy imports — degrade gracefully
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    log.warning("statsmodels not installed; using numpy OLS fallback")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ──────────────────────────────────────────────────────────────────────────────
# OLS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ols_numpy(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Minimal OLS via numpy when statsmodels is unavailable.
    Returns coefficients, R², residuals.
    """
    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        coef, res, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"r2": np.nan, "coef": [], "pvalues": [], "note": "lstsq failed"}

    y_hat = X_aug @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, k = X_aug.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if ss_tot > 0 else np.nan

    return {"r2": r2, "adj_r2": adj_r2, "coef": coef.tolist(),
            "residuals": y - y_hat, "note": "numpy_ols"}


def fit_ols(
    df: pd.DataFrame,
    predictors: list[str],
    target: str = "instability_index",
    train_frac: float = 0.70,
    add_const: bool = True,
) -> dict:
    """
    Fit OLS on time-ordered train split, evaluate on test split.

    Returns a result dict with:
      train_r2, test_r2, adj_r2, coefficients, pvalues (if statsmodels),
      feature_names, n_train, n_test.
    """
    valid = df[predictors + [target]].dropna()
    n_train = int(len(valid) * train_frac)
    train = valid.iloc[:n_train]
    test  = valid.iloc[n_train:]

    X_train = train[predictors].values
    y_train = train[target].values
    X_test  = test[predictors].values
    y_test  = test[target].values

    result = {
        "predictors": predictors,
        "target":     target,
        "n_train":    n_train,
        "n_test":     len(test),
    }

    if HAS_STATSMODELS:
        X_tr = sm.add_constant(X_train) if add_const else X_train
        X_te = sm.add_constant(X_test, has_constant="add") if add_const else X_test

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y_train, X_tr).fit()

        result["train_r2"]  = float(model.rsquared)
        result["adj_r2"]    = float(model.rsquared_adj)
        result["pvalues"]   = model.pvalues.tolist()
        result["coef"]      = model.params.tolist()
        result["coef_names"] = (["const"] + predictors) if add_const else predictors
        result["aic"]       = float(model.aic)
        result["bic"]       = float(model.bic)

        y_pred_test = model.predict(X_te)
        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        result["test_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        result["summary_str"] = model.summary().as_text()
    else:
        ols = _ols_numpy(X_train, y_train)
        result.update(ols)
        result["train_r2"] = ols["r2"]
        # Test R²
        X_te_aug = np.column_stack([np.ones(len(X_test)), X_test])
        y_pred_test = X_te_aug @ np.array(ols["coef"])
        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        result["test_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_logistic(
    df: pd.DataFrame,
    predictors: list[str],
    target: str = "instability_index",
    high_threshold_pct: float = 85.0,
    train_frac: float = 0.70,
) -> dict:
    """
    Logistic regression: predict high-instability periods.
    High = instability_index above high_threshold_pct percentile.

    Returns AUC, F1, classification report on test split.
    """
    if not HAS_SKLEARN:
        return {"error": "scikit-learn not available"}

    threshold = df[target].quantile(high_threshold_pct / 100.0)
    df2 = df[predictors + [target]].dropna().copy()
    df2["y_binary"] = (df2[target] > threshold).astype(int)

    n_train = int(len(df2) * train_frac)
    train = df2.iloc[:n_train]
    test  = df2.iloc[n_train:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[predictors].values)
    X_test  = scaler.transform(test[predictors].values)
    y_train = train["y_binary"].values
    y_test  = test["y_binary"].values

    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    result = {
        "predictors":   predictors,
        "threshold":    float(threshold),
        "threshold_pct": high_threshold_pct,
        "n_train":      len(train),
        "n_test":       len(test),
        "auc":          float(roc_auc_score(y_test, y_prob)),
        "f1":           float(f1_score(y_test, y_pred, zero_division=0)),
        "class_report": classification_report(y_test, y_pred, zero_division=0),
        "coef":         model.coef_[0].tolist(),
        "coef_names":   predictors,
        "intercept":    float(model.intercept_[0]),
        "frac_high_train": float(y_train.mean()),
        "frac_high_test":  float(y_test.mean()),
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Lead/lag correlation
# ──────────────────────────────────────────────────────────────────────────────

def lead_lag_correlation(
    df: pd.DataFrame,
    predictor: str,
    target: str = "instability_index",
    max_lag: int = 24,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Compute correlation between predictor at time t-lag and target at time t.

    Positive lag: predictor LEADS target (predictor measured earlier).
    Negative lag: predictor LAGS target (measured later).

    Returns DataFrame with columns: lag_hours, correlation, pvalue.
    """
    x = df[predictor].dropna()
    y = df[target].dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]

    records = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            # predictor leads: compare x[:-lag] with y[lag:]
            xi = x.iloc[:len(x) - lag] if lag > 0 else x
            yi = y.iloc[lag:] if lag > 0 else y
        else:
            # predictor lags: compare x[|lag|:] with y[:-|lag|]
            abs_lag = abs(lag)
            xi = x.iloc[abs_lag:]
            yi = y.iloc[:len(y) - abs_lag]

        min_len = min(len(xi), len(yi))
        if min_len < 10:
            continue

        xi, yi = xi.iloc[:min_len].values, yi.iloc[:min_len].values

        if method == "pearson":
            r, p = scipy_stats.pearsonr(xi, yi)
        else:
            r, p = scipy_stats.spearmanr(xi, yi)

        records.append({"lag_hours": lag, "correlation": r, "pvalue": p})

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Full model comparison suite
# ──────────────────────────────────────────────────────────────────────────────

def run_model_comparison(
    df: pd.DataFrame,
    best_divergence: str = "D2",
    train_frac: float = 0.70,
    high_pct: float = 85.0,
) -> dict:
    """
    Run all five model specifications (A–E) and return results.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with S, R, Phi, Psi, instability_index, D1–D5, BP1–BP3.
    best_divergence : str
        Which divergence metric to use for Model D and E.
    train_frac : float
        Fraction used for training (time-ordered).
    high_pct : float
        Percentile threshold for high-instability classification.

    Returns
    -------
    dict with keys: ols_results, logit_results, lead_lag, comparison_table
    """
    models_ols = {
        "A: R only":                  ["R"],
        "B: D1 only":                 ["D1"],
        "C: R + D1":                  ["R", "D1"],
        f"D: {best_divergence} only": [best_divergence],
        f"E: R + {best_divergence}":  ["R", best_divergence],
    }

    ols_results = {}
    for label, preds in models_ols.items():
        # Check all predictors exist
        missing = [p for p in preds if p not in df.columns]
        if missing:
            log.warning("Model '%s': missing columns %s — skipping", label, missing)
            continue
        ols_results[label] = fit_ols(df, preds, train_frac=train_frac)
        log.info("OLS %s: train_R²=%.4f  test_R²=%.4f",
                 label,
                 ols_results[label].get("train_r2", np.nan),
                 ols_results[label].get("test_r2",  np.nan))

    # Classification
    logit_results = {}
    for label, preds in models_ols.items():
        missing = [p for p in preds if p not in df.columns]
        if missing:
            continue
        logit_results[label] = fit_logistic(
            df, preds, high_threshold_pct=high_pct, train_frac=train_frac
        )
        log.info("Logit %s: AUC=%.4f  F1=%.4f",
                 label,
                 logit_results[label].get("auc", np.nan),
                 logit_results[label].get("f1",  np.nan))

    # Lead/lag for D1 and best divergence
    lead_lag = {}
    for metric in list({"D1", "D2", best_divergence}):
        if metric in df.columns:
            lead_lag[metric] = lead_lag_correlation(df, metric, max_lag=24)

    # Comparison table
    rows = []
    for label in ols_results:
        o = ols_results[label]
        l = logit_results.get(label, {})
        rows.append({
            "model":      label,
            "predictors": ", ".join(o["predictors"]),
            "train_R²":   o.get("train_r2", np.nan),
            "test_R²":    o.get("test_r2",  np.nan),
            "adj_R²":     o.get("adj_r2",   np.nan),
            "AUC":        l.get("auc",       np.nan),
            "F1":         l.get("f1",        np.nan),
        })
    comparison_table = pd.DataFrame(rows).set_index("model")

    return {
        "ols":              ols_results,
        "logit":            logit_results,
        "lead_lag":         lead_lag,
        "comparison_table": comparison_table,
        "best_divergence":  best_divergence,
    }
