"""
scripts/run_divergence_analysis.py
====================================
Full divergence analysis pipeline.

Tasks
-----
1. Build divergence metrics (D1–D5, BP1–BP3)
2. Predictive model comparison (OLS + logistic, A–E)
3. Regime-stratified analysis
4. Bounded-price diagnostics
5. Report generation

Usage
-----
  python scripts/run_divergence_analysis.py
  python scripts/run_divergence_analysis.py --window 48 --best-metric D2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constraint_field import load_config
from constraint_field.adapters.synthetic import SyntheticAdapter
from constraint_field.field import FieldBuilder, run_static_analysis
from constraint_field.analysis import (
    compute_divergence_metrics,
    divergence_summary,
    run_model_comparison,
    lead_lag_correlation,
    regime_divergence_summary,
    bounded_price_diagnostics,
    plot_divergence_timeseries,
    plot_lead_lag,
    plot_regime_divergence,
    plot_bounded_price,
    plot_model_comparison,
    plot_divergence_dashboard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("divergence_analysis")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="config/default.yaml")
    p.add_argument("--window",      type=int, default=24,
                   help="Rolling window for D4/BP2 (hours)")
    p.add_argument("--best-metric", default=None,
                   help="Override best divergence metric for Model D/E (default: auto)")
    p.add_argument("--train-frac",  type=float, default=0.70)
    p.add_argument("--out-dir",     default="outputs")
    return p.parse_args()


def select_best_metric(summary: pd.DataFrame) -> str:
    """
    Select the divergence metric with highest absolute correlation
    with instability (excluding binary metrics BP1).
    """
    candidates = summary[
        ~summary.index.isin(["BP1", "BP2", "BP3", "D5_pos", "D5_neg"])
    ]
    return candidates["corr_instability"].abs().idxmax()


def build_report(
    df: pd.DataFrame,
    div_summary: pd.DataFrame,
    model_results: dict,
    regime_summary: pd.DataFrame,
    bp_diag: dict,
    best_metric: str,
) -> str:
    """
    Compose the full text findings report.
    """
    ct  = model_results["comparison_table"]
    ll  = model_results["lead_lag"]
    ols = model_results["ols"]
    logit = model_results["logit"]

    # Best lead lag for D1
    best_lead = {}
    for m, ll_df in ll.items():
        lead_only = ll_df[ll_df["lag_hours"] > 0]
        if not lead_only.empty:
            best_row = lead_only.loc[lead_only["correlation"].idxmax()]
            best_lead[m] = best_row

    # Bounded price test
    mw = bp_diag.get("mannwhitney_test", {})
    corrs = bp_diag.get("correlations_with_instability", {})
    segs  = bp_diag.get("segment_stats", {})

    lines = [
        "=" * 68,
        "  FIELD DIVERGENCE ANALYSIS — FINDINGS REPORT",
        "=" * 68,
        "",
        "CORE QUESTION",
        "  Does the gap between field pressure (S) and price expression (R)",
        "  contain information about instability beyond what R alone carries?",
        "",
        "─" * 68,
        "1. DIVERGENCE METRIC SUMMARY",
        "─" * 68,
        "",
        div_summary[["mean","std","corr_instability"]].to_string(float_format="%.4f"),
        "",
        f"  Best single metric by |corr_with_instability|: {best_metric}",
        "",
        "─" * 68,
        "2. PREDICTIVE MODEL COMPARISON",
        "─" * 68,
        "",
        ct[["train_R²","test_R²","AUC","F1"]].to_string(float_format="%.4f"),
        "",
    ]

    # Interpret model comparison
    r_test_r2   = ct.loc["A: R only", "test_R²"]        if "A: R only" in ct.index else np.nan
    d1_test_r2  = ct.loc["B: D1 only", "test_R²"]       if "B: D1 only" in ct.index else np.nan
    c_test_r2   = ct.loc["C: R + D1", "test_R²"]        if "C: R + D1" in ct.index else np.nan
    e_label     = f"E: R + {best_metric}"
    e_test_r2   = ct.loc[e_label, "test_R²"]            if e_label in ct.index else np.nan

    lines += [
        f"  Model A (R only)    test R² = {r_test_r2:.4f}",
        f"  Model B (D1 only)   test R² = {d1_test_r2:.4f}",
        f"  Model C (R + D1)    test R² = {c_test_r2:.4f}",
        f"  Model E (R + {best_metric}) test R² = {e_test_r2:.4f}",
        "",
    ]

    if not np.isnan(c_test_r2) and not np.isnan(r_test_r2):
        gain = c_test_r2 - r_test_r2
        lines.append(
            f"  Adding D1 to R improves test R² by {gain:+.4f} "
            f"({'positive gain' if gain > 0 else 'no gain'})."
        )
    lines.append("")

    # AUC comparison
    r_auc = logit.get("A: R only", {}).get("auc", np.nan)
    c_auc = logit.get("C: R + D1", {}).get("auc", np.nan)
    lines += [
        f"  Classification AUC — R only: {r_auc:.4f}",
        f"  Classification AUC — R + D1: {c_auc:.4f}",
        "",
    ]

    lines += [
        "─" * 68,
        "3. LEAD/LAG ANALYSIS",
        "─" * 68,
        "",
    ]
    for m, row in best_lead.items():
        lines.append(
            f"  {m}: best lead = {int(row['lag_hours'])}h ahead  "
            f"corr={row['correlation']:.4f}  p={row['pvalue']:.4f}"
        )
    lines.append("")
    lines.append(
        "  Positive lead (metric measured before instability peaks) indicates"
        " anticipatory rather than concurrent signal."
    )
    lines.append("")

    lines += [
        "─" * 68,
        "4. REGIME ANALYSIS",
        "─" * 68,
        "",
        regime_summary[
            ["regime_label","n","mean_instability","frac_high_instability","frac_BP1"]
        ].to_string(float_format="%.4f"),
        "",
    ]

    lines += [
        "─" * 68,
        "5. BOUNDED-PRICE DIAGNOSTICS",
        "─" * 68,
        "",
        "  Correlations with instability index:",
    ]
    for col, c in corrs.items():
        lines.append(f"    {col:12s}: {c:+.4f}")
    lines.append("")
    lines.append("  Segment: high |Phi| + moderate R vs low |Phi| + moderate R")
    for seg in ["high_phi_low_R", "low_phi_low_R"]:
        s = segs.get(seg, {})
        lines.append(
            f"    {seg:22s}: n={s.get('n','?'):4}  "
            f"mean_inst={s.get('mean_instability', np.nan):.4f}  "
            f"frac_high={s.get('frac_high_instability', np.nan):.3f}"
        )
    if mw:
        lines += [
            "",
            f"  Mann-Whitney U test (high_phi_low_R > low_phi_low_R instability):",
            f"    p-value = {mw['pvalue']:.6f}  "
            f"{'SIGNIFICANT at α=0.05' if mw['significant_at_05'] else 'not significant'}",
            f"    {mw['interpretation']}",
        ]
    lines.append("")

    lines += [
        "─" * 68,
        "6. CONCLUSIONS",
        "─" * 68,
        "",
    ]

    # Conclusion 1: does divergence add info beyond R?
    adds_info = (not np.isnan(c_test_r2) and not np.isnan(r_test_r2)
                 and c_test_r2 > r_test_r2 + 0.005)
    lines.append(
        f"  Q1 — Does divergence add information beyond R?\n"
        f"       {'YES' if adds_info else 'WEAK/NO'}: "
        f"Model C (R + |Phi|) achieves test R²={c_test_r2:.4f} "
        f"vs Model A (R only) R²={r_test_r2:.4f}."
    )
    lines.append("")

    # Conclusion 2: does divergence lead instability?
    d1_lead = best_lead.get("D1", {})
    leads   = d1_lead.get("lag_hours", 0) > 0 and d1_lead.get("pvalue", 1) < 0.05
    lines.append(
        f"  Q2 — Does divergence lead instability?\n"
        f"       {'YES' if leads else 'INCONCLUSIVE'}: "
        f"D1 peak lead = {d1_lead.get('lag_hours','?')}h "
        f"(p={d1_lead.get('pvalue', np.nan):.4f})."
    )
    lines.append("")

    # Conclusion 3: price as partial signal
    bp_sig = mw and mw.get("significant_at_05", False)
    d2_corr = corrs.get("D2", 0)
    r_corr  = corrs.get("R", 0)
    lines.append(
        f"  Q3 — Is there evidence price only partially expresses field pressure?\n"
        f"       {'YES' if bp_sig else 'SUGGESTIVE'}: "
        f"At moderate R levels, high-|Phi| periods have "
        f"{'significantly' if bp_sig else 'noticeably'} higher instability.\n"
        f"       D2 (imbalance/(1+|R|)) corr with instability = {d2_corr:+.4f} "
        f"vs R corr = {r_corr:+.4f}.\n"
        f"       D2 specifically captures the bounded-price signature."
    )
    lines.append("")
    lines.append("=" * 68)

    return "\n".join(lines)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    out_fig = Path(args.out_dir) / "figures"
    out_dat = Path(args.out_dir) / "data"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_dat.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Field Divergence Analysis")
    log.info("=" * 60)

    # ── Build base panel ──────────────────────────────────────────────────
    log.info("[1/6] Building field panel …")
    synth = SyntheticAdapter(cache_dir=cfg["data"]["cache_dir"])
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    demand_df = synth.fetch(start, end)
    price_df  = synth.fetch_prices(start, end)
    flows_df  = synth.fetch_flows(start, end)

    builder = FieldBuilder(cfg["field"])
    panel   = builder.build(demand_df, price_df, flows_df)
    panel   = run_static_analysis(panel, cfg.get("analysis", {}))
    log.info("  Panel: %d rows, %d cols", *panel.shape)

    # ── Divergence metrics ────────────────────────────────────────────────
    log.info("[2/6] Computing divergence metrics …")
    panel = compute_divergence_metrics(panel, window=args.window)
    div_sum = divergence_summary(panel)
    log.info("\n%s", div_sum.to_string(float_format="%.4f"))
    div_sum.to_csv(out_dat / "divergence_summary.csv")

    # Select best metric
    best_metric = args.best_metric or select_best_metric(div_sum)
    log.info("  Best divergence metric: %s (|corr|=%.4f)",
             best_metric, abs(div_sum.loc[best_metric, "corr_instability"]))

    # ── Model comparison ──────────────────────────────────────────────────
    log.info("[3/6] Running model comparison …")

    # Install statsmodels if needed
    try:
        import statsmodels.api as sm
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "statsmodels", "-q", "--break-system-packages"])

    model_results = run_model_comparison(
        panel,
        best_divergence=best_metric,
        train_frac=args.train_frac,
    )
    ct = model_results["comparison_table"]
    log.info("\n%s", ct.to_string(float_format="%.4f"))
    ct.to_csv(out_dat / "model_comparison.csv")

    # ── Regime analysis ───────────────────────────────────────────────────
    log.info("[4/6] Regime analysis …")
    regime_sum = regime_divergence_summary(panel)
    log.info("\n%s", regime_sum.to_string(float_format="%.4f"))
    regime_sum.to_csv(out_dat / "regime_summary.csv")

    # ── Bounded-price diagnostics ─────────────────────────────────────────
    log.info("[5/6] Bounded-price diagnostics …")
    bp_diag = bounded_price_diagnostics(panel)
    log.info("  Correlations with instability: %s",
             {k: f"{v:.4f}" for k, v in
              bp_diag["correlations_with_instability"].items()})
    if bp_diag.get("mannwhitney_test"):
        mw = bp_diag["mannwhitney_test"]
        log.info("  Mann-Whitney: p=%.6f  significant=%s",
                 mw["pvalue"], mw["significant_at_05"])

    # ── Figures ───────────────────────────────────────────────────────────
    log.info("[6/6] Generating figures …")
    DPI = cfg["output"]["dpi"]

    # Figure 1: Divergence time-series
    fig = plot_divergence_timeseries(panel,
          metrics=["D1","D2","D3","D4","D5_pos","D5_neg"],
          title="Divergence Metrics vs Instability Index")
    fig.savefig(out_fig / "08_divergence_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 08_divergence_timeseries.png")

    # Figure 2: Lead/lag
    fig = plot_lead_lag(model_results["lead_lag"])
    fig.savefig(out_fig / "09_lead_lag.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 09_lead_lag.png")

    # Figure 3: Regime divergence
    fig = plot_regime_divergence(panel, regime_sum, metric=best_metric)
    fig.savefig(out_fig / "10_regime_divergence.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 10_regime_divergence.png")

    # Figure 4: Bounded-price scatter
    fig = plot_bounded_price(
        panel,
        phi_thresh=bp_diag["phi_thresholds"]["phi_high"],
        r_mod_thresh=bp_diag["phi_thresholds"]["r_mod"],
    )
    fig.savefig(out_fig / "11_bounded_price.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 11_bounded_price.png")

    # Figure 5: Model comparison bars
    fig = plot_model_comparison(ct)
    fig.savefig(out_fig / "12_model_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 12_model_comparison.png")

    # Figure 6: Full dashboard
    fig = plot_divergence_dashboard(panel, model_results, regime_sum, bp_diag)
    fig.savefig(out_fig / "13_divergence_dashboard.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved 13_divergence_dashboard.png")

    # ── Text report ───────────────────────────────────────────────────────
    report = build_report(
        panel, div_sum, model_results, regime_sum, bp_diag, best_metric
    )
    report_path = out_dat / "divergence_report.txt"
    report_path.write_text(report)
    print("\n" + report)
    log.info("  Saved %s", report_path)
    log.info("✓  Divergence analysis complete.  Outputs in %s/", args.out_dir)


if __name__ == "__main__":
    main()
