"""
scripts/run_demo.py
====================
End-to-end demonstration of the constraint-field research prototype.

Runs entirely on synthetic data (no network access needed).
Uses real EIA/CAISO adapters when --live flag is passed and
network is available.

Usage
-----
  python scripts/run_demo.py                # synthetic data, all defaults
  python scripts/run_demo.py --live         # attempt live EIA + CAISO fetch
  python scripts/run_demo.py --operator damped_wave
  python scripts/run_demo.py --start 2023-01-01 --end 2023-03-31

Output
------
  outputs/figures/   – all plots as PNG
  outputs/data/      – panel CSV and simulation results CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from constraint_field import load_config
from constraint_field.adapters import get_adapter, SyntheticAdapter
from constraint_field.field    import (
    FieldBuilder,
    run_static_analysis,
    summary_stress,
    plot_static_dashboard,
    plot_phase_portrait,
    plot_gradient_heatmap,
    plot_instability,
)
from constraint_field.inference import infer_all_E, get_E_inferrer
from constraint_field.dynamics  import Simulator, Shock
from constraint_field.compare   import (
    compute_comparison_metrics,
    shock_recovery_analysis,
    plot_simulation_comparison,
    plot_E_candidates,
    print_comparison_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_demo")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constraint field demo")
    p.add_argument("--config",   default="config/default.yaml")
    p.add_argument("--live",     action="store_true",
                   help="Attempt live data fetch (EIA + CAISO)")
    p.add_argument("--start",    default=None)
    p.add_argument("--end",      default=None)
    p.add_argument("--operator", default=None,
                   choices=["diffusion", "gradient_flow", "damped_wave"])
    p.add_argument("--out-dir",  default="outputs")
    p.add_argument("--no-shock", action="store_true",
                   help="Skip shock propagation experiment")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    start = args.start or cfg["data"]["start_date"]
    end   = args.end   or cfg["data"]["end_date"]

    out_fig = Path(args.out_dir) / "figures"
    out_dat = Path(args.out_dir) / "data"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_dat.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Constraint Field Research Prototype")
    log.info("  Region: %s  |  %s → %s", cfg["data"]["region"], start, end)
    log.info("=" * 60)

    # ── Step 1: Data ingestion ─────────────────────────────────────────────
    log.info("\n[1/6] Data ingestion …")

    if args.live:
        log.info("  Mode: LIVE (EIA + CAISO)")
        demand_adapter = get_adapter(
            "eia",
            region=cfg["data"]["region"],
            cache_dir=cfg["data"]["cache_dir"],
        )
        price_adapter = get_adapter(
            "caiso",
            cache_dir=cfg["data"]["cache_dir"],
            node_filter=cfg["data"]["caiso_oasis"]["node_filter"],
            market_run_id=cfg["data"]["caiso_oasis"]["market_run_id"],
        )
        try:
            demand_df = demand_adapter.fetch(start, end)
            price_df  = price_adapter.fetch(start, end)
            flows_df  = price_adapter.fetch_flows(start, end)
            log.info("  Live data fetched successfully.")
        except Exception as exc:
            log.warning("  Live fetch failed (%s). Falling back to synthetic.", exc)
            args.live = False

    if not args.live:
        log.info("  Mode: SYNTHETIC")
        synth = SyntheticAdapter(
            cache_dir=cfg["data"]["cache_dir"],
            peak_demand_mw=40_000,
            base_price=45.0,
            congestion_prob=0.025,
        )
        demand_df = synth.fetch(start, end)
        price_df  = synth.fetch_prices(start, end)
        flows_df  = synth.fetch_flows(start, end)

    log.info("  demand_df: %d rows, cols=%s", len(demand_df), list(demand_df.columns))
    log.info("  price_df : %d rows, cols=%s", len(price_df),  list(price_df.columns))
    log.info("  flows_df : %d rows, cols=%s", len(flows_df),  list(flows_df.columns))

    # ── Step 2: Build static field panel ───────────────────────────────────
    log.info("\n[2/6] Building static field panel (S, R) …")

    builder = FieldBuilder(cfg["field"])
    panel   = builder.build(demand_df, price_df, flows_df)

    log.info("  Panel: %d rows, columns=%s", len(panel), list(panel.columns))
    log.info("  S stats: mean=%.3f  std=%.3f", panel["S"].mean(), panel["S"].std())
    log.info("  R stats: mean=%.3f  std=%.3f", panel["R"].mean(), panel["R"].std())

    # Save raw panel
    panel.to_csv(out_dat / "field_panel.csv")
    log.info("  Saved: %s", out_dat / "field_panel.csv")

    # ── Step 3: Static field analysis ──────────────────────────────────────
    log.info("\n[3/6] Static field analysis …")

    panel = run_static_analysis(panel, cfg.get("analysis", {}))
    stress = summary_stress(panel)

    log.info("  Stress summary:\n%s", stress.to_string())
    stress.to_csv(out_dat / "stress_summary.csv")

    # Static visualisations
    fig_dash = plot_static_dashboard(panel, title=f"Static Field – {cfg['data']['region']}")
    fig_dash.savefig(out_fig / "01_static_dashboard.png", dpi=cfg["output"]["dpi"],
                     bbox_inches="tight")
    plt.close(fig_dash)
    log.info("  Saved: 01_static_dashboard.png")

    fig_phase = plot_phase_portrait(panel, color_by="cluster",
                                    title="S–R Phase Portrait (cluster coloured)")
    fig_phase.savefig(out_fig / "02_phase_portrait.png", dpi=cfg["output"]["dpi"],
                      bbox_inches="tight")
    plt.close(fig_phase)

    fig_grad = plot_gradient_heatmap(panel, title="Field Gradients ∂S/∂t and ∂R/∂t")
    fig_grad.savefig(out_fig / "03_gradient_heatmap.png", dpi=cfg["output"]["dpi"],
                     bbox_inches="tight")
    plt.close(fig_grad)

    fig_inst = plot_instability(panel)
    fig_inst.savefig(out_fig / "04_instability.png", dpi=cfg["output"]["dpi"],
                     bbox_inches="tight")
    plt.close(fig_inst)
    log.info("  Saved: 02–04 static analysis figures")

    # ── Step 4: E inference ────────────────────────────────────────────────
    log.info("\n[4/6] Inferring E (delivery fluidity) – all candidates …")

    E_all = infer_all_E(panel, cfg["inference"])
    E_all.to_csv(out_dat / "E_candidates.csv")

    log.info("  E candidates:")
    for col in E_all.columns:
        log.info("    %s: mean=%.3f  std=%.3f", col, E_all[col].mean(), E_all[col].std())

    fig_e = plot_E_candidates(E_all, panel,
                              title=f"E Candidate Comparison – {cfg['data']['region']}")
    fig_e.savefig(out_fig / "05_E_candidates.png", dpi=cfg["output"]["dpi"],
                  bbox_inches="tight")
    plt.close(fig_e)
    log.info("  Saved: 05_E_candidates.png")

    # E3 calibration diagnostic figure
    log.info("  Generating E3 calibration diagnostic ...")
    from constraint_field.inference.price_spread import PriceSpreadE
    e3_inferrer = PriceSpreadE(cfg["inference"].get("E3_price_spread", {}))
    _ = e3_inferrer.infer(panel)   # populates ._calibration cache
    fig_cal = e3_inferrer.plot_calibration(panel, figsize=(16, 14))
    fig_cal.savefig(out_fig / "05b_E3_calibration.png",
                    dpi=cfg["output"]["dpi"], bbox_inches="tight")
    plt.close(fig_cal)
    log.info("  Saved: 05b_E3_calibration.png")

    # Select active E for dynamics
    active_E_name = cfg["inference"].get("active", "E1")
    E_col_map = {"E1": "E1", "E2": "E2", "E3": "E3", "composite": "E_composite"}
    E_active  = E_all[E_col_map.get(active_E_name, "E1")]
    log.info("  Active E: %s (mean=%.3f)", active_E_name, E_active.mean())

    # ── Step 5: Dynamic propagation comparison ─────────────────────────────
    log.info("\n[5/6] Dynamic propagation – reduced vs upgraded …")

    # Operator selection
    op = args.operator or cfg["dynamics"].get("operator", "diffusion")
    log.info("  Operator: %s", op)

    dynamics_cfg = dict(cfg["dynamics"])
    dynamics_cfg["operator"] = op

    sim = Simulator(dynamics_cfg, seed=42)

    # Use middle third of panel as simulation window (skip warm-up)
    start_idx = len(panel) // 3
    log.info("  Simulation start index: %d / %d", start_idx, len(panel))

    # Inject a shock at t+24 (one day into simulation)
    shocks = [] if args.no_shock else [
        Shock(t_start=24, duration=3, magnitude=1.5, label="constraint shock")
    ]

    both = sim.compare(panel, E_active, start_idx=start_idx, shocks=shocks,
                       operator_name=op)
    reduced_run  = both["reduced"]
    upgraded_run = both["upgraded"]

    # Save simulation outputs
    reduced_run.to_csv(out_dat  / "sim_reduced.csv")
    upgraded_run.to_csv(out_dat / "sim_upgraded.csv")

    # Compute metrics
    metrics = compute_comparison_metrics(reduced_run, upgraded_run)
    metrics.to_csv(out_dat / "comparison_metrics.csv")
    log.info("  Metrics:\n%s", metrics.to_string())

    # Shock recovery
    if shocks:
        recovery = shock_recovery_analysis(reduced_run, upgraded_run,
                                           shock_t=shocks[0].t_start)
        log.info("  Recovery analysis:\n%s", recovery.to_string())
        recovery.to_csv(out_dat / "shock_recovery.csv")

    # Visualise
    E_slice = E_active.reindex(reduced_run.index)
    fig_comp = plot_simulation_comparison(
        reduced_run, upgraded_run, E_series=E_slice,
        title=f"Reduced vs Upgraded – {op} operator",
    )
    fig_comp.savefig(out_fig / "06_simulation_comparison.png",
                     dpi=cfg["output"]["dpi"], bbox_inches="tight")
    plt.close(fig_comp)
    log.info("  Saved: 06_simulation_comparison.png")

    # ── Step 6: Parameter sweep ────────────────────────────────────────────
    log.info("\n[6/6] Parameter sweep (diffusion.gamma × diffusion.eta) …")

    sweep_cfg = dict(cfg["dynamics"])
    sweep_cfg["operator"] = "diffusion"
    sweep_sim = Simulator(sweep_cfg, seed=42)

    sweep_results = sweep_sim.sweep(
        panel, E_active,
        param_grid={
            "diffusion.gamma": [0.05, 0.10, 0.20, 0.40],
            "diffusion.eta":   [0.02, 0.05, 0.10, 0.20],
        },
        start_idx=start_idx,
    )
    sweep_results.to_csv(out_dat / "parameter_sweep.csv", index=False)
    log.info("  Sweep: %d combinations\n%s", len(sweep_results),
             sweep_results.sort_values("rmse").head(5).to_string())

    # Sweep heatmap
    sweep_pivot = sweep_results.pivot_table(
        index="diffusion.gamma", columns="diffusion.eta", values="rmse"
    )
    fig_sw, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(sweep_pivot.values, aspect="auto",
                   cmap="YlOrRd", origin="upper")
    ax.set_xticks(range(len(sweep_pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in sweep_pivot.columns])
    ax.set_yticks(range(len(sweep_pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in sweep_pivot.index])
    ax.set_xlabel("eta (transmissibility)")
    ax.set_ylabel("gamma (damping)")
    ax.set_title("Parameter sweep – RMSE (diffusion operator)")
    plt.colorbar(im, ax=ax, label="RMSE")
    fig_sw.tight_layout()
    fig_sw.savefig(out_fig / "07_parameter_sweep.png",
                   dpi=cfg["output"]["dpi"], bbox_inches="tight")
    plt.close(fig_sw)
    log.info("  Saved: 07_parameter_sweep.png")

    # ── Final report ───────────────────────────────────────────────────────
    print_comparison_report(metrics, stress)

    log.info("\n✓ Demo complete.  Outputs in: %s/", args.out_dir)
    log.info("  Figures : %s", out_fig)
    log.info("  Data    : %s", out_dat)


if __name__ == "__main__":
    main()
