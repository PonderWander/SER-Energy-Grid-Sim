"""
scripts/run_hard_floor_corridor.py
=====================================
Re-run of the corridor-loading experiment with hard-floor E variants.

Three E variants:
  "soft_sigmoid"  (previous baseline)
  "hard_floor"    ε=0.02: E drops to 2% when stress >= threshold
  "linear_ramp"   E tapers linearly to 0 at threshold

Primary focus: SW spine bottleneck AZPS->WACM (cap=1.0 GW).
Hard floor fires at stress >= 1.0, producing retained accumulation.

All figures: HF_SW_deep_dive.png, HF_mode_comparison.png, HF_regime_transition.png
"""

from __future__ import annotations
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.special import expit

from constraint_field.graph.network import build_graph, node_order
from constraint_field.graph.edge_fluidity import E1_price_spread_edge
from constraint_field.graph.propagation import GraphPropagator, PropagationConfig
from scripts.run_spatial_experiments import build_gradient_rich_field
from scripts.run_corridor_loading import (
    init_corridor_phi, gradient_half_life, downstream_arrival,
    node_retention, spillover_share, edge_stress_series,
    CORRIDORS, LOADING_LEVELS, STEPS, ETA_C, GAMMA_C
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hard_floor")

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)
DPI = 150

STRESS_THRESHOLD = 1.0
E_FLOOR          = 0.02
SOFT_ALPHA       = 4.0
SOFT_MID         = 0.5
E_MODES = ["static", "soft_sigmoid", "linear_ramp", "hard_floor"]
MODE_STYLES = {
    "static":       ("#9E9E9E", "-",  1.0, "Static E"),
    "soft_sigmoid": ("#FF9800", "--", 1.2, "Soft sigmoid"),
    "linear_ramp":  ("#4CAF50", "-.", 1.2, "Linear ramp"),
    "hard_floor":   ("#F44336", "-",  1.8, "Hard floor (ε=0.02)"),
}


def compute_edge_E(G, nodes, Phi_t, e_base_row, mode):
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    result = {}
    for u, v in G.edges():
        col = f"{u}_{v}"; rev = f"{v}_{u}"
        key = col if col in e_base_row.index else rev
        e_base  = float(e_base_row.get(key, 0.5))
        cap     = G[u][v].get("capacity_gw", 1.0)
        if u in n_idx and v in n_idx:
            dphi   = abs(float(Phi_t[n_idx[u]]) - float(Phi_t[n_idx[v]]))
            stress = dphi / cap
        else:
            stress = 0.0
        if mode == "static":
            e_new = e_base
        elif mode == "soft_sigmoid":
            e_new = e_base * float(expit(-SOFT_ALPHA * (stress - SOFT_MID)))
        elif mode == "linear_ramp":
            e_new = 0.0 if stress >= STRESS_THRESHOLD else e_base * max(0.0, 1.0 - stress / STRESS_THRESHOLD)
        elif mode == "hard_floor":
            e_new = e_base * E_FLOOR if stress >= STRESS_THRESHOLD else e_base
        else:
            e_new = e_base
        result[key] = float(np.clip(e_new, 0.0, 1.0))
    return result


def run_corridor_sim(G, nodes, Phi0, e_base_row, mode, eta=ETA_C, gamma=GAMMA_C, steps=STEPS, seed=42):
    n     = len(nodes)
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    edges = list(G.edges())
    cfg_tmp = PropagationConfig(eta=eta, gamma=gamma, steps=1, use_E=False, noise_std=0.0, seed=seed)
    prop_tmp = GraphPropagator(G, nodes, cfg_tmp)
    L_rw     = prop_tmp.L_const
    traj      = np.zeros((steps + 1, n))
    traj[0]   = Phi0.copy()
    blocked_t = np.zeros(steps + 1)
    edge_labels = [f"{u}_{v}" for u, v in edges]
    stress_arr  = np.zeros((steps + 1, len(edges)))
    E_arr       = np.zeros((steps + 1, len(edges)))
    for t in range(1, steps + 1):
        phi_t = traj[t - 1].copy()
        e_dict = compute_edge_E(G, nodes, phi_t, e_base_row, mode)
        n_blocked = 0
        if mode == "static":
            L_cur = L_rw
        else:
            W = np.zeros((n, n))
            for ei, (u, v) in enumerate(edges):
                col = f"{u}_{v}"; rev = f"{v}_{u}"
                key = col if col in e_dict else rev
                e_val = e_dict.get(key, 0.5)
                cap   = G[u][v].get("capacity_gw", 1.0)
                dphi  = abs(phi_t[n_idx[u]] - phi_t[n_idx[v]]) if u in n_idx and v in n_idx else 0
                stress = dphi / cap
                stress_arr[t, ei] = stress
                E_arr[t, ei]      = e_val
                if mode == "hard_floor" and stress >= STRESS_THRESHOLD:
                    n_blocked += 1
                i, j = n_idx[u], n_idx[v]
                W[i, j] = W[j, i] = cap * e_val
            d = W.sum(axis=1)
            d_inv = np.where(d > 1e-9, 1.0 / d, 0.0)
            L_cur = np.diag(d_inv) @ (np.diag(d) - W)
        blocked_t[t] = n_blocked
        traj[t] = phi_t - eta * (L_cur @ phi_t) - gamma * phi_t
    stress_series = {edge_labels[ei]: stress_arr[:, ei] for ei in range(len(edges))}
    E_series      = {edge_labels[ei]: E_arr[:, ei] for ei in range(len(edges))}
    return traj, stress_series, E_series, blocked_t


def local_accumulation(traj, nodes, accumulation_node):
    ni     = nodes.index(accumulation_node)
    series = np.abs(traj[:, ni])
    peak_t = int(np.argmax(series))
    peak_v = float(series[peak_t])
    return peak_v, peak_t


def reroute_index(traj, nodes, G, primary_path, alt_nodes):
    prim_idx = [nodes.index(nd) for nd in primary_path if nd in nodes]
    alt_idx  = [nodes.index(nd) for nd in alt_nodes if nd in nodes]
    if not alt_idx or not prim_idx:
        return np.nan
    prim_phi = np.abs(traj[:, prim_idx]).mean()
    alt_phi  = np.abs(traj[:, alt_idx]).mean()
    return float(alt_phi / prim_phi) if prim_phi > 1e-9 else np.nan


def corridor_blocked_steps(blocked_t):
    n_blocked = int((blocked_t > 0).sum())
    return n_blocked, n_blocked / max(len(blocked_t) - 1, 1)


def run_experiment(G, nodes, E1_df, bg_std):
    SW_ALT_NODES = ["PACE", "PSCO", "NEVP", "IPCO"]
    SW_PATH      = CORRIDORS["SW_spine"]["path"]
    NW_ALT_NODES = ["IPCO", "NWMT", "NEVP"]
    NW_PATH      = CORRIDORS["NW_spine"]["path"]
    e_base_row   = E1_df.iloc[0]
    all_records  = []
    all_trajs    = {}
    for corr_name, corr_spec in CORRIDORS.items():
        path       = corr_spec["path"]
        is_SW      = (corr_name == "SW_spine")
        alt_nodes  = SW_ALT_NODES if is_SW else NW_ALT_NODES
        accum_node = "AZPS" if is_SW else "PACW"
        branch_nodes = set()
        for nd, brs in corr_spec["branches_from"].items():
            branch_nodes.update(brs)
        branch_nodes -= set(path)
        for loading in LOADING_LEVELS:
            loading_abs = loading * bg_std
            Phi0 = init_corridor_phi(G, nodes, path, loading_abs)
            for mode in E_MODES:
                traj, stress_s, E_s, blocked = run_corridor_sim(G, nodes, Phi0, e_base_row, mode)
                all_trajs[(corr_name, loading, mode)] = traj
                hl  = gradient_half_life(traj, nodes, corr_spec["upstream"], corr_spec["terminal"])
                arr = downstream_arrival(traj, nodes, corr_spec["terminal"], corr_spec["upstream"])
                t_eval  = min(int(hl) if not np.isinf(hl) else STEPS // 2, STEPS)
                ret     = node_retention(traj, nodes, path, t_eval)
                spl     = spillover_share(traj, nodes, path, list(branch_nodes), t_eval)
                rer     = reroute_index(traj, nodes, G, path, alt_nodes)
                peak_acc, peak_t_acc = local_accumulation(traj, nodes, accum_node)
                n_blk, frac_blk = corridor_blocked_steps(blocked)
                bot_key = f"{path[-2]}_{path[-1]}" if len(path) >= 2 else path[0]
                bot_rev = f"{path[-1]}_{path[-2]}" if len(path) >= 2 else bot_key
                bot_s   = stress_s.get(bot_key, stress_s.get(bot_rev, np.zeros(STEPS+1)))
                max_bot_stress = float(bot_s.max())
                all_records.append({
                    "corridor": corr_name, "loading": loading, "loading_abs": loading_abs, "mode": mode,
                    "half_life": hl, "arrival_t": arr, "retention": ret, "spillover": spl,
                    "reroute_index": rer, "local_accum": peak_acc, "local_accum_t": peak_t_acc,
                    "blocked_steps": n_blk, "frac_blocked": frac_blk, "max_bot_stress": max_bot_stress,
                })
                log.info("  %-10s %3.1fσ %-15s  hl=%.1f  ret=%.3f  spl=%.3f  blk=%d",
                         corr_name, loading, mode, hl,
                         ret if not np.isnan(ret) else -1,
                         spl if not np.isnan(spl) else -1, n_blk)
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_DAT / "hard_floor_loading.csv", index=False)
    return df, all_trajs


def plot_sw_deep_dive(df, trajs, G, nodes):
    sub = df[df["corridor"] == "SW_spine"]
    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)
    metric_panels = [
        ("half_life",     "Gradient half-life (steps)", gs[0, 0]),
        ("retention",     "Corridor retention",          gs[0, 1]),
        ("reroute_index", "Reroute index",               gs[0, 2]),
        ("local_accum",   "Peak local accum at AZPS",   gs[0, 3]),
    ]
    for col, ylabel, gs_pos in metric_panels:
        ax = fig.add_subplot(gs_pos)
        for mode, (color, ls, lw, label) in MODE_STYLES.items():
            sub_m = sub[sub["mode"] == mode].sort_values("loading")
            ax.plot(sub_m["loading"], sub_m[col], color=color, ls=ls, lw=lw, marker="o", ms=5, label=label)
        ax.set_xlabel("Loading (× σ)"); ax.set_ylabel(ylabel); ax.set_title(ylabel)
        if col == "half_life":
            ax.legend(fontsize=6)
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
    high_load = 2.5
    focus_nodes = ["CISO", "WALC", "AZPS", "WACM", "PACE", "NEVP"]
    traj_hard   = trajs.get(("SW_spine", high_load, "hard_floor"))
    traj_static = trajs.get(("SW_spine", high_load, "static"))
    node_colors = plt.cm.tab10(np.linspace(0, 0.7, len(focus_nodes)))
    for traj, ax, title in [
        (traj_hard,   fig.add_subplot(gs[1, :2]), f"Hard floor — loading={high_load:.1f}σ"),
        (traj_static, fig.add_subplot(gs[1, 2:]), f"Static E  — loading={high_load:.1f}σ"),
    ]:
        if traj is not None:
            for nd, nc in zip(focus_nodes, node_colors):
                if nd in nodes:
                    ni = nodes.index(nd)
                    ax.plot(range(len(traj)), traj[:, ni], color=nc,
                            lw=2.0 if nd in ["AZPS","WACM"] else 1.0,
                            ls="-" if nd in ["CISO","WALC","AZPS","WACM"] else "--",
                            label=nd)
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        ax.set_xlabel("Step"); ax.set_ylabel("Phi"); ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
    ax_str = fig.add_subplot(gs[2, :2])
    ax_E   = fig.add_subplot(gs[2, 2:])
    e_base = 0.58
    for mode, (color, ls, lw, label) in MODE_STYLES.items():
        traj_m = trajs.get(("SW_spine", high_load, mode))
        if traj_m is None:
            continue
        azps_i = nodes.index("AZPS"); wacm_i = nodes.index("WACM")
        cap    = G["AZPS"]["WACM"].get("capacity_gw", 1.0)
        stress = np.abs(traj_m[:, azps_i] - traj_m[:, wacm_i]) / cap
        ax_str.plot(range(len(stress)), stress, color=color, ls=ls, lw=lw, label=label)
        if mode == "static":    E_series = np.full(len(stress), e_base)
        elif mode == "soft_sigmoid": E_series = e_base * expit(-SOFT_ALPHA * (stress - SOFT_MID))
        elif mode == "linear_ramp":  E_series = np.where(stress >= STRESS_THRESHOLD, 0.0, e_base * (1 - stress / STRESS_THRESHOLD))
        elif mode == "hard_floor":   E_series = np.where(stress >= STRESS_THRESHOLD, e_base * E_FLOOR, e_base)
        else: E_series = np.full(len(stress), e_base)
        ax_E.plot(range(len(E_series)), E_series, color=color, ls=ls, lw=lw, label=label)
    ax_str.axhline(STRESS_THRESHOLD, color="red", lw=1.2, ls=":", alpha=0.8, label="Stress = 1.0")
    ax_str.set_xlabel("Step"); ax_str.set_ylabel("|ΔΦ(AZPS,WACM)| / cap")
    ax_str.set_title(f"(E) Bottleneck stress  loading={high_load:.1f}σ"); ax_str.legend(fontsize=6)
    ax_E.axhline(E_FLOOR * e_base, color="red", lw=0.8, ls=":", alpha=0.6, label=f"ε_floor={E_FLOOR}")
    ax_E.set_xlabel("Step"); ax_E.set_ylabel("E (AZPS→WACM)")
    ax_E.set_title("(F) Bottleneck edge fluidity"); ax_E.set_ylim(0, 0.7); ax_E.legend(fontsize=6)
    fig.suptitle("SW Spine Corridor Loading: Hard Floor vs Soft E", fontsize=12)
    fig.savefig(OUT_FIG / "HF_SW_deep_dive.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved HF_SW_deep_dive.png")


def plot_mode_comparison(df):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    metrics = [("half_life","Gradient half-life"), ("retention","Corridor retention"),
               ("reroute_index","Reroute index"), ("local_accum","Local accumulation (AZPS)")]
    for row, corr_name in enumerate(["SW_spine", "NW_spine"]):
        sub = df[df["corridor"] == corr_name]
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col_idx]
            for mode, (color, ls, lw, label) in MODE_STYLES.items():
                sub_m = sub[sub["mode"] == mode].sort_values("loading")
                ax.plot(sub_m["loading"], sub_m[metric], color=color, ls=ls, lw=lw, marker="o", ms=4, label=label)
            if corr_name == "SW_spine":
                ax.axvspan(1.3, 1.7, alpha=0.07, color="red")
            ax.set_xlabel("Loading (× σ)" if row == 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(f"{corr_name}: {ylabel}", fontsize=8)
            ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Mode Comparison: All Metrics", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "HF_mode_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved HF_mode_comparison.png")


def plot_regime_transition(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[(df["corridor"] == "SW_spine") & (df["mode"] == "hard_floor")].sort_values("loading")
    def norm01(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0
    ax.plot(sub["loading"], norm01(sub["half_life"]),    color="#2196F3", lw=2.0, marker="o", ms=6, label="Half-life (normalised)")
    ax.plot(sub["loading"], norm01(sub["retention"]),    color="#F44336", lw=2.0, marker="s", ms=6, label="Retention (normalised)")
    ax.plot(sub["loading"], norm01(sub["reroute_index"]),color="#4CAF50", lw=2.0, marker="^", ms=6, label="Reroute index (normalised)")
    ax.plot(sub["loading"], norm01(sub["local_accum"]), color="#FF9800", lw=2.0, marker="D", ms=6, label="Local accumulation at AZPS (normalised)")
    ax.axvline(1.4, color="red", lw=1.2, ls=":", alpha=0.7, label="≈ Stress > 1.0 (AZPS→WACM blocked)")
    ax.axvspan(1.3, 1.6, alpha=0.08, color="red")
    ax.set_xlabel("Upstream loading (× background σ)")
    ax.set_ylabel("Metric (each normalised to [0,1])")
    ax.set_title("SW Spine — Hard Floor E: Regime Transition")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.1, 1.15)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "HF_regime_transition.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved HF_regime_transition.png")


def write_report(df, bg_std):
    sep = "=" * 68
    lines = [sep, "  HARD-FLOOR E CORRIDOR LOADING — FINDINGS", sep, "",
             f"  Background σ = {bg_std:.4f}  Hard floor: E = ε={E_FLOOR} when stress ≥ {STRESS_THRESHOLD}",
             f"  η={ETA_C}  γ={GAMMA_C}  steps={STEPS}", ""]
    for corr in ["SW_spine", "NW_spine"]:
        sub_hf = df[(df["corridor"]==corr) & (df["mode"]=="hard_floor")].set_index("loading")
        sub_st = df[(df["corridor"]==corr) & (df["mode"]=="static")].set_index("loading")
        lines += [f"─"*68, f"  {corr}", f"─"*68, ""]
        for load in LOADING_LEVELS:
            hl_s  = sub_st.loc[load, "half_life"]  if load in sub_st.index  else np.nan
            hl_h  = sub_hf.loc[load, "half_life"]  if load in sub_hf.index  else np.nan
            ret_s = sub_st.loc[load, "retention"]  if load in sub_st.index  else np.nan
            ret_h = sub_hf.loc[load, "retention"]  if load in sub_hf.index  else np.nan
            stress= sub_hf.loc[load, "max_bot_stress"] if load in sub_hf.index else np.nan
            flag  = " *** BLOCKED" if not np.isnan(stress) and stress >= 1.0 else ""
            lines.append(f"  {load:.1f}σ: static_hl={hl_s:.1f}  hf_hl={hl_h:.1f}  "
                         f"static_ret={ret_s:.3f}  hf_ret={ret_h:.3f}  stress={stress:.3f}{flag}")
        lines.append("")
    lines.append(sep)
    report = "\n".join(lines)
    (OUT_DAT / "hard_floor_report.txt").write_text(report)
    print("\n" + report)
    return report


def main():
    log.info("Building graph and gradient-rich field …")
    G      = build_graph()
    nodes  = node_order(G)
    field  = build_gradient_rich_field(G, nodes, seed=42, T=1440)
    E1_df  = E1_price_spread_edge(G, field["R"])
    bg_std = float(field["Phi"].values.std(axis=1).mean())
    log.info("  Background σ = %.4f", bg_std)
    df, trajs = run_experiment(G, nodes, E1_df, bg_std)
    plot_sw_deep_dive(df, trajs, G, nodes)
    plot_mode_comparison(df)
    plot_regime_transition(df)
    write_report(df, bg_std)
    log.info("Complete.")


if __name__ == "__main__":
    main()
