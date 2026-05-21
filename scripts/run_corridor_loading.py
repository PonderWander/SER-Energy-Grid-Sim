"""
scripts/run_corridor_loading.py
================================
Corridor-loading experiment. Tests two major cross-sections of the
Western Interconnect:

  NW_spine: CISO -> PACW -> BPAT  (bypass: CISO->BPAT direct)
  SW_spine: CISO -> WALC -> AZPS -> WACM  (bottleneck: AZPS->WACM 1.0 GW)

Protocol: initialise monotone Phi gradient along each corridor,
ramp upstream loading in steps, measure dispersal regime.

Includes state-dependent E: E_ij decreases as edge-local stress rises.
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
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit

from constraint_field.graph.network import build_graph, node_order
from constraint_field.graph.edge_fluidity import E1_price_spread_edge
from constraint_field.graph.propagation import GraphPropagator, PropagationConfig
from scripts.run_spatial_experiments import build_gradient_rich_field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("corridor")

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)
DPI = 150

ETA_C   = 0.20
GAMMA_C = 0.02
STEPS   = 60
LOADING_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

SD_ALPHA      = 4.0
SD_STRESS_MID = 0.5

CORRIDORS = {
    "NW_spine": {
        "path":    ["CISO", "PACW", "BPAT"],
        "bypass":  ("CISO", "BPAT"),
        "branches_from": {
            "CISO": ["AZPS","NEVP","WALC","LDWP","IID","TIDC"],
            "PACW": ["IPCO","NEVP","PACE"],
            "BPAT": ["IPCO","NWMT"],
        },
        "terminal": "BPAT",
        "upstream": "CISO",
        "color": "#2196F3",
    },
    "SW_spine": {
        "path":    ["CISO", "WALC", "AZPS", "WACM"],
        "bypass":  None,
        "branches_from": {
            "CISO": ["PACW","BPAT","NEVP","LDWP","IID","TIDC"],
            "WALC": ["NEVP","IID","LDWP"],
            "AZPS": ["NEVP","PACE"],
            "WACM": ["PACE","PSCO"],
        },
        "bottleneck": ("AZPS", "WACM"),
        "terminal":  "WACM",
        "upstream":  "CISO",
        "color": "#F44336",
    },
}


def state_dependent_E(G, nodes, Phi_t, E_base_row, alpha=SD_ALPHA, stress_mid=SD_STRESS_MID):
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    result = {}
    for u, v in G.edges():
        col = f"{u}_{v}"; rev = f"{v}_{u}"
        base_key = col if col in E_base_row.index else rev
        e_base   = float(E_base_row.get(base_key, 0.5))
        cap      = G[u][v].get("capacity_gw", 1.0)
        if u in n_idx and v in n_idx:
            dphi   = abs(Phi_t[n_idx[u]] - Phi_t[n_idx[v]])
            stress = dphi / cap
            e_new  = e_base * float(expit(-alpha * (stress - stress_mid)))
        else:
            e_new = e_base
        result[col if col in E_base_row.index else rev] = float(np.clip(e_new, 0, 1))
    return result


def init_corridor_phi(G, nodes, corridor_path, loading, decay_along_path=True):
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    Phi0   = np.zeros(len(nodes))
    L      = len(corridor_path)
    for pos, nd in enumerate(corridor_path):
        if nd in n_idx:
            if decay_along_path:
                frac = pos / max(L - 1, 1)
                Phi0[n_idx[nd]] = loading * (1.0 - 2.0 * frac)
            else:
                Phi0[n_idx[nd]] = loading if pos == 0 else -loading
    return Phi0


def run_corridor(G, nodes, Phi0, E1_df, loading, corr_spec,
                 use_state_E=False, eta=ETA_C, gamma=GAMMA_C, steps=STEPS, seed=42):
    rng    = np.random.default_rng(seed)
    n      = len(nodes)
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    edges  = list(G.edges())
    traj   = np.zeros((steps + 1, n))
    E_traj = np.zeros((steps + 1, len(edges)))
    traj[0] = Phi0.copy()
    cfg   = PropagationConfig(eta=eta, gamma=gamma, steps=1, use_E=False, noise_std=0.0, seed=seed)
    prop  = GraphPropagator(G, nodes, cfg)
    L_rw  = prop.L_const
    e_base_row = E1_df.iloc[0]
    for t in range(1, steps + 1):
        phi_t = traj[t - 1].copy()
        if use_state_E:
            e_dict = state_dependent_E(G, nodes, phi_t, e_base_row)
            W = np.zeros((n, n))
            for ei, (u, v) in enumerate(edges):
                col = f"{u}_{v}"; rev = f"{v}_{u}"
                key = col if col in e_dict else rev
                e_val = e_dict.get(key, 0.5)
                cap   = G[u][v].get("capacity_gw", 1.0)
                E_traj[t, ei] = e_val
                i, j = n_idx[u], n_idx[v]
                W[i, j] = W[j, i] = cap * e_val
            d_arr = W.sum(axis=1)
            d_inv = np.where(d_arr > 1e-9, 1.0 / d_arr, 0.0)
            D_inv = np.diag(d_inv)
            L_cur = D_inv @ (np.diag(d_arr) - W)
        else:
            L_cur = L_rw
        traj[t] = phi_t - eta * (L_cur @ phi_t) - gamma * phi_t
    return traj, E_traj


def gradient_half_life(traj, nodes, upstream, downstream):
    ui  = nodes.index(upstream)
    di  = nodes.index(downstream)
    g0  = abs(traj[0, ui] - traj[0, di])
    if g0 < 1e-6:
        return np.inf
    half = g0 * 0.5
    for t in range(1, len(traj)):
        if abs(traj[t, ui] - traj[t, di]) <= half:
            return float(t)
    return float(len(traj))


def downstream_arrival(traj, nodes, terminal, upstream, threshold_frac=0.05):
    ti     = nodes.index(terminal)
    ui     = nodes.index(upstream)
    ref    = abs(traj[0, ui])
    cutoff = threshold_frac * ref
    for t in range(1, len(traj)):
        if abs(traj[t, ti]) > cutoff:
            return float(t)
    return float(len(traj))


def node_retention(traj, nodes, corridor_path, t):
    corr_idx = [nodes.index(nd) for nd in corridor_path if nd in nodes]
    all_phi  = np.abs(traj[t])
    total    = all_phi.sum()
    if total < 1e-9:
        return np.nan
    return float(all_phi[corr_idx].sum() / total)


def spillover_share(traj, nodes, corridor_path, branch_nodes, t):
    corr_set   = set(corridor_path)
    branch_idx = [nodes.index(nd) for nd in branch_nodes if nd in nodes and nd not in corr_set]
    all_phi    = np.abs(traj[t])
    total      = all_phi.sum()
    if total < 1e-9 or not branch_idx:
        return np.nan
    return float(all_phi[branch_idx].sum() / total)


def bypass_rerouting(traj, nodes, G, bypass_edge, corridor_path, t_window=None):
    if bypass_edge is None:
        return np.nan
    u_bp, v_bp = bypass_edge
    ui   = nodes.index(u_bp)
    vi   = nodes.index(v_bp)
    if len(corridor_path) < 3:
        return np.nan
    mid_i = nodes.index(corridor_path[1])
    T = len(traj) if t_window is None else min(t_window, len(traj))
    bypass_flux  = 0.0
    primary_flux = 0.0
    for t in range(1, T):
        phi = traj[t]
        cap_bp  = G[u_bp][v_bp].get("capacity_gw", 1.0) if G.has_edge(u_bp, v_bp) else 0
        bypass_flux += abs(phi[ui] - phi[vi]) * cap_bp
        cap1 = G[corridor_path[0]][corridor_path[1]].get("capacity_gw", 1.0) if G.has_edge(corridor_path[0], corridor_path[1]) else 0
        cap2 = G[corridor_path[1]][corridor_path[2]].get("capacity_gw", 1.0) if G.has_edge(corridor_path[1], corridor_path[2]) else 0
        f1   = abs(phi[nodes.index(corridor_path[0])] - phi[mid_i]) * cap1
        f2   = abs(phi[mid_i] - phi[nodes.index(corridor_path[2])]) * cap2
        primary_flux += min(f1, f2)
    if primary_flux < 1e-9:
        return np.nan
    return float(bypass_flux / primary_flux)


def edge_stress_series(traj, nodes, G, corridor_path):
    result = {}
    n_idx  = {nd: i for i, nd in enumerate(nodes)}
    for k in range(len(corridor_path) - 1):
        u, v  = corridor_path[k], corridor_path[k+1]
        cap   = G[u][v].get("capacity_gw", 1.0) if G.has_edge(u, v) else 1.0
        ui, vi = n_idx[u], n_idx[v]
        stress = np.abs(traj[:, ui] - traj[:, vi]) / cap
        result[f"{u}→{v}"] = stress
    return result


def run_experiment(G, nodes, E1_df, corr_name, corr_spec, bg_std):
    records  = []
    trajs    = {}
    branch_nodes = set()
    for nd, brs in corr_spec["branches_from"].items():
        branch_nodes.update(brs)
    branch_nodes -= set(corr_spec["path"])
    log.info("  Corridor: %s  path=%s  branches=%d",
             corr_name, "→".join(corr_spec["path"]), len(branch_nodes))
    for loading in LOADING_LEVELS:
        loading_abs = loading * bg_std
        Phi0 = init_corridor_phi(G, nodes, corr_spec["path"], loading_abs)
        for mode, use_sdE in [("static_E", False), ("state_E", True)]:
            traj, E_traj = run_corridor(G, nodes, Phi0, E1_df, loading_abs, corr_spec, use_state_E=use_sdE)
            key = (loading, mode)
            trajs[key] = traj
            hl  = gradient_half_life(traj, nodes, corr_spec["upstream"], corr_spec["terminal"])
            arr = downstream_arrival(traj, nodes, corr_spec["terminal"], corr_spec["upstream"])
            t_eval = min(int(hl) if not np.isinf(hl) else STEPS//2, STEPS)
            ret = node_retention(traj, nodes, corr_spec["path"], t_eval)
            spl = spillover_share(traj, nodes, corr_spec["path"], list(branch_nodes), t_eval)
            rer = bypass_rerouting(traj, nodes, G, corr_spec.get("bypass"), corr_spec["path"], t_window=t_eval) if corr_spec.get("bypass") else np.nan
            stresses = edge_stress_series(traj, nodes, G, corr_spec["path"])
            max_stress = max(s.max() for s in stresses.values()) if stresses else np.nan
            if np.isinf(hl) or hl >= STEPS * 0.9:
                regime = "retained"
            elif hl < STEPS * 0.3:
                regime = "rapid_dispersal"
            else:
                regime = "moderate_dispersal"
            records.append({
                "corridor": corr_name, "loading": loading, "loading_abs": loading_abs,
                "mode": mode, "half_life": hl, "arrival_t": arr, "retention": ret,
                "spillover": spl, "bypass_ratio": rer, "max_edge_stress": max_stress, "regime": regime,
            })
            log.info("    %-10s load=%.1fσ %-10s  hl=%.1f  arr=%.1f  ret=%.3f  spl=%.3f  [%s]",
                     corr_name, loading, mode, hl, arr,
                     ret if not np.isnan(ret) else -1, spl if not np.isnan(spl) else -1, regime)
    return records, trajs, list(branch_nodes)


def plot_loading_profiles(df, corr_name, col, label, ax, color_static, color_sdE):
    for mode, col_c, ls in [("static_E", color_static, "-"), ("state_E", color_sdE, "--")]:
        sub = df[(df["corridor"] == corr_name) & (df["mode"] == mode)]
        ax.plot(sub["loading"], sub[col], color=col_c, ls=ls, lw=1.4, marker="o", ms=5, label=f"{mode}")
    ax.set_xlabel("Loading (× σ)")
    ax.set_ylabel(label)
    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)


def plot_corridor_dashboard(df, trajs, G, nodes, corr_name, corr_spec, branch_nodes, bg_std):
    color_s = "#2196F3"
    color_d = "#F44336"
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)
    sub = df[df["corridor"] == corr_name]
    ax = fig.add_subplot(gs[0, 0])
    plot_loading_profiles(df, corr_name, "half_life", "Gradient half-life (steps)", ax, color_s, color_d)
    ax.set_title("(A) Gradient half-life")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 1])
    plot_loading_profiles(df, corr_name, "arrival_t", "Downstream arrival time (steps)", ax, color_s, color_d)
    ax.set_title("(B) Downstream arrival time")
    ax = fig.add_subplot(gs[0, 2])
    plot_loading_profiles(df, corr_name, "retention", "Corridor node retention", ax, color_s, color_d)
    ax.set_title("(C) Node retention")
    ax.set_ylim(0, 1)
    ax = fig.add_subplot(gs[0, 3])
    plot_loading_profiles(df, corr_name, "spillover", "Spillover to branches", ax, color_s, color_d)
    ax.set_title("(D) Branch spillover share")
    ax = fig.add_subplot(gs[1, 0])
    plot_loading_profiles(df, corr_name, "max_edge_stress", "|ΔΦ| / capacity", ax, color_s, color_d)
    ax.axhline(1.0, color="red", lw=0.9, ls=":", alpha=0.7, label="stress=1 (at-capacity)")
    ax.set_title("(E) Max corridor edge stress")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 1])
    if corr_spec.get("bypass"):
        plot_loading_profiles(df, corr_name, "bypass_ratio", "Bypass / primary path flux", ax, color_s, color_d)
        ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_title("(F) Bypass rerouting ratio")
        ax.legend(fontsize=7)
    else:
        sub_s = df[(df["corridor"] == corr_name) & (df["mode"] == "static_E")]
        sub_d = df[(df["corridor"] == corr_name) & (df["mode"] == "state_E")]
        ax.set_title("(F) Regime classification")
        for sub_r, col, label in [(sub_s, color_s, "static_E"), (sub_d, color_d, "state_E")]:
            regime_val = sub_r["regime"].map({"rapid_dispersal": 0, "moderate_dispersal": 1, "retained": 2})
            ax.scatter(sub_r["loading"], regime_val, color=col, s=80, alpha=0.85, label=label)
            ax.plot(sub_r["loading"], regime_val, color=col, lw=0.8, alpha=0.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["rapid", "moderate", "retained"], fontsize=7)
        ax.set_xlabel("Loading (× σ)")
        ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 2:])
    path = corr_spec["path"]
    path_colors = plt.cm.viridis(np.linspace(0, 1, len(path)))
    show_loads  = [LOADING_LEVELS[1], LOADING_LEVELS[3], LOADING_LEVELS[5]]
    for load_idx, (load, ls_traj) in enumerate([(l, s) for l, s in zip(show_loads, ["-","--",":"])]):
        traj_s = trajs.get((load, "static_E"))
        if traj_s is None:
            continue
        for ni, (nd, pc) in enumerate(zip(path, path_colors)):
            nd_i = nodes.index(nd) if nd in nodes else None
            if nd_i is None:
                continue
            label = f"{nd} (load={load:.1f}σ)" if ni == 0 else None
            ax.plot(range(len(traj_s)), traj_s[:, nd_i], color=pc, lw=1.0 + load_idx * 0.3,
                    ls=ls_traj, alpha=0.8, label=label)
    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Phi")
    ax.set_title("(G) Corridor node Phi over time")
    ax.legend(fontsize=6, ncol=2)
    ax = fig.add_subplot(gs[2, :2])
    high_load   = LOADING_LEVELS[-1]
    traj_sd     = trajs.get((high_load, "state_E"))
    if traj_sd is not None:
        stresses = edge_stress_series(traj_sd, nodes, G, path)
        edge_cols = ["#2196F3", "#F44336", "#4CAF50"]
        for (edge_lbl, stress), ec in zip(stresses.items(), edge_cols[:len(stresses)]):
            e_base = 0.58
            E_vals = e_base * expit(-SD_ALPHA * (stress - SD_STRESS_MID))
            ax.plot(range(len(stress)), stress, color=ec, lw=1.2, label=f"stress {edge_lbl}", ls="-")
            ax2 = ax.twinx()
            ax2.plot(range(len(E_vals)), E_vals, color=ec, lw=0.9, ls="--", alpha=0.6)
            ax2.set_ylabel("E (dashed)", fontsize=7)
            ax2.set_ylim(0, 1)
    ax.axhline(SD_STRESS_MID, color="red", lw=0.8, ls=":", alpha=0.6, label=f"stress_mid={SD_STRESS_MID}")
    ax.axhline(1.0, color="red", lw=0.8, ls="-", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Edge stress |ΔΦ|/cap")
    ax.set_title(f"(H) Edge stress + E (state-dep)  high loading {high_load:.1f}σ")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, 2:])
    t_snap = 15
    pos    = {nd: (G.nodes[nd]["lon"], G.nodes[nd]["lat"]) for nd in G.nodes()}
    n_colors = plt.cm.tab10(np.linspace(0, 0.6, len(LOADING_LEVELS)))
    for li, load in enumerate(LOADING_LEVELS[::2]):
        traj_s = trajs.get((load, "static_E"))
        if traj_s is None:
            continue
        phi_snap = traj_s[min(t_snap, len(traj_s)-1)]
        for ni, nd in enumerate(nodes):
            x, y = pos[nd]
            val  = phi_snap[ni]
            sz   = 50 + 300 * abs(val)
            col  = "#F44336" if val > 0 else "#2196F3"
            ax.scatter(x, y, s=sz, c=col, alpha=0.4 + 0.2 * li, zorder=5)
        for k in range(len(path)-1):
            u_p, v_p = path[k], path[k+1]
            xu, yu = pos[u_p]; xv, yv = pos[v_p]
            ax.plot([xu, xv], [yu, yv], color=n_colors[li], lw=1.5 + li, alpha=0.7,
                    label=f"load={load:.1f}σ" if k == 0 else "")
    for nd in nodes:
        ax.annotate(nd, pos[nd], fontsize=6, ha="center", xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"(I) Phi spatial snapshot t={t_snap}")
    ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"Corridor Loading Experiment: {corr_name}  (η={ETA_C}  γ={GAMMA_C}  steps={STEPS})", fontsize=12)
    fig.savefig(OUT_FIG / f"CL_{corr_name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved CL_%s.png", corr_name)


def plot_regime_summary(df, G, nodes):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for corr, col in [("NW_spine", "#2196F3"), ("SW_spine", "#F44336")]:
        for mode, ls in [("static_E", "-"), ("state_E", "--")]:
            sub = df[(df["corridor"] == corr) & (df["mode"] == mode)]
            ax.plot(sub["loading"], sub["half_life"], color=col, ls=ls, lw=1.4, marker="o", ms=5,
                    label=f"{corr} {mode}")
    ax.set_xlabel("Loading (× σ)")
    ax.set_ylabel("Gradient half-life (steps)")
    ax.set_title("(A) Half-life: NW vs SW")
    ax.legend(fontsize=6)
    ax = axes[1]
    for corr, col in [("NW_spine", "#2196F3"), ("SW_spine", "#F44336")]:
        for mode, ls in [("static_E", "-"), ("state_E", "--")]:
            sub = df[(df["corridor"] == corr) & (df["mode"] == mode)]
            ax.plot(sub["loading"], sub["spillover"], color=col, ls=ls, lw=1.4, marker="o", ms=5,
                    label=f"{corr} {mode}")
    ax.set_xlabel("Loading (× σ)")
    ax.set_ylabel("Spillover to branches")
    ax.set_title("(B) Branch spillover: NW vs SW")
    ax = axes[2]
    regime_map = {"rapid_dispersal": 0, "moderate_dispersal": 1, "retained": 2}
    row_labels = []
    mat = []
    for corr in ["NW_spine", "SW_spine"]:
        for mode in ["static_E", "state_E"]:
            sub = df[(df["corridor"] == corr) & (df["mode"] == mode)]
            row = [regime_map.get(r, 1) for r in sub["regime"]]
            mat.append(row)
            row_labels.append(f"{corr}\n{mode}")
    mat_arr = np.array(mat)
    im = ax.imshow(mat_arr, aspect="auto", cmap=plt.cm.RdYlGn_r, vmin=0, vmax=2)
    ax.set_xticks(range(len(LOADING_LEVELS)))
    ax.set_xticklabels([f"{l:.1f}σ" for l in LOADING_LEVELS], fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("(C) Dispersal regime map")
    for i in range(mat_arr.shape[0]):
        for j in range(mat_arr.shape[1]):
            label = ["rapid", "mod.", "ret."][mat_arr[i, j]]
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if mat_arr[i, j] != 1 else "black")
    fig.suptitle("Cross-Corridor Dispersal Regime Summary", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "CL_regime_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved CL_regime_summary.png")


def write_report(df, bg_std):
    sep = "=" * 68
    lines = [sep, "  CORRIDOR LOADING EXPERIMENT — FINDINGS", sep, "",
             f"  Background σ = {bg_std:.4f}", f"  Loading range: {LOADING_LEVELS[0]:.1f}–{LOADING_LEVELS[-1]:.1f}× σ",
             f"  η={ETA_C}  γ={GAMMA_C}  steps={STEPS}", ""]
    for corr in ["NW_spine", "SW_spine"]:
        sub_s = df[(df["corridor"] == corr) & (df["mode"] == "static_E")]
        sub_d = df[(df["corridor"] == corr) & (df["mode"] == "state_E")]
        lines += [f"─"*40, f"  {corr}", f"─"*40, ""]
        hl_s = sub_s.set_index("loading")["half_life"]
        hl_d = sub_d.set_index("loading")["half_life"]
        hl_trend = "INCREASING" if hl_s.iloc[-1] > hl_s.iloc[0] * 1.2 else ("DECREASING" if hl_s.iloc[-1] < hl_s.iloc[0] * 0.8 else "FLAT")
        lines.append(f"  Half-life ({hl_trend}):")
        for load in LOADING_LEVELS:
            hl_sv = hl_s.get(load, np.nan)
            hl_dv = hl_d.get(load, np.nan)
            lines.append(f"    {load:.1f}σ: static={hl_sv:.1f}  state-dep={hl_dv:.1f}")
        lines.append("")
    lines.append(sep)
    report = "\n".join(lines)
    (OUT_DAT / "corridor_report.txt").write_text(report)
    print("\n" + report)
    return report


def main():
    log.info("Building graph and gradient-rich field …")
    G     = build_graph()
    nodes = node_order(G)
    field = build_gradient_rich_field(G, nodes, seed=42, T=1440)
    E1_df = E1_price_spread_edge(G, field["R"])
    bg_std = float(field["Phi"].values.std(axis=1).mean())
    log.info("  Background σ = %.4f", bg_std)
    all_records = []
    all_trajs   = {}
    branch_map  = {}
    for corr_name, corr_spec in CORRIDORS.items():
        log.info("\n%s\nCorridor: %s", "─"*60, corr_name)
        records, trajs, branch_nodes = run_experiment(G, nodes, E1_df, corr_name, corr_spec, bg_std)
        all_records.extend(records)
        all_trajs[corr_name] = trajs
        branch_map[corr_name] = branch_nodes
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_DAT / "corridor_loading.csv", index=False)
    for corr_name, corr_spec in CORRIDORS.items():
        plot_corridor_dashboard(df, all_trajs[corr_name], G, nodes, corr_name, corr_spec, branch_map[corr_name], bg_std)
    plot_regime_summary(df, G, nodes)
    write_report(df, bg_std)
    log.info("Complete.")


if __name__ == "__main__":
    main()
