"""
scripts/run_sw_variants.py
===========================
Three SW spine variants + NW negative control.

V1: Hard floor ε=0, γ=0.02 (current γ)
V2: Hard floor ε=0, γ=0.01 (half γ — tests sustained retention plateau)
V3: Linear ramp, γ=0.02 (baseline comparator)

Measurements (regime-detection):
A. Retention plateau at AZPS (net_rise, dwell_75, AUC)
B. WACM isolation (corr_blocked, frac_isolated)
C. Branch dominance (absolute AUC PACE+NEVP vs WACM)
D. Regime classification per (loading, variant)

Outputs: SV_A_retention.png, SV_B_isolation.png, SV_C_branches.png,
         SV_D_regime.png, SV_summary.png, sw_variants_report.txt
"""

from __future__ import annotations
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.special import expit

from constraint_field.graph.network import build_graph, node_order
from constraint_field.graph.edge_fluidity import E1_price_spread_edge
from constraint_field.graph.propagation import GraphPropagator, PropagationConfig
from scripts.run_spatial_experiments import build_gradient_rich_field
from scripts.run_corridor_loading import (
    init_corridor_phi, CORRIDORS, LOADING_LEVELS, STEPS, ETA_C, GAMMA_C
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sw_variants")

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)
DPI = 150

STRESS_THRESHOLD = 1.0
GAMMA_LOW        = GAMMA_C * 0.5   # 0.01
BOTTLENECK_EDGE  = ("AZPS", "WACM")
ACCUM_NODE       = "AZPS"
TERMINAL_NODE    = "WACM"
BRANCH_NODES     = {"PACE": ("AZPS","PACE"), "NEVP": ("AZPS","NEVP")}
UPSTREAM_NODE    = "CISO"

VARIANTS = {
    "V1_hard_eps0":        {"eps": 0.0,  "gamma": GAMMA_C,   "label": "Hard floor ε=0, γ=0.02"},
    "V2_hard_eps0_lowγ":   {"eps": 0.0,  "gamma": GAMMA_LOW, "label": "Hard floor ε=0, γ=0.01"},
    "V3_linear_ramp":      {"eps": None, "gamma": GAMMA_C,   "label": "Linear ramp (baseline)"},
}
CORRIDORS_TO_RUN = {
    "SW_spine": CORRIDORS["SW_spine"],
    "NW_spine": CORRIDORS["NW_spine"],
}

VCOLS  = {"V1_hard_eps0":"#F44336","V2_hard_eps0_lowγ":"#9C27B0","V3_linear_ramp":"#4CAF50"}
VMARKS = {"V1_hard_eps0":"o","V2_hard_eps0_lowγ":"s","V3_linear_ramp":"^"}
VLINES = {"V1_hard_eps0":"-","V2_hard_eps0_lowγ":"-","V3_linear_ramp":"--"}


def simulate(G, nodes, Phi0, e_base_row, variant_cfg, eta=ETA_C, steps=STEPS, seed=42):
    eps   = variant_cfg["eps"]
    gamma = variant_cfg["gamma"]
    n     = len(nodes)
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    edges = list(G.edges())
    cfg_tmp  = PropagationConfig(eta=eta, gamma=gamma, steps=1, use_E=False, noise_std=0.0, seed=seed)
    prop_tmp = GraphPropagator(G, nodes, cfg_tmp)
    traj         = np.zeros((steps + 1, n))
    stress_bot   = np.zeros(steps + 1)
    E_bot        = np.zeros(steps + 1)
    blocked_mask = np.zeros(steps + 1, dtype=bool)
    traj[0] = Phi0.copy()
    bu, bv = BOTTLENECK_EDGE
    b_cap = G[bu][bv].get("capacity_gw", 1.0) if G.has_edge(bu, bv) else 1.0
    def _compute_L(phi_t):
        W = np.zeros((n, n))
        bot_stress = 0.0; bot_E = 0.0; n_blocked = 0
        for u, v in edges:
            col = f"{u}_{v}"; rev = f"{v}_{u}"
            key = col if col in e_base_row.index else rev
            e_b = float(e_base_row.get(key, 0.5))
            cap = G[u][v].get("capacity_gw", 1.0)
            if u in n_idx and v in n_idx:
                dphi = abs(phi_t[n_idx[u]] - phi_t[n_idx[v]])
                stress = dphi / cap
            else:
                stress = 0.0
            if eps is None:
                e_val = 0.0 if stress >= STRESS_THRESHOLD else e_b * (1.0 - stress / STRESS_THRESHOLD)
            else:
                if stress >= STRESS_THRESHOLD:
                    e_val = e_b * eps
                    if (u,v)==(bu,bv) or (v,u)==(bu,bv): n_blocked += 1
                else:
                    e_val = e_b
            if (u==bu and v==bv) or (u==bv and v==bu):
                bot_stress = stress; bot_E = e_val
            i, j = n_idx[u], n_idx[v]
            W[i,j] = W[j,i] = cap * e_val
        d = W.sum(axis=1); d_inv = np.where(d > 1e-9, 1.0/d, 0.0)
        L_cur = np.diag(d_inv) @ (np.diag(d) - W)
        return L_cur, bot_stress, bot_E, n_blocked > 0
    for t in range(1, steps + 1):
        phi_t = traj[t-1].copy()
        L_cur, bs, be, blocked = _compute_L(phi_t)
        stress_bot[t] = bs; E_bot[t] = be; blocked_mask[t] = blocked
        traj[t] = phi_t - eta * (L_cur @ phi_t) - gamma * phi_t
    dphi0 = abs(Phi0[n_idx[bu]] - Phi0[n_idx[bv]]) if bu in n_idx and bv in n_idx else 0
    stress_bot[0] = dphi0 / b_cap
    return traj, stress_bot, E_bot, blocked_mask


def measure_retention(traj, nodes, node, blocked_mask):
    ni = nodes.index(node) if node in nodes else None
    if ni is None: return {}
    series   = traj[:, ni]
    init_v   = float(series[0]); peak_v = float(series.max()); peak_t = int(series.argmax())
    net_rise = peak_v - init_v
    thresh_dwell = init_v + 0.75 * net_rise if net_rise > 1e-6 else peak_v
    dwell = int((series >= thresh_dwell).sum())
    uplift = np.maximum(series - init_v, 0)
    auc    = float(np.trapezoid(uplift))
    post = series[peak_t:]
    decay_rate = np.nan
    if len(post) > 3 and net_rise > 1e-6:
        norm_post = np.clip((post - init_v) / net_rise, 1e-9, None)
        t_ax = np.arange(len(norm_post))
        try:
            slope, _ = np.polyfit(t_ax, np.log(norm_post), 1)
            decay_rate = float(-slope)
        except Exception: pass
    return {"peak_phi": peak_v, "peak_t": peak_t, "net_rise": net_rise,
            "dwell_75": dwell, "auc": auc, "decay_rate": decay_rate, "init_phi": init_v}


def measure_isolation(traj, nodes, accum_node, terminal_node, blocked_mask):
    ai = nodes.index(accum_node) if accum_node in nodes else None
    ti = nodes.index(terminal_node) if terminal_node in nodes else None
    if ai is None or ti is None: return {}
    phi_a = traj[:, ai]; phi_t = traj[:, ti]
    init_wacm = phi_t[0]; ref = 0.05 * max(abs(init_wacm), 1e-6)
    arrival_delay = None
    for step in range(1, len(phi_t)):
        if abs(phi_t[step] - phi_t[step-1]) > ref:
            arrival_delay = step; break
    blocked_idx = np.where(blocked_mask)[0]
    corr_blocked = float(np.corrcoef(phi_a[blocked_idx], phi_t[blocked_idx])[0,1]) if len(blocked_idx) >= 3 else np.nan
    corr_full    = float(np.corrcoef(phi_a, phi_t)[0,1])
    n_isolated = 0
    for step in blocked_idx:
        if step < 1: continue
        da = abs(phi_a[step] - phi_a[step-1]); dw = abs(phi_t[step] - phi_t[step-1])
        if da > 1e-4 and dw < 0.10 * da: n_isolated += 1
    frac_isolated = n_isolated / max(len(blocked_idx), 1)
    return {"arrival_delay": arrival_delay, "peak_wacm": float(np.max(np.abs(phi_t))),
            "corr_blocked": corr_blocked, "corr_full": corr_full,
            "frac_isolated": frac_isolated, "n_blocked": int(blocked_mask.sum())}


def measure_branches(traj, nodes, G, blocked_mask, primary_terminal="WACM", branch_map=None):
    if branch_map is None: branch_map = {"PACE": 1.5, "NEVP": 0.8}
    ti = nodes.index(primary_terminal) if primary_terminal in nodes else None
    accum_i = nodes.index(ACCUM_NODE)
    branch_phi = {}; branch_auc = {}
    for bnd in branch_map:
        bi = nodes.index(bnd) if bnd in nodes else None
        if bi is None: continue
        series = traj[:, bi]
        branch_phi[bnd] = float(np.max(np.abs(series)))
        branch_auc[bnd] = float(np.trapezoid(np.abs(series)))
    primary_auc = float(np.trapezoid(np.abs(traj[:, ti]))) if ti else np.nan
    branch_total_auc = sum(branch_auc.values())
    blk_idx = np.where(blocked_mask)[0]
    if len(blk_idx) and ti:
        primary_blk  = float(np.mean(np.abs(traj[blk_idx, ti])))
        branch_blk   = {bnd: float(np.mean(np.abs(traj[blk_idx, nodes.index(bnd)]))) for bnd in branch_map if bnd in nodes}
        branch_blk_total = sum(branch_blk.values())
        branch_dominates = branch_blk_total > primary_blk
    else:
        primary_blk = np.nan; branch_blk = {}; branch_blk_total = np.nan; branch_dominates = False
    azps_peak = float(np.max(np.abs(traj[:, accum_i])))
    thresh_10  = 0.10 * azps_peak
    activation_times = {}
    for bnd in branch_map:
        bi = nodes.index(bnd) if bnd in nodes else None
        if bi is None: continue
        series = np.abs(traj[:, bi])
        for step in range(len(series)):
            if series[step] >= thresh_10:
                activation_times[bnd] = step; break
    sorted_act  = sorted(activation_times.items(), key=lambda x: x[1])
    first_wave  = sorted_act[0][0] if len(sorted_act) > 0 else None
    second_wave = sorted_act[1][0] if len(sorted_act) > 1 else None
    return {"branch_peak": branch_phi, "branch_auc": branch_auc, "primary_auc": primary_auc,
            "branch_total_auc": branch_total_auc, "primary_blk_mean": primary_blk,
            "branch_blk_mean": branch_blk, "branch_blk_total": branch_blk_total,
            "branch_dominates": branch_dominates, "first_wave": first_wave, "second_wave": second_wave,
            "activation_times": activation_times}


def classify_regime(ret, iso, bra, loading, stress_max):
    blocked    = iso.get("n_blocked", 0) > 0
    dwell      = ret.get("dwell_75", 0)
    frac_iso   = iso.get("frac_isolated", 0.0)
    branch_dom = bra.get("branch_dominates", False)
    decay      = ret.get("decay_rate", 0.0)
    if not blocked:
        return "throttled_dispersal" if stress_max >= 0.8 else "linear_dispersal"
    if branch_dom and frac_iso > 0.30: return "isolated_rerouting"
    if dwell >= 3: return "retained_accumulation"
    return "throttled_dispersal"


def run_all(G, nodes, E1_df, bg_std):
    records = []; traj_store = {}
    SW_ALT_NODES = ["PACE", "PSCO", "NEVP", "IPCO"]
    SW_PATH = CORRIDORS["SW_spine"]["path"]
    e_base  = E1_df.iloc[0]
    for corr_name, corr_spec in CORRIDORS_TO_RUN.items():
        path = corr_spec["path"]
        is_SW = (corr_name == "SW_spine")
        bmap  = {"PACE":1.5,"NEVP":0.8} if is_SW else {"IPCO":1.8,"NEVP":1.0}
        accum = ACCUM_NODE if is_SW else path[1]
        term  = TERMINAL_NODE if is_SW else path[-1]
        branch_nodes = set()
        for nd, brs in corr_spec["branches_from"].items(): branch_nodes.update(brs)
        branch_nodes -= set(path)
        for vname, vcfg in VARIANTS.items():
            for loading in LOADING_LEVELS:
                loading_abs = loading * bg_std
                Phi0 = init_corridor_phi(G, nodes, path, loading_abs)
                traj, stress_bot, E_bot, blocked_mask = simulate(G, nodes, Phi0, e_base, vcfg)
                traj_store[(corr_name, vname, loading)] = (traj, stress_bot, E_bot, blocked_mask)
                ret = measure_retention(traj, nodes, accum, blocked_mask)
                iso = measure_isolation(traj, nodes, accum, term, blocked_mask)
                bra = measure_branches(traj, nodes, G, blocked_mask, primary_terminal=term, branch_map=bmap)
                regime = classify_regime(ret, iso, bra, loading, stress_bot.max())
                row = {"corridor": corr_name, "variant": vname, "label": vcfg["label"],
                       "gamma": vcfg["gamma"], "loading": loading, "loading_abs": loading_abs,
                       "stress_max": float(stress_bot.max()), "n_blocked": int(blocked_mask.sum()),
                       "peak_phi_accum": ret.get("peak_phi",np.nan), "peak_t_accum": ret.get("peak_t",np.nan),
                       "net_rise": ret.get("net_rise",np.nan), "dwell_75": ret.get("dwell_75",np.nan),
                       "auc_accum": ret.get("auc",np.nan), "decay_rate": ret.get("decay_rate",np.nan),
                       "arrival_delay": iso.get("arrival_delay",np.nan), "peak_wacm": iso.get("peak_wacm",np.nan),
                       "corr_blocked": iso.get("corr_blocked",np.nan), "corr_full": iso.get("corr_full",np.nan),
                       "frac_isolated": iso.get("frac_isolated",np.nan),
                       "primary_auc": bra.get("primary_auc",np.nan), "branch_total_auc": bra.get("branch_total_auc",np.nan),
                       "branch_dominates": bra.get("branch_dominates",False),
                       "first_wave": bra.get("first_wave",None), "second_wave": bra.get("second_wave",None),
                       "peak_PACE": bra.get("branch_peak",{}).get("PACE",bra.get("branch_peak",{}).get("IPCO",np.nan)),
                       "peak_NEVP": bra.get("branch_peak",{}).get("NEVP",np.nan),
                       "branch_blk_total": bra.get("branch_blk_total",np.nan),
                       "primary_blk_mean": bra.get("primary_blk_mean",np.nan), "regime": regime}
                records.append(row)
                log.info("  %-10s %-22s %3.1fσ  blk=%2d  rise=%.3f  dwell=%2d  [%s]",
                         corr_name, vname, loading, int(blocked_mask.sum()),
                         ret.get("net_rise",0), int(ret.get("dwell_75",0)), regime)
    df = pd.DataFrame(records)
    df.to_csv(OUT_DAT / "sw_variants.csv", index=False)
    return df, traj_store


def _plot_metrics(df, corr, axes, metrics):
    sub = df[df["corridor"] == corr]
    for ax, (col, ylabel) in zip(axes, metrics):
        for vname in VARIANTS:
            sub_m = sub[sub["variant"] == vname].sort_values("loading")
            ax.plot(sub_m["loading"], sub_m[col], color=VCOLS[vname], ls=VLINES[vname],
                    lw=1.5, marker=VMARKS[vname], ms=5, label=VARIANTS[vname]["label"])
        ax.set_xlabel("Loading (× σ)"); ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.axvline(1.4, color="red", lw=0.7, ls=":", alpha=0.5)
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)


def fig_panel_A(df, traj_store, nodes):
    sw = df[df["corridor"] == "SW_spine"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col_idx, load in enumerate([1.5, 2.0, 2.5]):
        ax = axes[0, col_idx]
        for vname in VARIANTS:
            key = ("SW_spine", vname, load)
            data = traj_store.get(key, (None,))[0]
            if data is None: continue
            ai = nodes.index(ACCUM_NODE)
            ax.plot(range(len(data)), data[:, ai], color=VCOLS[vname], ls=VLINES[vname],
                    lw=1.6, marker=VMARKS[vname], ms=3, markevery=5, label=VARIANTS[vname]["label"])
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        ax.set_title(f"AZPS Phi — loading {load:.1f}σ", fontsize=9)
        ax.set_xlabel("Step"); ax.set_ylabel("Phi at AZPS")
        if col_idx == 0: ax.legend(fontsize=7)
    _plot_metrics(df, "SW_spine", axes[1, :], [
        ("peak_phi_accum","Peak |Phi| at AZPS"),
        ("dwell_75","Dwell time above 75% peak (steps)"),
        ("auc_accum","AUC of |Phi| at AZPS")])
    fig.suptitle("Panel A: Retention Plateau at AZPS (SW Spine)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "SV_A_retention.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved SV_A_retention.png")


def fig_panel_B(df, traj_store, nodes, high_load=2.5):
    sw = df[df["corridor"] == "SW_spine"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col_idx, vname in enumerate(VARIANTS):
        ax = axes[0, col_idx]
        key = ("SW_spine", vname, high_load)
        data, _, _, blocked = traj_store.get(key, (None,None,None,None))
        if data is None: continue
        ai = nodes.index(ACCUM_NODE); ti = nodes.index(TERMINAL_NODE)
        ax.plot(range(len(data)), data[:, ai], color="#F44336", lw=1.6, label="AZPS (accum)")
        ax.plot(range(len(data)), data[:, ti], color="#2196F3", lw=1.6, label="WACM (terminal)")
        if blocked is not None:
            for t in range(len(blocked)):
                if blocked[t]: ax.axvspan(t-0.5, t+0.5, alpha=0.12, color="red", zorder=0)
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
        ax.set_title(f"B. {VARIANTS[vname]['label']}\n{high_load:.1f}σ", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Phi"); ax.legend(fontsize=7)
    _plot_metrics(df, "SW_spine", axes[1, :], [
        ("corr_blocked","AZPS-WACM corr (blocked)"),
        ("frac_isolated","Fraction isolated steps"),
        ("peak_wacm","Peak |Phi| at WACM")])
    fig.suptitle("Panel B: WACM Isolation (SW Spine)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "SV_B_isolation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved SV_B_isolation.png")


def fig_panel_C(df, traj_store, nodes, high_load=2.5):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col_idx, vname in enumerate(VARIANTS):
        ax = axes[0, col_idx]
        key = ("SW_spine", vname, high_load)
        data = traj_store.get(key, (None,))[0]
        if data is None: continue
        for nd, col_c in [("PACE","#FF9800"),("NEVP","#9C27B0"),("WACM","#2196F3"),("AZPS","#F44336")]:
            if nd in nodes:
                ni = nodes.index(nd)
                ax.plot(range(len(data)), np.abs(data[:, ni]), color=col_c,
                        lw=1.8 if nd in ("WACM","AZPS") else 1.1,
                        ls="-" if nd in ("WACM","AZPS") else "--", label=nd)
        ax.set_title(f"C. {VARIANTS[vname]['label']}\n{high_load:.1f}σ", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("|Phi|"); ax.legend(fontsize=6)
    _plot_metrics(df, "SW_spine", axes[1, :], [
        ("primary_auc","AUC |Phi| at WACM (primary)"),
        ("branch_total_auc","AUC |Phi| PACE+NEVP (branch)"),
        ("peak_PACE","Peak |Phi| at PACE")])
    fig.suptitle("Panel C: Branch Dominance & Rerouting (SW Spine)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "SV_C_branches.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved SV_C_branches.png")


def fig_panel_D(df):
    sw = df[df["corridor"] == "SW_spine"]
    nw = df[df["corridor"] == "NW_spine"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    regime_map = {"linear_dispersal":0,"throttled_dispersal":1,"retained_accumulation":2,"isolated_rerouting":3}
    regime_colors = ["#4CAF50","#FF9800","#F44336","#9C27B0"]
    regime_labels = ["linear","throttled","retained","isolated+reroute"]
    cmap = mcolors.ListedColormap(regime_colors)
    for ax, df_sub, title in [(axes[0],sw,"D. SW Spine"),(axes[1],nw,"D. NW Spine (Control)")]:
        vnames = list(VARIANTS.keys()); loadings = sorted(df_sub["loading"].unique())
        mat = np.full((len(vnames), len(loadings)), 0)
        for vi, vn in enumerate(vnames):
            for li, load in enumerate(loadings):
                row = df_sub[(df_sub["variant"]==vn) & (df_sub["loading"]==load)]
                if not row.empty: mat[vi,li] = regime_map.get(row.iloc[0]["regime"],0)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks(range(len(loadings))); ax.set_xticklabels([f"{l:.1f}σ" for l in loadings], fontsize=8)
        ax.set_yticks(range(len(vnames))); ax.set_yticklabels([VARIANTS[v]["label"][:20] for v in vnames], fontsize=7)
        ax.set_title(title)
        for vi in range(len(vnames)):
            for li in range(len(loadings)):
                ax.text(li, vi, regime_labels[mat[vi,li]][:4], ha="center", va="center", fontsize=7, color="white")
    ax = axes[2]
    for corr, col in [("SW_spine","#F44336"),("NW_spine","#2196F3")]:
        sub = df[df["corridor"]==corr].groupby(["loading"])["stress_max"].first().reset_index()
        ax.plot(sub["loading"], sub["stress_max"], color=col, lw=1.8, marker="o", ms=6, label=corr)
    ax.axhline(STRESS_THRESHOLD, color="k", lw=1.0, ls="--", label=f"Threshold={STRESS_THRESHOLD}")
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Max bottleneck stress"); ax.set_title("D. Max stress: SW vs NW")
    ax.legend(fontsize=8)
    fig.suptitle("Panel D: Regime Classification + NW Control", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "SV_D_regime.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved SV_D_regime.png")


def fig_summary(df, traj_store, nodes, high_load=2.5):
    sw = df[df["corridor"] == "SW_spine"]
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    for vname, col, lbl in [("V2_hard_eps0_lowγ","#9C27B0","V2 (ε=0, γ=0.01)"),
                              ("V1_hard_eps0","#F44336","V1 (ε=0, γ=0.02)"),
                              ("V3_linear_ramp","#4CAF50","V3 (ramp)")]:
        key = ("SW_spine", vname, high_load)
        data = traj_store.get(key, (None,))[0]
        if data is None: continue
        ai = nodes.index(ACCUM_NODE)
        ax.plot(range(len(data)), np.abs(data[:,ai]), color=col, lw=2.0, label=lbl)
    ax.set_title(f"(A) AZPS retention — {high_load:.1f}σ"); ax.set_xlabel("Step"); ax.set_ylabel("|Phi| at AZPS"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 1])
    for vname, col, lbl in [("V2_hard_eps0_lowγ","#9C27B0","V2 WACM"),("V3_linear_ramp","#4CAF50","V3 WACM")]:
        key = ("SW_spine", vname, high_load)
        data, _, _, blk = traj_store.get(key, (None,None,None,None))
        if data is None: continue
        ti = nodes.index(TERMINAL_NODE)
        ax.plot(range(len(data)), data[:,ti], color=col, lw=1.8, label=lbl)
        if blk is not None:
            for t in range(len(blk)):
                if blk[t]: ax.axvspan(t-0.5, t+0.5, alpha=0.08, color="red", zorder=0)
    ax.set_title(f"(B) WACM decoupling — {high_load:.1f}σ"); ax.set_xlabel("Step"); ax.set_ylabel("Phi at WACM"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 0])
    sub_v2 = sw[sw["variant"]=="V2_hard_eps0_lowγ"].sort_values("loading")
    ax.plot(sub_v2["loading"], sub_v2["primary_auc"], color="#2196F3", lw=1.8, marker="o", ms=6, label="Primary (WACM) AUC")
    ax.plot(sub_v2["loading"], sub_v2["branch_total_auc"], color="#FF9800", lw=1.8, marker="s", ms=6, label="Branch (PACE+NEVP) AUC")
    cross = sub_v2[sub_v2["branch_dominates"] == True]
    if not cross.empty:
        ax.scatter(cross["loading"], cross["branch_total_auc"], s=120, color="red", zorder=6, marker="*", label="Branch dominates")
    ax.set_title("(C) Branch vs primary transfer — V2"); ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Cumulative |Phi| (AUC)"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 1])
    regime_map = {"linear_dispersal":0,"throttled_dispersal":1,"retained_accumulation":2,"isolated_rerouting":3}
    rlabels = ["linear","throttled","retained","iso+reroute"]
    cmap_r  = mcolors.ListedColormap(["#4CAF50","#FF9800","#F44336","#9C27B0"])
    vnames = list(VARIANTS.keys()); loadings = sorted(sw["loading"].unique())
    mat = np.full((len(vnames), len(loadings)), 0)
    for vi, vn in enumerate(vnames):
        for li, load in enumerate(loadings):
            row = sw[(sw["variant"]==vn) & (sw["loading"]==load)]
            if not row.empty: mat[vi,li] = regime_map.get(row.iloc[0]["regime"],0)
    im = ax.imshow(mat, aspect="auto", cmap=cmap_r, vmin=0, vmax=3)
    ax.set_xticks(range(len(loadings))); ax.set_xticklabels([f"{l:.1f}σ" for l in loadings])
    ax.set_yticks(range(len(vnames))); ax.set_yticklabels([VARIANTS[v]["label"][:24] for v in vnames], fontsize=7)
    ax.set_title("(D) Regime classification — SW spine")
    for vi in range(len(vnames)):
        for li in range(len(loadings)):
            ax.text(li, vi, rlabels[mat[vi,li]][:5], ha="center", va="center", fontsize=7, color="white")
    fig.suptitle("SW Spine Hard-Floor Variants: Retention, Isolation, Rerouting", fontsize=12)
    fig.savefig(OUT_FIG / "SV_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved SV_summary.png")


def write_report(df, bg_std):
    sep = "=" * 68
    sw  = df[df["corridor"] == "SW_spine"]
    nw  = df[df["corridor"] == "NW_spine"]
    lines = [sep, "  SW SPINE VARIANTS — REGIME-DETECTION FINDINGS", sep, "",
             f"  Background σ={bg_std:.4f}  η={ETA_C}",
             f"  V1: hard floor ε=0, γ={GAMMA_C}",
             f"  V2: hard floor ε=0, γ={GAMMA_LOW}",
             f"  V3: linear ramp to 0 at stress=1.0, γ={GAMMA_C}", ""]
    for cset_name, row_label in [("SW_spine","SW"), ("NW_spine","NW")]:
        sub = df[df["corridor"]==cset_name]
        lines += [f"─"*40, f"  {row_label} SPINE", f"─"*40, ""]
        for load in LOADING_LEVELS:
            for vname in VARIANTS:
                r = sub[(sub["variant"]==vname) & (sub["loading"]==load)]
                if not r.empty:
                    row = r.iloc[0]
                    lines.append(f"  {load:.1f}σ {vname}: regime={row.get('regime','?')}  "
                                 f"net_rise={row.get('net_rise',np.nan):.3f}  dwell={row.get('dwell_75',0):.0f}  "
                                 f"corr_blk={row.get('corr_blocked',np.nan):.3f}")
        lines.append("")
    lines.append(sep)
    report = "\n".join(lines)
    (OUT_DAT / "sw_variants_report.txt").write_text(report)
    print("\n" + report)
    return report


def main():
    log.info("Building graph and gradient-rich field …")
    G      = build_graph()
    nodes  = node_order(G)
    field  = build_gradient_rich_field(G, nodes, seed=42, T=1440)
    E1_df  = E1_price_spread_edge(G, field["R"])
    bg_std = float(field["Phi"].values.std(axis=1).mean())
    log.info("  σ=%.4f  γ variants: %.3f / %.3f", bg_std, GAMMA_C, GAMMA_LOW)
    df, traj_store = run_all(G, nodes, E1_df, bg_std)
    fig_panel_A(df, traj_store, nodes)
    fig_panel_B(df, traj_store, nodes)
    fig_panel_C(df, traj_store, nodes)
    fig_panel_D(df)
    fig_summary(df, traj_store, nodes)
    write_report(df, bg_std)
    log.info("Complete.")


if __name__ == "__main__":
    main()
