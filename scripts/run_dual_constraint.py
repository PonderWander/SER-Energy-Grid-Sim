"""
scripts/run_dual_constraint.py
================================
Dual-constraint cascade experiment on SW spine.

PRIMARY:   AZPS→WACM  cap=1.0  threshold=1.0
SECONDARY: PACE→WACM  cap=1.2  threshold=1.0
TERTIARY:  AZPS→NEVP  cap=0.8  threshold=1.0

Tests: multi-node accumulation, cascading isolation, network fragmentation.
Outputs: DC_accumulation.png, DC_cascade_timing.png, DC_isolation_map.png,
         DC_regime_map.png, DC_dashboard.png, dual_constraint_report.txt
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
import networkx as nx
import numpy as np
import pandas as pd

from constraint_field.graph.network import build_graph, node_order
from constraint_field.graph.edge_fluidity import E1_price_spread_edge
from constraint_field.graph.propagation import GraphPropagator, PropagationConfig
from scripts.run_spatial_experiments import build_gradient_rich_field
from scripts.run_corridor_loading import (
    init_corridor_phi, CORRIDORS, LOADING_LEVELS, STEPS, ETA_C
)
from scripts.run_sw_variants import GAMMA_LOW

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dual_constraint")

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)
DPI = 150

GAMMA  = GAMMA_LOW   # 0.01
ETA    = ETA_C       # 0.20
STEPS_DC = 80

CONSTRAINT_SETS = {
    "single": {("AZPS","WACM"): (1.0, 1.0)},
    "dual":   {("AZPS","WACM"): (1.0, 1.0), ("PACE","WACM"): (1.0, 1.2)},
    "triple": {("AZPS","WACM"): (1.0, 1.0), ("PACE","WACM"): (1.0, 1.2), ("AZPS","NEVP"): (1.0, 0.8)},
}
TRACK_NODES = ["CISO","WALC","AZPS","NEVP","PACE","WACM","PSCO"]
SW_PATH     = ["CISO","WALC","AZPS","WACM"]
CSET_COLORS = {"single":"#2196F3","dual":"#F44336","triple":"#9C27B0"}
CSET_LABELS = {
    "single":"Single (AZPS→WACM)",
    "dual":"Dual (AZPS→WACM + PACE→WACM)",
    "triple":"Triple (+ AZPS→NEVP)",
}
NODE_COLORS = {
    "CISO":"#1976D2","WALC":"#0097A7","AZPS":"#F44336",
    "PACE":"#FF9800","NEVP":"#9C27B0","WACM":"#4CAF50","PSCO":"#607D8B"
}


def simulate_dual(G, nodes, Phi0, e_base_row, constraints, eta=ETA, gamma=GAMMA, steps=STEPS_DC, seed=42):
    n = len(nodes); n_idx = {nd: i for i, nd in enumerate(nodes)}
    edges = list(G.edges())
    c_norm = {}
    for (u,v), params in constraints.items():
        key = f"{u}_{v}" if G.has_edge(u,v) else f"{v}_{u}"
        c_norm[key] = params; c_norm[f"{v}_{u}"] = params
    cfg_tmp  = PropagationConfig(eta=eta, gamma=gamma, steps=1, use_E=False, noise_std=0.0, seed=seed)
    prop_tmp = GraphPropagator(G, nodes, cfg_tmp)
    traj = np.zeros((steps+1, n)); traj[0] = Phi0.copy()
    c_keys       = list({f"{u}_{v}" if G.has_edge(u,v) else f"{v}_{u}" for u,v in constraints})
    block_record = {k: np.zeros(steps+1, dtype=bool) for k in c_keys}
    stress_record= {k: np.zeros(steps+1) for k in c_keys}
    def _build_L(phi_t):
        W = np.zeros((n, n))
        for u, v in edges:
            col = f"{u}_{v}"; rev = f"{v}_{u}"
            key = col if col in e_base_row.index else rev
            e_b = float(e_base_row.get(key, 0.5))
            cap_graph = G[u][v].get("capacity_gw", 1.0)
            dphi = abs(phi_t[n_idx[u]] - phi_t[n_idx[v]]) if u in n_idx and v in n_idx else 0.0
            ckey = col if col in c_norm else (rev if rev in c_norm else None)
            if ckey:
                threshold, cap_eff = c_norm[ckey]
                stress = dphi / cap_eff
                if col in stress_record: stress_record[col][-1] = stress
                elif rev in stress_record: stress_record[rev][-1] = stress
                if stress >= threshold:
                    e_val = 0.0
                    if col in block_record: block_record[col][-1] = True
                    elif rev in block_record: block_record[rev][-1] = True
                else:
                    e_val = e_b
            else:
                e_val = e_b
            i, j = n_idx[u], n_idx[v]; W[i,j] = W[j,i] = cap_graph * e_val
        d = W.sum(axis=1); d_inv = np.where(d > 1e-9, 1./d, 0.)
        return np.diag(d_inv) @ (np.diag(d) - W)
    for t in range(1, steps+1):
        phi_t = traj[t-1].copy()
        for k in block_record: block_record[k][-1] = False
        for k in stress_record: stress_record[k][-1] = 0.0
        L_cur = _build_L(phi_t)
        for k in block_record: block_record[k][t] = block_record[k][-1]
        for k in stress_record: stress_record[k][t] = stress_record[k][-1]
        traj[t] = phi_t - eta * (L_cur @ phi_t) - gamma * phi_t
    return traj, block_record, stress_record


def node_accumulation(traj, nodes, track_nodes):
    results = {}
    for nd in track_nodes:
        if nd not in nodes: continue
        ni = nodes.index(nd); series = traj[:, ni]
        init_v = float(series[0]); peak_v = float(series.max()); peak_t = int(series.argmax())
        net_rise = peak_v - init_v
        thresh_dwell = init_v + 0.75 * net_rise if net_rise > 1e-6 else peak_v
        dwell = int((series >= thresh_dwell).sum())
        results[nd] = {"init": init_v, "peak": peak_v, "net_rise": net_rise,
                       "peak_t": peak_t, "dwell_75": dwell}
    return results


def cascade_ordering(block_record):
    results = []
    for k, arr in block_record.items():
        if arr.any():
            onset    = int(arr.argmax())
            duration = int(arr.sum())
            results.append((onset, duration, k))
    return sorted(results, key=lambda x: -x[1])


def isolation_propagation(traj, nodes, block_record, ref_node="WACM"):
    ri = nodes.index(ref_node) if ref_node in nodes else None
    if ri is None: return {}
    all_blocked = np.zeros(len(traj), dtype=bool)
    for arr in block_record.values(): all_blocked |= arr
    blk_idx = np.where(all_blocked)[0]
    results = {}
    for nd in TRACK_NODES:
        ni = nodes.index(nd) if nd in nodes else None
        if ni is None or nd == ref_node: continue
        corr = float(np.corrcoef(traj[blk_idx,ni], traj[blk_idx,ri])[0,1]) if len(blk_idx) >= 3 else np.nan
        results[nd] = corr
    return results


def upstream_stress(traj, nodes, G):
    n_idx = {nd: i for i, nd in enumerate(nodes)}
    result = {}
    for u, v in [("CISO","WALC"),("WALC","AZPS")]:
        if not G.has_edge(u,v): continue
        cap = G[u][v]["capacity_gw"]
        ui, vi = n_idx[u], n_idx[v]
        result[f"{u}→{v}"] = np.abs(traj[:,ui] - traj[:,vi]) / cap
    return result


def classify_regime(accum_results, block_record, isolation_corrs, n_blocked_total):
    accum_nodes = [nd for nd, m in accum_results.items()
                   if m["net_rise"] > 0.05 and m["dwell_75"] > 3 and nd not in ("WACM",)]
    n_constraints_fired = sum(1 for arr in block_record.values() if arr.any())
    decoupled = [nd for nd, c in isolation_corrs.items() if not np.isnan(c) and c < 0.50]
    if n_constraints_fired >= 3 and len(accum_nodes) >= 3: return "network_fragmentation"
    elif n_constraints_fired >= 2 and len(decoupled) >= 2: return "cascading_isolation"
    elif len(accum_nodes) >= 2: return "multi_node_accumulation"
    elif len(accum_nodes) == 1: return "single_node_accumulation"
    else: return "linear_dispersal"


def run_all(G, nodes, E1_df, bg_std):
    records = []; traj_store = {}
    sw = CORRIDORS["SW_spine"]; e_base = E1_df.iloc[0]
    for cset_name, constraints in CONSTRAINT_SETS.items():
        for loading in LOADING_LEVELS:
            loading_abs = loading * bg_std
            Phi0 = init_corridor_phi(G, nodes, sw["path"], loading_abs)
            traj, block_record, stress_record = simulate_dual(G, nodes, Phi0, e_base, constraints)
            traj_store[(cset_name, loading)] = (traj, block_record, stress_record)
            accum   = node_accumulation(traj, nodes, TRACK_NODES)
            cascade = cascade_ordering(block_record)
            iso     = isolation_propagation(traj, nodes, block_record)
            up_str  = upstream_stress(traj, nodes, G)
            n_fired = sum(1 for arr in block_record.values() if arr.any())
            n_blk_total = sum(arr.sum() for arr in block_record.values())
            regime = classify_regime(accum, block_record, iso, int(n_blk_total))
            first_block_t   = cascade[0][0]  if cascade else np.nan
            first_block_dur = cascade[0][1]  if cascade else np.nan
            first_block_e   = cascade[0][2]  if cascade else None
            second_block_t  = cascade[1][0]  if len(cascade)>1 else np.nan
            second_block_dur= cascade[1][1]  if len(cascade)>1 else np.nan
            second_block_e  = cascade[1][2]  if len(cascade)>1 else None
            max_up_str = max((s.max() for s in up_str.values()), default=np.nan)
            row = {"cset": cset_name, "loading": loading, "loading_abs": loading_abs,
                   "regime": regime, "n_constraints": len(constraints), "n_fired": n_fired,
                   "n_blk_total": int(n_blk_total), "first_block_t": first_block_t,
                   "first_block_dur": first_block_dur, "first_block_e": first_block_e,
                   "second_block_t": second_block_t, "second_block_dur": second_block_dur,
                   "second_block_e": second_block_e, "max_upstream_str": max_up_str}
            for nd in TRACK_NODES:
                if nd in accum:
                    row[f"rise_{nd}"] = accum[nd]["net_rise"]
                    row[f"dwell_{nd}"] = accum[nd]["dwell_75"]
                    row[f"peak_{nd}"] = accum[nd]["peak"]
            for nd, corr in iso.items():
                row[f"iso_{nd}"] = corr
            records.append(row)
            accum_nodes = [nd for nd,m in accum.items() if m["net_rise"]>0.05 and m["dwell_75"]>3 and nd!="WACM"]
            log.info("  %-8s %3.1fσ  fired=%d  regime=%-28s  accum=%s", cset_name, loading, n_fired, regime, accum_nodes)
    df = pd.DataFrame(records)
    df.to_csv(OUT_DAT / "dual_constraint.csv", index=False)
    return df, traj_store


def fig_accumulation_profiles(df, traj_store, nodes, high_load=2.5):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for col_idx, (cset, ax) in enumerate(zip(CONSTRAINT_SETS, axes)):
        traj, block_rec, _ = traj_store[(cset, high_load)]
        for nd in TRACK_NODES:
            if nd not in nodes: continue
            ni = nodes.index(nd)
            ax.plot(range(len(traj)), traj[:,ni], color=NODE_COLORS.get(nd,"#607D8B"),
                    lw=2.0 if nd in ("AZPS","PACE","WACM") else 1.0,
                    ls="-" if nd in ("CISO","WALC","AZPS","WACM") else "--", label=nd)
        for (k,arr), sc in zip(block_rec.items(), ["#F44336","#FF9800","#9C27B0"]):
            for t in range(1,len(arr)):
                if arr[t]: ax.axvspan(t-0.5,t+0.5,alpha=0.07,color=sc,zorder=0)
        ax.axhline(0,color="k",lw=0.4,ls="--",alpha=0.3); ax.set_xlabel("Step")
        ax.set_title(f"{CSET_LABELS[cset]}\n{high_load:.1f}σ loading", fontsize=8)
        if col_idx == 1: ax.legend(fontsize=6,ncol=2,loc="upper right")
    fig.suptitle(f"Accumulation Profiles at {high_load:.1f}σ — Single vs Dual vs Triple", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "DC_accumulation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved DC_accumulation.png")


def fig_cascade_timing(df, traj_store, nodes):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    dual = df[df["cset"]=="dual"].sort_values("loading")
    ax.plot(dual["loading"], dual["first_block_dur"], color="#F44336", lw=1.8, marker="o", ms=6, label="Primary (AZPS→WACM) blocked steps")
    ax.plot(dual["loading"], dual["second_block_dur"], color="#FF9800", lw=1.8, marker="s", ms=6, label="Secondary (PACE→WACM) blocked steps")
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Steps blocked"); ax.set_title("(A) Cascade duration — dual constraint"); ax.legend(fontsize=7)
    ax = axes[1]
    for cset, col in CSET_COLORS.items():
        sub = df[df["cset"]==cset].sort_values("loading")
        if "rise_AZPS" in sub.columns:
            ax.plot(sub["loading"], sub["rise_AZPS"], color=col, lw=1.5, marker="o", ms=5, label=f"{cset} — AZPS")
        if "rise_PACE" in sub.columns:
            ax.plot(sub["loading"], sub["rise_PACE"], color=col, lw=1.0, ls="--", marker="s", ms=4, label=f"{cset} — PACE")
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Net rise from initial Phi"); ax.set_title("(B) AZPS and PACE net accumulation"); ax.legend(fontsize=5,ncol=2)
    ax = axes[2]
    for cset, col in CSET_COLORS.items():
        sub = df[df["cset"]==cset].sort_values("loading")
        ax.plot(sub["loading"], sub["max_upstream_str"], color=col, lw=1.5, marker="o", ms=5, label=CSET_LABELS[cset])
    ax.axhline(1.0, color="red", lw=0.8, ls=":", alpha=0.7, label="Threshold = 1.0")
    ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Max upstream edge stress"); ax.set_title("(C) Upstream corridor stress"); ax.legend(fontsize=6)
    fig.suptitle("Cascade Timing and Upstream Propagation", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "DC_cascade_timing.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved DC_cascade_timing.png")


def fig_isolation_map(df, traj_store, nodes, G, high_load=2.5):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    csets = list(CONSTRAINT_SETS.keys())
    pos   = {nd: (G.nodes[nd]["lon"], G.nodes[nd]["lat"]) for nd in G.nodes()}
    t_snap = 30
    for col_idx, cset in enumerate(csets):
        ax = axes[col_idx]
        traj, block_rec, _ = traj_store.get((cset, high_load), (None,None,None))
        if traj is None: ax.set_title(f"{cset} — no data"); continue
        phi_snap = traj[min(t_snap, len(traj)-1)]
        vmax = max(abs(phi_snap).max(), 0.5)
        cnorm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        nc = [plt.cm.RdBu_r(cnorm(phi_snap[nodes.index(nd)])) for nd in nodes]
        sz = [100 + 600*abs(phi_snap[nodes.index(nd)]) for nd in nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=sz, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        constraint_keys = set()
        for (u,v) in CONSTRAINT_SETS[cset]: constraint_keys.add((u,v)); constraint_keys.add((v,u))
        for u, v in G.edges():
            blocked_now = any(arr[min(t_snap, len(arr)-1)] for k, arr in block_rec.items()
                              if k.split("_")[0] in (u,v) and k.split("_")[1] in (u,v))
            is_c = (u,v) in constraint_keys or (v,u) in constraint_keys
            col  = "#F44336" if blocked_now else ("#FF9800" if is_c else "#BDBDBD")
            lw   = 3.0 if blocked_now else (2.0 if is_c else 0.8)
            ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]], color=col, lw=lw, alpha=0.85, zorder=3)
        n_fired = sum(1 for arr in block_rec.values() if arr[min(t_snap,len(arr)-1)])
        ax.set_title(f"{cset}  t={t_snap}  {high_load:.1f}σ\n{n_fired}/{len(CONSTRAINT_SETS[cset])} edges blocked", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Spatial Phi Snapshot — t={t_snap}, loading={high_load:.1f}σ", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "DC_isolation_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved DC_isolation_map.png")


def fig_regime_map(df):
    regime_ord = {"linear_dispersal":0,"single_node_accumulation":1,"multi_node_accumulation":2,"cascading_isolation":3,"network_fragmentation":4}
    regime_labels = ["linear","single-accum","multi-accum","cascade-iso","fragmented"]
    cmap = mcolors.ListedColormap(["#4CAF50","#2196F3","#FF9800","#F44336","#9C27B0"])
    csets = list(CONSTRAINT_SETS.keys()); loadings = sorted(df["loading"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    mat = np.zeros((len(csets), len(loadings)))
    for ri, cset in enumerate(csets):
        for ci, load in enumerate(loadings):
            row = df[(df["cset"]==cset) & (df["loading"]==load)]
            if not row.empty: mat[ri,ci] = regime_ord.get(row.iloc[0]["regime"],0)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(range(len(loadings))); ax.set_xticklabels([f"{l:.1f}σ" for l in loadings])
    ax.set_yticks(range(len(csets))); ax.set_yticklabels([CSET_LABELS[c] for c in csets], fontsize=8)
    ax.set_xlabel("Upstream loading"); ax.set_title("Regime Classification: Single / Dual / Triple Constraint")
    for ri in range(len(csets)):
        for ci in range(len(loadings)):
            ax.text(ci, ri, regime_labels[int(mat[ri,ci])], ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "DC_regime_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved DC_regime_map.png")


def fig_summary_dashboard(df, traj_store, nodes, G, high_load=2.5):
    sw = df[df["corridor"] if "corridor" in df.columns else "cset"]
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    for load, ls in [(1.5,"-."),(2.0,"--"),(2.5,"-")]:
        traj, brec, _ = traj_store.get(("dual", load), (None,None,None))
        if traj is None: continue
        ai = nodes.index("AZPS"); pi = nodes.index("PACE")
        ax.plot(range(len(traj)), traj[:,ai], color="#F44336", ls=ls, lw=1.5, label=f"AZPS {load:.1f}σ")
        ax.plot(range(len(traj)), traj[:,pi], color="#FF9800", ls=ls, lw=1.2, label=f"PACE {load:.1f}σ")
    ax.axhline(0,color="k",lw=0.4,ls="--",alpha=0.3)
    ax.set_title("(A) AZPS and PACE — dual constraint"); ax.set_xlabel("Step"); ax.set_ylabel("Phi"); ax.legend(fontsize=6,ncol=2)
    ax = fig.add_subplot(gs[0, 1])
    dual_df = df[df["cset"]=="dual"].sort_values("loading")
    for nd, col in [("AZPS","#F44336"),("PACE","#FF9800"),("WALC","#0097A7"),("CISO","#1976D2")]:
        col_name = f"rise_{nd}"
        if col_name in dual_df.columns:
            ax.plot(dual_df["loading"], dual_df[col_name], color=col, lw=1.6, marker="o", ms=5, label=nd)
    ax.axvline(1.4,color="gray",lw=0.7,ls=":",alpha=0.5); ax.set_xlabel("Loading (× σ)"); ax.set_ylabel("Net Phi rise")
    ax.set_title("(B) Accumulation by node — dual"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 0])
    traj, brec, _ = traj_store.get(("dual", 2.5), (None,None,None))
    if traj is not None:
        wacm_i = nodes.index("WACM")
        all_blk = np.zeros(len(traj), dtype=bool)
        for arr in brec.values(): all_blk |= arr
        blk_idx = np.where(all_blk)[0]
        nd_list = [n for n in TRACK_NODES if n != "WACM" and n in nodes]
        corrs = []
        for nd in nd_list:
            ni = nodes.index(nd)
            c = np.corrcoef(traj[blk_idx,ni],traj[blk_idx,wacm_i])[0,1] if len(blk_idx) >= 3 else np.nan
            corrs.append(c)
        colors = [NODE_COLORS.get(nd,"#607D8B") for nd in nd_list]
        ax.bar(nd_list, corrs, color=colors, alpha=0.85)
        ax.axhline(0.5, color="red", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)
    ax.set_ylabel("Correlation with WACM (blocked steps)"); ax.set_title("(C) Node isolation from WACM (dual, 2.5σ)"); ax.set_ylim(-1,1)
    ax = fig.add_subplot(gs[1, 1])
    regime_ord = {"linear_dispersal":0,"single_node_accumulation":1,"multi_node_accumulation":2,"cascading_isolation":3,"network_fragmentation":4}
    rlabels = ["linear","single","multi","cascade","fragment"]
    cmap_r  = mcolors.ListedColormap(["#4CAF50","#2196F3","#FF9800","#F44336","#9C27B0"])
    csets_l = list(CONSTRAINT_SETS.keys()); loadings = sorted(df["loading"].unique())
    mat = np.zeros((len(csets_l), len(loadings)))
    for ri, cset in enumerate(csets_l):
        for ci, load in enumerate(loadings):
            row = df[(df["cset"]==cset) & (df["loading"]==load)]
            if not row.empty: mat[ri,ci] = regime_ord.get(row.iloc[0]["regime"],0)
    im = ax.imshow(mat, aspect="auto", cmap=cmap_r, vmin=0, vmax=4)
    ax.set_xticks(range(len(loadings))); ax.set_xticklabels([f"{l:.1f}σ" for l in loadings])
    ax.set_yticks(range(len(csets_l))); ax.set_yticklabels([CSET_LABELS[c][:28] for c in csets_l], fontsize=7)
    ax.set_title("(D) Regime map")
    for ri in range(len(csets_l)):
        for ci in range(len(loadings)):
            ax.text(ci, ri, rlabels[int(mat[ri,ci])], ha="center", va="center", fontsize=7, color="white")
    fig.suptitle("Dual-Constraint Cascade: SW Spine", fontsize=12)
    fig.savefig(OUT_FIG / "DC_dashboard.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved DC_dashboard.png")


def write_report(df, bg_std):
    sep = "=" * 68
    lines = [sep, "  DUAL-CONSTRAINT CASCADE — FINDINGS", sep, "",
             f"  Background σ={bg_std:.4f}  η={ETA}  γ={GAMMA}  ε=0 (hard floor)",
             f"  PRIMARY:   AZPS→WACM  cap=1.0  threshold=1.0",
             f"  SECONDARY: PACE→WACM  cap=1.2  threshold=1.0",
             f"  TERTIARY:  AZPS→NEVP  cap=0.8  threshold=1.0  (triple only)", ""]
    for cset in CONSTRAINT_SETS:
        sub = df[df["cset"]==cset]
        lines += [f"─"*68, f"  {CSET_LABELS[cset]}", f"─"*68, ""]
        for _, row in sub.sort_values("loading").iterrows():
            azps_r = row.get("rise_AZPS",np.nan); pace_r = row.get("rise_PACE",np.nan)
            ft = row.get("first_block_t",np.nan); st = row.get("second_block_t",np.nan)
            fd = row.get("first_block_dur",np.nan); sd = row.get("second_block_dur",np.nan)
            lines.append(f"  {row['loading']:5.1f}σ  regime={row['regime']:<28}  fired={row['n_fired']:2.0f}"
                         f"  AZPS_rise={azps_r:.4f}  PACE_rise={pace_r:.4f}"
                         f"  1st_dur={fd:.0f}  2nd_dur={sd:.0f}")
        lines.append("")
    dual_25 = df[(df["cset"]=="dual") & (df["loading"]==2.5)]
    if not dual_25.empty:
        r = dual_25.iloc[0]
        fdu = r.get("first_block_dur",np.nan); sdu = r.get("second_block_dur",np.nan)
        fe = r.get("first_block_e","?"); se = r.get("second_block_e","?")
        if not np.isnan(fdu) and not np.isnan(sdu):
            lines.append(f"  CASCADE at 2.5σ: {fe} blocks for {int(fdu)} steps; {se} blocks for {int(sdu)} steps")
    lines.append(sep)
    report = "\n".join(lines)
    (OUT_DAT / "dual_constraint_report.txt").write_text(report)
    print("\n" + report)
    return report


def main():
    global G
    log.info("Building graph and field …")
    G      = build_graph()
    nodes_ = node_order(G)
    field  = build_gradient_rich_field(G, nodes_, seed=42, T=1440)
    E1_df  = E1_price_spread_edge(G, field["R"])
    bg_std = float(field["Phi"].values.std(axis=1).mean())
    log.info("  σ=%.4f  γ=%.3f  η=%.2f  steps=%d", bg_std, GAMMA, ETA, STEPS_DC)
    df, traj_store = run_all(G, nodes_, E1_df, bg_std)
    nodes = nodes_
    fig_accumulation_profiles(df, traj_store, nodes)
    fig_cascade_timing(df, traj_store, nodes)
    fig_isolation_map(df, traj_store, nodes, G)
    fig_regime_map(df)
    fig_summary_dashboard(df, traj_store, nodes, G)
    write_report(df, bg_std)
    log.info("Complete.")


if __name__ == "__main__":
    main()
