"""
scripts/run_spatial_experiments.py
====================================
Four targeted experiments with gradient-rich node forcing.

Change from previous suite
--------------------------
The Cholesky synthetic model produced between-node variance ratio of 0.0016
(temporal variance dominated; nodes were nearly spatially identical).
This made all spatial metrics degenerate.

Fix: gradient-rich forcing assigns persistent structural offsets by
climate zone, creating sustained inter-node Phi gradients that the
Laplacian diffusion operator can act upon.

No new operators. Laplacian diffusion only. η in diffusion-competitive regime.
Primary evaluation: spatial metrics (Moran's I, spatial std, distance-decay).

Experiments
-----------
1. Topology sensitivity
2. E ablation
3. Localised shock propagation
4. Graph-distance decay
"""

from __future__ import annotations

import logging
import sys
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
from scipy import stats as scipy_stats

from constraint_field.graph.network import build_graph, node_order, NODES
from constraint_field.graph.node_signals import SyntheticNodeSignals, build_node_field
from constraint_field.graph.edge_fluidity import (
    E1_price_spread_edge,
    weighted_adjacency,
    graph_laplacian,
    constant_laplacian,
)
from constraint_field.graph.propagation import GraphPropagator, PropagationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spatial_exp")

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)
DPI = 150

ETA   = 0.20
GAMMA = 0.02
STEPS = 72
START_T = 500

ZONE_R_OFFSET = {
    "AZPS":  +1.80, "WALC":  +1.40, "IID":   +1.60, "NEVP":  +0.90,
    "BPAT":  -1.80, "PACW":  -1.40, "IPCO":  -1.20, "NWMT":  -1.00,
    "CISO":   0.00, "LDWP":  -0.20, "TIDC":  +0.10,
    "PACE":  +0.50, "PSCO":  +0.70, "WACM":  +0.40,
}
ZONE_S_OFFSET = {
    "AZPS":  +0.90, "WALC":  +0.60, "IID":   +0.80, "NEVP":  +0.40,
    "BPAT":  -0.60, "PACW":  -0.50, "IPCO":  -0.40, "NWMT":  -0.30,
    "CISO":   0.00, "LDWP":  +0.20, "TIDC":  -0.10,
    "PACE":  +0.20, "PSCO":  +0.30, "WACM":  +0.10,
}


def build_gradient_rich_field(G, nodes, seed=42, T=1440):
    rng   = np.random.default_rng(seed)
    n     = len(nodes)
    idx   = pd.date_range("2023-01-01", periods=T, freq="1h", tz="UTC")
    hour_of_day = np.arange(T) % 24
    diurnal     = (0.4 * np.sin(2 * np.pi * hour_of_day / 24)
                   + 0.2 * np.sin(4 * np.pi * hour_of_day / 24))
    system_noise = rng.normal(0, 0.3, T)
    system_base  = diurnal + system_noise

    S_raw = np.zeros((T, n))
    for i, nd in enumerate(nodes):
        s_offset  = ZONE_S_OFFSET.get(nd, 0.0)
        idio      = rng.normal(0, 0.4, T)
        lat_shift = int((G.nodes[nd]["lat"] - 37.5) / 3)
        base_shift = np.roll(system_base, lat_shift)
        S_raw[:, i] = base_shift + s_offset + idio

    R_raw = np.zeros((T, n))
    for i, nd in enumerate(nodes):
        r_offset = ZONE_R_OFFSET.get(nd, 0.0)
        vol_mult  = 1.4 if r_offset > 1.0 else (0.7 if r_offset < -1.0 else 1.0)
        idio_p    = rng.normal(0, 0.5 * vol_mult, T)
        n_spikes  = rng.poisson(int(T * 0.015))
        spike_t   = rng.integers(0, T, n_spikes)
        spike_mag = rng.uniform(1.5, 4.0, n_spikes) * np.sign(r_offset + 0.1)
        spikes    = np.zeros(T)
        for st, sm in zip(spike_t, spike_mag):
            for k in range(min(6, T - st)):
                spikes[st + k] += sm * np.exp(-k * 0.5)
        R_raw[:, i] = system_base * 0.6 + r_offset + idio_p + spikes

    window = 168
    def rolling_zscore(arr):
        df  = pd.DataFrame(arr, index=idx)
        mu  = df.rolling(window, min_periods=max(1, window//4)).mean()
        sig = df.rolling(window, min_periods=max(1, window//4)).std().replace(0, np.nan)
        z   = (df - mu) / sig
        gz  = (df - df.mean()) / df.std().replace(0, 1)
        z   = z.fillna(gz)
        return z.clip(-3, 3).values

    S_norm = rolling_zscore(S_raw)
    R_norm = rolling_zscore(R_raw)
    S   = pd.DataFrame(S_norm, index=idx, columns=nodes)
    R   = pd.DataFrame(R_norm, index=idx, columns=nodes)
    Phi = R - S
    Psi = np.sqrt(S**2 + R**2)

    phi_arr         = Phi.values
    spatial_std     = phi_arr.std(axis=1).mean()
    between_var     = phi_arr.mean(axis=0).var()
    within_var      = np.array([phi_arr[:,i].var() for i in range(n)]).mean()
    ratio           = between_var / within_var if within_var > 0 else 0
    log.info(
        "Gradient-rich field: T=%d  nodes=%d\n"
        "  mean spatial std:       %.4f\n"
        "  between-node variance:  %.4f\n"
        "  within-node variance:   %.4f\n"
        "  gradient richness ratio:%.4f  (prev: 0.0016)",
        T, n, spatial_std, between_var, within_var, ratio,
    )
    return {"S": S, "R": R, "Phi": Phi, "Psi": Psi}


def adjacency_W(G, nodes):
    n = len(nodes)
    idx_map = {nd: i for i, nd in enumerate(nodes)}
    A = np.zeros((n, n))
    for u, v in G.edges():
        if u in idx_map and v in idx_map:
            A[idx_map[u], idx_map[v]] = A[idx_map[v], idx_map[u]] = 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def moran_i(values, W):
    z  = values - values.mean()
    W0 = W.sum()
    if W0 == 0 or z.std() < 1e-9:
        return 0.0
    return float(len(z) * (z @ W @ z) / (W0 * (z @ z)))


def spatial_metrics(traj, nodes, W, field_Phi=None):
    sim_cols = [f"{nd}_sim" for nd in nodes if f"{nd}_sim" in traj.columns]
    obs_cols = [f"{nd}_obs" for nd in nodes if f"{nd}_obs" in traj.columns]
    sim_arr  = traj[sim_cols].values.astype(float)
    obs_arr  = traj[obs_cols].values.astype(float) if obs_cols else None
    morans   = np.array([moran_i(sim_arr[t], W) for t in range(len(sim_arr))])
    sp_stds  = sim_arr.std(axis=1)
    ranges   = sim_arr.max(axis=1) - sim_arr.min(axis=1)
    if len(sp_stds) > 2:
        persistence = float(np.corrcoef(sp_stds[:-1], sp_stds[1:])[0, 1])
    else:
        persistence = np.nan
    if obs_arr is not None:
        resid = sim_arr - obs_arr
        rmse  = float(np.sqrt((resid**2).mean()))
    else:
        rmse = np.nan
    return {
        "moran_mean":    float(morans.mean()),
        "moran_std":     float(morans.std()),
        "moran_min":     float(morans.min()),
        "spatial_std":   float(sp_stds.mean()),
        "gradient_mag":  float(ranges.mean()),
        "persistence":   persistence,
        "rmse":          rmse,
    }


def node_distances(G, nodes):
    n       = len(nodes)
    idx_map = {nd: i for i, nd in enumerate(nodes)}
    D       = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    sp      = dict(nx.all_pairs_shortest_path_length(G))
    for u, hops in sp.items():
        for v, h in hops.items():
            if u in idx_map and v in idx_map:
                D[idx_map[u], idx_map[v]] = h
    return D


def geo_distances(G, nodes):
    import math
    n       = len(nodes)
    idx_map = {nd: i for i, nd in enumerate(nodes)}
    D       = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                continue
            lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
            lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
                 * math.sin(dlon/2)**2)
            D[i, j] = 6371 * 2 * math.asin(math.sqrt(a))
    return D


def run_sim(G, nodes, Phi, E_df, eta, gamma, steps, start_t,
            use_E=True, seed=42):
    cfg  = PropagationConfig(eta=eta, gamma=gamma, steps=steps,
                              use_E=use_E, noise_std=0.0, seed=seed)
    prop = GraphPropagator(G, nodes, cfg)
    return prop.run(Phi, E_df, start_t=start_t)


def make_shuffled_graph(G, nodes, seed=0):
    rng   = np.random.default_rng(seed)
    edges = list(G.edges())
    flat  = [n for u, v in edges for n in (u, v)]
    rng.shuffle(flat)
    pairs = [(flat[i], flat[i+1]) for i in range(0, len(flat)-1, 2)]
    G2    = nx.Graph()
    G2.add_nodes_from(G.nodes(data=True))
    for u, v in pairs:
        if u != v and not G2.has_edge(u, v):
            cap = G[u][v]["capacity_gw"] if G.has_edge(u, v) else 1.0
            G2.add_edge(u, v, capacity_gw=cap, observed=False)
    return G2


def make_random_graph(G, nodes, seed=1):
    n, m  = G.number_of_nodes(), G.number_of_edges()
    p     = 2 * m / (n * (n - 1))
    rng   = np.random.default_rng(seed)
    G2    = nx.Graph()
    G2.add_nodes_from(G.nodes(data=True))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j > i and rng.random() < p:
                G2.add_edge(u, v, capacity_gw=1.0, observed=False)
    return G2


def make_distance_graph(G, nodes, threshold_km=900):
    import math
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes(data=True))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j <= i:
                continue
            lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
            lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
                 * math.sin(dlon/2)**2)
            km = 6371 * 2 * math.asin(math.sqrt(a))
            if km <= threshold_km:
                G2.add_edge(u, v, capacity_gw=max(0.1, 1.0 - km/2000),
                            observed=False)
    return G2


def exp1_topology(G, nodes, field, E1, W_adj, hop_D, geo_D):
    log.info("─" * 60)
    log.info("EXP 1: Topology sensitivity  (η=%.2f  γ=%.2f)", ETA, GAMMA)
    Phi = field["Phi"]
    graphs = {
        "real":     G,
        "shuffled": make_shuffled_graph(G, nodes),
        "random":   make_random_graph(G, nodes),
        "distance": make_distance_graph(G, nodes, threshold_km=900),
    }
    records = []
    trajs   = {}
    for name, Gi in graphs.items():
        try:
            traj = run_sim(Gi, nodes, Phi, E1, ETA, GAMMA, STEPS, START_T, use_E=True)
            m    = spatial_metrics(traj, nodes, W_adj)
            trajs[name] = traj
            records.append({"graph": name, "edges": Gi.number_of_edges(), **m})
            log.info("  %-10s edges=%2d  moran=%.4f  sp_std=%.4f  grad=%.4f  persist=%.4f",
                     name, Gi.number_of_edges(),
                     m["moran_mean"], m["spatial_std"],
                     m["gradient_mag"], m["persistence"])
        except Exception as exc:
            log.warning("  %s FAILED: %s", name, exc)
    df = pd.DataFrame(records).set_index("graph")
    df.to_csv(OUT_DAT / "S1_topology.csv")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names  = df.index.tolist()
    colors = ["#2196F3", "#FF9800", "#9E9E9E", "#4CAF50"]
    metrics_plot = [
        ("moran_mean",   "Moran's I (mean)",     "Higher = more spatial autocorrelation"),
        ("spatial_std",  "Spatial std of Phi",   "Higher = stronger inter-node contrast"),
        ("gradient_mag", "Gradient magnitude",   "Mean cross-node range of Phi_sim"),
        ("persistence",  "Spatial persistence",  "Lag-1 AC of spatial_std series"),
    ]
    for ax, (col, ylabel, desc) in zip(axes, metrics_plot):
        vals = [df.loc[n, col] if n in df.index else np.nan for n in names]
        bars = ax.bar(names, vals, color=colors[:len(names)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + abs(max(vals, default=0)) * 0.02,
                        f"{val:.3f}", ha="center", fontsize=8)
        ax.set_title(f"{ylabel}\n{desc}", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    fig.suptitle(f"Exp 1: Topology Sensitivity  (η={ETA}  γ={GAMMA}  gradient-rich forcing)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "S1_topology.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved S1_topology.png")
    return df, trajs


def exp2_ablation(G, nodes, field, E1, W_adj):
    log.info("─" * 60)
    log.info("EXP 2: E ablation  (η=%.2f  γ=%.2f)", ETA, GAMMA)
    Phi = field["Phi"]
    rng = np.random.default_rng(0)
    E_ones     = pd.DataFrame(1.0, index=E1.index, columns=E1.columns)
    E_inverted = (1.0 - E1).astype(float)
    arr        = E1.values.copy()
    for t in range(len(arr)):
        rng.shuffle(arr[t])
    E_shuffled = pd.DataFrame(arr, index=E1.index, columns=E1.columns)
    variants = [
        ("E=1  (uniform)",     E_ones,     True),
        ("E=E1 (baseline)",    E1,         True),
        ("E=shuffled",         E_shuffled, True),
        ("E=inverted (1-E1)",  E_inverted, True),
        ("No E (reduced)",     E1,         False),
    ]
    records = []
    trajs   = {}
    for label, E_df, use_E in variants:
        traj = run_sim(G, nodes, Phi, E_df, ETA, GAMMA, STEPS, START_T, use_E=use_E)
        m    = spatial_metrics(traj, nodes, W_adj)
        trajs[label] = traj
        records.append({"variant": label, "use_E": use_E, **m})
        log.info("  %-24s  moran=%.4f  sp_std=%.4f  persist=%.4f",
                 label, m["moran_mean"], m["spatial_std"], m["persistence"])
    df = pd.DataFrame(records).set_index("variant")
    df.to_csv(OUT_DAT / "S2_ablation.csv")

    sim_cols = [f"{nd}_sim" for nd in nodes if f"{nd}_sim" in trajs["E=E1 (baseline)"].columns]
    diff     = (trajs["E=E1 (baseline)"][sim_cols].values
                - trajs["E=1  (uniform)"][sim_cols].values)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names  = df.index.tolist()
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#F44336", "#607D8B"]
    for ax, (col, ylabel) in zip(axes[:3], [
        ("moran_mean",   "Moran's I"),
        ("spatial_std",  "Spatial std"),
        ("persistence",  "Persistence"),
    ]):
        vals = [df.loc[n, col] for n in names]
        ax.bar(names, vals, color=colors[:len(names)], alpha=0.85)
        ax.set_xticklabels(names, fontsize=7, rotation=20, ha="right")
        ax.set_title(ylabel, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax = axes[3]
    im = ax.imshow(diff.T, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_xlabel("Sim step")
    ax.set_ylabel("Node")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=6)
    ax.set_title("Phi_sim(E1) − Phi_sim(E=1)\nnode × time", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Exp 2: E Ablation  (η={ETA}  γ={GAMMA}  gradient-rich forcing)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "S2_ablation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved S2_ablation.png")
    return df


def exp3_shock(G, nodes, field, E1, W_adj, hop_D):
    log.info("─" * 60)
    log.info("EXP 3: Shock propagation  (η=%.2f  γ=%.2f)", ETA, GAMMA)
    Phi       = field["Phi"]
    phi_sigma = float(Phi.values.std())
    shock_mag = 3.0 * phi_sigma
    n         = len(nodes)
    idx_map   = {nd: i for i, nd in enumerate(nodes)}
    shock_nd  = "CISO"
    shock_steps = 36
    E_ones      = pd.DataFrame(1.0, index=E1.index, columns=E1.columns)
    G_shuffled  = make_shuffled_graph(G, nodes)
    configs = [
        ("Real + E1",       G,          E1,     True),
        ("Real, no E",      G,          E1,     False),
        ("Real + E=1",      G,          E_ones, True),
        ("Shuffled + E1",   G_shuffled, E1,     True),
    ]
    shock = {"node": shock_nd, "t_start": 1, "magnitude": shock_mag, "duration": 1}
    trajs_shock = {}
    trajs_base  = {}
    for label, Gi, E_df, use_E in configs:
        cfg_s = PropagationConfig(eta=ETA, gamma=GAMMA, steps=shock_steps, use_E=use_E, noise_std=0.0, seed=42)
        prop_s  = GraphPropagator(Gi, nodes, cfg_s)
        t_shock = prop_s.run(Phi, E_df, start_t=START_T, shock=shock)
        trajs_shock[label] = t_shock
        cfg_b = PropagationConfig(eta=ETA, gamma=GAMMA, steps=shock_steps, use_E=use_E, noise_std=0.0, seed=42)
        prop_b  = GraphPropagator(Gi, nodes, cfg_b)
        t_base  = prop_b.run(Phi, E_df, start_t=START_T)
        trajs_base[label]  = t_base
    impulse = {}
    for label in trajs_shock:
        sim_s = trajs_shock[label][[f"{nd}_sim" for nd in nodes if f"{nd}_sim" in trajs_shock[label].columns]].values
        sim_b = trajs_base[label][[f"{nd}_sim"  for nd in nodes if f"{nd}_sim" in trajs_base[label].columns]].values
        impulse[label] = sim_s - sim_b
    ref_label = "Real + E1"
    imp_ref   = impulse[ref_label]
    shock_i   = idx_map.get(shock_nd, 0)
    arrival_records = []
    for i, nd in enumerate(nodes):
        series    = imp_ref[:, i]
        peak_t    = int(np.argmax(np.abs(series)))
        peak_mag  = float(np.abs(series[peak_t]))
        hops      = int(hop_D[shock_i, i]) if shock_i < hop_D.shape[0] else -1
        arrival_records.append({"node": nd, "hops_from_shock": hops, "peak_t": peak_t, "peak_magnitude": peak_mag})
    arr_df = pd.DataFrame(arrival_records).sort_values("hops_from_shock")
    arr_df.to_csv(OUT_DAT / "S3_arrivals.csv", index=False)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    show_nodes = [shock_nd] + [n for n in ["BPAT","AZPS","NEVP","PACE","LDWP"] if n in nodes and n != shock_nd][:3]
    node_colors = ["#000000", "#F44336", "#2196F3", "#4CAF50"]
    config_labels = list(impulse.keys())
    for ci, label in enumerate(config_labels):
        ax = fig.add_subplot(gs[0, ci])
        imp = impulse[label]
        for ni, (nd, nc) in enumerate(zip(show_nodes, node_colors)):
            if nd in nodes:
                nd_i = list(nodes).index(nd)
                lw   = 1.8 if nd == shock_nd else 1.0
                ls   = "-"  if nd == shock_nd else "--"
                ax.plot(range(imp.shape[0]), imp[:, nd_i], color=nc, lw=lw, ls=ls, label=nd)
        ax.axvline(1, color="gray", lw=0.8, ls=":", alpha=0.6, label="shock t=1")
        ax.axhline(0, color="k",   lw=0.4, ls="--", alpha=0.3)
        ax.set_title(f"{label}", fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Impulse (Δ Phi_sim)")
        if ci == 0:
            ax.legend(fontsize=6, loc="upper right")
    ax_hop = fig.add_subplot(gs[1, :2])
    hops  = arr_df["hops_from_shock"].values
    pmags = arr_df["peak_magnitude"].values
    ax_hop.scatter(hops, pmags, s=80, color="#2196F3", alpha=0.85, zorder=5)
    for _, row in arr_df.iterrows():
        ax_hop.annotate(row["node"], (row["hops_from_shock"], row["peak_magnitude"]),
                        fontsize=7, ha="left", va="bottom", xytext=(3, 2), textcoords="offset points")
    if len(set(hops)) > 1:
        try:
            popt = np.polyfit(hops[pmags > 0.01], np.log(np.maximum(pmags[pmags > 0.01], 1e-6)), 1)
            h_fit = np.linspace(hops.min(), hops.max(), 50)
            ax_hop.plot(h_fit, np.exp(np.polyval(popt, h_fit)), color="#F44336", lw=1.2, ls="--", alpha=0.7,
                        label=f"exp fit  λ={popt[0]:.3f}/hop")
            ax_hop.legend(fontsize=8)
        except Exception:
            pass
    ax_hop.set_xlabel("Hops from shock node (CISO)")
    ax_hop.set_ylabel("Peak impulse magnitude")
    ax_hop.set_title("Impulse decay with graph distance")
    ax_mi = fig.add_subplot(gs[1, 2:])
    mi_colors = ["#2196F3", "#9E9E9E", "#FF9800", "#F44336"]
    for label, col in zip(config_labels, mi_colors):
        imp = impulse[label]
        mi_series = [moran_i(imp[t], W_adj) for t in range(imp.shape[0])]
        ax_mi.plot(range(len(mi_series)), mi_series, color=col, lw=1.2, alpha=0.85, label=label)
    ax_mi.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_mi.axvline(1, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_mi.set_xlabel("Step")
    ax_mi.set_ylabel("Moran's I of impulse field")
    ax_mi.set_title("Spatial coherence of propagating shock")
    ax_mi.legend(fontsize=7)
    fig.suptitle(f"Exp 3: Shock Propagation from {shock_nd}  (η={ETA}  γ={GAMMA}  mag=3σ={shock_mag:.2f})", fontsize=11)
    fig.savefig(OUT_FIG / "S3_shock.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved S3_shock.png")
    return arr_df, impulse


def exp4_distance_decay(G, nodes, field, E1, W_adj, hop_D, geo_D, impulse_data):
    log.info("─" * 60)
    log.info("EXP 4: Graph-distance decay")
    Phi = field["Phi"]
    n   = len(nodes)
    phi_arr = Phi.values
    phi_arr = (phi_arr - phi_arr.mean(axis=0)) / (phi_arr.std(axis=0) + 1e-9)
    pair_records = []
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j <= i:
                continue
            corr     = float(np.corrcoef(phi_arr[:, i], phi_arr[:, j])[0, 1])
            hop_dist = int(hop_D[i, j])
            km_dist  = float(geo_D[i, j])
            pair_records.append({"u": u, "v": v, "phi_corr": corr, "hops": hop_dist, "km": km_dist})
    pairs_df = pd.DataFrame(pair_records)
    pairs_df.to_csv(OUT_DAT / "S4_pairs.csv", index=False)
    G_shuffled = make_shuffled_graph(G, nodes)
    results_sim = {}
    for label, Gi, use_E in [
        ("Real + E1",      G,          True),
        ("Real, no E",     G,          False),
        ("Shuffled + E1",  G_shuffled, True),
    ]:
        traj     = run_sim(Gi, nodes, Phi, E1, ETA, GAMMA, STEPS, START_T, use_E=use_E)
        sim_cols = [f"{nd}_sim" for nd in nodes if f"{nd}_sim" in traj.columns]
        sim_arr  = traj[sim_cols].values
        sim_arr  = (sim_arr - sim_arr.mean(axis=0)) / (sim_arr.std(axis=0) + 1e-9)
        results_sim[label] = sim_arr

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax = axes[0, 0]
    for h in sorted(pairs_df["hops"].unique()):
        sub = pairs_df[pairs_df["hops"] == h]
        ax.scatter([h] * len(sub), sub["phi_corr"], alpha=0.5, s=30,
                   color=plt.cm.viridis(h / pairs_df["hops"].max()))
    means = pairs_df.groupby("hops")["phi_corr"].mean()
    ax.plot(means.index, means.values, "r-o", lw=1.4, ms=6, zorder=5, label="Mean correlation")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Graph hops between nodes")
    ax.set_ylabel("Phi cross-correlation")
    ax.set_title("(A) Input Phi: correlation vs hops\n(gradient-rich forcing)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.scatter(pairs_df["km"], pairs_df["phi_corr"], alpha=0.5, s=30, c=pairs_df["hops"], cmap="viridis")
    km_bins = pd.cut(pairs_df["km"], bins=6)
    km_means = pairs_df.groupby(km_bins, observed=True)["phi_corr"].mean()
    km_mids  = [iv.mid for iv in km_means.index]
    ax.plot(km_mids, km_means.values, "r-o", lw=1.4, ms=6, zorder=5)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Geographic distance (km)")
    ax.set_ylabel("Phi cross-correlation")
    ax.set_title("(B) Input Phi: correlation vs km")
    ax = axes[0, 2]
    sim_colors = {"Real + E1": "#2196F3", "Real, no E": "#9E9E9E", "Shuffled + E1": "#FF9800"}
    for label, sim_arr in results_sim.items():
        sim_pairs = []
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if j <= i:
                    continue
                corr     = float(np.corrcoef(sim_arr[:, i], sim_arr[:, j])[0, 1])
                hop_dist = int(hop_D[i, j])
                sim_pairs.append({"hops": hop_dist, "corr": corr})
        sp_df = pd.DataFrame(sim_pairs)
        means = sp_df.groupby("hops")["corr"].mean()
        ax.plot(means.index, means.values, "-o", lw=1.4, ms=6, color=sim_colors[label], label=label)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Graph hops")
    ax.set_ylabel("Sim Phi correlation")
    ax.set_title("(C) Sim Phi: correlation vs hops\nReal vs shuffled topology")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    imp_label = "Real + E1"
    if imp_label in impulse_data:
        imp   = impulse_data[imp_label]
        pmags = np.abs(imp).max(axis=0)
        hops_arr = np.array([int(hop_D[nodes.index("CISO"), i]) if "CISO" in nodes else 0 for i in range(n)])
        ax.scatter(hops_arr, pmags, s=60, color="#2196F3", alpha=0.85)
        for i, nd in enumerate(nodes):
            ax.annotate(nd, (hops_arr[i], pmags[i]), fontsize=6, ha="left",
                        xytext=(2, 1), textcoords="offset points")
        mask = pmags > 0.005
        if mask.sum() > 2:
            try:
                popt = np.polyfit(hops_arr[mask], np.log(pmags[mask] + 1e-9), 1)
                h_f  = np.linspace(0, hops_arr.max(), 40)
                ax.plot(h_f, np.exp(np.polyval(popt, h_f)), "r--", lw=1.2, alpha=0.7,
                        label=f"λ={popt[0]:.3f}/hop")
                ax.legend(fontsize=8)
            except Exception:
                pass
    ax.set_xlabel("Hops from CISO")
    ax.set_ylabel("Peak impulse magnitude")
    ax.set_title("(D) Shock impulse decay\nvs graph-hop distance")
    ax = axes[1, 1]
    for label, sim_arr in results_sim.items():
        mi_series = [moran_i(sim_arr[t], W_adj) for t in range(len(sim_arr))]
        ax.plot(range(len(mi_series)), mi_series, color=sim_colors[label], lw=1.2, alpha=0.85, label=label)
    ax.set_xlabel("Sim step")
    ax.set_ylabel("Moran's I")
    ax.set_title("(E) Moran's I over simulation\n(spatial autocorr persistence)")
    ax.legend(fontsize=7)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax = axes[1, 2]
    sim_arr_real = results_sim.get("Real + E1", results_sim[list(results_sim.keys())[0]])
    corr_matrix = np.corrcoef(sim_arr_real.T)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(nodes, fontsize=5, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(nodes, fontsize=5)
    ax.set_title("(F) Node-pair Phi_sim correlation\n(Real + E1)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Exp 4: Graph-Distance Decay  (η={ETA}  γ={GAMMA}  gradient-rich forcing)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "S4_distance_decay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Saved S4_distance_decay.png")
    return pairs_df


def write_report(topo_df, ablation_df, arr_df, pairs_df):
    sep = "=" * 68
    lines = [sep, "  SPATIAL EXPERIMENTS — FINDINGS (gradient-rich forcing)",
             f"  η={ETA}  γ={GAMMA}  η/γ={ETA/GAMMA:.1f}  (diffusion-competitive regime)", sep, ""]
    if topo_df is not None and not topo_df.empty:
        real_mi = topo_df.loc["real","moran_mean"] if "real" in topo_df.index else np.nan
        shuf_mi = topo_df.loc["shuffled","moran_mean"] if "shuffled" in topo_df.index else np.nan
        rand_mi = topo_df.loc["random","moran_mean"] if "random" in topo_df.index else np.nan
        dist_mi = topo_df.loc["distance","moran_mean"] if "distance" in topo_df.index else np.nan
        real_sp = topo_df.loc["real","spatial_std"] if "real" in topo_df.index else np.nan
        shuf_sp = topo_df.loc["shuffled","spatial_std"] if "shuffled" in topo_df.index else np.nan
        mi_gap  = real_mi - shuf_mi if not (np.isnan(real_mi) or np.isnan(shuf_mi)) else np.nan
        lines += ["1. TOPOLOGY SENSITIVITY", "─" * 40, "",
                  f"  Moran's I: real={real_mi:.4f}  shuffled={shuf_mi:.4f}  random={rand_mi:.4f}  distance={dist_mi:.4f}",
                  f"  Spatial std: real={real_sp:.4f}  shuffled={shuf_sp:.4f}",
                  f"  Topology MI gap (real − shuffled): {mi_gap:.4f}",
                  f"  → Topology matters: {'YES' if not np.isnan(mi_gap) and abs(mi_gap) > 0.05 else 'WEAK'}", ""]
    lines += [sep]
    report = "\n".join(lines)
    (OUT_DAT / "spatial_report.txt").write_text(report)
    print("\n" + report)
    return report


def main():
    log.info("Building graph and gradient-rich field …")
    G     = build_graph()
    nodes = node_order(G)
    field = build_gradient_rich_field(G, nodes, seed=42, T=1440)
    E1    = E1_price_spread_edge(G, field["R"])
    W_adj = adjacency_W(G, nodes)
    hop_D = node_distances(G, nodes)
    geo_D = geo_distances(G, nodes)
    log.info("  η=%.2f  γ=%.2f  η/γ=%.1f  (diffusion-competitive)", ETA, GAMMA, ETA/GAMMA)
    topo_df,  _      = exp1_topology(G, nodes, field, E1, W_adj, hop_D, geo_D)
    ablation_df      = exp2_ablation(G, nodes, field, E1, W_adj)
    arr_df, impulse  = exp3_shock(G, nodes, field, E1, W_adj, hop_D)
    pairs_df         = exp4_distance_decay(G, nodes, field, E1, W_adj, hop_D, geo_D, impulse)
    write_report(topo_df, ablation_df, arr_df, pairs_df)
    log.info("Complete. Figures in %s  Data in %s", OUT_FIG, OUT_DAT)


if __name__ == "__main__":
    main()
