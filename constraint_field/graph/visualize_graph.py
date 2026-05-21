"""
constraint_field.graph.visualize_graph
========================================
Spatial and temporal visualisations for the graph constraint field.

Figures produced
----------------
1. Static network map coloured by node Phi_t at a chosen snapshot
2. Edge fluidity map coloured by E_{ij,t}
3. Propagation comparison: reduced vs upgraded node trajectories
4. Bottleneck / barrier map (low-E edges highlighted)
5. Animation of Phi propagation over time (matplotlib FuncAnimation)
6. Full graph dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Color maps
PHI_CMAP   = "RdBu_r"     # Phi: red=positive (over-constraint), blue=negative
E_CMAP     = "RdYlGn"     # E: green=fluid, red=congested
S_CMAP     = "YlOrRd"     # S: demand pressure
R_CMAP     = "PuRd"       # R: constraint signal


def _node_positions(G: nx.Graph) -> dict[str, tuple[float, float]]:
    """Extract lon/lat positions as (x, y) for each node."""
    return {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}


def _obs_color(obs: str) -> str:
    return {"DOCUMENTED": "#2196F3", "APPROXIMATED": "#FF9800",
            "SYNTHETIC": "#9E9E9E", "OBSERVED": "#4CAF50"}.get(obs, "#607D8B")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Snapshot field map
# ──────────────────────────────────────────────────────────────────────────────

def plot_field_snapshot(
    G:       nx.Graph,
    field:   dict[str, pd.DataFrame],
    E_df:    pd.DataFrame,
    t:       int = 0,
    figsize: tuple = (16, 7),
    title:   str | None = None,
) -> plt.Figure:
    """
    Two-panel spatial snapshot at time index t.
    Left: node Phi coloured map.  Right: edge E coloured map.
    """
    nodes = list(G.nodes())
    pos   = _node_positions(G)
    Phi   = field["Phi"]
    S_df  = field["S"]
    R_df  = field["R"]

    phi_t  = Phi.iloc[t]
    s_t    = S_df.iloc[t]
    r_t    = R_df.iloc[t]
    ts_str = str(Phi.index[t])[:13]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: Phi snapshot ──────────────────────────────────────────────
    ax = axes[0]
    phi_vals = np.array([phi_t.get(n, 0) for n in nodes])
    vmax     = max(abs(phi_vals).max(), 0.5)

    node_colors = [phi_t.get(n, 0) for n in nodes]
    norm        = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap        = plt.cm.get_cmap(PHI_CMAP)
    nc          = [cmap(norm(v)) for v in node_colors]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                             labels={n: n for n in nodes}, font_size=7)
    # Edges coloured by observability
    for u, v in G.edges():
        obs = G[u][v].get("observability", "APPROXIMATED")
        ec  = _obs_color(obs)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=ec, width=1.5, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Phi = R - S", shrink=0.7)

    # Legend for edge observability
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=_obs_color("DOCUMENTED"),  lw=2, label="Documented"),
        Line2D([0], [0], color=_obs_color("APPROXIMATED"), lw=2, label="Approximated"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7)
    ax.set_title(f"Phi field snapshot — {ts_str}")
    ax.axis("off")

    # ── Right: Edge E snapshot ───────────────────────────────────────────
    ax2 = axes[1]
    # Draw nodes with S value
    s_vals  = np.array([s_t.get(n, 0) for n in nodes])
    s_norm  = mcolors.Normalize(vmin=s_vals.min(), vmax=s_vals.max())
    s_cmap  = plt.cm.get_cmap(S_CMAP)
    nc2     = [s_cmap(s_norm(v)) for v in s_vals]

    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=nc2, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax2,
                             labels={n: n for n in nodes}, font_size=7)

    # Edges coloured and weighted by E value
    e_cmap = plt.cm.get_cmap(E_CMAP)
    e_norm = mcolors.Normalize(vmin=0, vmax=1)
    for u, v in G.edges():
        col = f"{u}_{v}"
        rev = f"{v}_{u}"
        if col in E_df.columns:
            e_val = float(E_df.iloc[t].get(col, E_df.iloc[t].get(rev, 0.5)))
        else:
            e_val = 0.5
        ec    = e_cmap(e_norm(e_val))
        width = 1.0 + 4.0 * e_val   # thicker = more fluid
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax2,
                               edge_color=[ec], width=width, alpha=0.85)

    sm2 = plt.cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax2, label="Edge fluidity E_{ij}", shrink=0.7)
    ax2.set_title(f"Edge fluidity E — {ts_str}\n(thicker = more fluid)")
    ax2.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Propagation comparison: reduced vs upgraded
# ──────────────────────────────────────────────────────────────────────────────

def plot_propagation_comparison(
    reduced:  pd.DataFrame,
    upgraded: pd.DataFrame,
    nodes:    list[str],
    n_show:   int = 4,
    figsize:  tuple = (16, 10),
    title:    str = "Graph Propagation: Reduced vs Upgraded",
) -> plt.Figure:
    """
    Time-series comparison for selected nodes.
    """
    show_nodes = nodes[:n_show]
    fig, axes  = plt.subplots(n_show, 1, figsize=figsize, sharex=True)
    if n_show == 1:
        axes = [axes]

    for ax, nd in zip(axes, show_nodes):
        obs_col  = f"{nd}_obs"
        sim_r_col = f"{nd}_sim"   # same col name in reduced df
        sim_u_col = f"{nd}_sim"   # same col name in upgraded df

        if obs_col in reduced.columns:
            ax.plot(reduced.index, reduced[obs_col],
                    color="#212121", lw=1.2, alpha=0.8, label="Observed Phi")

        if sim_r_col in reduced.columns:
            ax.plot(reduced.index, reduced[sim_r_col],
                    color="#607D8B", lw=1.0, ls="--", alpha=0.8,
                    label="Reduced (const L)")

        if sim_u_col in upgraded.columns:
            ax.plot(upgraded.index, upgraded[sim_u_col],
                    color="#4CAF50", lw=1.0, ls="-", alpha=0.8,
                    label="Upgraded (dynamic E)")

        # Shade shock period
        if "shock_active" in reduced.columns:
            mask = reduced["shock_active"] > 0
            for xi in reduced.index[mask]:
                ax.axvspan(xi, xi + pd.Timedelta("1h"),
                           alpha=0.15, color="#F44336", zorder=0)

        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylabel(f"Phi\n{nd}", fontsize=8)
        if nd == show_nodes[0]:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(title)

    axes[-1].set_xlabel("Time (UTC)")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Bottleneck map
# ──────────────────────────────────────────────────────────────────────────────

def plot_bottleneck_map(
    G:           nx.Graph,
    bottleneck_df: pd.DataFrame,
    field:       dict[str, pd.DataFrame],
    t:           int = 0,
    figsize:     tuple = (10, 7),
    title:       str = "Bottleneck / Barrier Map (Low-E Edges)",
) -> plt.Figure:
    """
    Highlight bottleneck edges (low mean E) on the network map.
    """
    nodes = list(G.nodes())
    pos   = _node_positions(G)
    Phi   = field["Phi"]
    phi_t = Phi.iloc[t]

    fig, ax = plt.subplots(figsize=figsize)

    # Node colour = Phi
    phi_vals = [phi_t.get(n, 0) for n in nodes]
    vmax     = max(abs(v) for v in phi_vals) or 1.0
    norm     = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap     = plt.cm.get_cmap(PHI_CMAP)
    nc       = [cmap(norm(v)) for v in phi_vals]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Edge colour = mean E; width = capacity; label bottlenecks
    bt_dict = {f"{r['u']}_{r['v']}": r["mean_E"]
               for _, r in bottleneck_df.iterrows()}
    e_cmap = plt.cm.get_cmap(E_CMAP)

    for u, v in G.edges():
        col    = f"{u}_{v}"
        rev    = f"{v}_{u}"
        mean_e = bt_dict.get(col, bt_dict.get(rev, 0.5))
        cap    = G[u][v].get("capacity_gw", 1.0)
        ec     = e_cmap(mean_e)
        width  = 1.0 + 3.5 * cap / 5.0
        style  = "-" if mean_e >= 0.3 else "--"   # dashed = bottleneck
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=[ec], width=width,
                               style=style, alpha=0.85)
        # Annotate bottleneck edges
        if mean_e < 0.35:
            mx = (G.nodes[u]["lon"] + G.nodes[v]["lon"]) / 2
            my = (G.nodes[u]["lat"] + G.nodes[v]["lat"]) / 2
            ax.text(mx, my, f"E={mean_e:.2f}", fontsize=6.5,
                    ha="center", color="darkred",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    sm = plt.cm.ScalarMappable(cmap=e_cmap,
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean edge fluidity E_{ij}", shrink=0.75)

    ax.set_title(title + "\n(dashed edges = bottlenecks, E < 0.35)")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Animation
# ──────────────────────────────────────────────────────────────────────────────

def make_phi_animation(
    G:       nx.Graph,
    field:   dict[str, pd.DataFrame],
    E_df:    pd.DataFrame,
    n_frames: int = 72,
    interval: int = 150,   # ms per frame
    figsize: tuple = (10, 7),
) -> animation.FuncAnimation:
    """
    Animate Phi propagation over time.

    Returns a matplotlib FuncAnimation object.
    Save with: anim.save('phi_animation.gif', writer='pillow', fps=6)
    """
    nodes   = list(G.nodes())
    pos     = _node_positions(G)
    Phi     = field["Phi"]
    n_frames = min(n_frames, len(Phi))

    # Global colour scale
    phi_arr = Phi.values[:n_frames]
    vmax    = max(np.abs(phi_arr).max(), 0.5)
    norm    = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap    = plt.cm.get_cmap(PHI_CMAP)
    e_cmap  = plt.cm.get_cmap(E_CMAP)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        phi_t = Phi.iloc[frame]
        nc    = [cmap(norm(phi_t.get(n, 0))) for n in nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc,
                               node_size=600, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        for u, v in G.edges():
            col = f"{u}_{v}"
            rev = f"{v}_{u}"
            if col in E_df.columns:
                e_val = float(E_df.iloc[frame].get(col, E_df.iloc[frame].get(rev, 0.5)))
            else:
                e_val = 0.5
            ec    = e_cmap(e_val)
            width = 1.0 + 4.0 * e_val
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                   edge_color=[ec], width=width, alpha=0.8)

        ts = str(Phi.index[frame])[:13]
        ax.set_title(f"Phi propagation — {ts}  (frame {frame+1}/{n_frames})",
                     fontsize=10)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=False
    )
    plt.close(fig)  # prevent duplicate display
    return anim


# ──────────────────────────────────────────────────────────────────────────────
# 5. Full graph dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_graph_dashboard(
    G:            nx.Graph,
    field:        dict[str, pd.DataFrame],
    E_df:         pd.DataFrame,
    reduced:      pd.DataFrame,
    upgraded:     pd.DataFrame,
    bottleneck_df: pd.DataFrame,
    metrics_df:   pd.DataFrame,
    nodes:        list[str],
    snapshot_t:   int = 48,
    figsize:      tuple = (20, 22),
    title:        str = "Graph Constraint Field — Full Dashboard",
) -> plt.Figure:
    """
    6-panel summary dashboard for the graph model.
    """
    fig  = plt.figure(figsize=figsize)
    gs   = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35)
    pos  = _node_positions(G)
    Phi  = field["Phi"]
    S_df = field["S"]
    n_nodes = len(nodes)

    # ── Panel A: Phi snapshot ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    phi_t = Phi.iloc[snapshot_t]
    vals  = np.array([phi_t.get(n, 0) for n in nodes])
    vmax  = max(abs(vals).max(), 0.5)
    norm  = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap  = plt.cm.get_cmap(PHI_CMAP)
    nc    = [cmap(norm(phi_t.get(n, 0))) for n in nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax_a, node_color=nc, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax_a, font_size=7)
    for u, v in G.edges():
        obs = G[u][v].get("observability", "APPROXIMATED")
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax_a,
                               edge_color=_obs_color(obs), width=1.5, alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_a, label="Phi", shrink=0.7)
    ax_a.set_title(f"(A) Phi snapshot t={snapshot_t}  [{str(Phi.index[snapshot_t])[:13]}]")
    ax_a.axis("off")

    # ── Panel B: Edge fluidity snapshot ──────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2:])
    e_cmap = plt.cm.get_cmap(E_CMAP)
    e_norm = mcolors.Normalize(vmin=0, vmax=1)
    s_t    = S_df.iloc[snapshot_t]
    s_vals = [s_t.get(n, 0) for n in nodes]
    s_norm = mcolors.Normalize(vmin=min(s_vals), vmax=max(s_vals))
    s_cmap = plt.cm.get_cmap(S_CMAP)
    nc_b   = [s_cmap(s_norm(v)) for v in s_vals]
    nx.draw_networkx_nodes(G, pos, ax=ax_b, node_color=nc_b, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax_b, font_size=7)
    for u, v in G.edges():
        col = f"{u}_{v}"
        rev = f"{v}_{u}"
        e_val = float(E_df.iloc[snapshot_t].get(col, E_df.iloc[snapshot_t].get(rev, 0.5)))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax_b,
                               edge_color=[e_cmap(e_norm(e_val))],
                               width=1.0 + 4*e_val, alpha=0.85)
    sm2 = plt.cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax_b, label="Edge E", shrink=0.7)
    ax_b.set_title("(B) Edge fluidity E (colour+width)")
    ax_b.axis("off")

    # ── Panel C: Node Phi time-series (CISO + 2 others) ─────────────────
    ax_c = fig.add_subplot(gs[1, :])
    show = nodes[:min(4, n_nodes)]
    colors = ["#F44336","#2196F3","#4CAF50","#FF9800"]
    for nd, col in zip(show, colors):
        if nd in Phi.columns:
            ax_c.plot(Phi.index, Phi[nd], color=col, lw=0.9, alpha=0.8, label=nd)
    ax_c.axhline(0, color="k", lw=0.4, ls="--", alpha=0.35)
    ax_c.set_title("(C) Node Phi time-series")
    ax_c.set_ylabel("Phi = R - S")
    ax_c.legend(fontsize=7, loc="upper right", ncol=4)

    # ── Panel D: Propagation comparison for top node ──────────────────────
    ax_d = fig.add_subplot(gs[2, :2])
    nd0 = nodes[0]
    for df, lbl, col in [(reduced, "Reduced", "#607D8B"), (upgraded, "Upgraded", "#4CAF50")]:
        sim_col = f"{nd0}_sim"
        if sim_col in df.columns:
            ax_d.plot(df.index, df[sim_col], color=col, lw=1.0, ls="--", label=f"{lbl} sim")
    obs_col = f"{nd0}_obs"
    if obs_col in reduced.columns:
        ax_d.plot(reduced.index, reduced[obs_col], color="#212121", lw=1.2, label="Observed")
    ax_d.set_title(f"(D) Propagation comparison — {nd0}")
    ax_d.set_ylabel("Phi")
    ax_d.legend(fontsize=7)

    # ── Panel E: RMSE comparison ──────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 2:])
    overall = metrics_df[metrics_df["node"] == "OVERALL"]
    if not overall.empty:
        x = np.arange(2)
        rmse_r = float(overall[overall["model"] == "reduced"]["rmse"].iloc[0])
        rmse_u = float(overall[overall["model"] == "upgraded"]["rmse"].iloc[0])
        ax_e.bar(x, [rmse_r, rmse_u], color=["#607D8B","#4CAF50"], alpha=0.85)
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(["Reduced\n(const L)", "Upgraded\n(dynamic E)"])
        ax_e.set_ylabel("Overall RMSE")
        ax_e.set_title("(E) Reduced vs Upgraded RMSE")
        for xi, val in zip(x, [rmse_r, rmse_u]):
            ax_e.text(xi, val + 0.005, f"{val:.4f}", ha="center", fontsize=9)

    # ── Panel F: Bottleneck table ─────────────────────────────────────────
    ax_f = fig.add_subplot(gs[3, :])
    ax_f.axis("off")
    tbl = bottleneck_df[["edge","corridor","capacity_gw","mean_E",
                          "frac_lt_03","observability"]].head(10)
    cell_text = [[str(v) if not isinstance(v, float) else f"{v:.3f}"
                  for v in row] for row in tbl.values]
    t = ax_f.table(cellText=cell_text, colLabels=tbl.columns.tolist(),
                   loc="center", cellLoc="center")
    t.scale(1, 1.4)
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    # Highlight worst bottlenecks
    for i in range(min(3, len(tbl))):
        for j in range(len(tbl.columns)):
            t[i+1, j].set_facecolor("#fff3e0")
    ax_f.set_title("(F) Edge bottleneck ranking (lowest mean E first)", pad=10)

    fig.suptitle(title, fontsize=13, y=1.005)
    return fig
