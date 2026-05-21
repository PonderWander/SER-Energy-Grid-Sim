"""
constraint_field.graph.propagation
=====================================
Graph-Laplacian propagation of the Phi field over the BA network.

Update rule
-----------
The discrete-time graph diffusion equation:

    Phi_{t+1} = Phi_t - eta * L_t * Phi_t - gamma * Phi_t + epsilon_t

where:
  L_t   = D_t - W_t   (weighted graph Laplacian at time t)
  eta   = diffusion coefficient (scales propagation speed)
  gamma = damping coefficient (mean-reversion toward 0)
  epsilon_t = optional stochastic forcing

Intuition
---------
- The term -eta * L_t * Phi_t drives Phi toward spatial smoothness:
  nodes with high Phi relative to their neighbours lose Phi to them,
  and vice versa. This models constraint-pressure spreading across
  the network via available transmission paths.

- E_{ij,t} modulates W_t, so congested edges (low E) carry less
  Phi. A corridor with E=0 is fully blocked; with E=1 it transmits
  at full capacity weight.

- gamma * Phi_t is a dissipation term: field pressure decays toward
  zero in the absence of sustained forcing. This models the fact that
  constraint events are transient.

- Comparing simulations with constant L (reduced) vs dynamic L_t
  (upgraded with E) reveals how edge fluidity shapes propagation.

Stability condition
-------------------
For stable diffusion: eta * lambda_max(L) < 1
where lambda_max is the largest eigenvalue of L.
The simulator checks and warns if this is violated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .edge_fluidity import weighted_adjacency, graph_laplacian, constant_laplacian

log = logging.getLogger(__name__)


@dataclass
class PropagationConfig:
    eta:       float = 0.05    # diffusion coefficient
    gamma:     float = 0.10    # damping / dissipation
    noise_std: float = 0.01    # stochastic forcing std
    use_E:     bool  = True    # True = dynamic E; False = constant L (reduced)
    steps:     int   = 72      # simulation horizon (hours)
    seed:      int   = 42


class GraphPropagator:
    """
    Simulate Phi propagation over the graph network.

    Parameters
    ----------
    G     : nx.Graph with capacity_gw edge attributes
    nodes : ordered list of node IDs
    cfg   : PropagationConfig
    """

    def __init__(
        self,
        G:     nx.Graph,
        nodes: list[str],
        cfg:   PropagationConfig | None = None,
    ):
        self.G     = G
        self.nodes = nodes
        self.n     = len(nodes)
        self.cfg   = cfg or PropagationConfig()
        self.rng   = np.random.default_rng(self.cfg.seed)

        # Precompute constant Laplacian (for reduced baseline)
        # Row-normalise (D^{-1} L) so lambda_max <= 2 regardless of
        # edge capacity weights, ensuring numerical stability.
        L_raw = constant_laplacian(G, nodes)
        self.L_const = self._row_normalise(L_raw)
        self._check_stability(self.L_const, "constant")

    @staticmethod
    def _row_normalise(L: np.ndarray) -> np.ndarray:
        """
        Row-normalise the Laplacian: L_rw = D^{-1} L.
        This bounds lambda_max <= 2 regardless of edge weights,
        guaranteeing stability for eta < 0.5.
        Preserves the zero eigenvector (constant field is still fixed point).
        """
        diag = np.diag(L).copy()
        D_inv = np.diag(np.where(diag > 1e-9, 1.0 / diag, 0.0))
        return D_inv @ L

    def _check_stability(self, L: np.ndarray, label: str) -> None:
        """Warn if eta * lambda_max > 1 (unstable diffusion)."""
        eigvals     = np.linalg.eigvals(L).real
        lambda_max  = eigvals.max()
        spectral_dt = self.cfg.eta * lambda_max
        if spectral_dt >= 1.0:
            log.warning(
                "Stability check (%s): eta * lambda_max = %.3f >= 1. "
                "Diffusion may be unstable. Consider reducing eta.",
                label, spectral_dt
            )
        else:
            log.info("Stability OK (%s): eta * lambda_max = %.4f", label, spectral_dt)

    def run(
        self,
        Phi_df:  pd.DataFrame,
        E_df:    pd.DataFrame | None = None,
        start_t: int = 0,
        shock:   dict | None = None,
    ) -> pd.DataFrame:
        """
        Run graph propagation simulation.

        Parameters
        ----------
        Phi_df  : node-indexed Phi DataFrame (rows=time, cols=nodes)
        E_df    : edge fluidity DataFrame (rows=time, cols="u_v")
                  Required if cfg.use_E=True.
        start_t : start row index in Phi_df
        shock   : optional dict with keys:
                  'node'      : node ID to perturb
                  't_start'   : relative timestep
                  'magnitude' : Phi perturbation magnitude

        Returns
        -------
        pd.DataFrame with columns: {node}_sim, {node}_obs, {node}_resid
        for each node, plus 'shock_active'.
        """
        n_steps = min(self.cfg.steps, len(Phi_df) - start_t)
        slice_  = Phi_df.iloc[start_t: start_t + n_steps]
        idx     = slice_.index

        Phi_obs = slice_[self.nodes].values.copy()  # (T, n)
        Phi_sim = np.zeros_like(Phi_obs)
        Phi_sim[0] = Phi_obs[0]                     # initialise from observation

        node_idx = {nd: i for i, nd in enumerate(self.nodes)}
        shock_active = np.zeros(n_steps)

        for t in range(1, n_steps):
            phi_t = Phi_sim[t - 1].copy()

            # Choose Laplacian
            if self.cfg.use_E and E_df is not None:
                abs_t = start_t + t
                if abs_t < len(E_df):
                    W  = weighted_adjacency(self.G, E_df, abs_t, self.nodes)
                    L_raw, _ = graph_laplacian(W)
                    L = self._row_normalise(L_raw)
                else:
                    L = self.L_const
            else:
                L = self.L_const

            # Diffusion term
            diffusion = -self.cfg.eta * (L @ phi_t)

            # Damping
            damping = -self.cfg.gamma * phi_t

            # Noise
            noise = self.rng.normal(0, self.cfg.noise_std, self.n)

            # Shock injection
            shock_force = np.zeros(self.n)
            if shock and shock.get("t_start", -1) <= t < shock.get("t_start", -1) + shock.get("duration", 1):
                nid = shock.get("node", self.nodes[0])
                if nid in node_idx:
                    shock_force[node_idx[nid]] = shock["magnitude"]
                shock_active[t] = 1.0

            Phi_sim[t] = phi_t + diffusion + damping + noise + shock_force

        # Assemble results
        results = {"time": idx, "shock_active": shock_active}
        for i, nd in enumerate(self.nodes):
            results[f"{nd}_sim"] = Phi_sim[:, i]
            results[f"{nd}_obs"] = Phi_obs[:, i]
            results[f"{nd}_resid"] = Phi_sim[:, i] - Phi_obs[:, i]

        df = pd.DataFrame(results).set_index("time")
        rmse = np.sqrt(np.mean((Phi_sim - Phi_obs) ** 2))
        log.info(
            "Propagation [use_E=%s]: steps=%d  RMSE=%.4f  max|resid|=%.4f",
            self.cfg.use_E, n_steps, rmse, np.abs(Phi_sim - Phi_obs).max()
        )
        return df

    def compare(
        self,
        Phi_df:  pd.DataFrame,
        E_df:    pd.DataFrame,
        start_t: int = 0,
        shock:   dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run both reduced and upgraded and return both results."""
        self.cfg.use_E = False
        reduced = self.run(Phi_df, E_df, start_t, shock)

        self.cfg.use_E = True
        upgraded = self.run(Phi_df, E_df, start_t, shock)

        return {"reduced": reduced, "upgraded": upgraded}


# ──────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ──────────────────────────────────────────────────────────────────────────────

def propagation_metrics(
    reduced:  pd.DataFrame,
    upgraded: pd.DataFrame,
    nodes:    list[str],
) -> pd.DataFrame:
    """
    Compare reduced vs upgraded propagation per node and overall.

    Metrics: RMSE, MAE, max_residual, residual_ac1
    """
    from scipy.stats import pearsonr

    rows = []
    for label, df in [("reduced", reduced), ("upgraded", upgraded)]:
        all_resid = []
        for nd in nodes:
            resid_col = f"{nd}_resid"
            if resid_col not in df.columns:
                continue
            r = df[resid_col].values
            all_resid.append(r)
            ac1, _ = pearsonr(r[1:], r[:-1]) if len(r) > 1 else (np.nan, np.nan)
            rows.append({
                "model": label, "node": nd,
                "rmse":  np.sqrt(np.mean(r**2)),
                "mae":   np.abs(r).mean(),
                "max_resid": np.abs(r).max(),
                "ac1":   ac1,
            })
        # Overall
        if all_resid:
            flat = np.concatenate(all_resid)
            rows.append({
                "model": label, "node": "OVERALL",
                "rmse":  np.sqrt(np.mean(flat**2)),
                "mae":   np.abs(flat).mean(),
                "max_resid": np.abs(flat).max(),
                "ac1":   np.nan,
            })
    return pd.DataFrame(rows)


def bottleneck_analysis(
    G:    nx.Graph,
    E_df: pd.DataFrame,
    nodes: list[str],
) -> pd.DataFrame:
    """
    Identify persistent bottleneck edges: those with lowest mean E.

    Returns DataFrame sorted by mean_E ascending.
    """
    rows = []
    for u, v in G.edges():
        col = f"{u}_{v}"
        rev = f"{v}_{u}"
        if col in E_df.columns:
            e_series = E_df[col]
        elif rev in E_df.columns:
            e_series = E_df[rev]
        else:
            continue
        rows.append({
            "edge":       col,
            "u":          u, "v": v,
            "corridor":   G[u][v].get("corridor", ""),
            "capacity_gw": G[u][v].get("capacity_gw", np.nan),
            "observability": G[u][v].get("observability", "?"),
            "mean_E":     float(e_series.mean()),
            "std_E":      float(e_series.std()),
            "min_E":      float(e_series.min()),
            "frac_lt_03": float((e_series < 0.3).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_E").reset_index(drop=True)
