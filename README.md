# SER Energy Grid Sim

A field-based constraint dynamics model for electricity system analysis.

The model represents the Western Interconnect as a 14-node graph of balancing
authorities. At each node it tracks three observed variables — demand pressure
(**S**), price signal (**R**), and field imbalance (**Φ = R − S**) — and one
inferred variable, edge-level delivery fluidity (**E** ∈ [0,1]). Spatial
dynamics are governed by a row-normalised graph Laplacian, with E weighting
edge transmissibility in response to local constraint stress.

---

## Key findings

| Finding | Result |
|---------|--------|
| **Bounded-price effect** | High-\|Φ\|/moderate-R hours have 3.8× higher instability than matched low-\|Φ\| hours. Cohen's d = 3.35, p < 1e-6. Survives ToD residualisation and all four S–R cluster regimes. |
| **Topology dominates** | Real graph Moran's I = 0.57 vs shuffled = −0.13. Edge topology, not E weighting, drives spatial autocorrelation. |
| **Hop-ordered propagation** | Shock arrivals are hop-ordered (Kendall τ = 0.787 ± 0.071) across all 24 source × threshold conditions. |
| **SW bottleneck** | AZPS→WACM (cap 1.0 GW) blocks at loading ≥ 1.5σ. Hard floor ε=0 produces retained accumulation at AZPS (net rise +0.479σ, dwell 57/60 steps) and WACM isolation (correlation 0.086). |
| **Cascade** | With dual constraint (AZPS→WACM + PACE→WACM), system enters cascading isolation at 1.5σ. With triple constraint at 2.5σ, network fragmentation at AZPS. |

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/run_experiments.py
```

All scripts are deterministic (seed=42). Outputs go to `outputs/figures/` and
`outputs/data/`. See [USAGE.md](USAGE.md) for full configuration and extension
instructions.

---

## Package structure

```
constraint_field/               Core package
  adapters/                     Data adapters
    base.py                     Abstract adapter interface
    eia.py                      EIA API (demand, LMP by BA) — production-ready
    caiso.py                    CAISO API (nodal LMP, flows) — production-ready
    synthetic.py                Synthetic data generator (Cholesky + structural offsets)
  analysis/                     Statistical analysis modules
    divergence.py               D1–D5 divergence metrics, bounded-price test
    decontaminated_lead.py      PIT-based decontamination, ToD residualisation
    models.py                   Regression models for lead/lag analysis
    regimes.py                  S–R cluster regime classification
    visualize_divergence.py     Figure generation for divergence analysis
    visualize_decontaminated.py Figure generation for decontaminated lead
  dynamics/                     Propagation operators
    operators.py                DiffusionOperator, GradientFlowOperator, DampedWaveOperator
    simulator.py                Simulator with shock injection, reduced vs upgraded comparison
  field/                        Scalar field layer
    builder.py                  Rolling z-score normalisation, Φ/Ψ construction
    analysis.py                 Field statistics and instability index
    visualize.py                Field dashboard and phase portrait figures
  graph/                        Graph layer (primary spatial model)
    network.py                  14-node Western Interconnect topology (NODES, EDGES)
    node_signals.py             SyntheticNodeSignals, build_node_field
    edge_fluidity.py            E1 / E2 / E3 fluidity computation; row-normalised Laplacian
    propagation.py              GraphPropagator with configurable E, shock injection
    topology.py                 Graph topology utilities
    visualize_graph.py          Network map, bottleneck, propagation figures
  inference/
    base.py                     Abstract fluidity estimator interface
    calibration.py              SpreadCalibrator (4 transforms × 13 scales)
    price_spread.py             E1: logistic price-spread transmissibility
    flow_efficiency.py          E2: interchange / capacity efficiency
    congestion.py               E3: congestion proxy

scripts/                        Experiment runners (in development order)
  run_demo.py                   Scalar prototype
  run_divergence_analysis.py    Bounded-price effect (D1–D5)
  run_decontaminated_lead.py    Decontaminated 1h lead
  run_graph_model.py            Early graph prototype
  run_graph_analysis.py         14-node graph: E calibration, bottlenecks (figs 21–25)
  run_experiments.py            7-experiment suite (E1–E7 figures)
  run_spatial_experiments.py    Gradient-rich forcing (S1–S4 figures)
  run_threshold_propagation.py  Hop-ordered propagation (T1–T4 figures)
  run_corridor_loading.py       NW/SW corridor loading (CL_* figures)
  run_hard_floor_corridor.py    Hard floor E variants (HF_* figures)
  run_sw_variants.py            SW spine V1/V2/V3 (SV_* figures)
  run_dual_constraint.py        Dual/triple cascade (DC_* figures)
```

---

## Model parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| η (eta) | 0.20 | Diffusion rate. Row-normalised Laplacian bounds λ_max ≤ 2; stable for η < 0.5 |
| γ V1 | 0.02 | Damping (mean reversion) |
| γ V2 | 0.01 | Half-damping; used in SW variant and cascade experiments |
| ε (floor) | 0.0 | Hard floor: E → 0 when edge stress ≥ threshold |
| threshold | 1.0 | Stress = \|ΔΦ\| / capacity_gw; edge blocks when stress ≥ 1.0 |
| bg_std | ≈1.13 | Background spatial σ of gradient-rich Φ field (seed=42, T=1440h) |
| T | 1440h | Gradient-rich field length (60 days) |
| loading range | 0.5–3.0 × σ | Corridor upstream loading sweep |

---

## The network

14 balancing authorities covering the Western Interconnect, 28 edges with
physical capacity weights. SW bottleneck: AZPS→WACM (1.0 GW). NW spine:
CISO→PACW→BPAT (4.5 / 4.0 GW; never binds). The SW corridor (CISO→WALC→AZPS→WACM)
is the primary experimental subject; PACE (cap 1.5 GW from AZPS) is the
first-wave rerouting path when the bottleneck blocks.

---

## Connecting to real data

EIA and CAISO adapters are production-ready. To replace synthetic data:

```python
from constraint_field.adapters.eia import EIAAdapter
from constraint_field.graph.node_signals import build_node_field

eia    = EIAAdapter(api_key="YOUR_KEY")
demand = eia.fetch_demand("2023-01-01", "2023-03-31")
prices = eia.fetch_lmp("2023-01-01", "2023-03-31")
field  = build_node_field(demand, prices)   # same structure as synthetic field
```

The resulting `field` dict (`S`, `R`, `Phi`, `Psi`) is a drop-in replacement
for the gradient-rich synthetic field used in all spatial and corridor experiments.
Real nodal LMP data would be expected to produce substantially stronger spatial
gradients and a cleaner test of the bounded-price effect.

---

## Citation

If you use this code, please cite using CITATION.cff
