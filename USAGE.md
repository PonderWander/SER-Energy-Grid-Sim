# USAGE

Practical guide to running, configuring, and extending the `constraint_field` prototype.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Running individual experiments](#running-individual-experiments)
- [Experiment outputs](#experiment-outputs)
- [Configuration reference](#configuration-reference)
- [The 14-node network](#the-14-node-network)
- [Field variables](#field-variables)
- [Edge fluidity (E)](#edge-fluidity-e)
- [Constraint modes](#constraint-modes)
- [Using real data](#using-real-data)
- [Extending the model](#extending-the-model)

---

## Installation

```bash
git clone <repo>
cd constraint_field_complete
pip install -r requirements.txt
```

No build step required. All scripts use `PYTHONPATH=.` to resolve the package from
the project root.

**Dependencies:**
```
pandas>=2.0   numpy>=1.24   scipy>=1.11   matplotlib>=3.7
networkx>=3.1  scikit-learn>=1.3  statsmodels  requests
```

---

## Quick start

Run the full spatial experiment suite (the fastest meaningful entry point):

```bash
PYTHONPATH=. python scripts/run_spatial_experiments.py
```

This builds the gradient-rich synthetic field (seed=42, T=1440h), runs four
topology/ablation/shock/decay experiments, and writes figures and CSVs to
`outputs/figures/` and `outputs/data/`.

To run the complete corridor-loading and cascade sequence:

```bash
PYTHONPATH=. python scripts/run_corridor_loading.py
PYTHONPATH=. python scripts/run_hard_floor_corridor.py
PYTHONPATH=. python scripts/run_sw_variants.py
PYTHONPATH=. python scripts/run_dual_constraint.py
```

All scripts are deterministic from `seed=42` and produce no side-effects outside
the `outputs/` directory.

---

## Running individual experiments

Scripts are ordered by development iteration. Each is self-contained: it builds
the graph and field internally, runs its experiments, and writes all outputs.

### 1. Scalar prototype

```bash
PYTHONPATH=. python scripts/run_demo.py
```

Single-node proof of concept. Builds S, R, Φ from synthetic demand/price series,
runs diffusion simulation, produces basic dashboard figures.

---

### 2. Divergence analysis

```bash
PYTHONPATH=. python scripts/run_divergence_analysis.py
```

Computes D1–D5 divergence metrics and tests the bounded-price effect:
hours with high |Φ| and moderate R vs matched hours with low |Φ| and moderate R.

Key result: Cohen's d = 3.35, p < 1e-6. Survives all four S–R cluster regimes.

**Output:** `outputs/data/divergence_summary.csv`, `divergence_report.txt`

---

### 3. Decontaminated lead analysis

```bash
PYTHONPATH=. python scripts/run_decontaminated_lead.py
```

Tests whether the 1h lead (D1→I1, D2→I1) survives time-of-day residualisation
and rolling-window decontamination. Confirms r = 0.647 (D1→I1) is not an artifact.

**Output:** `outputs/data/regime_decontaminated.csv`, `decontaminated_report.txt`

---

### 4. Graph model prototype

```bash
PYTHONPATH=. python scripts/run_graph_model.py
```

Early 14-node prototype. Establishes the row-normalised Laplacian (λ_max ≤ 2,
stable for η < 0.5) as the correct operator, replacing the capacity-weighted raw
Laplacian (λ_max ≈ 54, unstable at η = 0.04).

---

### 5. Graph analysis

```bash
PYTHONPATH=. python scripts/run_graph_analysis.py
```

Full 14-node Western Interconnect analysis: E1/E2/E3 calibration, propagation
comparison (reduced vs upgraded), bottleneck identification.

Key result: E1 = 0.606, E2 = 0.734, E3 = 0.526 (post-calibration).
Bottleneck edges: IPCO→NWMT, AZPS→WACM, WALC→LDWP.

**Output:** `outputs/figures/` figures 21–25, `graph_report.txt`

---

### 6. Seven-experiment suite

```bash
PYTHONPATH=. python scripts/run_experiments.py
```

Runs all seven core experiments on the 14-node graph:

| # | Experiment | Key finding |
|---|-----------|-------------|
| 1 | η × γ parameter sweep | Topology structure, not E weighting, drives spatial autocorrelation |
| 2 | Topology sensitivity | Real vs shuffled vs random vs distance graph |
| 3 | E ablation | E contributes < 0.00054 ΔRMSE across full sweep |
| 4 | Shock propagation | Hop-ordered arrival, 22.5% attenuation per hop |
| 5 | Gradient forcing | Between-node variance ratio = 0.0016 (homogeneous) |
| 6 | State-dependent E | Soft sigmoid E, stress-responsive |
| 7 | Regime report | Summary classification |

**Output:** `E1_param_sweep.png` through `E7_regime_map.png`; `exp1_sweep.csv` through `exp7_report.txt`

---

### 7. Spatial experiments (gradient-rich forcing)

```bash
PYTHONPATH=. python scripts/run_spatial_experiments.py
```

Replaces the Cholesky synthetic model with structurally offset node signals
(SW desert nodes: R offset +1.4–1.8σ; NW hydro nodes: R offset −1.0–1.8σ).
Between-node variance ratio rises from 0.0016 to meaningful levels.

**Parameters:**
```python
ETA    = 0.20
GAMMA  = 0.02
STEPS  = 72
START_T = 500   # timestep to start simulation within the 1440h field
```

**Experiments:**

| ID | Name | Key result |
|----|------|-----------|
| S1 | Topology sensitivity | Moran's I: real = 0.568, shuffled = −0.132. Gap = 0.70 |
| S2 | E ablation | MI gap (E1 vs E=1) = 0.006 — E is weak |
| S3 | Shock propagation | Kendall τ = 0.787 ± 0.071 across all 24 conditions |
| S4 | Distance decay | Phi correlation decays with hop distance |

**Output:** `S1_topology.png`, `S2_ablation.png`, `S3_shock.png`, `S4_distance_decay.png`

---

### 8. Threshold propagation

```bash
PYTHONPATH=. python scripts/run_threshold_propagation.py
```

Tests hop-ordering consistency across six source nodes and four loading thresholds.

**Parameters:**
```python
SHOCK_NODES = ["CISO", "AZPS", "PACW", "NEVP", "BPAT", "PSCO"]
THRESHOLDS  = [1.0, 1.5, 2.0, 2.5]   # × background spatial std
SHOCK_STEPS = 36
```

Key result: hop-ordering consistent in 100% of conditions. Top mediator node: WALC (50%).
Pacific Intertie spine (CISO→PACW→BPAT) is the dominant corridor.

**Output:** `T1_consistency_matrix.png`, `T2_mediation_maps.png`, `T3_threshold_profiles.png`,
`T4_source_comparison.png`

---

### 9. Corridor loading

```bash
PYTHONPATH=. python scripts/run_corridor_loading.py
```

Initialises a monotone Φ gradient along each corridor and sweeps upstream loading
from 0.5σ to 3.0σ.

**Corridors:**
```
NW_spine: CISO → PACW → BPAT           (bypass: CISO→BPAT, caps 4.5/4.0 GW)
SW_spine: CISO → WALC → AZPS → WACM   (bottleneck: AZPS→WACM, cap 1.0 GW)
```

**Parameters:**
```python
ETA_C          = 0.20
GAMMA_C        = 0.02
STEPS          = 60
LOADING_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # × background std
```

**Two modes per condition:** `static_E` (constant E1) and `state_E` (soft sigmoid,
E decreases as edge stress rises, α=4.0, stress_mid=0.5).

Key result: SW bottleneck AZPS→WACM reaches stress = 1.13 at loading 1.5σ, 2.26 at 3.0σ.
Spillover into PACE and NEVP increases monotonically. NW spine never blocks.

**Output:** `CL_NW_spine.png`, `CL_SW_spine.png`, `CL_regime_summary.png`

---

### 10. Hard floor corridor

```bash
PYTHONPATH=. python scripts/run_hard_floor_corridor.py
```

Compares four E-computation modes on the SW spine:

```python
E_MODES = ["static", "soft_sigmoid", "linear_ramp", "hard_floor"]

STRESS_THRESHOLD = 1.0   # E collapses at stress >= 1.0
E_FLOOR          = 0.02  # hard floor: E → 2% of base value
```

Key result: hard floor produces retained accumulation at AZPS (peak +0.43σ, dwell 56/60 steps)
and WACM isolation (correlation drops to 0.32). Linear ramp stays in throttled dispersal
throughout.

**Output:** `HF_SW_deep_dive.png`, `HF_mode_comparison.png`, `HF_regime_transition.png`

---

### 11. SW spine variants

```bash
PYTHONPATH=. python scripts/run_sw_variants.py
```

Three hard-floor ε=0 variants (no residual transmission at threshold):

```python
VARIANTS = {
    "V1_hard_eps0":       {"eps": 0.0,  "gamma": 0.02},  # current γ
    "V2_hard_eps0_lowγ":  {"eps": 0.0,  "gamma": 0.01},  # half damping
    "V3_linear_ramp":     {"eps": None, "gamma": 0.02},  # baseline comparator
}
```

NW spine is run under the same settings as a negative control (threshold never fires).

**Measurements (regime-detection only — no RMSE):**

| Panel | Metric | Purpose |
|-------|--------|---------|
| A | AZPS net_rise, dwell_75, AUC | Retention plateau |
| B | corr_blocked, frac_isolated | WACM decoupling |
| C | branch_auc (PACE+NEVP) vs primary_auc (WACM) | Rerouting |
| D | Regime classification heatmap | Loading × variant map |

Key result: V2 plateau = +0.479σ (net rise), dwell = 57/60 steps. WACM correlation
drops to 0.086 during blocked steps. PACE is first-wave rerouting path; NEVP second-wave.

**Output:** `SV_A_retention.png`, `SV_B_isolation.png`, `SV_C_branches.png`,
`SV_D_regime.png`, `SV_summary.png`

---

### 12. Dual constraint cascade

```bash
PYTHONPATH=. python scripts/run_dual_constraint.py
```

Introduces a second constraint on PACE→WACM (the primary rerouting path) to test
whether cascading isolation and multi-node accumulation emerge.

```python
CONSTRAINT_SETS = {
    "single": {("AZPS","WACM"): (1.0, 1.0)},
    "dual":   {("AZPS","WACM"): (1.0, 1.0),
               ("PACE","WACM"): (1.0, 1.2)},   # cap = real physical capacity
    "triple": {("AZPS","WACM"): (1.0, 1.0),
               ("PACE","WACM"): (1.0, 1.2),
               ("AZPS","NEVP"): (1.0, 0.8)},   # tertiary: full AZPS isolation
}

TRACK_NODES = ["CISO","WALC","AZPS","NEVP","PACE","WACM","PSCO"]
STEPS_DC    = 80   # extra steps to observe cascade completion
```

**Constraint parameters:** `(stress_threshold, cap_effective)`.
Stress = |ΔΦ_ij| / cap_effective. Edge blocks when stress ≥ threshold.

**Regime outcomes:**

| Loading | Single | Dual | Triple |
|---------|--------|------|--------|
| 0.5–1.0σ | linear | linear | linear |
| 1.5–2.0σ | single-node accumulation | cascading isolation | cascading isolation |
| 2.5–3.0σ | single-node accumulation | cascading isolation | **network fragmentation** |

At 2.5σ: AZPS→WACM blocks for 17 steps; PACE→WACM for 6 steps (11-step window
where primary stays blocked but secondary has re-opened). Max upstream stress
(WALC→AZPS) = 1.883 at 3.0σ — approaches threshold.

**Output:** `DC_accumulation.png`, `DC_cascade_timing.png`, `DC_isolation_map.png`,
`DC_regime_map.png`, `DC_dashboard.png`

---

## Experiment outputs

All outputs are written to:

```
outputs/
  figures/     PNG figures (150 DPI)
  data/        CSV result tables and TXT reports
```

The directory is created automatically on first run. Outputs are not versioned —
re-running a script overwrites its previous outputs.

To redirect outputs, edit `OUT_FIG` and `OUT_DAT` at the top of any script:

```python
OUT_FIG = Path("my_outputs/figures")
OUT_DAT = Path("my_outputs/data")
```

---

## Configuration reference

### Core diffusion parameters

All scripts import `ETA_C` and `GAMMA_C` from `run_corridor_loading`. To change
globally, edit those constants there; downstream scripts inherit them via import.

| Parameter | Default | Where set | Notes |
|-----------|---------|-----------|-------|
| `ETA` / `ETA_C` | 0.20 | `run_corridor_loading.py` | Diffusion rate. Row-normalised Laplacian guarantees λ_max ≤ 2, so η < 0.5 is always stable |
| `GAMMA` / `GAMMA_C` | 0.02 | `run_corridor_loading.py` | Damping (mean reversion). V2 uses 0.01 |
| `GAMMA_LOW` | 0.01 | `run_sw_variants.py` | Half-damping variant (V2) |
| `STRESS_THRESHOLD` | 1.0 | `run_sw_variants.py`, `run_hard_floor_corridor.py` | Edge stress level at which E collapses |
| `E_FLOOR` | 0.02 | `run_hard_floor_corridor.py` | Residual E at threshold (0.0 in SW variants) |
| `STEPS` | 60 / 72 | per script | Simulation length in timesteps |
| `LOADING_LEVELS` | [0.5…3.0] | `run_corridor_loading.py` | Loading multipliers × background σ |

### Gradient-rich field constants

Edit `ZONE_R_OFFSET` and `ZONE_S_OFFSET` in `run_spatial_experiments.py` to change
the structural price-zone offsets that drive inter-node Φ gradients.

```python
ZONE_R_OFFSET = {
    "AZPS": +1.80,   # SW desert — high price pressure
    "BPAT": -1.80,   # NW hydro  — low price pressure
    "CISO":  0.00,   # anchor node
    ...
}
```

Setting all offsets to 0 recovers the Cholesky homogeneous baseline (between-node
variance ratio ≈ 0.0016), which was used in earlier iterations before gradient-rich
forcing was introduced.

### Constraint set configuration

In `run_dual_constraint.py`, add or remove constraint edges by editing `CONSTRAINT_SETS`:

```python
CONSTRAINT_SETS = {
    "my_scenario": {
        ("AZPS", "WACM"): (1.0, 1.0),    # (stress_threshold, cap_effective)
        ("PACE", "WACM"): (0.8, 1.2),    # lower threshold = fires earlier
        ("WALC", "AZPS"): (1.0, 2.5),    # upstream constraint
    },
}
```

Any edge in the 14-node graph can be constrained. `cap_effective` can differ from
the graph's physical capacity to represent partial congestion.

---

## The 14-node network

14 balancing authorities (BAs) representing the Western Interconnect:

| BA | Region | Peak GW | Role |
|----|--------|---------|------|
| CISO | Pacific | 52.0 | Largest BA; anchor node |
| BPAT | Pacific NW | 11.2 | BPA; high hydro |
| PACW | Pacific NW | 8.9 | PacifiCorp West |
| IPCO | Pacific NW | 3.1 | Idaho Power |
| NWMT | Mountain NW | 2.4 | NorthWestern |
| AZPS | Desert SW | 8.1 | APS; SW bottleneck node |
| WACM | Desert SW | 6.8 | WAPA Colorado; downstream terminal |
| WALC | Desert SW | 4.3 | WAPA Lower Colorado |
| NEVP | Mountain | 10.5 | NV Energy |
| PACE | Mountain | 12.1 | PacifiCorp East; first-wave rerouting |
| PSCO | Mountain | 7.4 | Xcel Energy Colorado |
| LDWP | Pacific | 2.9 | LADWP |
| IID | Desert | 0.9 | Imperial Irrigation |
| TIDC | Pacific | 0.6 | Turlock Irrigation |

The graph has 28 directed edges with physical capacity weights (GW). Topology and
capacity data are in `constraint_field/graph/network.py` (`NODES`, `EDGES`).

---

## Field variables

Three observed variables, one inferred:

| Variable | Definition | Range |
|----------|-----------|-------|
| **S** | Rolling z-scored demand (168h window, clipped ±3σ) | [−3, +3] |
| **R** | Rolling z-scored LMP price (same window) | [−3, +3] |
| **Φ = R − S** | Field imbalance: constraint pressure not accounted for by demand | [−6, +6] |
| **E** | Edge-level delivery fluidity ∈ [0, 1] (inferred, not observed) | [0, 1] |

The three E candidates:

| Candidate | Formula | Mean (gradient-rich) |
|-----------|---------|---------------------|
| E1 | Price-spread transmissibility (logistic-scaled) | 0.583 |
| E2 | Flow efficiency (normalised interchange / capacity) | 0.734 |
| E3 | Congestion proxy (inverse price spread) | 0.415 |

E1 is used as the baseline in all corridor and cascade experiments.

---

## Constraint modes

The `simulate` function in `run_sw_variants.py` and the `_build_L` function in
`run_dual_constraint.py` support three E-collapse modes:

| Mode | Behaviour | Use case |
|------|-----------|---------|
| `eps=None` (linear ramp) | E tapers linearly from full to 0 as stress → threshold | Smooth capacity degradation |
| `eps=0.0` (hard floor) | E → exactly 0 when stress ≥ threshold | Physical binding constraint |
| `eps=0.02` (soft floor) | E → 2% of base value at threshold | Near-binding with residual leakage |

The hard floor (ε=0) is the operationally meaningful case: it models a corridor
that is physically blocked above its capacity proxy and produces the accumulation
and isolation signatures of a binding transmission constraint.

---

## Using real data

The EIA and CAISO adapters in `constraint_field/adapters/` are production-ready.

### EIA (demand and price by BA)

```python
from constraint_field.adapters.eia import EIAAdapter

eia = EIAAdapter(api_key="YOUR_EIA_KEY")
demand = eia.fetch_demand("2023-01-01", "2023-03-31")   # DataFrame: T × n_BAs
prices = eia.fetch_lmp("2023-01-01", "2023-03-31")
```

### CAISO (California-specific, higher resolution)

```python
from constraint_field.adapters.caiso import CAISOAdapter

caiso = CAISOAdapter()
lmp   = caiso.fetch_lmp("2023-01-01", "2023-03-31", nodes=["TH_NP15", "TH_SP15"])
flows = caiso.fetch_flows("2023-01-01", "2023-03-31")
```

### Building the field from real data

Once you have real demand and price panels (T × n_BAs), pass them into the
field builder in place of the synthetic signals:

```python
from constraint_field.graph.node_signals import build_node_field

field = build_node_field(demand_df, price_df)
# Returns: {"S": ..., "R": ..., "Phi": ..., "Psi": ...}
# Same structure as the synthetic field — plug directly into any script.
```

The divergence analysis (`run_divergence_analysis.py`) and decontaminated lead
(`run_decontaminated_lead.py`) scripts are the most likely to yield a publishable
signal with real nodal LMP data, since the bounded-price effect (Cohen's d = 3.35
in synthetic data) requires genuine price gradients to validate spatially.

---

## Extending the model

### Adding a new corridor

In `run_corridor_loading.py`, add an entry to `CORRIDORS`:

```python
CORRIDORS["SE_spine"] = {
    "path":        ["PSCO", "WACM", "AZPS"],
    "bypass":      None,
    "branches_from": {
        "PSCO": ["PACE"],
        "WACM": ["PACE", "NEVP"],
    },
    "bottleneck":  ("WACM", "AZPS"),
    "terminal":    "AZPS",
    "upstream":    "PSCO",
    "color":       "#4CAF50",
}
```

The corridor loading, hard-floor, and SW-variants scripts all iterate over
`CORRIDORS`, so the new corridor will be picked up automatically.

### Adding a new constraint scenario

In `run_dual_constraint.py`, add to `CONSTRAINT_SETS`:

```python
CONSTRAINT_SETS["quad"] = {
    ("AZPS","WACM"): (1.0, 1.0),
    ("PACE","WACM"): (1.0, 1.2),
    ("AZPS","NEVP"): (1.0, 0.8),
    ("WALC","AZPS"): (1.0, 2.5),   # upstream constraint
}
```

### Changing the network

Edit `NODES` and `EDGES` in `constraint_field/graph/network.py`. The graph is
a plain `networkx.Graph` with node attributes `lat`, `lon`, `peak_gw` and edge
attribute `capacity_gw`. All downstream code uses the graph object, so topology
changes propagate automatically.
