"""
experiments/cross_corridor/config.py
======================================
Corridor definitions, E-regime specs, loading variants, and all tunable
parameters for the cross-corridor stress routing experiment.

All node/edge names reference constraint_field/graph/network.py (NODES, EDGES).
No names are hard-coded in the metric or simulation modules; they import from here.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Corridor membership
# ─────────────────────────────────────────────────────────────────────────────

# SW corridor: Desert Southwest spine + bottleneck zone
SW_CORRIDOR = {
    "nodes":   ["CISO", "WALC", "AZPS", "WACM", "PACE", "NEVP", "IID", "LDWP"],
    "path":    ["CISO", "WALC", "AZPS", "WACM"],   # primary spine
    "bottleneck": ("AZPS", "WACM"),                 # capacity 1.0 GW
    "label":   "SW",
    "upstream": "CISO",
    "terminal": "WACM",
    # Edges that define the SW corridor (canonical direction)
    "edges": [
        ("CISO", "WALC"),
        ("WALC", "AZPS"),
        ("AZPS", "WACM"),
        ("AZPS", "PACE"),
        ("AZPS", "NEVP"),
        ("WALC", "NEVP"),
        ("WALC", "IID"),
        ("WALC", "LDWP"),
        ("CISO", "LDWP"),
    ],
}

# NW corridor: Pacific Northwest spine
NW_CORRIDOR = {
    "nodes":   ["CISO", "PACW", "BPAT", "IPCO", "NWMT", "NEVP", "LDWP"],
    "path":    ["CISO", "PACW", "BPAT"],           # primary spine
    "bottleneck": None,                             # NW spine never blocks
    "label":   "NW",
    "upstream": "CISO",
    "terminal": "BPAT",
    "edges": [
        ("CISO", "PACW"),
        ("CISO", "BPAT"),
        ("PACW", "BPAT"),
        ("PACW", "IPCO"),
        ("PACW", "NEVP"),
        ("BPAT", "IPCO"),
        ("BPAT", "NWMT"),
        ("IPCO", "NWMT"),
    ],
}

# Connector nodes: appear in both corridor node sets
CONNECTOR_NODES = ["CISO", "NEVP", "LDWP", "WALC"]

# All corridor nodes (union)
ALL_CORRIDOR_NODES = list(dict.fromkeys(SW_CORRIDOR["nodes"] + NW_CORRIDOR["nodes"]))

# ─────────────────────────────────────────────────────────────────────────────
# Loading variants
# ─────────────────────────────────────────────────────────────────────────────

LOADING_SIGMA = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Each variant specifies (sw_loading_frac, nw_loading_frac) as multipliers
# applied to the shared loading_sigma level.
# E.g. SW_HEAVY at loading=2.0σ → SW gets 2.0σ, NW gets 0.5σ
LOADING_VARIANTS = {
    "symmetric":  (1.0, 1.0),
    "SW_heavy":   (1.0, 0.33),
    "NW_heavy":   (0.33, 1.0),
    "diffuse":    (0.5,  0.5),   # network-wide lower amplitude
}

# ─────────────────────────────────────────────────────────────────────────────
# E regimes
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: human label + construction key used in build_E_regime()
E_REGIMES = {
    "uniform":        "E=1 (uniform transmissibility)",
    "calibrated_E1":  "Static calibrated E1",
    "state_dep":      "State-dependent E (soft sigmoid)",
    "shuffled":       "Shuffled E (preserves distribution, breaks spatial structure)",
    "inverted":       "Inverted E (SW/NW temporal structure swapped)",
}

# Soft-sigmoid parameters (state-dependent E)
SD_ALPHA    = 4.0
SD_MID      = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion parameters
# ─────────────────────────────────────────────────────────────────────────────

ETA   = 0.20   # diffusion rate (row-normalised Laplacian; stable for η < 0.5)
GAMMA = 0.02   # damping / mean-reversion

STEPS = 60     # simulation length per condition
SEED  = 42

# ─────────────────────────────────────────────────────────────────────────────
# Metric thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Stress co-occurrence: Φ levels at which nodes are counted as "activated"
COOCCURRENCE_THRESHOLDS = [1.0, 1.5, 2.0]   # in normalised Φ units

# Path activation: fraction of source-node peak Φ that a downstream node
# must reach to be counted as part of the active transfer path
PATH_ACTIVATION_FRAC = 0.10

# Leakage: Φ entering the other corridor must exceed this fraction of initial
# injected stress to count as cross-corridor leakage
LEAKAGE_MIN_FRAC = 0.02

# ─────────────────────────────────────────────────────────────────────────────
# Output paths (relative to project root, overridable)
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

OUT_FIG = Path("outputs/figures")
OUT_DAT = Path("outputs/data")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)

DPI = 150
