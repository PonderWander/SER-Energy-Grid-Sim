"""
constraint_field.graph.topology
=================================
Network topology for the graph-based constraint field model.

Observability classification (used throughout this module)
----------------------------------------------------------
OBSERVED      : directly measured or reported in public data
DOCUMENTED    : based on published regulatory filings, NERC maps,
                or EIA Form 411 / EIA-930 interchange reporting
APPROXIMATED  : inferred from geography, grid planning documents,
                or statistical coupling — not directly measured
SYNTHETIC     : generated for demonstration when real data unavailable

Every node, edge, and attribute is labelled with its observability class.

Option A: Coarse WECC / Western Interconnection BA graph
---------------------------------------------------------
Nodes = 11 major WECC balancing authorities that report hourly
        interchange to EIA-930 (Form EIA-411 NERC).
Edges = interchange paths documented in EIA-930 "from_ba" / "to_ba"
        pairs from public annual data summaries and NERC reliability
        assessments. Where direct EIA-930 interchange records exist
        between a pair, edge is DOCUMENTED. Geographic adjacency
        added as APPROXIMATED where no direct interchange record found
        but transmission corridors are well-established in WECC
        planning documents (e.g. WECC 2024 ADS).

Data sources used to construct this topology
--------------------------------------------
1. EIA-930 Balancing Authority List (public):
   https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx
2. NERC 2023 Long-Term Reliability Assessment — Fig. 3-1 BA map
3. EIA Form EIA-411 interchange summary tables (public annual data)
4. WECC 2024 Anchor Data Set (ADS) — corridor descriptions
5. Geographic adjacency from NERC BA boundary shapefile
   (https://hifld-geoplatform.opendata.arcgis.com/datasets/
    electric-planning-area/explore)

Note: This file does NOT claim to reconstruct the physical transmission
network. It constructs a coarse administrative-level graph of
balancing authorities and their interchange relationships, which is
a strictly coarser and less precise representation than the actual
AC power flow network.

Option B: CAISO sub-regional pricing node approximation
-------------------------------------------------------
CAISO publishes the following nodal/zonal information publicly:
  - Aggregate Pricing Nodes (APNodes) mapped to 3 load zones:
    NP15 (Northern CA), SP15 (Southern CA), ZP26 (Central CA)
  - Trading Hubs: TH_NP15, TH_SP15, TH_ZP26
  - Inter-tie nodes: ~15 published scheduling points

The intra-CAISO sub-graph uses these 6 nodes (3 zones + 3 hubs)
with edges based on:
  - Published CAISO congestion path definitions (FERC filings)
  - Geographic adjacency of the three load zones
  - Statistical LMP correlation (APPROXIMATED from public price data)

This is explicitly a simplified approximation of the actual
CAISO nodal network, which has ~5,000 physical nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

Observability = Literal["OBSERVED", "DOCUMENTED", "APPROXIMATED", "SYNTHETIC"]


# ──────────────────────────────────────────────────────────────────────────────
# Node and Edge dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeSpec:
    id:           str
    name:         str
    ba_code:      str          # EIA BA abbreviation
    region:       str          # geographic region
    lon:          float        # approximate centroid longitude
    lat:          float        # approximate centroid latitude
    peak_gw:      float        # approximate peak demand (GW) from EIA-860
    observability: Observability = "DOCUMENTED"
    notes:        str = ""


@dataclass
class EdgeSpec:
    u:            str          # from node id
    v:            str          # to node id
    capacity_gw:  float        # approximate transfer capability (GW)
    observability: Observability = "DOCUMENTED"
    corridor:     str = ""     # descriptive corridor name
    notes:        str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Option A: WECC Coarse BA Graph (11 nodes)
# ──────────────────────────────────────────────────────────────────────────────

WECC_NODES: list[NodeSpec] = [
    NodeSpec("CISO", "CAISO",              "CISO", "Southwest", -120.5, 37.3, 52.0,
             "DOCUMENTED", "EIA-930 reporter; NERC WECC region"),
    NodeSpec("PACE", "PacifiCorp East",    "PACE", "Northwest",  -111.1, 42.0, 11.5,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("PACW", "PacifiCorp West",    "PACW", "Northwest",  -122.5, 44.5,  9.5,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("BPAT", "Bonneville PA",      "BPAT", "Northwest",  -122.0, 45.7, 13.0,
             "DOCUMENTED", "EIA-930 reporter; major hydro BA"),
    NodeSpec("NEVP", "NV Energy",          "NEVP", "Southwest",  -115.1, 36.2,  8.5,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("AZPS", "APS (Arizona)",      "AZPS", "Southwest",  -112.1, 33.5, 13.5,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("WALC", "Western APS/Walc",   "WALC", "Southwest",  -114.5, 34.0,  3.5,
             "DOCUMENTED", "EIA-930 reporter; includes WAPA-Desert Southwest"),
    NodeSpec("IPCO", "Idaho Power",        "IPCO", "Northwest",  -115.7, 43.6,  3.4,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("NWMT", "NorthWestern MT",    "NWMT", "Northwest",  -110.4, 46.9,  1.8,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("PSCO", "Xcel Energy CO",     "PSCO", "Mountain",   -105.0, 39.7,  7.0,
             "DOCUMENTED", "EIA-930 reporter"),
    NodeSpec("PNM",  "PNM Resources NM",   "PNM",  "Southwest",  -106.6, 35.1,  3.0,
             "DOCUMENTED", "EIA-930 reporter"),
]

# Edges: documented interchange paths from EIA-930 annual summaries
# capacity_gw is approximate ATC from WECC ADS / OASIS transfer limits
WECC_EDGES: list[EdgeSpec] = [
    # CAISO corridors — well-documented in CAISO OASIS & WECC ADS
    EdgeSpec("CISO", "BPAT", 4.8, "DOCUMENTED",
             "CA-OR Intertie (COI)", "Pacific AC/DC Intertie; EIA-930 interchange pair"),
    EdgeSpec("CISO", "NEVP", 2.0, "DOCUMENTED",
             "CA-NV path",    "Documented EIA-930 interchange"),
    EdgeSpec("CISO", "AZPS", 1.6, "DOCUMENTED",
             "CA-AZ path",    "Documented EIA-930 interchange"),
    EdgeSpec("CISO", "WALC", 0.8, "DOCUMENTED",
             "CA-WALC path",  "Documented EIA-930 interchange"),
    EdgeSpec("CISO", "PACW", 1.2, "DOCUMENTED",
             "CA-PacW",       "Documented EIA-930 interchange"),

    # Pacific Northwest
    EdgeSpec("BPAT", "PACW", 3.5, "DOCUMENTED",
             "BPA-PacW",      "Documented EIA-930 interchange"),
    EdgeSpec("BPAT", "PACE", 1.5, "DOCUMENTED",
             "BPA-PacE",      "Documented EIA-930 interchange"),
    EdgeSpec("BPAT", "IPCO", 1.0, "DOCUMENTED",
             "BPA-IPCO",      "Documented EIA-930 interchange"),
    EdgeSpec("BPAT", "NWMT", 0.6, "APPROXIMATED",
             "BPA-NWMont",    "Geographic adjacency; WECC ADS corridor"),
    EdgeSpec("PACW", "IPCO", 0.8, "DOCUMENTED",
             "PacW-IPCO",     "Documented EIA-930 interchange"),
    EdgeSpec("PACE", "IPCO", 0.9, "DOCUMENTED",
             "PacE-IPCO",     "Documented EIA-930 interchange"),
    EdgeSpec("PACE", "NWMT", 0.5, "APPROXIMATED",
             "PacE-NWMont",   "WECC ADS; geographic adjacency"),
    EdgeSpec("PACE", "PSCO", 1.1, "DOCUMENTED",
             "PacE-Xcel",     "Documented EIA-930 interchange"),

    # Southwest
    EdgeSpec("NEVP", "AZPS", 1.2, "DOCUMENTED",
             "NV-AZ",         "Documented EIA-930 interchange"),
    EdgeSpec("NEVP", "WALC", 0.7, "APPROXIMATED",
             "NV-WALC",       "WECC ADS corridor; geographic adjacency"),
    EdgeSpec("AZPS", "WALC", 1.0, "DOCUMENTED",
             "AZ-WALC",       "Documented EIA-930 interchange"),
    EdgeSpec("AZPS", "PNM",  0.8, "DOCUMENTED",
             "AZ-NM",         "Documented EIA-930 interchange"),
    EdgeSpec("WALC", "PNM",  0.5, "APPROXIMATED",
             "WALC-NM",       "WECC ADS; geographic adjacency"),
    EdgeSpec("PSCO", "PNM",  0.6, "DOCUMENTED",
             "CO-NM",         "Documented EIA-930 interchange"),
]


def build_wecc_graph() -> nx.Graph:
    """
    Construct the WECC coarse BA graph as a NetworkX graph.

    Node attributes: name, region, lon, lat, peak_gw, observability
    Edge attributes: capacity_gw, observability, corridor

    Returns
    -------
    nx.Graph  (undirected; interchange is bidirectional)
    """
    G = nx.Graph(
        name="WECC Coarse BA Graph",
        description="11-node WECC balancing authority graph",
        observability_note=(
            "Edges are DOCUMENTED (from EIA-930 interchange pairs) or "
            "APPROXIMATED (from WECC ADS corridor descriptions and "
            "geographic adjacency). Capacity values are approximate ATC "
            "from WECC planning documents, not OASIS TTC."
        )
    )

    for node in WECC_NODES:
        G.add_node(
            node.id,
            name=node.name,
            ba_code=node.ba_code,
            region=node.region,
            lon=node.lon,
            lat=node.lat,
            peak_gw=node.peak_gw,
            observability=node.observability,
            notes=node.notes,
        )

    for edge in WECC_EDGES:
        G.add_edge(
            edge.u, edge.v,
            capacity_gw=edge.capacity_gw,
            observability=edge.observability,
            corridor=edge.corridor,
            notes=edge.notes,
        )

    log.info(
        "WECC graph built: %d nodes, %d edges  "
        "(%d documented, %d approximated)",
        G.number_of_nodes(),
        G.number_of_edges(),
        sum(1 for _, _, d in G.edges(data=True) if d["observability"] == "DOCUMENTED"),
        sum(1 for _, _, d in G.edges(data=True) if d["observability"] == "APPROXIMATED"),
    )
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Option B: CAISO Sub-Regional Graph (6 nodes)
# ──────────────────────────────────────────────────────────────────────────────

CAISO_NODES: list[NodeSpec] = [
    NodeSpec("NP15", "NP15 Load Zone",   "CISO", "NorCal",  -121.5, 38.5, 28.0,
             "DOCUMENTED", "CAISO published load zone; FERC-jurisdictional"),
    NodeSpec("SP15", "SP15 Load Zone",   "CISO", "SoCal",   -118.2, 34.1, 21.0,
             "DOCUMENTED", "CAISO published load zone"),
    NodeSpec("ZP26", "ZP26 Load Zone",   "CISO", "CenCal",  -120.0, 36.7,  4.0,
             "DOCUMENTED", "CAISO published load zone"),
    NodeSpec("TH_NP", "TH NP15 Hub",    "CISO", "NorCal",  -121.5, 38.5,  0.0,
             "DOCUMENTED", "CAISO published trading hub"),
    NodeSpec("TH_SP", "TH SP15 Hub",    "CISO", "SoCal",   -118.2, 34.1,  0.0,
             "DOCUMENTED", "CAISO published trading hub"),
    NodeSpec("TH_ZP", "TH ZP26 Hub",    "CISO", "CenCal",  -120.0, 36.7,  0.0,
             "DOCUMENTED", "CAISO published trading hub"),
]

CAISO_EDGES: list[EdgeSpec] = [
    # Zone-to-zone: based on CAISO published congestion path definitions
    EdgeSpec("NP15", "SP15", 4.5, "DOCUMENTED",
             "Path 26 (NP15-SP15)",
             "CAISO Path 26; published transfer limit in CAISO OASIS"),
    EdgeSpec("NP15", "ZP26", 2.0, "DOCUMENTED",
             "NP15-ZP26 corridor",
             "CAISO published congestion path"),
    EdgeSpec("SP15", "ZP26", 1.8, "DOCUMENTED",
             "SP15-ZP26 corridor",
             "CAISO published congestion path"),
    # Hub-to-zone mappings (definitional — hubs are within zones)
    EdgeSpec("TH_NP", "NP15", 99.0, "DOCUMENTED",
             "Hub-Zone definitional",
             "TH_NP15 is the NP15 zone hub by CAISO definition"),
    EdgeSpec("TH_SP", "SP15", 99.0, "DOCUMENTED",
             "Hub-Zone definitional",
             "TH_SP15 is the SP15 zone hub by CAISO definition"),
    EdgeSpec("TH_ZP", "ZP26", 99.0, "DOCUMENTED",
             "Hub-Zone definitional",
             "TH_ZP26 is the ZP26 zone hub by CAISO definition"),
]


def build_caiso_graph() -> nx.Graph:
    """Construct the CAISO sub-regional 6-node graph."""
    G = nx.Graph(
        name="CAISO Sub-Regional Graph",
        description="6-node CAISO zone/hub graph",
        observability_note=(
            "Zone nodes and hub-zone edges are DOCUMENTED in CAISO tariff "
            "and FERC filings. Zone-to-zone edge capacities are approximate "
            "transfer limits from CAISO OASIS scheduling limits. "
            "This is a coarse approximation of the ~5000-node CAISO network."
        )
    )
    for node in CAISO_NODES:
        G.add_node(node.id, **{k: v for k, v in node.__dict__.items() if k != "id"})
    for edge in CAISO_EDGES:
        G.add_edge(edge.u, edge.v,
                   **{k: v for k, v in edge.__dict__.items() if k not in ("u", "v")})
    log.info("CAISO graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G
