"""constraint_field.graph — graph-based constraint field model."""

from .topology      import build_wecc_graph, build_caiso_graph, WECC_NODES, WECC_EDGES
from .node_signals  import SyntheticNodeSignals, build_node_field
from .edge_fluidity import (
    E1_price_spread_edge, E2_flow_efficiency_edge, E3_congestion_proxy_edge,
    weighted_adjacency, graph_laplacian, constant_laplacian,
)
from .propagation   import GraphPropagator, PropagationConfig, propagation_metrics, bottleneck_analysis
from .visualize_graph import (
    plot_field_snapshot, plot_propagation_comparison,
    plot_bottleneck_map, make_phi_animation, plot_graph_dashboard,
)

__all__ = [
    "build_wecc_graph", "build_caiso_graph",
    "SyntheticNodeSignals", "build_node_field",
    "E1_price_spread_edge", "E2_flow_efficiency_edge", "E3_congestion_proxy_edge",
    "weighted_adjacency", "graph_laplacian", "constant_laplacian",
    "GraphPropagator", "PropagationConfig", "propagation_metrics", "bottleneck_analysis",
    "plot_field_snapshot", "plot_propagation_comparison",
    "plot_bottleneck_map", "make_phi_animation", "plot_graph_dashboard",
]