"""
constraint_field.adapters
=========================
Data ingestion layer.  All adapters expose the same interface:

    adapter.fetch(start, end) -> pd.DataFrame   # demand / generation
    adapter.fetch_prices(start, end)             # prices  (where applicable)
    adapter.fetch_flows(start, end)              # interchange (where applicable)

Use the factory function `get_adapter()` for config-driven instantiation.
"""

from .base       import BaseAdapter
from .eia        import EIAAdapter
from .caiso      import CAISOAdapter
from .synthetic  import SyntheticAdapter


def get_adapter(source: str, **kwargs) -> BaseAdapter:
    """
    Factory that returns the appropriate adapter from a string name.

    Parameters
    ----------
    source : str
        One of: "eia", "caiso", "synthetic"
    **kwargs
        Passed through to the adapter constructor.

    Examples
    --------
    >>> adapter = get_adapter("synthetic", peak_demand_mw=35_000)
    >>> df = adapter.fetch("2023-01-01", "2023-01-31")
    """
    registry = {
        "eia":       EIAAdapter,
        "caiso":     CAISOAdapter,
        "synthetic": SyntheticAdapter,
    }
    key = source.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown adapter '{source}'.  Available: {list(registry)}"
        )
    return registry[key](**kwargs)


__all__ = [
    "BaseAdapter",
    "EIAAdapter",
    "CAISOAdapter",
    "SyntheticAdapter",
    "get_adapter",
]
