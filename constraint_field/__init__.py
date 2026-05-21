"""
constraint_field
================
Field-based constraint dynamics model for electricity system analysis.

Layers
------
Static  : S (load pressure) + R (constraint signal) – observed, reduced
Dynamic : S + R + E (delivery fluidity) – inferred corollary upgrade

Public API
----------
>>> from constraint_field import load_config, run_demo
"""

__version__ = "0.1.0"

from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR   = Path(__file__).parent.parent / "data"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML configuration.  Defaults to config/default.yaml."""
    import yaml
    if path is None:
        path = CONFIG_DIR / "default.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)
