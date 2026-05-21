"""
experiments/__init__.py
Ensures the project root (parent of this directory) is on sys.path
so that `constraint_field.*` and `scripts.*` are importable from anywhere
within the experiments package tree.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.resolve())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
