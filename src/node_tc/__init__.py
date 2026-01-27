from __future__ import annotations

__all__ = ["NODETC", "NODETrajectoryCluster"]

def __getattr__(name: str):
    if name == "NODETC":
        from .model import NODETC
        return NODETC
    if name == "NODETrajectoryCluster":
        from .estimator import NODETrajectoryCluster
        return NODETrajectoryCluster
    raise AttributeError(f"module 'node_tc' has no attribute {name}")

