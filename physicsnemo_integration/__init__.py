"""
physicsnemo_integration: NVIDIA PhysicsNeMo benchmarking.

Modules:
    wrapper   — Adapter to wrap PhysicsNeMo models in unified interface
    benchmark — Run benchmarks comparing custom vs PhysicsNeMo models
"""

from physicsnemo_integration.wrapper import PhysicsNeMoWrapper, MeshGraphNetLayer
from physicsnemo_integration.benchmark import PhysicsNeMoBenchmark

__all__ = ["PhysicsNeMoWrapper", "MeshGraphNetLayer", "PhysicsNeMoBenchmark"]
