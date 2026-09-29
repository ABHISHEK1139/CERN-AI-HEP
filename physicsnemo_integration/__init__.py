"""
physicsnemo_integration: NVIDIA PhysicsNeMo benchmarking.

Modules:
    wrapper   — Adapter to wrap PhysicsNeMo models in unified interface
    benchmark — Run benchmarks comparing custom vs PhysicsNeMo models
"""

from physicsnemo_integration.benchmark import PhysicsNeMoBenchmark
from physicsnemo_integration.wrapper import MeshGraphNetLayer, PhysicsNeMoWrapper

__all__ = ["MeshGraphNetLayer", "PhysicsNeMoBenchmark", "PhysicsNeMoWrapper"]
