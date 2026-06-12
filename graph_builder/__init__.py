"""
graph_builder: Convert collision events into PyTorch Geometric graph objects.

Modules:
    features         — Feature engineering and normalization
    graph_constructor — Event-to-graph conversion with edge strategies
    dataset          — PyG InMemoryDataset and DataLoader integration
"""

from graph_builder.features import FeatureExtractor
from graph_builder.graph_constructor import EventGraphConstructor
from graph_builder.dataset import CollisionEventDataset

__all__ = [
    "FeatureExtractor",
    "EventGraphConstructor",
    "CollisionEventDataset",
]
