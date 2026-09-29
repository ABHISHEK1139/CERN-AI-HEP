"""
Test helpers.

Kept out of ``conftest.py`` so test modules can import them explicitly.
``conftest`` is auto-loaded as a plugin; this module is imported normally.
"""

import torch
from torch_geometric.data import Data

__all__ = ["MockAutoencoder", "make_graph", "make_graphs", "make_loader"]


def make_graph(
    n_nodes: int = 5,
    feature_dim: int = 11,
    label: int | None = 0,
    seed: int = 0,
) -> Data:
    """A small ring-connected graph with a ring adjacency (no self-loops)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, feature_dim, generator=g)
    idx = torch.arange(n_nodes)
    src = torch.cat([idx, (idx + 1) % n_nodes])
    dst = torch.cat([(idx + 1) % n_nodes, idx])
    data = Data(x=x, edge_index=torch.stack([src, dst]).long())
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data


def make_graphs(n: int = 20, seed: int = 0, label: int | None = "auto", **kwargs) -> list:
    """``n`` graphs, alternating labels 0/1 by default.

    Args:
        n: Number of graphs.
        seed: Base seed; graph ``i`` uses ``seed + i`` so different fixtures
            produce genuinely different data.
        label: Fixed label for every graph, or ``"auto"`` to alternate 0/1.
    """
    return [
        make_graph(
            label=(i % 2 if label == "auto" else label),
            seed=seed + i,
            **kwargs,
        )
        for i in range(n)
    ]


def make_loader(graphs, batch_size: int = 4, shuffle: bool = False):
    from torch_geometric.loader import DataLoader

    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


class MockAutoencoder(torch.nn.Module):
    """Minimal autoencoder stand-in: per-graph score is the batch position."""

    def forward(self, data):
        scores = torch.arange(data.num_graphs, dtype=torch.float)
        return {
            "per_graph_loss": scores,
            "loss": scores.mean(),
        }
