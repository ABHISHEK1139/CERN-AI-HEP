"""
PyTorch Geometric dataset for collision event graphs.

Provides a proper PyG InMemoryDataset with train/val/test splits
and DataLoader integration.

Usage:
    dataset = CollisionEventDataset(root="data/graphs/")
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from graph_builder.splitting import SplittableDatasetMixin

logger = logging.getLogger(__name__)


class CollisionEventDataset(SplittableDatasetMixin, InMemoryDataset):
    """
    PyTorch Geometric dataset for collision event graphs.

    Expects pre-built graphs from EventGraphConstructor.
    """

    def __init__(
        self,
        root: str,
        graphs: list[Data] | None = None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory for dataset.
            graphs: Optional pre-built graph list. If None, loads from disk.
            transform: PyG transform to apply at access time.
            pre_transform: PyG transform to apply at processing time.

        Note:
            An explicitly supplied ``graphs`` list is authoritative and always
            wins over any previously cached ``processed/dataset.pt``. The cache
            is only consulted when ``graphs is None``. Reading the cache first
            used to make a freshly regenerated graph set a silent no-op.
        """
        self._graphs = graphs
        super().__init__(root, transform, pre_transform)

        if graphs is not None:
            if len(graphs) == 0:
                raise ValueError("CollisionEventDataset received an empty graphs list.")
            self.data, self.slices = self.collate(list(graphs))
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            torch.save((self._data, self.slices), self.processed_paths[0])
        elif Path(self.processed_paths[0]).exists():
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location="cpu", weights_only=False
            )
        else:
            # Try loading from graphs.pt
            graphs_path = Path(root) / "graphs.pt"
            if graphs_path.exists():
                loaded = torch.load(graphs_path, map_location="cpu", weights_only=False)
                if len(loaded) == 0:
                    raise ValueError(f"{graphs_path} contains 0 graphs.")
                self.data, self.slices = self.collate(loaded)
                Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
                torch.save((self._data, self.slices), self.processed_paths[0])
            else:
                raise FileNotFoundError(
                    "No graphs found. Run graph_constructor first, or provide graphs list."
                )

    @property
    def raw_file_names(self):
        return ["graphs.pt"]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        pass  # Processing handled in __init__

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        n = len(self)
        if n == 0:
            return {"n_graphs": 0}

        sample = self.get(0)
        node_dim = sample.x.shape[1] if getattr(sample, "x", None) is not None else 0
        has_edge_attr = getattr(sample, "edge_attr", None) is not None
        edge_dim = sample.edge_attr.shape[1] if has_edge_attr else 0

        # Collect stats
        num_nodes = []
        num_edges = []
        labels = []

        for i in range(n):
            g = self.get(i)
            num_nodes.append(g.num_nodes)
            ei = getattr(g, "edge_index", None)
            num_edges.append(ei.shape[1] if ei is not None else 0)
            if getattr(g, "y", None) is not None:
                try:
                    labels.append(int(g.y.flatten()[0].item()))
                except (IndexError, ValueError, AttributeError):
                    pass

        labels = np.array(labels) if labels else np.array([])

        return {
            "n_graphs": n,
            "node_feature_dim": node_dim,
            "edge_feature_dim": edge_dim,
            "has_edge_features": has_edge_attr,
            "nodes_per_graph": {
                "mean": float(np.mean(num_nodes)),
                "std": float(np.std(num_nodes)),
                "min": int(np.min(num_nodes)),
                "max": int(np.max(num_nodes)),
            },
            "edges_per_graph": {
                "mean": float(np.mean(num_edges)),
                "std": float(np.std(num_edges)),
            },
            "label_distribution": {
                "normal": int((labels == 0).sum()) if len(labels) > 0 else 0,
                "anomaly": int((labels == 1).sum()) if len(labels) > 0 else 0,
            },
        }
