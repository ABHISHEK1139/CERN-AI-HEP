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
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class CollisionEventDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for collision event graphs.

    Expects pre-built graphs from EventGraphConstructor.
    """

    def __init__(
        self,
        root: str,
        graphs: Optional[List[Data]] = None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory for dataset.
            graphs: Optional pre-built graph list. If None, loads from disk.
            transform: PyG transform to apply at access time.
            pre_transform: PyG transform to apply at processing time.
        """
        self._graphs = graphs
        super().__init__(root, transform, pre_transform)

        if Path(self.processed_paths[0]).exists():
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location="cpu", weights_only=False
            )
        elif graphs is not None:
            if len(graphs) == 0:
                raise ValueError("CollisionEventDataset received an empty graphs list.")
            self.data, self.slices = self.collate(graphs)
            # Save for future use
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            torch.save((self._data, self.slices), self.processed_paths[0])
        else:
            # Try loading from graphs.pt
            graphs_path = Path(root) / "graphs.pt"
            if graphs_path.exists():
                graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)
                if len(graphs) == 0:
                    raise ValueError(f"{graphs_path} contains 0 graphs.")
                self.data, self.slices = self.collate(graphs)
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

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["CollisionEventDataset", "CollisionEventDataset", "CollisionEventDataset"]:
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        n = len(self)
        indices = np.random.RandomState(seed).permutation(n)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return (
            self.index_select(train_idx.tolist()),
            self.index_select(val_idx.tolist()),
            self.index_select(test_idx.tolist()),
        )

    def index_select(self, indices: List[int]) -> "CollisionEventDataset":
        """Select subset by indices, returning a new dataset."""
        graphs = [self.get(i) for i in indices]
        if not graphs:
            raise ValueError("index_select received empty indices.")
        subset = CollisionEventDataset.__new__(CollisionEventDataset)
        # Preserve InMemoryDataset internals so collate()/get() keep working.
        subset.transform = self.transform
        subset.pre_transform = self.pre_transform
        subset._indices = None
        # InMemoryDataset.collate is an instance method in older PyG but a
        # staticmethod in newer releases; support both.
        try:
            subset.data, subset.slices = self.collate(graphs)
        except TypeError:
            subset.data, subset.slices = CollisionEventDataset.collate(graphs)
        subset._data_list = None
        return subset

    def get_loaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train/val/test DataLoaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_ds, val_ds, test_ds = self.get_splits(
            train_ratio, val_ratio, test_ratio, seed
        )
        for _name, _ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            if len(_ds) == 0:
                logger.warning(
                    f"Empty {_name} split from {len(self)} graphs "
                    f"(ratios {train_ratio}/{val_ratio}/{test_ratio})."
                )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        logger.info(
            f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
            f"batch_size={batch_size}"
        )

        return train_loader, val_loader, test_loader

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
