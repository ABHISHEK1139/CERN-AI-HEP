"""
PyTorch Geometric dataset for JetClass (Particle Cloud).

Unlike LHCO/CMS where nodes=jets in an event,
JetClass treats each jet as a separate graph where nodes=particles.

Features per particle (16):
    - part_px, part_py, part_pz, part_energy (4D kinematics)
    - part_deta, part_dphi (relative to jet axis)
    - part_d0val, part_d0err, part_dzval, part_dzerr (impact parameters)
    - part_charge (track charge)
    - part_isChargedHadron, part_isNeutralHadron, part_isPhoton,
      part_isElectron, part_isMuon (particle ID flags)

Edges: k-Nearest Neighbors in (deta, dphi) space.

Labels: 10-class jet origin (QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl).
For anomaly detection we treat QCD as background (label=0) and everything else as signal (label=1).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import uproot
import awkward as ak
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# Particle-level feature branches in JetClass ROOT files
PARTICLE_FEATURES = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi",
    "part_d0val", "part_d0err", "part_dzval", "part_dzerr",
    "part_charge",
    "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
    "part_isElectron", "part_isMuon",
]

# Label branches (one-hot encoded in JetClass)
LABEL_BRANCHES = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]


class JetClassDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for JetClass particle clouds.

    Each graph represents a single jet:
        - Nodes = constituent particles
        - Node features = 16-dim kinematic + ID vector
        - Edges = k-NN in (deta, dphi) space
        - Label = 0 (QCD background) or 1 (any signal)
    """

    def __init__(
        self,
        root: str,
        root_file_paths: List[str],
        k_neighbors: int = 8,
        max_particles: int = 128,
        transform=None,
        pre_transform=None,
        sample_size: Optional[int] = None,
        tag: str = "default",
    ):
        """
        Args:
            root: Directory where processed graphs are stored.
            root_file_paths: List of JetClass .root file paths.
            k_neighbors: Number of nearest neighbors for kNN graph.
            max_particles: Maximum particles per jet (zero-padded jets trimmed).
            sample_size: Number of jets to sample (None for all).
            tag: A tag string to differentiate processed file names.
        """
        self.root_file_paths = root_file_paths
        self.k_neighbors = k_neighbors
        self.max_particles = max_particles
        self.sample_size = sample_size
        self.tag = tag

        self._processed_file_name = f"jetclass_{tag}_{sample_size if sample_size else 'all'}.pt"

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self._processed_file_name]

    @staticmethod
    def _build_knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build a k-NN graph from positional coordinates using pure PyTorch.
        Uses GPU acceleration if available.

        Args:
            pos: Node positions [N, D].
            k: Number of nearest neighbors.

        Returns:
            edge_index: [2, N*k] tensor of directed edges (source, target) on CPU.
        """
        # Move to GPU for fast pairwise distance computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_dev = pos.to(device)

        # Pairwise L2 distances [N, N]
        dist = torch.cdist(pos_dev, pos_dev, p=2)
        # Set self-distance to infinity to exclude self-loops
        dist.fill_diagonal_(float('inf'))
        # Get k nearest neighbors for each node
        _, indices = dist.topk(k, largest=False, dim=-1)  # [N, k]

        n = pos.size(0)
        # Build edge_index: source repeats, target from indices
        source = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        target = indices.reshape(-1)

        edge_index = torch.stack([source, target], dim=0).cpu()
        return edge_index

    def process(self):
        logger.info(f"Processing JetClass data from {len(self.root_file_paths)} file(s)...")

        all_features = []  # List of awkward arrays per file
        all_labels = []

        for fpath in self.root_file_paths:
            logger.info(f"  Reading {fpath}...")
            file = uproot.open(fpath)

            # JetClass uses "tree" as tree name
            tree_key = next((k for k in file.keys() if "tree" in k.lower()), None)
            if not tree_key:
                raise ValueError(f"Could not find 'tree' in {fpath}")

            tree = file[tree_key]

            # Read particle features (ragged arrays: variable particles per jet)
            feat_arrays = tree.arrays(PARTICLE_FEATURES)
            label_arrays = tree.arrays(LABEL_BRANCHES)

            all_features.append(feat_arrays)
            all_labels.append(label_arrays)

        # Concatenate across files
        features = ak.concatenate(all_features)
        labels = ak.concatenate(all_labels)

        n_total = len(features)
        logger.info(f"Total jets across all files: {n_total}")

        # Sample if requested
        if self.sample_size is not None and self.sample_size < n_total:
            np.random.seed(42)
            indices = np.random.choice(n_total, self.sample_size, replace=False)
            features = features[indices]
            labels = labels[indices]
            logger.info(f"Sampled {self.sample_size} jets.")

        # Determine binary labels: QCD=0, everything else=1
        is_qcd = ak.to_numpy(labels["label_QCD"]).astype(bool)
        binary_labels = (~is_qcd).astype(np.int64)

        logger.info(f"Label distribution: QCD(bg)={is_qcd.sum()}, Signal={binary_labels.sum()}")

        # Build PyG graphs
        data_list = []
        n_jets = len(features)

        logger.info(f"Constructing {n_jets} particle-cloud graphs (k={self.k_neighbors})...")

        for i in range(n_jets):
            if i % 10000 == 0 and i > 0:
                logger.info(f"  Processed {i}/{n_jets} jets...")

            # Extract particle features for this jet
            node_feats = []
            for feat_name in PARTICLE_FEATURES:
                col = ak.to_numpy(features[feat_name][i]).astype(np.float32)
                node_feats.append(col)

            # Stack into [n_particles, 16]
            node_feats = np.stack(node_feats, axis=-1)

            # Remove zero-padded particles (all features = 0)
            mask = np.any(node_feats != 0, axis=-1)
            node_feats = node_feats[mask]

            n_particles = len(node_feats)
            if n_particles < 2:
                continue  # Skip jets with fewer than 2 particles

            # Trim to max_particles
            if n_particles > self.max_particles:
                node_feats = node_feats[:self.max_particles]
                n_particles = self.max_particles

            x = torch.tensor(node_feats, dtype=torch.float)

            # Build k-NN graph in (deta, dphi) space (columns 4 and 5)
            eta_phi = x[:, 4:6].contiguous()
            k = min(self.k_neighbors, n_particles - 1)
            edge_index = self._build_knn_graph(eta_phi, k)

            y = torch.tensor([binary_labels[i]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved {len(data_list)} jet graphs to {self.processed_paths[0]}")

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["JetClassDataset", "JetClassDataset", "JetClassDataset"]:
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

    def index_select(self, indices: List[int]) -> "JetClassDataset":
        graphs = [self.get(i) for i in indices]
        subset = JetClassDataset.__new__(JetClassDataset)
        subset.transform = self.transform
        subset.pre_transform = self.pre_transform
        subset._indices = None
        subset.data, subset.slices = self.collate(graphs)
        subset._data_list = None
        return subset

    def get_loaders(
        self,
        batch_size: int = 256,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds, val_ds, test_ds = self.get_splits(
            train_ratio, val_ratio, test_ratio, seed
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        logger.info(f"JetClass DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return train_loader, val_loader, test_loader
