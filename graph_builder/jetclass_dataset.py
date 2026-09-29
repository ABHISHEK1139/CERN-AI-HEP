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

import hashlib
import logging

import awkward as ak
import numpy as np
import torch
import uproot
from torch_geometric.data import Data, InMemoryDataset

from graph_builder.splitting import SplittableDatasetMixin

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


class JetClassDataset(SplittableDatasetMixin, InMemoryDataset):
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
        root_file_paths: list[str],
        k_neighbors: int = 8,
        max_particles: int = 128,
        transform=None,
        pre_transform=None,
        sample_size: int | None = None,
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

        Raises:
            ValueError: If ``root_file_paths`` is empty. Processing would
                otherwise fail deep inside awkward with an opaque error.
        """
        if not root_file_paths:
            raise ValueError(
                "JetClassDataset needs at least one ROOT file. Found none in "
                "the requested paths; download JetClass first (see README)."
            )
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}.")
        if max_particles < 2:
            raise ValueError(
                f"max_particles must be >= 2 to form kNN edges, got {max_particles}."
            )
        self.root_file_paths = list(root_file_paths)
        self.k_neighbors = k_neighbors
        self.max_particles = max_particles
        self.sample_size = sample_size
        self.tag = tag

        # Include k/max_particles AND a hash of the source file list in the cache
        # key: reusing a k=8 cache for a k=16 ablation (or a different file set
        # under the same tag) silently invalidates the experiment.
        files_key = hashlib.md5(
            "|".join(sorted(str(f) for f in root_file_paths)).encode()
        ).hexdigest()[:8]
        self._processed_file_name = (
            f"jetclass_{tag}_{sample_size if sample_size else 'all'}"
            f"_k{k_neighbors}_nmax{max_particles}_{files_key}.pt"
        )

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], map_location="cpu", weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self._processed_file_name]

    @staticmethod
    def _build_knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build a k-NN graph from positional coordinates using pure PyTorch (CPU).

        Per-jet graphs are tiny; staying on CPU avoids per-jet H2D ping-pong.

        Args:
            pos: Node positions [N, D].
            k: Number of nearest neighbors.

        Returns:
            edge_index: [2, N*k] tensor of directed edges (source, target) on CPU.
        """
        n = pos.size(0)
        k = min(k, n - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long)

        # Pairwise L2 distances [N, N]
        dist = torch.cdist(pos, pos, p=2)
        # Set self-distance to infinity to exclude self-loops
        dist.fill_diagonal_(float('inf'))
        # Get k nearest neighbors for each node
        _, indices = dist.topk(k, largest=False, dim=-1)  # [N, k]

        # Build edge_index: source repeats, target from indices
        source = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
        target = indices.reshape(-1)

        return torch.stack([source, target], dim=0).cpu()

    def process(self):
        logger.info(f"Processing JetClass data from {len(self.root_file_paths)} file(s)...")
        # NOTE: whole files are read into RAM here. For 100M-scale training use
        # JetClassIterableDataset (chunked streaming) or preprocess_6m.py instead.

        all_features = []  # List of awkward arrays per file
        all_labels = []

        for fpath in self.root_file_paths:
            logger.info(f"  Reading {fpath}...")
            with uproot.open(fpath) as file:

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
        if n_total == 0:
            raise ValueError("JetClass processing found 0 jets in the input files.")

        # Sample if requested (isolated RNG: do not pollute global np.random state)
        if self.sample_size is not None:
            if self.sample_size < 1:
                raise ValueError(f"sample_size must be >= 1, got {self.sample_size}.")
            if self.sample_size < n_total:
                rng = np.random.RandomState(42)
                indices = rng.choice(n_total, self.sample_size, replace=False)
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
            # Sanitize: NaN/inf poisons MSE loss; zero-padding check must ignore them
            node_feats = np.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)

            # Remove zero-padded particles (all features = 0)
            mask = np.any(node_feats != 0, axis=-1)
            node_feats = node_feats[mask]

            n_particles = len(node_feats)
            if n_particles < 2:
                continue  # Skip jets with fewer than 2 particles

            # Trim to max_particles, keeping highest-pT constituents
            # (pT = hypot(px, py) from cols 0,1) instead of arbitrary first-N.
            if n_particles > self.max_particles:
                pt = np.hypot(node_feats[:, 0], node_feats[:, 1])
                keep = np.argsort(pt)[::-1][:self.max_particles]
                node_feats = node_feats[np.sort(keep)]
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
