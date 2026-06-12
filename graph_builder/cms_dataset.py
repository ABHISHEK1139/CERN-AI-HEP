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


class CMSDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for CMS Open Data (NanoAOD).
    
    Reads standard ROOT files using uproot.
    Builds graphs from the leading 2 Jets.
    Features: pt, eta, phi, mass.
    """

    def __init__(
        self,
        root: str,
        root_file_path: str,
        label: int,
        transform=None,
        pre_transform=None,
        sample_size: Optional[int] = None,
    ):
        """
        Args:
            root: Directory where processed graphs are stored.
            root_file_path: Path to the CMS NanoAOD .root file.
            label: Label to assign to all events in this file (e.g. 0 for background, 1 for anomaly).
            sample_size: Number of events to sample (None for all).
        """
        self.root_file_path = root_file_path
        self.label = label
        self.sample_size = sample_size
        
        # We define a custom processed file name based on the root file name and sample size
        file_base = Path(self.root_file_path).stem
        self._processed_file_name = f"cms_{file_base}_label{label}_{sample_size if sample_size else 'all'}.pt"
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self._processed_file_name]

    def process(self):
        logger.info(f"Loading CMS Data from {self.root_file_path}...")
        
        # Open ROOT file and select the "Events" tree
        file = uproot.open(self.root_file_path)
        
        # Finding the correct tree name (usually Events or Events;1)
        tree_key = next((k for k in file.keys() if "Events" in k), None)
        if not tree_key:
            raise ValueError(f"Could not find 'Events' tree in {self.root_file_path}")
            
        tree = file[tree_key]
        
        logger.info("Reading Jet branches into memory...")
        # Read only required branches to save memory
        branches = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"]
        arrays = tree.arrays(branches)
        
        # Convert to awkward arrays
        pt = arrays["Jet_pt"]
        eta = arrays["Jet_eta"]
        phi = arrays["Jet_phi"]
        mass = arrays["Jet_mass"]
        
        # Filter: We only want events that have at least 2 jets
        mask = ak.num(pt) >= 2
        pt = pt[mask]
        eta = eta[mask]
        phi = phi[mask]
        mass = mass[mask]
        
        logger.info(f"Filtered to {len(pt)} events with >= 2 jets.")
        
        if self.sample_size is not None and self.sample_size < len(pt):
            # Since awkward arrays don't have a direct random sample, we do it via numpy indices
            np.random.seed(42)
            indices = np.random.choice(len(pt), self.sample_size, replace=False)
            pt = pt[indices]
            eta = eta[indices]
            phi = phi[indices]
            mass = mass[indices]
            logger.info(f"Sampled {self.sample_size} events.")
            
        data_list = []
        # Fully connected 2-node graph: 0->1 and 1->0
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        logger.info("Constructing PyG graphs...")
        # Iterating over the awkward arrays
        # Extract the first two jets for each event
        pt_np = pt[:, :2].to_numpy()
        eta_np = eta[:, :2].to_numpy()
        phi_np = phi[:, :2].to_numpy()
        mass_np = mass[:, :2].to_numpy()
        
        for i in range(len(pt_np)):
            # Create node feature matrix [2, 4]
            x = torch.tensor([
                [pt_np[i, 0], eta_np[i, 0], phi_np[i, 0], mass_np[i, 0]],
                [pt_np[i, 1], eta_np[i, 1], phi_np[i, 1], mass_np[i, 1]]
            ], dtype=torch.float)
            
            y = torch.tensor([self.label], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved {len(data_list)} graphs to {self.processed_paths[0]}")

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["CMSDataset", "CMSDataset", "CMSDataset"]:
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

    def index_select(self, indices: List[int]) -> "CMSDataset":
        graphs = [self.get(i) for i in indices]
        subset = CMSDataset.__new__(CMSDataset)
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

        logger.info(f"CMS DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return train_loader, val_loader, test_loader
