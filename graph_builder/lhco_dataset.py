import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class LHCODataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for the LHCO 2020 Anomaly Detection Dataset.

    Reads the features.h5 file, where each event has 2 jets and a label.
    Each jet has 7 features: px, py, pz, m, tau1, tau2, tau3.
    Constructs a 2-node graph for each event.
    """

    def __init__(
        self,
        root: str,
        h5_path: str = "data/lhco/events_anomalydetection_v2.features.h5",
        transform=None,
        pre_transform=None,
        sample_size: Optional[int] = None,
    ):
        self.h5_path = h5_path
        self.sample_size = sample_size
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # We don't strictly use PyG's raw mechanism because it's a single file elsewhere.
        return []

    @property
    def processed_file_names(self):
        return [f"lhco_dataset{'_' + str(self.sample_size) if self.sample_size else ''}.pt"]

    def process(self):
        logger.info(f"Loading data from {self.h5_path}...")
        df = pd.read_hdf(self.h5_path)

        if self.sample_size is not None and self.sample_size < len(df):
            # Take a stratified sample to preserve signal/bg ratio
            df_bg = df[df['label'] == 0]
            df_sig = df[df['label'] == 1]
            n_sig = int(self.sample_size * (len(df_sig) / len(df)))
            n_bg = self.sample_size - n_sig
            df = pd.concat([
                df_bg.sample(n=n_bg, random_state=42),
                df_sig.sample(n=n_sig, random_state=42)
            ]).sample(frac=1.0, random_state=42)  # shuffle

        logger.info(f"Building graphs for {len(df)} events...")
        
        # Columns: pxj1, pyj1, pzj1, mj1, tau1j1, tau2j1, tau3j1 (same for j2), label
        j1_cols = ['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1']
        j2_cols = ['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']
        
        node_features_j1 = torch.tensor(df[j1_cols].values, dtype=torch.float)
        node_features_j2 = torch.tensor(df[j2_cols].values, dtype=torch.float)
        labels = torch.tensor(df['label'].values, dtype=torch.long)

        data_list = []
        # Fully connected 2-node graph: 0->1 and 1->0
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        for i in range(len(df)):
            x = torch.stack([node_features_j1[i], node_features_j2[i]])
            y = labels[i].view(1)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info("LHCO graphs processing complete.")

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["LHCODataset", "LHCODataset", "LHCODataset"]:
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

    def index_select(self, indices: List[int]) -> "LHCODataset":
        graphs = [self.get(i) for i in indices]
        subset = LHCODataset.__new__(LHCODataset)
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

        logger.info(f"LHCO DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return train_loader, val_loader, test_loader
