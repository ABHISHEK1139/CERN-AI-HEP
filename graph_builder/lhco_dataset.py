"""
PyTorch Geometric dataset for the LHCO 2020 Anomaly Detection Dataset.

Reads the features.h5 file, where each event has 2 jets and a label.
Each jet has 7 features: px, py, pz, m, tau1, tau2, tau3.
Constructs a 2-node graph for each event.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from graph_builder.splitting import SplittableDatasetMixin

logger = logging.getLogger(__name__)

# Jet 1 / Jet 2 feature columns, in node-feature order.
LHCO_J1_COLS = ['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1']
LHCO_J2_COLS = ['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']
LHCO_INPUT_DIM = len(LHCO_J1_COLS)


class LHCODataset(SplittableDatasetMixin, InMemoryDataset):
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
        sample_size: int | None = None,
    ):
        """
        Args:
            root: Directory where processed graphs are stored.
            h5_path: Path to the LHCO features HDF5 file.
            sample_size: Optional stratified subsample size.

        Raises:
            ValueError: If ``sample_size`` is not a positive int.
        """
        if sample_size is not None and sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size}.")
        self.h5_path = h5_path
        self.sample_size = sample_size
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(
            self.processed_paths[0], map_location="cpu", weights_only=False
        )

    @property
    def raw_file_names(self):
        # We don't strictly use PyG's raw mechanism because it's a single file elsewhere.
        return []

    @property
    def processed_file_names(self):
        # Include source mtime/size so an updated H5 never reuses a stale cache.
        tag = f"_{self.sample_size}" if self.sample_size else ""
        try:
            st = Path(self.h5_path).stat()
            tag += f"_m{int(st.st_mtime)}_s{st.st_size}"
        except OSError:
            pass
        return [f"lhco_dataset{tag}.pt"]

    def process(self):
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(
                f"LHCO H5 not found: {self.h5_path}. "
                "Download events_anomalydetection_v2.features.h5 first (see README)."
            )
        logger.info("Loading data from %s...", self.h5_path)
        df = pd.read_hdf(self.h5_path)

        missing = [c for c in LHCO_J1_COLS + LHCO_J2_COLS + ['label'] if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.h5_path} is missing expected LHCO columns: {missing}. "
                f"Available: {list(df.columns)[:20]}..."
            )

        if self.sample_size is not None and self.sample_size < len(df):
            # Take a stratified sample to preserve signal/bg ratio
            df_bg = df[df['label'] == 0]
            df_sig = df[df['label'] == 1]
            frac_sig = len(df_sig) / len(df) if len(df) > 0 else 0.0
            n_sig = min(len(df_sig), int(self.sample_size * frac_sig))
            n_bg = min(len(df_bg), self.sample_size - n_sig)
            if n_bg + n_sig <= 0:
                raise ValueError(
                    f"sample_size={self.sample_size} too small to sample both classes "
                    f"(bg={len(df_bg)}, sig={len(df_sig)})."
                )
            df = pd.concat([
                df_bg.sample(n=n_bg, random_state=42) if n_bg > 0 else df_bg.iloc[0:0],
                df_sig.sample(n=n_sig, random_state=42) if n_sig > 0 else df_sig.iloc[0:0],
            ]).sample(frac=1.0, random_state=42)  # shuffle

        logger.info("Building graphs for %d events...", len(df))

        node_features_j1 = torch.tensor(df[LHCO_J1_COLS].values, dtype=torch.float)
        node_features_j2 = torch.tensor(df[LHCO_J2_COLS].values, dtype=torch.float)
        labels = torch.tensor(df['label'].values, dtype=torch.long)

        data_list = []
        # Fully connected 2-node graph: 0->1 and 1->0 (cloned per graph:
        # sharing one tensor aliases it across all Data objects).
        base_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        for i in range(len(df)):
            x = torch.stack([node_features_j1[i], node_features_j2[i]])
            if not torch.isfinite(x).all():
                continue
            y = labels[i].view(1)

            data = Data(x=x, edge_index=base_edge_index.clone(), y=y)
            data_list.append(data)

        if not data_list:
            raise ValueError("LHCO processing produced 0 graphs (all events filtered).")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info("LHCO graphs processing complete.")
