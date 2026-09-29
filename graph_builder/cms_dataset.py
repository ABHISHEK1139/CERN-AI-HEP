"""
PyTorch Geometric dataset for CMS Open Data (NanoAOD).

Reads standard ROOT files using uproot and builds a 2-jet graph per event with
node features [pt, eta, phi, mass].
"""

import logging
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from torch_geometric.data import Data, InMemoryDataset

from graph_builder.splitting import SplittableDatasetMixin

logger = logging.getLogger(__name__)


class CMSDataset(SplittableDatasetMixin, InMemoryDataset):
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
        sample_size: int | None = None,
    ):
        """
        Args:
            root: Directory where processed graphs are stored.
            root_file_path: Path to the CMS NanoAOD .root file.
            label: Label to assign to all events in this file (e.g. 0 for background, 1 for anomaly).
            sample_size: Number of events to sample (None for all).

        Raises:
            ValueError: If ``label`` is not 0 or 1, or ``sample_size`` < 1.
        """
        if label not in (0, 1):
            raise ValueError(
                f"label must be 0 (background) or 1 (anomaly), got {label!r}."
            )
        if sample_size is not None and sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size}.")

        self.root_file_path = root_file_path
        self.label = label
        self.sample_size = sample_size

        # We define a custom processed file name based on the root file name and sample size.
        # Include the parent directory and source mtime/size to avoid collisions
        # on identical stems and stale reuse after file updates.
        file_path = Path(self.root_file_path)
        file_base = f"{file_path.parent.name}_{file_path.stem}"
        src_tag = ""
        try:
            st = file_path.stat()
            src_tag = f"_m{int(st.st_mtime)}_s{st.st_size}"
        except OSError:
            pass
        self._processed_file_name = f"cms_{file_base}_label{label}_{sample_size if sample_size else 'all'}{src_tag}.pt"

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

    def process(self):
        if not Path(self.root_file_path).exists():
            raise FileNotFoundError(
                f"CMS ROOT not found: {self.root_file_path}. "
                "Download CMS Open Data first (see README / scripts/)."
            )
        logger.info(f"Loading CMS Data from {self.root_file_path}...")

        # Open ROOT file and select the "Events" tree
        with uproot.open(self.root_file_path) as file:

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
        if len(pt) == 0:
            raise ValueError(
                f"No events with >= 2 jets in {self.root_file_path}."
            )

        if self.sample_size is not None and self.sample_size < len(pt):
                # Since awkward arrays don't have a direct random sample, we do it via numpy indices
                # (isolated RNG: do not pollute global np.random state)
                rng = np.random.RandomState(42)
                indices = rng.choice(len(pt), self.sample_size, replace=False)
                pt = pt[indices]
                eta = eta[indices]
                phi = phi[indices]
                mass = mass[indices]
                logger.info(f"Sampled {self.sample_size} events.")

        data_list = []
        # Fully connected 2-node graph: 0->1 and 1->0 (cloned per graph).
        base_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        logger.info("Constructing PyG graphs...")
        # Iterating over the awkward arrays
        # Extract the first two jets for each event. Use ak.to_list to handle
        # ragged/variable-length and missing values robustly.
        pt_lists = ak.to_list(pt[:, :2])
        eta_lists = ak.to_list(eta[:, :2])
        phi_lists = ak.to_list(phi[:, :2])
        mass_lists = ak.to_list(mass[:, :2])

        for i in range(len(pt_lists)):
            row_pt, row_eta, row_phi, row_mass = (
                pt_lists[i], eta_lists[i], phi_lists[i], mass_lists[i]
            )
            if row_pt is None or len(row_pt) < 2:
                continue
            try:
                vals = [
                    [float(row_pt[0]), float(row_eta[0]), float(row_phi[0]), float(row_mass[0])],
                    [float(row_pt[1]), float(row_eta[1]), float(row_phi[1]), float(row_mass[1])],
                ]
            except (TypeError, ValueError):
                continue
            # Create node feature matrix [2, 4]
            x = torch.tensor(vals, dtype=torch.float)
            if not torch.isfinite(x).all():
                continue

            y = torch.tensor([self.label], dtype=torch.long)

            data = Data(x=x, edge_index=base_edge_index.clone(), y=y)
            data_list.append(data)

        if not data_list:
            raise ValueError(
                "CMS processing produced 0 graphs (no events with >= 2 valid jets)."
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved {len(data_list)} graphs to {self.processed_paths[0]}")
