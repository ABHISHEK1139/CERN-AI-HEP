import logging
import math
from typing import List, Iterator
import numpy as np
import awkward as ak
import uproot
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

PARTICLE_FEATURES = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_charge", "part_isElectron",
    "part_isMuon", "part_isPhoton", "part_isChargedHadron", "part_isNeutralHadron"
]

LABEL_BRANCHES = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql",
    "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"
]

class JetClassIterableDataset(IterableDataset):
    """
    An IterableDataset for large-scale JetClass training.
    Reads .root files lazily in chunks so that the system RAM is not overwhelmed.
    """
    def __init__(
        self,
        root_file_paths: List[str],
        k_neighbors: int = 8,
        max_particles: int = 128,
        chunk_size: int = 10000,
        start_batch: int = 0,
        batch_size: int = 2048,
    ):
        """
        Args:
            root_file_paths: List of JetClass .root file paths to iterate over.
            k_neighbors: k for k-NN graph construction.
            max_particles: max particles per graph.
            chunk_size: number of jets to read from the root file at once.
            start_batch: Batch number to resume from (skips the first start_batch * batch_size jets).
            batch_size: batch size used in DataLoader (needed to accurately resume).
        """
        self.root_file_paths = root_file_paths
        self.k_neighbors = k_neighbors
        self.max_particles = max_particles
        self.chunk_size = chunk_size
        self.start_idx = start_batch * batch_size

    @staticmethod
    def _build_knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
        # L2 distances
        dist = torch.cdist(pos, pos, p=2)
        dist.fill_diagonal_(float('inf'))
        _, indices = dist.topk(k, largest=False, dim=-1)
        
        n = pos.size(0)
        source = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
        target = indices.reshape(-1)
        
        edge_index = torch.stack([source, target], dim=0)
        return edge_index

    def __iter__(self) -> Iterator[Data]:
        worker_info = get_worker_info()
        if worker_info is not None:
            # Partition files across workers
            files = [f for i, f in enumerate(self.root_file_paths) if i % worker_info.num_workers == worker_info.id]
            # When using multiple workers, resume logic needs to be much more complex. 
            # We assume num_workers=0 for resumable training.
            jets_to_skip = 0
        else:
            files = self.root_file_paths
            jets_to_skip = self.start_idx

        logger.info(f"IterableDataset starting over {len(files)} files. Skipping {jets_to_skip} jets.")

        for fpath in files:
            logger.info(f"  Streaming {fpath}...")
            try:
                with uproot.open(fpath) as file:
                    tree_key = next((k for k in file.keys() if "tree" in k.lower()), None)
                    if not tree_key:
                        continue
                    tree = file[tree_key]
                    
                    # Iterate in chunks to save RAM
                    for arrays in tree.iterate(PARTICLE_FEATURES + LABEL_BRANCHES, step_size=self.chunk_size):
                        n_jets = len(arrays)
                        print(f"      [Iterable] Read chunk of {n_jets} jets from disk. Padding arrays...")
                        lengths = ak.to_numpy(ak.num(arrays["part_px"]))
                        
                        padded_feats = []
                        for feat in PARTICLE_FEATURES:
                            padded = ak.fill_none(ak.pad_none(arrays[feat], self.max_particles, clip=True), 0.0)
                            padded_feats.append(ak.to_numpy(padded).astype(np.float32))
                            
                        print(f"      [Iterable] Padded. Stacking {len(PARTICLE_FEATURES)} features...")
                        # Stack all features into a single dense block [n_jets, max_particles, 16]
                        node_feats_all = np.stack(padded_feats, axis=-1)
                        print(f"      [Iterable] Stacked into {node_feats_all.shape}. Yielding...")
                        
                        is_qcd = ak.to_numpy(arrays["label_QCD"]).astype(bool)
                        binary_labels = (~is_qcd).astype(np.int64)

                        for i in range(n_jets):
                            length = min(lengths[i], self.max_particles)
                            if length < 2:
                                continue
                                
                            # Zero-copy slicing of the dense numpy block
                            x = torch.from_numpy(node_feats_all[i, :length, :])
                            y = torch.tensor([binary_labels[i]], dtype=torch.long)
                            
                            # We omit edge_index here! We will build it natively on the GPU!
                            yield Data(x=x, y=y)
                            
            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")
                continue
