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
        chunk_size: int = 50000,
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
                        
                        # Handle resuming/skipping
                        if jets_to_skip >= n_jets:
                            jets_to_skip -= n_jets
                            continue
                            
                        # If we partially skip within this chunk
                        start_offset = 0
                        if jets_to_skip > 0:
                            start_offset = jets_to_skip
                            jets_to_skip = 0

                        is_qcd = ak.to_numpy(arrays["label_QCD"]).astype(bool)
                        binary_labels = (~is_qcd).astype(np.int64)

                        for i in range(start_offset, n_jets):
                            node_feats = []
                            for feat_name in PARTICLE_FEATURES:
                                col = ak.to_numpy(arrays[feat_name][i]).astype(np.float32)
                                node_feats.append(col)
                            
                            node_feats = np.stack(node_feats, axis=-1)
                            mask = np.any(node_feats != 0, axis=-1)
                            node_feats = node_feats[mask]
                            
                            n_particles = len(node_feats)
                            if n_particles < 2:
                                continue
                                
                            if n_particles > self.max_particles:
                                node_feats = node_feats[:self.max_particles]
                                n_particles = self.max_particles
                                
                            x = torch.tensor(node_feats, dtype=torch.float)
                            eta_phi = x[:, 4:6].contiguous()
                            k = min(self.k_neighbors, n_particles - 1)
                            edge_index = self._build_knn_graph(eta_phi, k)
                            y = torch.tensor([binary_labels[i]], dtype=torch.long)
                            
                            yield Data(x=x, edge_index=edge_index, y=y)
                            
            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")
                continue
