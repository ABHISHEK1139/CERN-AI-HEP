import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.gcn import GCNEncoder
from graph_builder.jetclass_dataset import JetClassDataset


class BatchData:
    def __init__(self, x, batch, y, edge_index=None):
        self.x = x
        self.batch = batch
        self.y = y
        self.edge_index = edge_index

class FastChunkedDataset:
    """Streams pre-processed dense chunks [n_jets, max_particles, 16] as batches.

    ``max_particles`` is read from each chunk's own tensor shape rather than
    hard-coded, because the previous literal ``128`` silently produced wrong
    (or crashing) masking for any chunk pre-processed with a different cap.
    """

    def __init__(self, chunk_files, batch_size=2048, device=None):
        self.chunk_files = list(chunk_files)
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __iter__(self):
        for fpath in self.chunk_files:
            chunk = torch.load(fpath, map_location=self.device, weights_only=True)
            x_all = chunk['x'].to(self.device)
            lengths = chunk['lengths'].to(self.device)
            y_all = chunk['y'].to(self.device)

            if x_all.dim() != 3:
                raise ValueError(
                    f"{fpath}: expected x of shape [n_jets, max_particles, features], "
                    f"got {tuple(x_all.shape)}."
                )
            max_particles = x_all.size(1)

            num_jets = x_all.size(0)
            for start_idx in range(0, num_jets, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_jets)
                B = end_idx - start_idx

                x_batch = x_all[start_idx:end_idx]
                lengths_batch = torch.clamp(lengths[start_idx:end_idx], max=max_particles)
                y_batch = y_all[start_idx:end_idx]

                # Vectorised collation on the device
                mask = (
                    torch.arange(max_particles, device=self.device).unsqueeze(0)
                    < lengths_batch.unsqueeze(1)
                )
                x_collated = x_batch[mask]
                batch_idx_tensor = (
                    torch.arange(B, device=self.device)
                    .unsqueeze(1)
                    .expand(B, max_particles)[mask]
                )

                yield BatchData(x=x_collated, batch=batch_idx_tensor, y=y_batch)

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, data):
        x = data.x
        z = self.encoder(x)
        out = self.decoder(z)
        from torch_geometric.utils import scatter
        loss = nn.MSELoss(reduction='none')(out, x).mean(dim=1)
        graph_loss = scatter(loss, data.batch, reduce='mean')
        return {'per_graph_loss': graph_loss}

def build_knn_graph_gpu(x, batch, k=8):
    from torch_geometric.utils import to_dense_batch
    dense_pos, mask = to_dense_batch(x[:, 4:6], batch) # extract eta, phi
    B, N_max, _ = dense_pos.shape

    dist = torch.cdist(dense_pos, dense_pos)
    # Mask padded rows (invalid sources) AND padded columns (invalid targets).
    dist.masked_fill_(~mask.unsqueeze(1), float('inf'))
    dist.masked_fill_(~mask.unsqueeze(2), float('inf'))
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))

    actual_k = min(k, N_max - 1)
    if actual_k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    _, topk_idx = dist.topk(actual_k, dim=-1, largest=False)

    total_nodes = x.size(0)
    global_idx = torch.arange(total_nodes, device=x.device)
    dense_global_idx, _ = to_dense_batch(global_idx, batch, fill_value=-1)

    valid = mask.unsqueeze(-1).expand(B, N_max, actual_k)
    source_global = dense_global_idx.unsqueeze(-1).expand(B, N_max, actual_k)[valid]

    target_global = torch.gather(dense_global_idx.unsqueeze(1).expand(B, N_max, N_max), 2, topk_idx)[valid]

    dist_topk = torch.gather(dist, 2, topk_idx)[valid]
    valid_edges = (target_global != -1) & (source_global != -1) & torch.isfinite(dist_topk)
    return torch.stack([source_global[valid_edges], target_global[valid_edges]], dim=0)

def get_model(arch, input_dim=16, hidden_dim=64, latent_dim=32):
    """Instantiate an ablation model by architecture name.

    The previous version had no terminal ``else``, so an unknown ``arch``
    returned ``None`` and the caller crashed much later with an opaque
    ``AttributeError: 'NoneType' object has no attribute 'to'``.

    Raises:
        ValueError: If ``arch`` is not recognised.
    """
    if arch == "mlp":
        return MLPAutoencoder(input_dim, hidden_dim, latent_dim)
    if arch == "gcn":
        enc = GCNEncoder(input_dim, hidden_dim, latent_dim, num_layers=3)
        dec = GraphDecoder(latent_dim, hidden_dim, input_dim)
        return GraphAutoencoder(enc, dec)
    if arch == "edgeconv":
        enc = EdgeConvEncoder(input_dim, hidden_dim, latent_dim, num_layers=3)
        dec = GraphDecoder(latent_dim, hidden_dim, input_dim)
        return GraphAutoencoder(enc, dec)
    raise ValueError(f"Unknown arch {arch!r}. Available: mlp, gcn, edgeconv.")

def train_and_eval(model, train_ds, val_loader, device, k=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    print("  Training 1 epoch on 6 million...")
    n_batches = 0
    for i, data in enumerate(train_ds):
        # Build graph natively on GPU
        if not isinstance(model, MLPAutoencoder):
            data.edge_index = build_knn_graph_gpu(data.x, data.batch, k=k)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                loss = self_loss(model, data)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = self_loss(model, data)
            loss.backward()
            optimizer.step()
        n_batches += 1

        if i % 100 == 0:
            print(f"    Batch {i}: Loss {loss.item():.4f}")

    if n_batches == 0:
        print("  Warning: 0 training batches (no chunks yielded data).")
        return None

    model.eval()
    print("  Evaluating...")
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            if not isinstance(model, MLPAutoencoder):
                data.edge_index = build_knn_graph_gpu(data.x, data.batch, k=k)

            res = model(data)
            all_scores.extend(res['per_graph_loss'].detach().cpu().numpy())
            all_labels.extend(data.y.cpu().numpy().flatten())

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    if len(labels) != len(scores):
        raise ValueError(
            f"Label/score length mismatch ({len(labels)} labels vs {len(scores)} scores)."
        )

    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        print("  Validation set has a single class; AUROC undefined.")
        return None
    # Raw AUROC, no post-hoc flip: flipping on test scores inflates the metric.
    raw_auroc = roc_auc_score(labels, scores)
    if raw_auroc < 0.5:
        print(f"  Note: raw AUROC {raw_auroc:.4f} < 0.5 (inverse scoring).")
    return raw_auroc


def self_loss(model, data):
    """Uniform loss accessor across the MLP and GNN autoencoders."""
    out = model(data)
    if isinstance(model, MLPAutoencoder):
        return out['per_graph_loss'].mean()
    return out['loss']

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    bg_files = sorted(glob("data/jetclass/ZJetsToNuNu_*.root"))
    if not bg_files:
        print("Note: no data/jetclass/ZJetsToNuNu_*.root found; training uses processed chunks.")

    # Validation data (use val_5M like before for quick eval)
    val_bg_files = sorted(glob("data/jetclass/val_5M/ZJetsToNuNu_*.root"))[:1]
    val_sig_files = sorted(glob("data/jetclass/val_5M/HTo*.root"))[:1]
    if not val_bg_files or not val_sig_files:
        raise FileNotFoundError(
            "Validation ROOT files missing under data/jetclass/val_5M/ "
            f"(found {len(val_bg_files)} background, {len(val_sig_files)} signal). "
            "Download JetClass before running the ablation."
        )

    val_dataset_bg = JetClassDataset(root="data/jetclass/graphs_val", root_file_paths=val_bg_files, k_neighbors=8, sample_size=2000, tag="val_bg")
    val_dataset_sig = JetClassDataset(root="data/jetclass/graphs_val", root_file_paths=val_sig_files, k_neighbors=8, sample_size=2000, tag="val_sig")
    if len(val_dataset_bg) == 0 or len(val_dataset_sig) == 0:
        raise ValueError("Validation datasets are empty after preprocessing.")

    mixed_val = torch.utils.data.ConcatDataset([val_dataset_bg, val_dataset_sig])
    val_loader = DataLoader(mixed_val, batch_size=512, shuffle=False)

    experiments = [
        {"name": "MLP (Baseline)", "arch": "mlp", "k": 8},
        {"name": "GCN", "arch": "gcn", "k": 8},
        {"name": "EdgeConv (k=8)", "arch": "edgeconv", "k": 8},
    ]

    results = {}
    for exp in experiments:
        name = exp["name"]
        arch = exp["arch"]
        k = exp["k"]

        print(f"\n--- Running {name} on 6M dataset ---")
        chunk_files = sorted(glob("data/jetclass/processed_chunks/chunk_*.pt"))
        if len(chunk_files) == 0:
            raise FileNotFoundError(
                "No processed chunks found under data/jetclass/processed_chunks/. "
                "Run experiments/preprocess_6m.py first."
            )

        train_ds = FastChunkedDataset(chunk_files, batch_size=2048, device=device)

        model = get_model(arch).to(device)
        auroc = train_and_eval(model, train_ds, val_loader, device, k=k)

        if auroc is None:
            print(f"  {name}: skipped (no usable training/validation data).")
            continue
        results[name] = auroc
        print(f"  {name} AUROC: {auroc:.4f}")

    print("\n--- Final 6M Ablation Results ---")
    if not results:
        print("(no results)")
    for name, auroc in results.items():
        print(f"{name}: {auroc:.4f}")

if __name__ == "__main__":
    main()
