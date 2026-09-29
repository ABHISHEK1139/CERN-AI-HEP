import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.gcn import GCNEncoder
from graph_builder.jetclass_dataset import JetClassDataset


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

def train_and_eval(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # Train 1 epoch
    print("  Training 1 epoch...")
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if isinstance(model, MLPAutoencoder):
            loss = model(data)['per_graph_loss'].mean()
        else:
            loss = model(data)['loss']
        loss.backward()
        optimizer.step()

    model.eval()
    print("  Evaluating...")
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            res = model(data)
            all_scores.extend(res['per_graph_loss'].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy().flatten())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if len(np.unique(labels)) < 2:
        return 0.5
    # Raw AUROC, no post-hoc flip: flipping on test scores inflates the metric.
    raw_auroc = roc_auc_score(labels, scores)
    if raw_auroc < 0.5:
        print(f"  Note: raw AUROC {raw_auroc:.4f} < 0.5 (inverse scoring).")
    return raw_auroc

def get_model(arch, input_dim=16, hidden_dim=64, latent_dim=32):
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
    raise ValueError(f"Unknown arch '{arch}'. Available: mlp, gcn, edgeconv.")

def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from glob import glob
    train_files = glob("data/jetclass/val_5M/ZJetsToNuNu_*.root")[:1]
    val_files = glob("data/jetclass/val_5M/HTo*.root")[:1]

    if not train_files or not val_files:
        raise FileNotFoundError(
            "JetClass ROOT files not found under data/jetclass/val_5M/ "
            f"(found {len(train_files)} background, {len(val_files)} signal). "
            "Download JetClass before running the ablation (see README)."
        )

    results = {}

    # Ablation settings
    experiments = [
        {"name": "MLP (Baseline)", "arch": "mlp", "k": 8},
        {"name": "GCN", "arch": "gcn", "k": 8},
        {"name": "EdgeConv (k=4)", "arch": "edgeconv", "k": 4},
        {"name": "EdgeConv (k=8)", "arch": "edgeconv", "k": 8},
        {"name": "EdgeConv (k=16)", "arch": "edgeconv", "k": 16},
    ]

    for exp in experiments:
        name = exp["name"]
        k = exp["k"]
        arch = exp["arch"]

        print(f"\n--- Running {name} ---")

        # Load data with specific k.
        # Train/val background come from one disjointly-split pool so that
        # validation jets never leak into training (same-file resampling
        # with a fixed seed would otherwise overlap).
        print(f"  Loading dataset with k={k}...")
        pool = JetClassDataset(root=f"data/jetclass/graphs_k{k}", root_file_paths=train_files, k_neighbors=k, sample_size=12000, tag="pool_bg")
        n_pool = len(pool)
        if n_pool < 2:
            print(f"  Skipping {name}: only {n_pool} jets available.")
            continue
        n_train = min(10000, int(0.8 * n_pool))
        n_val_bg = min(2000, n_pool - n_train)
        perm = np.random.RandomState(42).permutation(n_pool)
        train_dataset = pool.index_select(perm[:n_train].tolist())
        val_dataset_bg = pool.index_select(perm[n_train:n_train + n_val_bg].tolist())
        val_dataset_sig = JetClassDataset(root=f"data/jetclass/graphs_k{k}", root_file_paths=val_files, k_neighbors=k, sample_size=2000, tag="val_sig")

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        mixed_val = torch.utils.data.ConcatDataset([val_dataset_bg, val_dataset_sig])
        val_loader = DataLoader(mixed_val, batch_size=256, shuffle=False)

        model = get_model(arch).to(device)

        auroc = train_and_eval(model, train_loader, val_loader, device)
        results[name] = auroc
        print(f"  {name} AUROC: {auroc:.4f}")

    print("\n--- Final Ablation Results ---")
    for name, auroc in results.items():
        print(f"{name}: {auroc:.4f}")

if __name__ == "__main__":
    run_ablation()
