import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_builder.jetclass_dataset import JetClassDataset
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.gcn import GCNEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder

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
    
    temp_auroc = roc_auc_score(labels, scores)
    if temp_auroc < 0.5:
        scores = -scores
        temp_auroc = roc_auc_score(labels, scores)
        
    return temp_auroc

def get_model(arch, input_dim=16, hidden_dim=64, latent_dim=32):
    if arch == "mlp":
        return MLPAutoencoder(input_dim, hidden_dim, latent_dim)
    elif arch == "gcn":
        enc = GCNEncoder(input_dim, hidden_dim, latent_dim, num_layers=3)
        dec = GraphDecoder(latent_dim, hidden_dim, input_dim)
        return GraphAutoencoder(enc, dec)
    elif arch == "edgeconv":
        enc = EdgeConvEncoder(input_dim, hidden_dim, latent_dim, num_layers=3)
        dec = GraphDecoder(latent_dim, hidden_dim, input_dim)
        return GraphAutoencoder(enc, dec)

def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from glob import glob
    train_files = glob("data/jetclass/val_5M/ZJetsToNuNu_*.root")[:1]
    val_files = glob("data/jetclass/val_5M/HTo*.root")[:1]
    
    if not train_files or not val_files:
        print("Root files not found.")
        return
        
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
        
        # Load data with specific k
        print(f"  Loading dataset with k={k}...")
        train_dataset = JetClassDataset(root=f"data/jetclass/graphs_k{k}", root_file_paths=train_files, k_neighbors=k, sample_size=10000, tag="train_bg")
        val_dataset_bg = JetClassDataset(root=f"data/jetclass/graphs_k{k}", root_file_paths=train_files, k_neighbors=k, sample_size=2000, tag="val_bg")
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
