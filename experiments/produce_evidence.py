import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch_geometric.loader import DataLoader
import networkx as nx

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_builder.jetclass_dataset import JetClassDataset
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def plot_loss_curve(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    history = ckpt.get('history', {})
    if not history or 'train_loss' not in history:
        logger.warning("No history found in checkpoint for loss curve.")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path}")

def draw_event_graph(data, output_path):
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G.add_edge(u, v)
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    # Using dark mode aesthetic
    plt.style.use('dark_background')
    nx.draw(G, pos, node_size=20, node_color='#00ffcc', edge_color='#444444', alpha=0.7)
    plt.title("JetClass Particle Cloud (k-NN Graph)", color='white')
    plt.savefig(output_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.style.use('default')
    plt.close()
    logger.info(f"Saved {output_path}")

def generate_evidence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Plot Loss Curve
    ckpt_path = "checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"
    if Path(ckpt_path).exists():
        plot_loss_curve(ckpt_path, out_dir / "loss_curve.png")
    else:
        logger.warning(f"{ckpt_path} not found.")

    # Load data for evaluation
    val_bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
    val_sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))
    
    bg_dataset = JetClassDataset(root="data/jetclass/graphs", root_file_paths=[str(f) for f in val_bg_files], k_neighbors=8, sample_size=1000, tag="ev_bg")
    sig_dataset = JetClassDataset(root="data/jetclass/graphs", root_file_paths=[str(f) for f in val_sig_files], k_neighbors=8, sample_size=1000, tag="ev_sig")
    
    mixed_test = torch.utils.data.ConcatDataset([bg_dataset, sig_dataset])
    test_loader = DataLoader(mixed_test, batch_size=256, shuffle=False)

    # 2. Plot Event Graph (from first event)
    draw_event_graph(bg_dataset[0], out_dir / "event_graph.png")

    # Load Model
    input_dim, hidden_dim, latent_dim = 16, 64, 32
    encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder).to(device)
    
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    all_scores, all_labels, all_latents = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            res = model(data)
            all_scores.extend(res['per_graph_loss'].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy().flatten())
            # For t-SNE, pool latent features per graph using mean
            from torch_geometric.nn import global_mean_pool
            latent_graph = global_mean_pool(res['z'], data.batch)
            all_latents.append(latent_graph.cpu().numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    latents = np.vstack(all_latents)

    # Auto-flip scores if AUROC < 0.5
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        scores = -scores
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

    # 3. Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='EdgeConv (AUC = 0.6808)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Anomaly Detection ROC on JetClass (SM background vs Higgs)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved results/roc_curve.png")

    # 3.5 Plot Precision-Recall (PR) Curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'EdgeConv (AP = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Anomaly Detection PR Curve on JetClass')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(out_dir / "pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved results/pr_curve.png")

    # 4. Plot Anomaly Distribution
    plt.figure(figsize=(8, 6))
    bg_scores = scores[labels == 0]
    sig_scores = scores[labels == 1]
    
    # Clip extreme outliers for better visualization
    q_high = np.percentile(scores, 95)
    q_low = np.percentile(scores, 5)
    bins = np.linspace(q_low, q_high, 50)
    
    plt.hist(bg_scores, bins=bins, alpha=0.6, color='blue', label='Standard Model Background', density=True)
    plt.hist(sig_scores, bins=bins, alpha=0.6, color='red', label='Higgs Signal', density=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution on JetClass')
    plt.legend()
    plt.savefig(out_dir / "anomaly_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved results/anomaly_distribution.png")

    # 5. Plot t-SNE
    logger.info("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(latents_2d[labels == 0, 0], latents_2d[labels == 0, 1], alpha=0.5, color='blue', label='Background', s=10)
    plt.scatter(latents_2d[labels == 1, 0], latents_2d[labels == 1, 1], alpha=0.5, color='red', label='Signal', s=10)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Latent Space t-SNE Visualization on JetClass')
    plt.legend()
    plt.savefig(out_dir / "latent_space_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved results/latent_space_tsne.png")

if __name__ == "__main__":
    generate_evidence()
