import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import math

from graph_builder.jetclass_dataset import JetClassDataset
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_jet_kinematics(data):
    """
    Computes Jet pT, Mass, and Particle count for a single jet graph.
    Node features: px=0, py=1, pz=2, energy=3
    """
    x = data.x.numpy()
    px = np.sum(x[:, 0])
    py = np.sum(x[:, 1])
    pz = np.sum(x[:, 2])
    e = np.sum(x[:, 3])
    
    pt = math.sqrt(px**2 + py**2)
    mass_sq = e**2 - px**2 - py**2 - pz**2
    mass = math.sqrt(max(0, mass_sq))
    n_particles = x.shape[0]
    
    return pt, mass, n_particles

def run_physics_analysis():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    ckpt_path = "checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"
    
    # Load data
    val_bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
    val_sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))
    
    try:
        bg_dataset = JetClassDataset(root="data/jetclass/graphs", root_file_paths=[str(f) for f in val_bg_files], k_neighbors=8, sample_size=100, tag="ev_bg")
        sig_dataset = JetClassDataset(root="data/jetclass/graphs", root_file_paths=[str(f) for f in val_sig_files], k_neighbors=8, sample_size=100, tag="ev_sig")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    mixed_test = torch.utils.data.ConcatDataset([bg_dataset, sig_dataset])
    test_loader = DataLoader(mixed_test, batch_size=1, shuffle=False)
    
    # Load Model
    input_dim, hidden_dim, latent_dim = 16, 64, 32
    encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder).to(device)
    
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    results = []
    logger.info("Running inference to collect physics observables...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_dev = data.to(device)
            res = model(data_dev)
            score = res['per_graph_loss'].item()
            
            # Compute physics
            pt, mass, n_particles = compute_jet_kinematics(data)
            
            results.append({
                'score': score,
                'pt': pt,
                'mass': mass,
                'n_particles': n_particles,
                'label': data.y.item()
            })

    # Sort by anomaly score
    results.sort(key=lambda x: x['score'])
    
    # Define "Normal" (bottom 50%) and "Anomalous" (top 10%)
    normal = results[:int(len(results)*0.5)]
    anomalous = results[int(len(results)*0.9):]
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Particle Count
    axs[0].hist([x['n_particles'] for x in normal], bins=20, alpha=0.5, label='Normal (Low Score)', density=True, color='blue')
    axs[0].hist([x['n_particles'] for x in anomalous], bins=20, alpha=0.5, label='Anomalous (High Score)', density=True, color='red')
    axs[0].set_xlabel('Particle Multiplicity')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Jet Particle Multiplicity')
    axs[0].legend()
    
    # Plot 2: Mass
    axs[1].hist([x['mass'] for x in normal], bins=30, range=(0, 300), alpha=0.5, label='Normal (Low Score)', density=True, color='blue')
    axs[1].hist([x['mass'] for x in anomalous], bins=30, range=(0, 300), alpha=0.5, label='Anomalous (High Score)', density=True, color='red')
    axs[1].set_xlabel('Jet Mass (GeV)')
    axs[1].set_title('Jet Mass Distribution')
    axs[1].legend()
    
    # Plot 3: pT
    axs[2].hist([x['pt'] for x in normal], bins=30, range=(200, 1000), alpha=0.5, label='Normal (Low Score)', density=True, color='blue')
    axs[2].hist([x['pt'] for x in anomalous], bins=30, range=(200, 1000), alpha=0.5, label='Anomalous (High Score)', density=True, color='red')
    axs[2].set_xlabel('Jet pT (GeV)')
    axs[2].set_title('Jet Transverse Momentum (pT)')
    axs[2].legend()
    
    plt.tight_layout()
    plot_path = out_dir / "physics_analysis.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved {plot_path}")

if __name__ == "__main__":
    try:
        run_physics_analysis()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
