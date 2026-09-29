import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anomaly_engine.checkpoint import load_state_dict
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from graph_builder.jetclass_dataset import JetClassDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CKPT_PATH = "checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"

def compute_jet_kinematics(data):
    """
    Computes Jet pT, Mass, and Particle count for a single jet graph.

    Node features (JetClass 16-feature layout): px=0, py=1, pz=2, energy=3.

    IMPORTANT: in JetClass, node 0 *is* the jet itself and nodes 1..N-1 are
    its constituents. The previous implementation summed px/py/pz/energy over
    every node, which counts the jet's own four-momentum a second time on top
    of the sum of its constituents — inflating pT, energy and the reconstructed
    invariant mass. Constituents alone are the physically meaningful sum here.

    Returns:
        (pt, mass, n_constituents) as floats/ints.
    """
    x = data.x.detach().cpu().numpy()
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Expected a non-empty [N, F] node feature matrix, got {x.shape}.")
    if x.shape[1] < 4:
        raise ValueError(
            f"JetClass kinematics need at least 4 feature columns (px,py,pz,E); got {x.shape[1]}."
        )

    n_total = x.shape[0]
    # Drop the jet node itself when it is present.
    constituents = x[1:] if n_total > 1 else x

    px = float(np.sum(constituents[:, 0]))
    py = float(np.sum(constituents[:, 1]))
    pz = float(np.sum(constituents[:, 2]))
    e = float(np.sum(constituents[:, 3]))

    pt = math.hypot(px, py)
    mass_sq = e * e - px * px - py * py - pz * pz
    # Negative mass^2 is unphysical and comes from rounding in float32; clamp
    # instead of taking sqrt of a negative (which returns NaN).
    mass = math.sqrt(max(0.0, mass_sq))
    n_constituents = int(constituents.shape[0])

    return pt, mass, n_constituents

def run_physics_analysis():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    val_bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
    val_sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))
    if not val_bg_files or not val_sig_files:
        raise FileNotFoundError(
            f"JetClass validation ROOT files missing under data/jetclass/val_5M/ "
            f"(found {len(val_bg_files)} background, {len(val_sig_files)} signal)."
        )

    bg_dataset = JetClassDataset(root="data/jetclass/graphs", root_file_paths=[str(f) for f in val_bg_files], k_neighbors=8, sample_size=100, tag="ev_bg")
    sig_dataset = JetClassDataset(root="data/jetclass/graphs", root_file_paths=[str(f) for f in val_sig_files], k_neighbors=8, sample_size=100, tag="ev_sig")
    if len(bg_dataset) == 0 or len(sig_dataset) == 0:
        raise ValueError("JetClass datasets are empty; no physics analysis possible.")

    mixed_test = torch.utils.data.ConcatDataset([bg_dataset, sig_dataset])
    test_loader = DataLoader(mixed_test, batch_size=1, shuffle=False)

    # Load Model. Required: random weights would make every correlation below
    # an artefact of initialisation rather than a property of the model.
    input_dim, hidden_dim, latent_dim = 16, 64, 32
    encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder).to(device)

    load_state_dict(CKPT_PATH, model, device=device)
    logger.info("Loaded trained weights from %s", CKPT_PATH)

    results = []
    logger.info("Running inference to collect physics observables...")
    with torch.no_grad():
        for data in test_loader:
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

    # Exploratory quantile split (NOT a calibrated decision rule): bottom 50%
    # vs top 10% only visualizes score-observable correlations. For research
    # decisions use AnomalyScorer.select_threshold on validation scores.
    normal = results[:int(len(results)*0.5)]
    anomalous = results[int(len(results)*0.9):]
    if not normal or not anomalous:
        raise ValueError(
            f"Too few events to split into score quantiles (n={len(results)})."
        )

    # Plotting
    _, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Particle Count
    axs[0].hist([x['n_particles'] for x in normal], bins=20, alpha=0.5, label='Normal (Low Score)', density=True, color='blue')
    axs[0].hist([x['n_particles'] for x in anomalous], bins=20, alpha=0.5, label='Anomalous (High Score)', density=True, color='red')
    axs[0].set_xlabel('Particle Multiplicity')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Jet Particle Multiplicity')
    axs[0].legend()

    # Plot 2: Mass (range clips overflow; overflow counts are logged below)
    mass_all = [x['mass'] for x in results]
    mass_lo, mass_hi = 0, 300
    logger.info(f"Mass overflow beyond [{mass_lo}, {mass_hi}]: "
                f"{sum(m < mass_lo or m > mass_hi for m in mass_all)}/{len(mass_all)}")
    axs[1].hist([x['mass'] for x in normal], bins=30, range=(mass_lo, mass_hi), alpha=0.5, label='Normal (Low Score)', density=True, color='blue')
    axs[1].hist([x['mass'] for x in anomalous], bins=30, range=(mass_lo, mass_hi), alpha=0.5, label='Anomalous (High Score)', density=True, color='red')
    axs[1].set_xlabel('Jet Mass (GeV)')
    axs[1].set_title('Jet Mass Distribution')
    axs[1].legend()

    # Plot 3: pT (range clips overflow; overflow counts are logged below)
    pt_all = [x['pt'] for x in results]
    pt_lo, pt_hi = 200, 1000
    logger.info(f"pT overflow beyond [{pt_lo}, {pt_hi}]: "
                f"{sum(p < pt_lo or p > pt_hi for p in pt_all)}/{len(pt_all)}")
    axs[2].hist([x['pt'] for x in normal], bins=30, range=(pt_lo, pt_hi), alpha=0.5, label='Normal (Low Score)', density=True, color='blue')
    axs[2].hist([x['pt'] for x in anomalous], bins=30, range=(pt_lo, pt_hi), alpha=0.5, label='Anomalous (High Score)', density=True, color='red')
    axs[2].set_xlabel('Jet pT (GeV)')
    axs[2].set_title('Jet Transverse Momentum (pT)')
    axs[2].legend()

    plt.tight_layout()
    plot_path = out_dir / "physics_analysis.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Saved {plot_path}")

if __name__ == "__main__":
    # Non-zero on failure so CI and shell pipelines notice. The previous bare
    # `except: logger.error(...)` exited 0 even when nothing had been analysed.
    try:
        run_physics_analysis()
    except Exception as exc:
        logger.error("Physics analysis failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc
