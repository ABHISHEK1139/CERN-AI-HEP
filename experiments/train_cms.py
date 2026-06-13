import argparse
import logging
from pathlib import Path

import torch
import torch_geometric
import numpy as np
from graph_builder.cms_dataset import CMSDataset
from anomaly_engine.models.gcn import GCNEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.trainer import Trainer
from anomaly_engine.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Graph Autoencoder on CMS Open Data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample-bg", type=int, default=None, help="Sample size for TTbar background")
    parser.add_argument("--sample-sig", type=int, default=None, help="Sample size for Higgs signal")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading CMS TTbar Background Dataset...")
    bg_dataset = CMSDataset(
        root="data/cms/graphs",
        root_file_path="data/cms/ttbar/TTbar.root",
        label=0,
        sample_size=args.sample_bg
    )
    
    logger.info("Loading CMS Higgs Signal Dataset...")
    sig_dataset = CMSDataset(
        root="data/cms/graphs",
        root_file_path="data/cms/higgs/GluGluToHToTauTau.root",
        label=1,
        sample_size=args.sample_sig
    )

    # Split background
    train_loader, val_loader, bg_test_loader = bg_dataset.get_loaders(
        batch_size=args.batch_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Combine background test and signal test
    logger.info("Constructing mixed test set...")
    # The get_splits method already returned loaders, but we can just use the datasets directly to merge
    bg_train_ds, bg_val_ds, bg_test_ds = bg_dataset.get_splits(0.8, 0.1, 0.1)
    
    # Concat datasets
    mixed_test_dataset = torch.utils.data.ConcatDataset([bg_test_ds, sig_dataset])
    test_loader = torch_geometric.loader.DataLoader(
        mixed_test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Initialize model
    # CMS features: pt, eta, phi, mass
    input_dim = 4
    hidden_dim = 32
    latent_dim = 16

    encoder = GCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder)

    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir="checkpoints/cms_autoencoder",
    )
    
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        run_name="cms_gcn",
        resume=args.resume
    )

    # Evaluate
    logger.info("Evaluating Anomaly Detection Performance on CMS mixed test set...")
    evaluator = Evaluator(device=device)
    results = evaluator.evaluate_autoencoder(model, test_loader)
    
    logger.info("--- Test Results ---")
    logger.info(f"AUROC: {results.get('auroc', 0.0):.4f}")
    if 'auprc' in results:
        logger.info(f"AUPRC: {results['auprc']:.4f}")
    if 'score_separation' in results:
        logger.info(f"Separation (Anomaly - Normal): {results['score_separation']:.4f}")
        
    # Save ROC plot
    logger.info("Saving ROC Curve...")
    Path("results").mkdir(exist_ok=True)
    
    # Re-run to get scores for plotting
    model.eval()
    model.to(device)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            res = model(data)
            all_scores.extend(res['per_graph_loss'].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy().flatten())
            
    evaluator.plot_roc_curve(np.array(all_labels), np.array(all_scores), "GCN Autoencoder (CMS)", "results/cms_roc.png")
    evaluator.plot_score_distributions(np.array(all_scores), np.array(all_labels), "results/cms_scores.png")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
