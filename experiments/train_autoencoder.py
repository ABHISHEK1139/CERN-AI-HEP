import argparse
import logging
from pathlib import Path

import torch
import torch_geometric
from graph_builder.lhco_dataset import LHCODataset
from anomaly_engine.models.gcn import GCNEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.trainer import Trainer
from anomaly_engine.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Graph Autoencoder on LHCO Data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for fast debugging")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading LHCO Dataset...")
    dataset = LHCODataset(root="data/lhco/graphs", sample_size=args.sample)
    
    # We want to train on purely BACKGROUND (label=0) for proper unsupervised anomaly detection.
    # We will split background into train and val.
    # Signal (label=1) will be used exclusively for testing the ROC AUC.
    bg_idx = (dataset.data.y == 0).nonzero(as_tuple=True)[0].tolist()
    sig_idx = (dataset.data.y == 1).nonzero(as_tuple=True)[0].tolist()
    
    logger.info(f"Found {len(bg_idx)} background events and {len(sig_idx)} signal events.")
    
    bg_dataset = dataset.index_select(bg_idx)
    
    # Split bg into train/val
    train_loader, val_loader, bg_test_loader = bg_dataset.get_loaders(
        batch_size=args.batch_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Combine bg_test and sig_test for final evaluation
    import numpy as np
    mixed_idx = np.concatenate([bg_idx[int(0.9 * len(bg_idx)):], sig_idx])
    mixed_test_dataset = dataset.index_select(mixed_idx.tolist())
    test_loader = torch_geometric.loader.DataLoader(mixed_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    # LHCO jets have 7 features: px, py, pz, m, tau1, tau2, tau3
    input_dim = 7
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
        checkpoint_dir="checkpoints/lhco_autoencoder",
    )
    
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        run_name="lhco_gcn",
        resume=args.resume
    )

    # Evaluate
    logger.info("Evaluating Anomaly Detection Performance on mixed test set...")
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
            
    evaluator.plot_roc_curve(np.array(all_labels), np.array(all_scores), "GCN Autoencoder", "results/lhco_roc.png")
    evaluator.plot_score_distributions(np.array(all_scores), np.array(all_labels), "results/lhco_scores.png")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
