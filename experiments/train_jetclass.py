"""
Train Graph Autoencoder on JetClass particle clouds.

Strategy:
    - Background: QCD jets (Standard Model, label=0)
    - Signal: Higgs/W/Z/Top jets (BSM-like, label=1)
    - Train autoencoder on QCD background only
    - Evaluate anomaly detection on mixed QCD + signal test set

Usage:
    python experiments/train_jetclass.py --epochs 50 --sample 5000
    python experiments/train_jetclass.py --epochs 50  # full dataset
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch_geometric.loader import DataLoader

from graph_builder.jetclass_dataset import JetClassDataset
from graph_builder.jetclass_iterable import JetClassIterableDataset
from anomaly_engine.models.gcn import GCNEncoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.trainer import Trainer
from anomaly_engine.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Graph Autoencoder on JetClass")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (jets)")
    parser.add_argument("--k", type=int, default=8, help="k for kNN graph")
    parser.add_argument("--arch", type=str, default="gcn", choices=["gcn", "edgeconv"], help="Encoder architecture")
    parser.add_argument("--large", action="store_true", help="Use large-scale IterableDataset")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save interval (batches) for large runs")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ---- Data Loading ----
    if args.large:
        # Large-scale iterable dataset
        bg_files = sorted(Path("data/jetclass").glob("ZJetsToNuNu_*.root"))
        sig_files = sorted(Path("data/jetclass").glob("HTo*.root"))
        
        logger.info(f"Large-scale mode: Found {len(bg_files)} background files and {len(sig_files)} signal files.")
        
        # In large mode, we don't have validation splits out-of-the-box in the iterable dataset.
        # We will just train on the background files, and use the val set for mixed evaluation.
        # However, to keep it simple, we'll just train on the iterable background dataset.
        train_ds = JetClassIterableDataset(
            root_file_paths=[str(f) for f in bg_files],
            k_neighbors=args.k,
            batch_size=args.batch_size,
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size,
            pin_memory=True
        )
        
        # We reuse the original JetClassDataset for validation/testing (small scale)
        val_bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
        val_sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))
        
        logger.info("Loading JetClass Validation Sets for Evaluation...")
        val_dataset = JetClassDataset(
            root="data/jetclass/graphs",
            root_file_paths=[str(f) for f in val_bg_files],
            k_neighbors=args.k,
            sample_size=2000,
            tag="qcd_bg_val",
        )
        sig_dataset = JetClassDataset(
            root="data/jetclass/graphs",
            root_file_paths=[str(f) for f in val_sig_files],
            k_neighbors=args.k,
            sample_size=1000,
            tag="higgs_sig_val",
        )
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        mixed_test = torch.utils.data.ConcatDataset([val_dataset, sig_dataset])
        test_loader = DataLoader(mixed_test, batch_size=args.batch_size, shuffle=False)
        
    else:
        # Small-scale in-memory dataset
        bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
        sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))

        if not bg_files:
            raise FileNotFoundError("No ZJetsToNuNu background files found in data/jetclass/val_5M/")

        bg_sample = args.sample
        sig_sample = int(args.sample * 0.2) if args.sample else None

        bg_dataset = JetClassDataset(
            root="data/jetclass/graphs",
            root_file_paths=[str(f) for f in bg_files],
            k_neighbors=args.k,
            sample_size=bg_sample,
            tag="qcd_bg",
        )

        sig_dataset = JetClassDataset(
            root="data/jetclass/graphs",
            root_file_paths=[str(f) for f in sig_files],
            k_neighbors=args.k,
            sample_size=sig_sample,
            tag="higgs_sig",
        )

        bg_train_ds, bg_val_ds, bg_test_ds = bg_dataset.get_splits(0.8, 0.1, 0.1)

        train_loader = DataLoader(bg_train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(bg_val_ds, batch_size=args.batch_size, shuffle=False)

        mixed_test = torch.utils.data.ConcatDataset([bg_test_ds, sig_dataset])
        test_loader = DataLoader(mixed_test, batch_size=args.batch_size, shuffle=False)

        logger.info(f"Train: {len(bg_train_ds)}, Val: {len(bg_val_ds)}, "
                    f"Test BG: {len(bg_test_ds)}, Test Sig: {len(sig_dataset)}")

    # ---- Model ----
    # JetClass has 16 particle features
    input_dim = 16
    hidden_dim = 64
    latent_dim = 32

    if args.arch == "edgeconv":
        encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
        run_name = "jetclass_edgeconv"
    else:
        encoder = GCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
        run_name = "jetclass_gcn"

    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Train ----
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir="checkpoints/jetclass_autoencoder",
    )

    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        run_name=run_name,
        resume=args.resume,
        save_steps=args.save_steps if args.large else None,
    )

    # ---- Evaluate ----
    logger.info("Evaluating Anomaly Detection on JetClass mixed test set...")
    evaluator = Evaluator(device=device)
    results = evaluator.evaluate_autoencoder(model, test_loader)

    logger.info("--- Test Results ---")
    logger.info(f"AUROC: {results.get('auroc', 0.0):.4f}")
    if 'auprc' in results:
        logger.info(f"AUPRC: {results['auprc']:.4f}")
    if 'score_separation' in results:
        logger.info(f"Separation (Anomaly - Normal): {results['score_separation']:.4f}")

    # ---- Plots ----
    logger.info("Generating plots...")
    Path("results").mkdir(exist_ok=True)

    model.eval()
    model.to(device)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            res = model(data)
            all_scores.extend(res['per_graph_loss'].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy().flatten())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # In JetClass, QCD background (label 0) is often harder to reconstruct than Higgs signal (label 1).
    # If the AUROC is < 0.5, it means the model is separating them but the scoring convention is flipped.
    # We will flip the scores for the plots.
    from sklearn.metrics import roc_auc_score
    temp_auroc = roc_auc_score(labels, scores)
    if temp_auroc < 0.5:
        logger.info(f"Flipping scores (Original AUROC {temp_auroc:.4f} < 0.5)")
        scores = -scores
        # Re-evaluate with flipped scores
        evaluator.plot_roc_curve(labels, scores, f"{args.arch.upper()} Autoencoder (JetClass)", f"results/jetclass_{args.arch}_roc.png")
        evaluator.plot_score_distributions(scores, labels, f"results/jetclass_{args.arch}_scores.png")
        
        # Log the flipped AUROC
        flipped_auroc = roc_auc_score(labels, scores)
        logger.info(f"Corrected AUROC: {flipped_auroc:.4f}")
    else:
        evaluator.plot_roc_curve(labels, scores, f"{args.arch.upper()} Autoencoder (JetClass)", f"results/jetclass_{args.arch}_roc.png")
        evaluator.plot_score_distributions(scores, labels, f"results/jetclass_{args.arch}_scores.png")

    logger.info("Done!")


if __name__ == "__main__":
    main()
