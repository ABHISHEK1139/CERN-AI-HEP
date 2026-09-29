"""
Train Graph Autoencoder on JetClass particle clouds.

Strategy (QCD-vs-non-QCD ranking benchmark, not BSM discovery):
    - Background proxy: QCD jets (Standard Model, label=0)
    - Non-QCD proxy: Higgs/W/Z/Top jets (other known SM classes, label=1)
    - Train autoencoder on QCD background only
    - Evaluate anomaly-ranking on mixed QCD + non-QCD test set

Usage:
    python experiments/train_jetclass.py --epochs 50 --sample 5000
    python experiments/train_jetclass.py --epochs 50  # full dataset
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from anomaly_engine.evaluate import Evaluator
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.gcn import GCNEncoder
from anomaly_engine.trainer import Trainer
from graph_builder.jetclass_dataset import JetClassDataset
from graph_builder.jetclass_iterable import JetClassIterableDataset

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

        if not bg_files:
            raise FileNotFoundError(
                "Large mode requires data/jetclass/ZJetsToNuNu_*.root. "
                "Download JetClass data first (see README / scripts/)."
            )

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
        if not val_bg_files or not val_sig_files:
            raise FileNotFoundError(
                "Large mode also needs held-out evaluation data under "
                f"data/jetclass/val_5M/ (found {len(val_bg_files)} background, "
                f"{len(val_sig_files)} signal files). This path was previously "
                "unguarded and failed deep inside uproot."
            )

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

        # Split the QCD validation pool into disjoint val / test-background halves
        # so validation graphs never leak into the reported test set.
        n_val = len(val_dataset)
        if n_val < 2:
            raise ValueError(f"Need >= 2 validation jets, got {n_val}.")
        half = n_val // 2
        bg_val_ds = val_dataset.index_select(list(range(half)))
        bg_test_ds = val_dataset.index_select(list(range(half, n_val)))
        val_loader = DataLoader(bg_val_ds, batch_size=args.batch_size, shuffle=False)
        mixed_test = torch.utils.data.ConcatDataset([bg_test_ds, sig_dataset])
        test_loader = DataLoader(mixed_test, batch_size=args.batch_size, shuffle=False)

    else:
        # Small-scale in-memory dataset
        bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
        sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))

        if not bg_files:
            raise FileNotFoundError("No ZJetsToNuNu background files found in data/jetclass/val_5M/")
        if not sig_files:
            raise FileNotFoundError(
                "No HTo* signal files found in data/jetclass/val_5M/. An "
                "anomaly-detection evaluation needs a signal class; without "
                "one the AUROC is undefined."
            )

        bg_sample = args.sample
        sig_sample = max(1, int(args.sample * 0.2)) if args.sample else None

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
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # The trainer returns the *full* per-epoch curve (the best-checkpoint reload
    # no longer truncates it), so the convergence figure is complete.
    evaluator.plot_training_curves(
        history,
        title=f"{args.arch.upper()} Autoencoder (JetClass)",
        output_path=str(results_dir / f"jetclass_{args.arch}_loss.png"),
    )

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

    # Report the RAW test AUROC first: post-hoc flipping on test scores would
    # inflate the metric (the flip rule must be fixed a priori to be valid).
    # Plots below may use direction-corrected scores, clearly labeled as such.
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        logger.warning("Single-class test set; skipping AUROC computation.")
        raw_auroc = 0.5
    else:
        raw_auroc = roc_auc_score(labels, scores)
    logger.info(f"Raw test AUROC (no flip): {raw_auroc:.4f}")
    if raw_auroc < 0.5:
        logger.warning(
            "Raw AUROC < 0.5: the model separates the classes but scores them "
            "inversely (background reconstructs worse than signal). Plots use "
            "direction-corrected scores; headline number stays the raw value."
        )
        plot_scores = -scores
        plot_suffix = " (direction-corrected)"
    else:
        plot_scores = scores
        plot_suffix = ""
    evaluator.plot_roc_curve(labels, plot_scores, f"{args.arch.upper()} Autoencoder (JetClass){plot_suffix}", f"results/jetclass_{args.arch}_roc.png")
    evaluator.plot_score_distributions(plot_scores, labels, f"results/jetclass_{args.arch}_scores.png")

    logger.info("Done!")


if __name__ == "__main__":
    main()
