"""
Train a graph autoencoder for anomaly detection.

The autoencoder trains on normal events only, learns to reconstruct them,
then flags events with high reconstruction error as anomalies.

Usage:
    python experiments/train_autoencoder.py --encoder gcn --epochs 100
    python experiments/train_autoencoder.py --encoder gat --data data/graphs/
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from event_ingestion.synthetic import SyntheticEventGenerator
from graph_builder.graph_constructor import EventGraphConstructor
from graph_builder.dataset import CollisionEventDataset
from anomaly_engine.models import get_autoencoder
from anomaly_engine.trainer import Trainer
from anomaly_engine.anomaly_scorer import AnomalyScorer
from anomaly_engine.evaluate import Evaluator


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config, data_dir=None):
    """Prepare data, returning separate loaders for normal-only training."""
    graph_dir = Path(data_dir) if data_dir else Path(config["data"]["graphs"]["output"])
    graphs_file = graph_dir / "graphs.pt"

    if graphs_file.exists():
        graphs = torch.load(graphs_file)
    else:
        logging.info("Generating synthetic data...")
        syn_config = config["data"]["synthetic"]
        gen = SyntheticEventGenerator(seed=syn_config["seed"])
        events, labels = gen.generate(
            n_normal=syn_config["n_normal"],
            n_anomaly=syn_config["n_anomaly"],
        )

        constructor = EventGraphConstructor(
            strategy=config["data"]["graphs"]["strategy"],
            k=config["data"]["graphs"]["k"],
        )
        graphs = constructor.convert_dataset(events, labels)

        graph_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, graphs_file)

    # Split into train/val/test
    n = len(graphs)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test_graphs = [graphs[i] for i in indices[n_train + n_val:]]

    # For autoencoder: train on NORMAL events only
    train_normal = [g for g in train_graphs if g.y.item() == 0]
    logging.info(
        f"Training on {len(train_normal)} normal events "
        f"(filtered {len(train_graphs) - len(train_normal)} anomalies)"
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_normal, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train graph autoencoder")
    parser.add_argument("--encoder", type=str, default="gcn",
                       choices=["gcn", "graphsage", "gat"],
                       help="Encoder architecture")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config, args.data)

    # Create autoencoder
    model_config = config["model"]
    model = get_autoencoder(
        args.encoder,
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        num_layers=model_config["num_layers"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Autoencoder: {args.encoder} encoder ({n_params:,} parameters)")

    # Train
    train_config = config["training"]
    trainer = Trainer(
        model,
        device=args.device,
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        patience=train_config["patience"],
    )

    history = trainer.train_autoencoder(
        train_loader, val_loader,
        epochs=train_config["epochs"],
        run_name=f"autoencoder_{args.encoder}",
    )

    # Anomaly scoring
    scorer = AnomalyScorer(model, device=args.device)
    scores, labels, event_ids = scorer.score_dataset(test_loader)

    # Generate report
    report = scorer.generate_report(scores, labels, event_ids, top_k=args.top_k)
    scorer.print_report(report)

    # Visualizations
    evaluator = Evaluator(device=args.device)
    figures_dir = Path(config["output"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_training_curves(
        history,
        title=f"Autoencoder ({args.encoder.upper()})",
        output_path=str(figures_dir / f"autoencoder_training_{args.encoder}.png"),
    )

    if len(labels) > 0 and len(np.unique(labels)) > 1:
        evaluator.plot_score_distributions(
            scores, labels,
            output_path=str(figures_dir / f"anomaly_scores_{args.encoder}.png"),
        )
        evaluator.plot_roc_curve(
            labels, scores,
            model_name=f"Autoencoder ({args.encoder.upper()})",
            output_path=str(figures_dir / f"roc_{args.encoder}.png"),
        )
        evaluator.plot_latent_space(
            model, test_loader,
            output_path=str(figures_dir / f"latent_space_{args.encoder}.png"),
        )


if __name__ == "__main__":
    main()
