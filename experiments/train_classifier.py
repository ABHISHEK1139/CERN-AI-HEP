"""
Train a graph classifier (signal vs background).

Usage:
    python experiments/train_classifier.py --model gcn --epochs 100
    python experiments/train_classifier.py --model graphsage --data data/graphs/
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_ingestion.synthetic import SyntheticEventGenerator
from graph_builder.graph_constructor import EventGraphConstructor
from graph_builder.dataset import CollisionEventDataset
from anomaly_engine.models import get_classifier
from anomaly_engine.trainer import Trainer
from anomaly_engine.evaluate import Evaluator


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config: dict, data_dir: str = None):
    """Prepare dataset: generate synthetic if needed, build graphs, create loaders."""
    import json

    graph_dir = Path(data_dir) if data_dir else Path(config["data"]["graphs"]["output"])
    graphs_file = graph_dir / "graphs.pt"
    fingerprint_file = graph_dir / "synthetic_fingerprint.json"

    syn_config = config["data"]["synthetic"]
    graph_config = config["data"]["graphs"]
    fingerprint = {
        "n_normal": syn_config["n_normal"],
        "n_anomaly": syn_config["n_anomaly"],
        "seed": syn_config["seed"],
        "strategy": graph_config["strategy"],
        "k": graph_config["k"],
    }
    try:
        cached = json.loads(fingerprint_file.read_text())
    except (OSError, ValueError):
        cached = None
    if cached != fingerprint and graphs_file.exists():
        logging.info("Synthetic/graph config changed; regenerating graphs...")
        graphs_file.unlink()

    if graphs_file.exists():
        logging.info(f"Loading existing graphs from {graphs_file}")
        graphs = torch.load(graphs_file, map_location="cpu", weights_only=False)
    else:
        logging.info("No graphs found. Generating synthetic data...")

        # Generate synthetic events
        gen = SyntheticEventGenerator(seed=syn_config["seed"])
        events, labels = gen.generate(
            n_normal=syn_config["n_normal"],
            n_anomaly=syn_config["n_anomaly"],
        )

        # Build graphs
        constructor = EventGraphConstructor(
            strategy=graph_config["strategy"],
            k=graph_config["k"],
            delta_r_threshold=graph_config.get("delta_r", 1.5),
        )
        graphs = constructor.convert_dataset(events, labels)

        # Save
        graph_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, graphs_file)
        fingerprint_file.write_text(json.dumps(fingerprint, indent=2))
        logging.info(f"Saved {len(graphs)} graphs to {graphs_file}")

    # Create dataset and loaders
    dataset = CollisionEventDataset(root=str(graph_dir), graphs=graphs)
    split_config = config["splits"]
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=config["training"]["batch_size"],
        train_ratio=split_config["train"],
        val_ratio=split_config["val"],
        test_ratio=split_config["test"],
        seed=split_config["seed"],
    )

    return train_loader, val_loader, test_loader, dataset


def main():
    parser = argparse.ArgumentParser(description="Train graph classifier")
    parser.add_argument("--model", type=str, default="gcn",
                       choices=["gcn", "graphsage", "gat", "mlp", "cnn"],
                       help="Model architecture")
    parser.add_argument("--data", type=str, default=None, help="Graph data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config with CLI overrides
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Prepare data
    train_loader, val_loader, test_loader, dataset = prepare_data(config, args.data)
    stats = dataset.get_stats()
    logging.info(f"Dataset: {stats['n_graphs']} graphs, "
                 f"node_dim={stats['node_feature_dim']}, "
                 f"labels: {stats['label_distribution']}")

    # Create model
    model_config = config["model"]
    model = get_classifier(
        args.model,
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        num_classes=model_config["num_classes"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {args.model} ({n_params:,} parameters)")

    # Train
    train_config = config["training"]
    trainer = Trainer(
        model,
        device=args.device,
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        patience=train_config["patience"],
        max_grad_norm=train_config["max_grad_norm"],
        checkpoint_dir=config["output"]["checkpoints"],
        use_mlflow=config["mlflow"]["enabled"],
    )

    history = trainer.train_classifier(
        train_loader, val_loader,
        epochs=train_config["epochs"],
        run_name=f"classifier_{args.model}",
        resume=args.resume,
    )

    # Evaluate
    evaluator = Evaluator(device=args.device)
    results = evaluator.evaluate_classifier(model, test_loader)

    print(f"\n{'='*50}")
    print(f"RESULTS: {args.model.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    if "auroc" in results:
        print(f"  AUROC:     {results['auroc']:.4f}")
    print(f"  Params:    {n_params:,}")
    print(f"{'='*50}")

    # Save plots
    figures_dir = Path(config["output"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_training_curves(
        history,
        title=f"{args.model.upper()} Classifier",
        output_path=str(figures_dir / f"training_{args.model}.png"),
    )


if __name__ == "__main__":
    main()
