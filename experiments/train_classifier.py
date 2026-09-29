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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anomaly_engine.evaluate import Evaluator
from experiments.data_pipeline import build_classifier, ensure_graph_dataset, make_trainer


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} is not a mapping (got {type(config).__name__}).")
    for section in ("data", "model", "training", "splits", "output"):
        if section not in config:
            raise ValueError(f"Config {path} is missing the required '{section}' section.")
    return config


def prepare_data(config: dict, data_dir: str | None = None):
    """Prepare dataset: generate synthetic if needed, build graphs, create loaders."""

    return ensure_graph_dataset(config, data_dir)


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
    model = build_classifier(args.model, config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model: %s (%s parameters)", args.model, f"{n_params:,}")

    # Train
    trainer = make_trainer(
        model, config, args.device, config["output"]["checkpoints"]
    )

    history = trainer.train_classifier(
        train_loader, val_loader,
        epochs=config["training"]["epochs"],
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
