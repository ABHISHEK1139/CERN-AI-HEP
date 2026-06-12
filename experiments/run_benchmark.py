"""
Run full benchmark across all model architectures.

Compares: MLP, CNN, GCN, GraphSAGE, GAT, PhysicsNeMo
Produces: comparison table, figures, JSON results

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --data data/graphs/ --epochs 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from event_ingestion.synthetic import SyntheticEventGenerator
from graph_builder.graph_constructor import EventGraphConstructor
from graph_builder.dataset import CollisionEventDataset
from anomaly_engine.evaluate import Evaluator
from physicsnemo_integration.benchmark import PhysicsNeMoBenchmark


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config, data_dir=None):
    """Prepare dataset for benchmarking."""
    graph_dir = Path(data_dir) if data_dir else Path(config["data"]["graphs"]["output"])
    graphs_file = graph_dir / "graphs.pt"

    if graphs_file.exists():
        graphs = torch.load(graphs_file)
    else:
        logging.info("Generating synthetic data for benchmark...")
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

    dataset = CollisionEventDataset(root=str(graph_dir), graphs=graphs)
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=config["training"]["batch_size"],
    )

    return train_loader, val_loader, test_loader, dataset


def main():
    parser = argparse.ArgumentParser(description="Run full model benchmark")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to benchmark (default: all)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    if args.epochs:
        config["benchmark"]["epochs"] = args.epochs

    # Prepare data
    train_loader, val_loader, test_loader, dataset = prepare_data(config, args.data)
    stats = dataset.get_stats()
    logging.info(f"Dataset: {stats}")

    # Run benchmark
    model_config = config["model"]
    benchmark = PhysicsNeMoBenchmark(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        device=args.device,
    )

    models_to_test = args.models or config["benchmark"]["models"]
    results = benchmark.run_benchmark(
        train_loader, val_loader, test_loader,
        models_to_test=models_to_test,
        epochs=config["benchmark"]["epochs"],
    )

    # Print comparison
    benchmark.print_comparison()

    # Save results
    benchmark.save_results(args.output)

    # Generate comparison plot
    evaluator = Evaluator(device=args.device)
    figures_dir = Path(config["output"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Filter out errored results
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        evaluator.plot_comparison_table(
            valid_results,
            output_path=str(figures_dir / "model_comparison.png"),
        )

    logging.info(f"\nBenchmark complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
