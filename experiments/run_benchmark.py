"""
Run full benchmark across all model architectures.

Compares: MLP, CNN, GCN, GraphSAGE, GAT, PhysicsNeMo
Produces: comparison table, figures, JSON results

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --data data/graphs/ --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anomaly_engine.evaluate import Evaluator
from experiments.data_pipeline import ensure_graph_dataset
from physicsnemo_integration.benchmark import PhysicsNeMoBenchmark


def load_config(config_path: str | None = None):
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} is not a mapping (got {type(config).__name__}).")
    for section in ("data", "model", "training", "benchmark", "output"):
        if section not in config:
            raise ValueError(f"Config {path} is missing the required '{section}' section.")
    return config


def prepare_data(config, data_dir: str | None = None):
    """Prepare dataset for benchmarking (shared with train_classifier.py)."""
    return ensure_graph_dataset(config, data_dir)


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
    if args.epochs is not None:
        config["benchmark"]["epochs"] = args.epochs

    # Prepare data
    train_loader, val_loader, test_loader, dataset = prepare_data(config, args.data)
    logging.info("Dataset: %s", dataset.get_stats())

    # Run benchmark
    model_config = config["model"]
    benchmark = PhysicsNeMoBenchmark(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        dropout=model_config.get("dropout", 0.2),
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
    else:
        logging.error("Every model failed; skipping the comparison figure.")

    logging.info("Benchmark complete. Results saved to %s", args.output)


if __name__ == "__main__":
    main()
