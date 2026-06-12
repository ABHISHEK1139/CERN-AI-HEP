"""
PhysicsNeMo benchmarking.

Runs the same dataset through PhysicsNeMo and custom models,
collects metrics, and generates comparison reports.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from anomaly_engine.evaluate import Evaluator
from anomaly_engine.trainer import Trainer
from anomaly_engine.models import get_classifier, CLASSIFIERS
from physicsnemo_integration.wrapper import PhysicsNeMoWrapper

logger = logging.getLogger(__name__)


class PhysicsNeMoBenchmark:
    """Run benchmark comparing custom GNN models vs PhysicsNeMo."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        device: str = "auto",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.evaluator = Evaluator(device=str(self.device))
        self.results = {}

    def run_benchmark(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        models_to_test: Optional[List[str]] = None,
        epochs: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run full benchmark across all models.

        Args:
            train_loader: Training data.
            val_loader: Validation data.
            test_loader: Test data.
            models_to_test: List of model names. None = all models + PhysicsNeMo.
            epochs: Training epochs per model.

        Returns:
            Dict of {model_name: metrics}.
        """
        if models_to_test is None:
            models_to_test = list(CLASSIFIERS.keys()) + ["physicsnemo"]

        logger.info(f"Benchmarking {len(models_to_test)} models: {models_to_test}")

        for model_name in models_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*60}")

            try:
                result = self._train_and_evaluate(
                    model_name, train_loader, val_loader, test_loader, epochs
                )
                self.results[model_name] = result
                logger.info(
                    f"{model_name}: accuracy={result.get('accuracy', 0):.3f}, "
                    f"auroc={result.get('auroc', 0):.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}

        return self.results

    def _train_and_evaluate(
        self,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, Any]:
        """Train a single model and evaluate."""
        start_time = time.time()

        # Create model
        kwargs = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

        if model_name == "physicsnemo":
            model = PhysicsNeMoWrapper(**kwargs)
        else:
            model = get_classifier(model_name, **kwargs)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Train
        trainer = Trainer(
            model,
            device=str(self.device),
            learning_rate=1e-3,
            patience=10,
            checkpoint_dir=f"checkpoints/{model_name}",
        )
        history = trainer.train_classifier(
            train_loader, val_loader, epochs=epochs, run_name=model_name
        )

        # Evaluate
        metrics = self.evaluator.evaluate_classifier(model, test_loader)

        elapsed = time.time() - start_time
        metrics["training_time_sec"] = elapsed
        metrics["n_parameters"] = n_params
        metrics["model_name"] = model_name

        return metrics

    def print_comparison(self) -> None:
        """Print comparison table."""
        if not self.results:
            print("No results yet. Run benchmark first.")
            return

        print("\n" + "=" * 80)
        print("MODEL COMPARISON — GNN Anomaly Detection for LHC Events")
        print("=" * 80)
        print(
            f"{'Model':<15} {'Accuracy':>10} {'AUROC':>8} {'F1':>8} "
            f"{'Precision':>10} {'Recall':>8} {'Params':>10} {'Time(s)':>8}"
        )
        print("-" * 80)

        for name, metrics in self.results.items():
            if "error" in metrics:
                print(f"{name:<15} {'ERROR':>10}  {metrics['error']}")
                continue

            print(
                f"{name:<15} "
                f"{metrics.get('accuracy', 0):>10.4f} "
                f"{metrics.get('auroc', 0):>8.4f} "
                f"{metrics.get('f1', 0):>8.4f} "
                f"{metrics.get('precision', 0):>10.4f} "
                f"{metrics.get('recall', 0):>8.4f} "
                f"{metrics.get('n_parameters', 0):>10,} "
                f"{metrics.get('training_time_sec', 0):>8.1f}"
            )

        print("=" * 80)

    def save_results(self, output_path: str) -> None:
        """Save benchmark results to JSON."""
        import json
        from pathlib import Path

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=convert)

        logger.info(f"Results saved to {output_path}")
