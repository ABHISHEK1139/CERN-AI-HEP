"""
Shared synthetic-data / graph-dataset preparation for the experiment scripts.

``train_classifier.py`` and ``run_benchmark.py`` carried byte-identical copies of
this logic. Two things were wrong in both copies:

1. When the fingerprint changed, the script deleted ``graphs.pt`` and rebuilt
   it, but then constructed ``CollisionEventDataset(root=..., graphs=graphs)``
   — and that class preferred a *previously cached* ``processed/dataset.pt``
   over the graphs it was handed. So regeneration silently did nothing and
   every "new" run trained on the old graphs. :func:`ensure_graph_dataset`
   now clears the stale PyG cache alongside the graph file.
2. ``config["mlflow"]["experiment"]`` was never read; the trainer always fell
   back to its ``cern-ai`` default.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch

from anomaly_engine.models import available_classifiers
from event_ingestion.synthetic import SyntheticEventGenerator
from graph_builder.dataset import CollisionEventDataset
from graph_builder.graph_constructor import EventGraphConstructor

logger = logging.getLogger(__name__)


def _graph_fingerprint(syn_cfg: dict[str, Any], graph_cfg: dict[str, Any]) -> dict[str, Any]:
    """Identity of a generated dataset: every knob that changes the graphs."""
    return {
        "n_normal": syn_cfg["n_normal"],
        "n_anomaly": syn_cfg["n_anomaly"],
        "seed": syn_cfg["seed"],
        "strategy": graph_cfg["strategy"],
        "k": graph_cfg["k"],
        "delta_r": graph_cfg.get("delta_r", 1.5),
    }


def ensure_graph_dataset(
    config: dict[str, Any],
    data_dir: str | None = None,
):
    """Load or generate the synthetic graph dataset and build DataLoaders.

    Args:
        config: Parsed experiment YAML.
        data_dir: Override for ``config['data']['graphs']['output']``.

    Returns:
        ``(train_loader, val_loader, test_loader, dataset)``.
    """
    graph_dir = Path(data_dir) if data_dir else Path(config["data"]["graphs"]["output"])
    graphs_file = graph_dir / "graphs.pt"
    fingerprint_file = graph_dir / "synthetic_fingerprint.json"
    # CollisionEventDataset's own collated cache.
    pyg_cache_dir = graph_dir / "processed"

    syn_cfg = config["data"]["synthetic"]
    graph_cfg = config["data"]["graphs"]
    fingerprint = _graph_fingerprint(syn_cfg, graph_cfg)

    try:
        cached = json.loads(fingerprint_file.read_text())
    except (OSError, ValueError):
        cached = None

    if cached != fingerprint and (graphs_file.exists() or pyg_cache_dir.exists()):
        logger.info(
            "Synthetic/graph config changed (fingerprint %s -> %s); "
            "discarding cached graphs and the collated dataset cache.",
            cached, fingerprint,
        )
        graphs_file.unlink(missing_ok=True)
        # Without this, CollisionEventDataset would serve the stale
        # processed/dataset.pt and silently ignore the regenerated graphs.
        shutil.rmtree(pyg_cache_dir, ignore_errors=True)

    if graphs_file.exists():
        logger.info("Loading existing graphs from %s", graphs_file)
        graphs = torch.load(graphs_file, map_location="cpu", weights_only=False)
    else:
        logger.info("No graphs found. Generating synthetic data...")

        gen = SyntheticEventGenerator(seed=syn_cfg["seed"])
        events, labels = gen.generate(
            n_normal=syn_cfg["n_normal"],
            n_anomaly=syn_cfg["n_anomaly"],
        )

        constructor = EventGraphConstructor(
            strategy=graph_cfg["strategy"],
            k=graph_cfg["k"],
            delta_r_threshold=graph_cfg.get("delta_r", 1.5),
        )
        graphs = constructor.convert_dataset(events, labels)
        if not graphs:
            raise RuntimeError(
                f"Graph construction produced 0 graphs from "
                f"{len(events)} synthetic events. Check the strategy/k in the "
                f"config and the min_particles cut."
            )

        graph_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, graphs_file)
        fingerprint_file.write_text(json.dumps(fingerprint, indent=2))
        logger.info("Saved %d graphs to %s", len(graphs), graphs_file)

    dataset = CollisionEventDataset(root=str(graph_dir), graphs=graphs)
    split_cfg = config.get("splits", {})
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=config["training"]["batch_size"],
        train_ratio=split_cfg.get("train", 0.7),
        val_ratio=split_cfg.get("val", 0.15),
        test_ratio=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
    )
    return train_loader, val_loader, test_loader, dataset


def make_trainer(model, config: dict[str, Any], device: str, run_prefix: str):
    """Build a :class:`Trainer` wired from the experiment config.

    Honours ``mlflow.enabled`` *and* ``mlflow.experiment``; the experiment name
    was previously dropped on the floor.
    """
    from anomaly_engine.trainer import Trainer

    train_cfg = config["training"]
    mlflow_cfg = config.get("mlflow", {}) or {}
    return Trainer(
        model,
        device=device,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        patience=train_cfg["patience"],
        max_grad_norm=train_cfg["max_grad_norm"],
        checkpoint_dir=run_prefix,
        use_mlflow=bool(mlflow_cfg.get("enabled", False)),
        experiment_name=mlflow_cfg.get("experiment", "cern-ai"),
    )


def build_classifier(name: str, config: dict[str, Any]):
    """Instantiate a classifier from ``config['model']`` by registry name."""
    from anomaly_engine.models import get_classifier

    if name not in available_classifiers():
        raise ValueError(
            f"Unknown model {name!r}. Available: {available_classifiers()}"
        )
    model_cfg = config["model"]
    return get_classifier(
        name,
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
    )
