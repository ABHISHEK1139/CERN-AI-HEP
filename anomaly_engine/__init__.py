"""
anomaly_engine: GNN models, graph autoencoder, anomaly scoring, and training pipeline.

Modules:
    models/         — GCN, GraphSAGE, GAT, baselines, autoencoder
    trainer         — Training loop with MLflow integration
    anomaly_scorer  — Reconstruction error and anomaly ranking
    evaluate        — Metrics, ROC curves, latent space visualization
"""

from anomaly_engine.trainer import Trainer
from anomaly_engine.anomaly_scorer import AnomalyScorer
from anomaly_engine.evaluate import Evaluator

__all__ = ["Trainer", "AnomalyScorer", "Evaluator"]
