"""
Evaluation utilities for classification and anomaly detection.

- Classification: accuracy, precision, recall, F1, AUC, confusion matrix
- Anomaly detection: AUROC, score distributions, threshold analysis
- Visualization: ROC curves, t-SNE of latent space, training curves
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import torch
from torch_geometric.loader import DataLoader

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation and visualization for graph models."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    # ----------------------------------------------------------------
    # Classification Evaluation
    # ----------------------------------------------------------------

    @torch.no_grad()
    def evaluate_classifier(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Evaluate a classifier on a test set.

        Returns:
            Dict with accuracy, precision, recall, F1, AUC, confusion matrix.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix,
        )

        model = model.to(self.device)
        model.eval()

        all_preds = []
        all_probs = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            out = model(data)
            if isinstance(out, dict):
                raise TypeError(
                    "evaluate_classifier needs a classifier returning logits, "
                    "but got a dict (e.g. autoencoder). Use evaluate_autoencoder()."
                )
            logits = out
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)

            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=-1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # anomaly probability
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            else:
                # Single-logit binary head: sigmoid probability, 0.5 threshold.
                p = torch.sigmoid(logits[:, 0])
                all_probs.extend(p.cpu().numpy())
                all_preds.extend((logits[:, 0] > 0).long().cpu().numpy())
            all_labels.extend(data.y.cpu().numpy().flatten())

        preds = np.array(all_preds)
        probs = np.array(all_probs)
        labels = np.array(all_labels)

        if len(labels) == 0:
            raise ValueError("evaluate_classifier received an empty DataLoader.")

        n_classes = len(np.unique(labels))
        average = "binary" if n_classes == 2 else "macro"
        results = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, average=average, zero_division=0)),
            "recall": float(recall_score(labels, preds, average=average, zero_division=0)),
            "f1": float(f1_score(labels, preds, average=average, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }

        if n_classes == 2:
            results["auroc"] = float(roc_auc_score(labels, probs))

        return results

    # ----------------------------------------------------------------
    # Anomaly Detection Evaluation
    # ----------------------------------------------------------------

    @torch.no_grad()
    def evaluate_autoencoder(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Evaluate autoencoder anomaly detection performance.

        Returns:
            Dict with AUROC, AUPRC, score statistics.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        model = model.to(self.device)
        model.eval()

        all_scores = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            result = model(data)
            if not isinstance(result, dict) or "per_graph_loss" not in result:
                raise TypeError(
                    "evaluate_autoencoder needs an autoencoder returning "
                    "dict(per_graph_loss=...). Use evaluate_classifier() for classifiers."
                )
            scores = result["per_graph_loss"].cpu().numpy()
            all_scores.extend(scores)

            if data.y is not None:
                all_labels.extend(data.y.cpu().numpy().flatten())

        scores = np.array(all_scores)
        labels = np.array(all_labels)

        if len(scores) == 0:
            raise ValueError("evaluate_autoencoder received an empty DataLoader.")

        results = {
            "mean_recon_error": float(np.mean(scores)),
            "std_recon_error": float(np.std(scores)),
        }

        if len(labels) > 0 and len(np.unique(labels)) > 1:
            results["auroc"] = float(roc_auc_score(labels, scores))
            results["auprc"] = float(average_precision_score(labels, scores))

            # Score separation
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            results["normal_mean"] = float(np.mean(normal_scores))
            results["anomaly_mean"] = float(np.mean(anomaly_scores))
            results["score_separation"] = float(
                np.mean(anomaly_scores) - np.mean(normal_scores)
            )

        return results

    # ----------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Curves",
        output_path: Optional[str] = None,
    ):
        """Plot training and validation loss curves."""
        import matplotlib.pyplot as plt

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        lrs = history.get("lr", [])
        if not train_loss or not val_loss:
            raise ValueError("plot_training_curves needs history with train_loss/val_loss.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax = axes[0]
        ax.plot(train_loss, label="Train Loss", color="#3498db", linewidth=2)
        ax.plot(val_loss, label="Val Loss", color="#e74c3c", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{title} — Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1]
        ax.plot(lrs if lrs else [1.0] * len(train_loss), color="#2ecc71", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_roc_curve(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        model_name: str = "Model",
        output_path: Optional[str] = None,
    ):
        """Plot ROC curve."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color="#3498db", linewidth=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Anomaly Detection")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_score_distributions(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        output_path: Optional[str] = None,
    ):
        """Plot anomaly score distributions for normal vs anomalous events."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, bins=50, alpha=0.7, color="#3498db",
                label=f"Normal (n={len(normal_scores)})", density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color="#e74c3c",
                label=f"Anomaly (n={len(anomaly_scores)})", density=True)

        ax.set_xlabel("Reconstruction Error (Anomaly Score)")
        ax.set_ylabel("Density")
        ax.set_title("Anomaly Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def plot_latent_space(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        output_path: Optional[str] = None,
        method: str = "tsne",
    ):
        """
        Visualize latent space using t-SNE or UMAP.

        Args:
            model: Trained autoencoder or encoder.
            loader: DataLoader.
            output_path: Path to save figure.
            method: 'tsne' or 'umap'.
        """
        import matplotlib.pyplot as plt
        from torch_geometric.nn import global_mean_pool

        model = model.to(self.device)
        model.eval()

        all_embeddings = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)

            if getattr(data, "x", None) is None:
                raise ValueError("plot_latent_space needs data.x node features.")
            if hasattr(model, "encoder"):
                batch = getattr(data, "batch", None)
                if batch is None:
                    batch = torch.zeros(
                        data.x.size(0), dtype=torch.long, device=data.x.device
                    )
                z = model.encoder(data.x, data.edge_index, batch)
            else:
                # Generic model (e.g. PhysicsNeMo wrapper expects a Data object)
                try:
                    z = model(data.x, data.edge_index, data.batch)
                except Exception:
                    # Fall back: wrapper-style forward on Data, then pool
                    graph_emb = model(data)
                    all_embeddings.append(graph_emb.cpu().numpy())
                    if data.y is not None:
                        all_labels.extend(data.y.cpu().numpy().flatten())
                    continue

            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            graph_emb = global_mean_pool(z, batch)
            all_embeddings.append(graph_emb.cpu().numpy())

            if data.y is not None:
                all_labels.extend(data.y.cpu().numpy().flatten())

        if not all_embeddings:
            raise ValueError("plot_latent_space received an empty DataLoader.")
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.array(all_labels)

        if len(embeddings) < 2:
            raise ValueError("plot_latent_space needs at least 2 graphs.")

        # Dimensionality reduction
        if method == "tsne":
            from sklearn.manifold import TSNE
            perplexity = max(1, min(30, len(embeddings) - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
            except ImportError:
                from sklearn.manifold import TSNE
                perplexity = max(1, min(30, len(embeddings) - 1))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                method = "tsne"

        coords = reducer.fit_transform(embeddings)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        if len(labels) > 0:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
                      s=10, alpha=0.5, c="#3498db", label="Normal")
            ax.scatter(coords[anomaly_mask, 0], coords[anomaly_mask, 1],
                      s=20, alpha=0.8, c="#e74c3c", label="Anomaly", marker="x")
            ax.legend()
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.5, c="#3498db")

        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Latent Space Visualization ({method.upper()})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None,
    ):
        """
        Plot comparison table of multiple models.

        Args:
            results: {model_name: {metric: value}}.
        """
        import matplotlib.pyplot as plt

        if not results:
            raise ValueError("plot_comparison_table received empty results.")
        models = list(results.keys())
        metrics = ["auroc", "accuracy", "f1", "precision", "recall"]
        available_metrics = [m for m in metrics if m in results[models[0]]]
        if not available_metrics:
            raise ValueError("plot_comparison_table found no plottable metrics.")

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(available_metrics))
        width = 0.8 / len(models)
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, model_name in enumerate(models):
            values = [results[model_name].get(m, 0) for m in available_metrics]
            ax.bar(x + i * width, values, width, label=model_name,
                   color=colors[i % len(colors)], edgecolor="black", alpha=0.8)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in available_metrics])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
