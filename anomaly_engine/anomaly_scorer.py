"""
Anomaly scoring and ranking.

Computes reconstruction error from a trained graph autoencoder,
ranks events by anomaly score, and applies threshold selection.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Score and rank events by anomaly likelihood."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "auto",
    ):
        """
        Args:
            model: Trained GraphAutoencoder.
            device: Computation device.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_dataset(
        self, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for all events in a dataset.

        Args:
            loader: DataLoader of event graphs.

        Returns:
            Tuple of:
                - scores: Anomaly scores array [N]
                - labels: True labels array [N] (if available)
                - event_ids: Event ID array [N]
        """
        all_scores = []
        all_labels = []
        all_event_ids = []

        for data in loader:
            data = data.to(self.device)
            result = self.model(data)

            scores = result["per_graph_loss"].cpu().numpy()
            all_scores.extend(scores)

            if data.y is not None:
                labels = data.y.cpu().numpy().flatten()
                all_labels.extend(labels)

            if hasattr(data, "event_id"):
                if isinstance(data.event_id, (list, tuple)):
                    all_event_ids.extend(data.event_id)
                else:
                    all_event_ids.append(data.event_id)

        scores = np.array(all_scores)
        labels = np.array(all_labels) if all_labels else np.array([])
        event_ids = np.array(all_event_ids) if all_event_ids else np.arange(len(scores))

        return scores, labels, event_ids

    def rank_anomalies(
        self,
        scores: np.ndarray,
        event_ids: np.ndarray,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Rank events by anomaly score and return top-k.

        Args:
            scores: Anomaly scores.
            event_ids: Event identifiers.
            top_k: Number of top anomalies to return.

        Returns:
            List of dicts with event_id, score, rank.
        """
        sorted_idx = np.argsort(scores)[::-1]  # highest score first
        top_k = min(top_k, len(scores))

        results = []
        for rank, idx in enumerate(sorted_idx[:top_k]):
            results.append({
                "rank": rank + 1,
                "event_id": int(event_ids[idx]),
                "anomaly_score": float(scores[idx]),
            })

        return results

    def select_threshold(
        self,
        scores: np.ndarray,
        method: str = "percentile",
        percentile: float = 95.0,
        n_sigma: float = 3.0,
    ) -> float:
        """
        Select anomaly threshold.

        Args:
            scores: Anomaly scores array.
            method: 'percentile' or 'sigma'.
            percentile: Percentile threshold (for percentile method).
            n_sigma: Number of standard deviations (for sigma method).

        Returns:
            Threshold value.
        """
        if method == "percentile":
            threshold = np.percentile(scores, percentile)
        elif method == "sigma":
            threshold = np.mean(scores) + n_sigma * np.std(scores)
        else:
            raise ValueError(f"Unknown method: {method}")

        n_above = (scores > threshold).sum()
        logger.info(
            f"Threshold ({method}): {threshold:.6f} "
            f"({n_above}/{len(scores)} events flagged, {n_above/len(scores):.1%})"
        )

        return threshold

    def generate_report(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        event_ids: np.ndarray,
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate a full anomaly detection report.

        Args:
            scores: Anomaly scores.
            labels: True labels (0=normal, 1=anomaly).
            event_ids: Event identifiers.
            top_k: Number of top anomalies.

        Returns:
            Report dict with rankings, thresholds, and metrics.
        """
        report = {
            "n_events": len(scores),
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            },
            "top_anomalies": self.rank_anomalies(scores, event_ids, top_k),
            "thresholds": {
                "p95": float(np.percentile(scores, 95)),
                "p99": float(np.percentile(scores, 99)),
                "3sigma": float(np.mean(scores) + 3 * np.std(scores)),
            },
        }

        # If labels available, compute detection metrics
        if len(labels) > 0:
            from sklearn.metrics import roc_auc_score, average_precision_score

            if len(np.unique(labels)) > 1:
                report["auroc"] = float(roc_auc_score(labels, scores))
                report["auprc"] = float(average_precision_score(labels, scores))

            # Top-k precision
            sorted_idx = np.argsort(scores)[::-1]
            top_k_labels = labels[sorted_idx[:top_k]]
            report["top_k_precision"] = float(top_k_labels.mean())
            report["n_anomalies_in_top_k"] = int(top_k_labels.sum())

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted anomaly detection report."""
        print("=" * 60)
        print("ANOMALY DETECTION REPORT")
        print("=" * 60)
        print(f"  Events analyzed:  {report['n_events']:,}")
        print()
        print("  Score statistics:")
        s = report["score_stats"]
        print(f"    Mean:   {s['mean']:.6f}")
        print(f"    Std:    {s['std']:.6f}")
        print(f"    Min:    {s['min']:.6f}")
        print(f"    Max:    {s['max']:.6f}")
        print(f"    Median: {s['median']:.6f}")
        print()
        print("  Thresholds:")
        t = report["thresholds"]
        print(f"    95th percentile: {t['p95']:.6f}")
        print(f"    99th percentile: {t['p99']:.6f}")
        print(f"    3-sigma:         {t['3sigma']:.6f}")

        if "auroc" in report:
            print()
            print("  Detection metrics:")
            print(f"    AUROC:  {report['auroc']:.4f}")
            print(f"    AUPRC:  {report['auprc']:.4f}")
            print(f"    Top-k precision: {report['top_k_precision']:.3f}")
            print(f"    Anomalies in top-{len(report['top_anomalies'])}: "
                  f"{report['n_anomalies_in_top_k']}")

        print()
        print("  Top 10 most anomalous events:")
        for entry in report["top_anomalies"][:10]:
            print(f"    #{entry['rank']:3d}  Event {entry['event_id']:6d}  "
                  f"Score: {entry['anomaly_score']:.6f}")
        print("=" * 60)
