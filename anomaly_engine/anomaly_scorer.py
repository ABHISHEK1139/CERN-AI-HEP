"""
Anomaly scoring and ranking.

Computes reconstruction error from a trained graph autoencoder,
ranks events by anomaly score, and applies threshold selection.
"""

import logging
from typing import Any

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        all_scores: list[float] = []
        all_labels: list[float] = []
        all_event_ids: list[Any] = []
        labels_aligned = True

        for data in loader:
            data = data.to(self.device)
            result = self.model(data)
            if not isinstance(result, dict) or "per_graph_loss" not in result:
                raise TypeError(
                    "AnomalyScorer needs an autoencoder returning "
                    "dict(per_graph_loss=...)."
                )

            scores = result["per_graph_loss"].detach().cpu().numpy().flatten()
            all_scores.extend(scores.tolist())

            y = getattr(data, "y", None)
            if y is None:
                labels_aligned = False
            else:
                all_labels.extend(y.detach().cpu().numpy().flatten().tolist())

            if hasattr(data, "event_id"):
                eid = data.event_id
                # PyG Batch collates ints into a tensor [B]; expand per-graph.
                if isinstance(eid, torch.Tensor):
                    all_event_ids.extend(eid.detach().cpu().flatten().tolist())
                elif isinstance(eid, (list, tuple)):
                    all_event_ids.extend(list(eid))
                elif isinstance(eid, np.ndarray):
                    all_event_ids.extend(eid.flatten().tolist())
                else:
                    try:
                        all_event_ids.append(int(eid))
                    except (TypeError, ValueError):
                        all_event_ids.append(eid)
            else:
                # No event_id on this batch: fall back to positional indices so
                # the id list never drifts out of step with the score list.
                offset = len(all_event_ids)
                all_event_ids.extend(range(offset, offset + len(scores)))

        scores = np.array(all_scores)
        labels = np.array(all_labels) if all_labels else np.array([])
        event_ids = np.array(all_event_ids) if all_event_ids else np.arange(len(scores))

        if not labels_aligned and labels.size:
            logger.warning(
                "Some batches lacked data.y; %d of %d labels are usable.",
                labels.size, scores.size,
            )

        return scores, labels, event_ids

    def rank_anomalies(
        self,
        scores: np.ndarray,
        event_ids: np.ndarray,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Rank events by anomaly score and return top-k.

        Args:
            scores: Anomaly scores.
            event_ids: Event identifiers.
            top_k: Number of top anomalies to return.

        Returns:
            List of dicts with event_id, score, rank.
        """
        scores = np.asarray(scores).flatten()
        event_ids = np.asarray(event_ids).flatten()
        if len(scores) != len(event_ids):
            raise ValueError(
                f"rank_anomalies needs one event_id per score, got "
                f"{len(event_ids)} ids and {len(scores)} scores."
            )
        if top_k <= 0 or len(scores) == 0:
            return []
        top_k = min(int(top_k), len(scores))

        # Highest score first. Stable ordering via mergesort keeps ties in
        # ascending index order, so repeated runs rank identically.
        sorted_idx = np.argsort(-scores, kind="stable")

        results = []
        for rank, idx in enumerate(sorted_idx[:top_k]):
            eid = event_ids[idx]
            try:
                eid = int(eid)
            except (TypeError, ValueError):
                pass
            results.append({
                "rank": rank + 1,
                "event_id": eid,
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
        if len(scores) == 0:
            raise ValueError("select_threshold received empty scores.")
        scores = np.asarray(scores).flatten()
        if method == "percentile":
            if not 0.0 <= percentile <= 100.0:
                raise ValueError(f"percentile must be in [0, 100], got {percentile}.")
            threshold = float(np.percentile(scores, percentile))
        elif method == "sigma":
            if n_sigma < 0:
                raise ValueError(f"n_sigma must be >= 0, got {n_sigma}.")
            threshold = float(np.mean(scores) + n_sigma * np.std(scores))
        else:
            raise ValueError(
                f"Unknown method: {method!r}. Expected 'percentile' or 'sigma'."
            )

        n_above = int((scores > threshold).sum())
        logger.info(
            "Threshold (%s): %.6f (%d/%d events flagged, %.1f%%)",
            method, threshold, n_above, len(scores), 100.0 * n_above / len(scores),
        )

        return threshold

    def generate_report(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        event_ids: np.ndarray,
        top_k: int = 100,
    ) -> dict[str, Any]:
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
        scores = np.asarray(scores).flatten()
        labels = np.asarray(labels).flatten()
        event_ids = np.asarray(event_ids).flatten()
        if len(scores) == 0:
            raise ValueError("generate_report received empty scores.")
        if len(event_ids) != len(scores):
            raise ValueError(
                f"generate_report needs one event_id per score, got "
                f"{len(event_ids)} ids and {len(scores)} scores."
            )
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

        # If per-graph labels are fully available, compute detection metrics.
        # (Batches without data.y are skipped in score_dataset, so a length
        # mismatch means labels are partial — skip metrics instead of misaligning.)
        if len(labels) == len(scores) and len(labels) > 0:
            from sklearn.metrics import average_precision_score, roc_auc_score

            if len(np.unique(labels)) > 1:
                report["auroc"] = float(roc_auc_score(labels, scores))
                report["auprc"] = float(average_precision_score(labels, scores))

            # Top-k precision. Guard the empty-selection case: slicing
            # `[:0]` and calling .mean() returned NaN plus a RuntimeWarning.
            k = max(0, min(int(top_k), len(scores)))
            top_k_labels = labels[np.argsort(scores)[::-1][:k]]
            if k > 0:
                report["top_k"] = k
                report["top_k_precision"] = float(top_k_labels.mean())
                report["n_anomalies_in_top_k"] = int(top_k_labels.sum())
            else:
                report["top_k"] = 0
                report["top_k_precision"] = None
                report["n_anomalies_in_top_k"] = 0
        elif len(labels) > 0:
            logger.warning(
                "Labels cover only %d of %d events; skipping detection metrics "
                "to avoid misaligning scores against labels.",
                len(labels), len(scores),
            )

        return report

    def print_report(self, report: dict[str, Any]) -> None:
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
            k = report.get("top_k", len(report["top_anomalies"]))
            precision = report.get("top_k_precision")
            if precision is not None:
                print(f"    Top-{k} precision: {precision:.3f}")
                print(f"    Anomalies in top-{k}: {report.get('n_anomalies_in_top_k', 0)}")
        elif "auroc" not in report:
            print()
            print("  Detection metrics: unavailable (no per-event labels).")

        print()
        print("  Top 10 most anomalous events:")
        for entry in report["top_anomalies"][:10]:
            print(f"    #{entry['rank']:3d}  Event {entry['event_id']!s:>6}  "
                  f"Score: {entry['anomaly_score']:.6f}")
        print("=" * 60)
