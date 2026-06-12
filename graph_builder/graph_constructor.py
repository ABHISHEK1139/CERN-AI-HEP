"""
Event-to-graph conversion for collision events.

Converts particle collision events into PyTorch Geometric Data objects:
- Nodes = particles (features: type, pT, η, φ, mass, charge, energy)
- Edges = relationships between particles (kNN, ΔR threshold, or fully connected)

Usage:
    constructor = EventGraphConstructor(strategy="knn", k=8)
    graph = constructor.event_to_graph(event)

CLI:
    python -m graph_builder.graph_constructor --input data/synthetic/ --output data/graphs/
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from event_ingestion.config import EventConfig, EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from graph_builder.features import FeatureExtractor

logger = logging.getLogger(__name__)


class EventGraphConstructor:
    """Convert collision events to PyTorch Geometric graph objects."""

    def __init__(
        self,
        strategy: str = "knn",
        k: int = 8,
        delta_r_threshold: float = 1.5,
        include_edge_features: bool = True,
        config: Optional[EventConfig] = None,
    ):
        """
        Args:
            strategy: Edge construction strategy: 'knn', 'fully_connected', 'delta_r'.
            k: Number of nearest neighbors for kNN.
            delta_r_threshold: ΔR threshold for delta_r strategy.
            include_edge_features: Whether to compute edge features.
            config: Event configuration.
        """
        self.strategy = strategy
        self.k = k
        self.delta_r_threshold = delta_r_threshold
        self.include_edge_features = include_edge_features
        self.config = config or EventConfig()
        self.feature_extractor = FeatureExtractor()

    def event_to_graph(
        self, event: Dict[str, Any], label: Optional[int] = None
    ) -> "torch_geometric.data.Data":
        """
        Convert a single event to a PyTorch Geometric Data object.

        Args:
            event: Event dict from EventLoader.
            label: Optional label (0=normal, 1=anomaly).

        Returns:
            PyG Data object with node features, edge_index, edge_attr, and label.
        """
        from torch_geometric.data import Data

        particles = event["particles"]
        n_particles = len(particles)

        if n_particles < 2:
            # Need at least 2 particles to form edges
            logger.warning(f"Event {event.get('event_id', '?')}: only {n_particles} particles, skipping")
            return None

        # ---- Node features ----
        x = self.feature_extractor.extract_event_features(particles)
        x = torch.tensor(x, dtype=torch.float32)

        # ---- Edge construction ----
        if self.strategy == "knn":
            edge_index = self._build_knn_edges(particles, min(self.k, n_particles - 1))
        elif self.strategy == "fully_connected":
            edge_index = self._build_fully_connected_edges(n_particles)
        elif self.strategy == "delta_r":
            edge_index = self._build_delta_r_edges(particles, self.delta_r_threshold)
        else:
            raise ValueError(f"Unknown edge strategy: {self.strategy}")

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # ---- Edge features ----
        edge_attr = None
        if self.include_edge_features and edge_index.size(1) > 0:
            edge_features = []
            for e in range(edge_index.size(1)):
                i, j = edge_index[0, e].item(), edge_index[1, e].item()
                ef = self.feature_extractor.compute_edge_features(
                    particles[i], particles[j]
                )
                edge_features.append(ef)
            edge_attr = torch.tensor(np.stack(edge_features), dtype=torch.float32)

        # ---- Build Data object ----
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_particles,
        )

        # Label
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        elif "is_anomaly" in event:
            data.y = torch.tensor(
                [1 if event["is_anomaly"] else 0], dtype=torch.long
            )

        # Metadata
        data.event_id = event.get("event_id", -1)

        return data

    def _build_knn_edges(
        self, particles: List[Dict[str, Any]], k: int
    ) -> np.ndarray:
        """Build k-nearest neighbor edges in η-φ space."""
        delta_r_matrix = self.feature_extractor.compute_delta_r_matrix(particles)
        n = len(particles)

        src, dst = [], []
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            distances = delta_r_matrix[i].copy()
            distances[i] = np.inf  # exclude self-loop
            neighbors = np.argsort(distances)[:k]

            for j in neighbors:
                src.append(i)
                dst.append(j)
                # Bidirectional
                src.append(j)
                dst.append(i)

        # Remove duplicates
        edges = set(zip(src, dst))
        if edges:
            src, dst = zip(*edges)
            return np.array([list(src), list(dst)])
        return np.zeros((2, 0), dtype=np.int64)

    def _build_fully_connected_edges(self, n: int) -> np.ndarray:
        """Build fully connected graph (all pairs)."""
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array([src, dst])

    def _build_delta_r_edges(
        self, particles: List[Dict[str, Any]], threshold: float
    ) -> np.ndarray:
        """Build edges between particles within ΔR threshold."""
        delta_r_matrix = self.feature_extractor.compute_delta_r_matrix(particles)
        n = len(particles)

        src, dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if delta_r_matrix[i, j] < threshold:
                    src.extend([i, j])
                    dst.extend([j, i])

        if src:
            return np.array([src, dst])
        # Fallback: if no edges within threshold, connect nearest pairs
        return self._build_knn_edges(particles, k=min(3, n - 1))

    def convert_dataset(
        self,
        events: List[Dict[str, Any]],
        labels: Optional[np.ndarray] = None,
    ) -> List:
        """
        Convert a list of events to a list of PyG Data objects.

        Args:
            events: List of event dicts.
            labels: Optional label array.

        Returns:
            List of PyG Data objects.
        """
        graphs = []
        n_skipped = 0

        for i, event in enumerate(events):
            label = int(labels[i]) if labels is not None else None
            graph = self.event_to_graph(event, label=label)
            if graph is not None:
                graphs.append(graph)
            else:
                n_skipped += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"Converted {i + 1}/{len(events)} events to graphs")

        logger.info(
            f"Converted {len(graphs)} events to graphs "
            f"(skipped {n_skipped})"
        )
        return graphs


def main():
    """CLI entry point for graph construction."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build event graphs")
    parser.add_argument("--input", type=str, required=True, help="Input data directory")
    parser.add_argument("--output", type=str, required=True, help="Output graph directory")
    parser.add_argument("--strategy", type=str, default="knn", choices=["knn", "fully_connected", "delta_r"])
    parser.add_argument("--k", type=int, default=8, help="k for kNN")
    parser.add_argument("--delta-r", type=float, default=1.5, help="ΔR threshold")
    args = parser.parse_args()

    from event_ingestion.loader import EventLoader

    # Load events
    input_path = Path(args.input)
    loader = EventLoader()

    npz_files = list(input_path.glob("*.npz"))
    root_files = list(input_path.glob("*.root"))
    data_files = npz_files + root_files

    if not data_files:
        print(f"No .npz or .root files found in {input_path}")
        return

    all_events = []
    all_labels = []
    for f in data_files:
        if f.suffix == ".npz":
            data = np.load(f, allow_pickle=True)
            events = data["events"].tolist()
            labels = data["labels"]
            all_events.extend(events)
            all_labels.extend(labels)
        else:
            events = loader.load_root(f)
            all_events.extend(events)
            all_labels.extend([0] * len(events))  # unknown label

    labels = np.array(all_labels)

    # Build graphs
    constructor = EventGraphConstructor(
        strategy=args.strategy, k=args.k, delta_r_threshold=args.delta_r
    )
    graphs = constructor.convert_dataset(all_events, labels)

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(graphs, output_path / "graphs.pt")
    torch.save(labels, output_path / "labels.pt")

    print(f"\nSaved {len(graphs)} graphs to {output_path / 'graphs.pt'}")
    print(f"Node feature dim: {graphs[0].x.shape[1]}")
    print(f"Avg edges per graph: {np.mean([g.edge_index.shape[1] for g in graphs]):.0f}")


if __name__ == "__main__":
    main()
