"""
Baseline models (non-graph) for comparison.

Implements:
- MLPClassifier: Flat feature MLP — ignores graph structure
- CNNClassifier: 1D CNN over particle sequences — partial structure

These establish baselines to demonstrate the value of graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron baseline.

    Operates on aggregated (pooled) node features, ignoring graph structure.
    This baseline demonstrates that graph topology matters.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Pool node features then classify
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # mean + max pooled
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        """Forward pass: pool then classify."""
        x = data.x
        batch = data.batch

        # Aggregate node features (ignore edges entirely)
        pooled = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=-1)

        return self.network(pooled)

    def predict(self, data):
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data):
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)


class CNNClassifier(nn.Module):
    """
    1D CNN baseline.

    Treats particles as a sequence sorted by pT, applies 1D convolutions.
    Captures local patterns in the particle sequence but not true graph structure.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        num_classes: int = 2,
        max_particles: int = 50,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_particles = max_particles

        # 1D CNN: treat particle features as channels, sequence as particles
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        """Forward pass: pad to fixed length, conv, classify."""
        x = data.x  # [total_nodes, input_dim]
        batch = data.batch

        # Reconstruct per-graph tensors with padding
        batch_size = batch.max().item() + 1
        padded = torch.zeros(
            batch_size, self.max_particles, self.input_dim,
            device=x.device, dtype=x.dtype
        )

        for b in range(batch_size):
            mask = batch == b
            nodes = x[mask]
            n = min(nodes.shape[0], self.max_particles)
            # Sort by pT (feature index 5 = first physics feature after one-hot)
            if n > 0:
                pt_idx = 5  # pT is at index NUM_PARTICLE_TYPES (5)
                sorted_idx = nodes[:, pt_idx].argsort(descending=True)
                nodes = nodes[sorted_idx]
                padded[b, :n] = nodes[:n]

        # [B, max_particles, input_dim] -> [B, input_dim, max_particles]
        padded = padded.permute(0, 2, 1)

        features = self.conv_layers(padded).squeeze(-1)  # [B, hidden_dim]
        return self.classifier(features)

    def predict(self, data):
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data):
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)
