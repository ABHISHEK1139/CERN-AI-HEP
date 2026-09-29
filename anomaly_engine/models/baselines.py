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

from anomaly_engine.models.norm import SafeBatchNorm1d

# JetClass 16-feature layout: pT is recoverable as hypot(part_px, part_py), i.e.
# columns 0 and 1. The synthetic 11-feature layout is
# [5 type one-hot, log1p(pt), eta, phi, log1p(mass), charge, log1p(energy)],
# where pT is column 5.
_PT_COLUMN_CANDIDATES = (0, 5)


def pt_column_index(n_features: int) -> int:
    """Return the node-feature column that best represents pT for sorting.

    Prefers the JetClass layout (px at column 0) and falls back to the
    synthetic layout (log1p(pt) at column 5).
    """
    for col in _PT_COLUMN_CANDIDATES:
        if n_features > col:
            return col
    return 0


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

        # Pool node features then classify.
        # SafeBatchNorm1d: a batch_size=1 graph batch collapses the pooled
        # tensor to a single row, which plain BatchNorm1d rejects.
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # mean + max pooled
            nn.ReLU(),
            SafeBatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            SafeBatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        """Forward pass: pool then classify."""
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("MLPClassifier requires data.x node features.")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

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

    Treats particles as a sequence sorted by pT and applies 1D convolutions.
    Captures local patterns in the particle sequence but not true graph structure.

    Zero-padding is masked out of every BatchNorm statistic, so graphs with
    differing constituent counts contribute equally regardless of how much
    padding they carry.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        num_classes: int = 2,
        max_particles: int = 128,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        if max_particles < 1:
            raise ValueError(f"max_particles must be >= 1, got {max_particles}.")
        self.input_dim = input_dim
        # Default raised from 50 to 128 to match the JetClass constituent cap;
        # 50 silently truncated three quarters of every real jet.
        self.max_particles = max_particles

        # 1D CNN: treat particle features as channels, sequence as particles
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            SafeBatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            SafeBatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _pad_and_sort(self, x: torch.Tensor, batch: torch.Tensor, batch_size: int):
        """Build a dense [B, max_particles, F] tensor sorted by pT, plus a mask."""
        padded = x.new_zeros(batch_size, self.max_particles, x.size(1))
        mask = torch.zeros(batch_size, self.max_particles, dtype=torch.bool, device=x.device)
        pt_idx = pt_column_index(x.size(1))

        for b in range(batch_size):
            nodes = x[batch == b]
            n = min(nodes.shape[0], self.max_particles)
            if n == 0:
                continue
            order = nodes[:, pt_idx].argsort(descending=True)[:n]
            padded[b, :n] = nodes[order]
            mask[b, :n] = True
        return padded, mask

    def forward(self, data):
        """Forward pass: pad to fixed length, conv, mask-aware pool, classify."""
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("CNNClassifier requires data.x node features.")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = getattr(data, "num_graphs", None)
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1 if x.size(0) > 0 else 0
        if batch_size == 0:
            return x.new_empty((0, self.classifier[-1].out_features))

        padded, mask = self._pad_and_sort(x, batch, batch_size)
        # [B, max_particles, F] -> [B, F, max_particles]
        padded = padded.transpose(1, 2)

        h = self.conv_layers(padded)  # [B, hidden, L]
        # Mask-aware global average pool: exclude padded positions so that
        # short graphs are not diluted by their own padding.
        m = mask.unsqueeze(1).to(h.dtype)  # [B, 1, L]
        denom = m.sum(dim=-1).clamp(min=1.0)  # [B, 1]
        features = (h * m).sum(dim=-1) / denom  # [B, hidden]

        return self.classifier(features)

    def predict(self, data):
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data):
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)
