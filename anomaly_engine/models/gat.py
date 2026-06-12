"""
Graph Attention Network (GAT) models.

Implements:
- GATClassifier: Graph-level classification with attention
- GATEncoder: GAT encoder for autoencoder

Reference: Velickovic et al., "Graph Attention Networks", ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATEncoder(nn.Module):
    """GAT encoder with multi-head attention."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: multi-head attention
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Final layer: single head
        self.convs.append(GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout))
        self.norms.append(nn.BatchNorm1d(latent_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def encode_graph(self, x, edge_index, batch):
        node_emb = self.forward(x, edge_index, batch)
        return global_mean_pool(node_emb, batch) + global_max_pool(node_emb, batch)


class GATClassifier(nn.Module):
    """GAT for graph-level binary classification."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        graph_emb = self.encoder.encode_graph(
            data.x, data.edge_index, data.batch
        )
        return self.classifier(graph_emb)

    def predict(self, data):
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data):
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)
