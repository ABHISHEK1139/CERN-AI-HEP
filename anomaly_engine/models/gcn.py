"""
Graph Convolutional Network (GCN) models.

Implements:
- GCNClassifier: Graph-level classification (signal vs background)
- GCNEncoder: GCN encoder for graph autoencoder

Reference: Kipf & Welling, "Semi-Supervised Classification with
Graph Convolutional Networks", ICLR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCNEncoder(nn.Module):
    """GCN-based graph encoder for autoencoder."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, latent_dim))
        self.norms.append(nn.BatchNorm1d(latent_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        """
        Encode node features.

        Args:
            x: Node features [N, input_dim].
            edge_index: Edge indices [2, E].
            batch: Batch assignment vector.

        Returns:
            Node embeddings [N, latent_dim].
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def encode_graph(self, x, edge_index, batch):
        """
        Encode full graph to a single vector.

        Returns:
            Graph embedding [B, latent_dim].
        """
        node_emb = self.forward(x, edge_index, batch)
        # Combine mean and max pooling
        graph_emb = global_mean_pool(node_emb, batch) + global_max_pool(node_emb, batch)
        return graph_emb


class GCNClassifier(nn.Module):
    """GCN for graph-level binary classification."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Batch object.

        Returns:
            Logits [B, num_classes].
        """
        graph_emb = self.encoder.encode_graph(
            data.x, data.edge_index, data.batch
        )
        return self.classifier(graph_emb)

    def predict(self, data):
        """Get predicted class."""
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data):
        """Get class probabilities."""
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)
