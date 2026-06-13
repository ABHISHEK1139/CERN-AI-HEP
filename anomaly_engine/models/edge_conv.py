"""
Edge Convolution (EdgeConv) models for Particle Cloud / Point Cloud datasets.

Implements:
- EdgeConvEncoder: EdgeConv encoder for graph autoencoder

Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", TOG 2019.
(Also known as the building block for ParticleNet in High Energy Physics).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool


class EdgeConvEncoder(nn.Module):
    """
    EdgeConv-based graph encoder for point clouds.
    
    For each edge (i, j), EdgeConv computes features using an MLP on:
    [x_i, x_j - x_i]
    
    This explicitly learns local geometric/kinematic relationships.
    """

    def __init__(
        self,
        input_dim: int = 16,
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer MLP for EdgeConv
        mlp1 = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(EdgeConv(nn=mlp1, aggr='mean'))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp_hidden = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(EdgeConv(nn=mlp_hidden, aggr='mean'))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        mlp_out = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.convs.append(EdgeConv(nn=mlp_out, aggr='mean'))
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
        # Combine mean and max pooling for robust global representation
        graph_emb = global_mean_pool(node_emb, batch) + global_max_pool(node_emb, batch)
        return graph_emb
