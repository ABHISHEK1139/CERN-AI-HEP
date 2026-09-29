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

from anomaly_engine.models.norm import SafeBatchNorm1d, apply_norm, resolve_norm


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
        norm: str = "batch",
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of node features.
            hidden_dim: Width of hidden layers.
            latent_dim: Width of the output latent space.
            num_layers: Number of EdgeConv message-passing layers (>= 2).
            dropout: Dropout probability applied between layers.
            norm: 'batch' (default), 'layer', or 'none'. 'batch' uses
                :class:`SafeBatchNorm1d`, which matters here because the norms
                inside the EdgeConv MLPs normalize over the *edge* axis — a
                2-particle event yields a single edge and plain BatchNorm
                raises on it.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 (input + output layer), got {num_layers}.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if norm in ("batch", "batchnorm", "bn"):
                layers.append(SafeBatchNorm1d(hidden_dim))
            elif norm in ("layer", "layernorm", "ln"):
                layers.append(nn.LayerNorm(hidden_dim))
            elif norm not in ("none", "identity", "off"):
                raise ValueError(
                    f"Unknown norm '{norm}'. Expected one of: 'batch', 'layer', 'none'."
                )
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer MLP for EdgeConv
        self.convs.append(EdgeConv(nn=_make_mlp(2 * input_dim, hidden_dim), aggr='mean'))
        self.norms.append(resolve_norm(hidden_dim, norm))

        # Hidden layers
        for _ in range(max(num_layers - 2, 0)):
            self.convs.append(EdgeConv(nn=_make_mlp(2 * hidden_dim, hidden_dim), aggr='mean'))
            self.norms.append(resolve_norm(hidden_dim, norm))

        # Output layer
        self.convs.append(EdgeConv(nn=_make_mlp(2 * hidden_dim, latent_dim), aggr='mean'))
        self.norms.append(resolve_norm(latent_dim, norm))

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
            x = apply_norm(norm, x)
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
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        node_emb = self.forward(x, edge_index, batch)
        # Combine mean and max pooling for robust global representation
        graph_emb = global_mean_pool(node_emb, batch) + global_max_pool(node_emb, batch)
        return graph_emb
