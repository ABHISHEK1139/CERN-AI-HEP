"""
Graph Autoencoder for anomaly detection.

Architecture:
    Graph → GNN Encoder → Latent Space → MLP Decoder → Reconstructed Features
    Reconstruction Error = Anomaly Score

Normal events: low reconstruction error (model has seen similar patterns)
Anomalous events: high reconstruction error (model cannot reconstruct well)

This is an unsupervised approach — no anomaly labels needed during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GraphDecoder(nn.Module):
    """
    Decoder: reconstruct node features from latent embeddings.

    Takes per-node latent vectors and reconstructs the original node features.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 11,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        """
        Decode latent vectors to reconstructed features.

        Args:
            z: Latent node embeddings [N, latent_dim].

        Returns:
            Reconstructed features [N, output_dim].
        """
        return self.decoder(z)


class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder for unsupervised anomaly detection.

    Pipeline:
        Input Features → GNN Encoder → Latent Space → MLP Decoder → Reconstructed Features
        Loss = MSE(input, reconstructed) per node, aggregated per graph
    """

    def __init__(self, encoder: nn.Module, decoder: GraphDecoder):
        """
        Args:
            encoder: Any GNN encoder (GCN, GraphSAGE, GAT) with forward(x, edge_index, batch).
            decoder: GraphDecoder instance.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        """
        Full encode-decode pass.

        Args:
            data: PyG Data/Batch object.

        Returns:
            Dict with:
                - x_recon: Reconstructed node features [N, D]
                - z: Latent embeddings [N, latent_dim]
                - loss: Mean reconstruction loss (scalar)
                - per_node_loss: Per-node reconstruction error [N]
                - per_graph_loss: Per-graph reconstruction error [B]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode
        z = self.encoder(x, edge_index, batch)

        # Decode
        x_recon = self.decoder(z)

        # Reconstruction loss
        per_node_loss = F.mse_loss(x_recon, x, reduction="none").mean(dim=-1)  # [N]

        # Aggregate per graph using efficient scatter
        from torch_geometric.utils import scatter
        per_graph_loss = scatter(per_node_loss, batch, reduce='mean')

        loss = per_graph_loss.mean()

        return {
            "x_recon": x_recon,
            "z": z,
            "loss": loss,
            "per_node_loss": per_node_loss,
            "per_graph_loss": per_graph_loss,
        }

    def get_anomaly_scores(self, data) -> torch.Tensor:
        """
        Compute anomaly scores (reconstruction error per graph).

        Higher score = more anomalous.

        Args:
            data: PyG Batch object.

        Returns:
            Anomaly scores tensor [B].
        """
        with torch.no_grad():
            result = self.forward(data)
        return result["per_graph_loss"]

    def get_latent(self, data) -> torch.Tensor:
        """
        Get graph-level latent representations.

        Args:
            data: PyG Batch object.

        Returns:
            Latent vectors [B, latent_dim].
        """
        with torch.no_grad():
            z = self.encoder(data.x, data.edge_index, data.batch)
            return global_mean_pool(z, data.batch)

    def encode_nodes(self, data) -> torch.Tensor:
        """Get per-node latent representations."""
        with torch.no_grad():
            return self.encoder(data.x, data.edge_index, data.batch)
