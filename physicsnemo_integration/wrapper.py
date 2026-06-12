"""
PhysicsNeMo model wrapper.

Adapts PhysicsNeMo (NVIDIA Modulus) GNN architectures to the same
interface as custom models, enabling direct comparison.

PhysicsNeMo provides physics-informed neural network architectures
including MeshGraphNet and other GNN variants designed for
scientific simulation tasks.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

logger = logging.getLogger(__name__)


class PhysicsNeMoWrapper(nn.Module):
    """
    Wrapper to adapt PhysicsNeMo models to the project's interface.

    If PhysicsNeMo is not installed, falls back to a MeshGraphNet-inspired
    architecture implemented in pure PyTorch Geometric.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 6,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_physicsnemo: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_physicsnemo = use_physicsnemo

        self._physicsnemo_available = False

        if use_physicsnemo:
            try:
                self._init_physicsnemo(input_dim, hidden_dim, latent_dim, num_layers)
                self._physicsnemo_available = True
                logger.info("Using PhysicsNeMo MeshGraphNet backend")
            except ImportError:
                logger.warning(
                    "PhysicsNeMo not installed. "
                    "Install with: pip install nvidia-physicsnemo\n"
                    "Falling back to PyG MeshGraphNet-style architecture."
                )

        if not self._physicsnemo_available:
            self._init_fallback(input_dim, hidden_dim, latent_dim, num_layers, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _init_physicsnemo(self, input_dim, hidden_dim, latent_dim, num_layers):
        """Initialize with actual PhysicsNeMo MeshGraphNet."""
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        self.mesh_graph_net = MeshGraphNet(
            input_dim_nodes=input_dim,
            input_dim_edges=4,  # ΔR, Δη, Δφ, relative_pT
            output_dim_nodes=latent_dim,
            processor_size=num_layers,
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            hidden_dim_edge_encoder=hidden_dim,
            hidden_dim_node_decoder=hidden_dim,
        )

    def _init_fallback(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        """
        Fallback: MeshGraphNet-inspired architecture using PyTorch Geometric.

        MeshGraphNet uses:
        1. Node/edge encoders
        2. Message passing processors with residual connections
        3. Node decoder
        """
        from torch_geometric.nn import MessagePassing

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # edge_attr dim = 4
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Processor: stack of message passing layers with residuals
        self.processors = nn.ModuleList()
        for _ in range(num_layers):
            self.processors.append(
                MeshGraphNetLayer(hidden_dim, dropout)
            )

        # Node decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, data):
        """Forward pass — returns classification logits."""
        graph_emb = self.encode_graph(data)
        return self.classifier(graph_emb)

    def encode_graph(self, data):
        """Encode graph to a single vector."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        if self._physicsnemo_available:
            # PhysicsNeMo path
            node_emb = self.mesh_graph_net(x, edge_index, edge_attr)
        else:
            # Fallback path
            node_emb = self._fallback_forward(x, edge_index, edge_attr)

        # Pool to graph level
        return global_mean_pool(node_emb, batch) + global_max_pool(node_emb, batch)

    def _fallback_forward(self, x, edge_index, edge_attr):
        """Fallback MeshGraphNet forward pass."""
        # Encode
        h = self.node_encoder(x)

        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = None

        # Process
        for processor in self.processors:
            h = processor(h, edge_index, e)

        # Decode
        return self.node_decoder(h)

    def predict(self, data):
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data):
        logits = self.forward(data)
        return F.softmax(logits, dim=-1)


class MeshGraphNetLayer(nn.Module):
    """
    Single MeshGraphNet processor layer.

    Performs edge update → node update with residual connections.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # src + dst + edge
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # node + aggregated_messages
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [N, hidden_dim].
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, hidden_dim] or None.
        """
        src, dst = edge_index

        # Edge update
        if edge_attr is not None:
            edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        else:
            # If no edge features, use zero padding
            edge_input = torch.cat([
                x[src], x[dst],
                torch.zeros_like(x[src])
            ], dim=-1)

        edge_update = self.edge_mlp(edge_input)
        if edge_attr is not None:
            edge_attr = self.edge_norm(edge_attr + edge_update)  # residual
        else:
            edge_attr = self.edge_norm(edge_update)

        # Aggregate messages (mean)
        from torch_geometric.utils import scatter
        messages = scatter(edge_attr, dst, dim=0, dim_size=x.size(0), reduce="mean")

        # Node update
        node_input = torch.cat([x, messages], dim=-1)
        node_update = self.node_mlp(node_input)
        x = self.node_norm(x + self.dropout(node_update))  # residual

        return x
