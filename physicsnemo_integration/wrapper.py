"""
PhysicsNeMo model wrapper.

Adapts PhysicsNeMo (NVIDIA Modulus) GNN architectures to the same
interface as custom models, enabling direct comparison.

PhysicsNeMo provides physics-informed neural network architectures
including MeshGraphNet and other GNN variants designed for
scientific simulation tasks.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool

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
                self._physicsnemo_available = self._probe_physicsnemo(
                    input_dim, latent_dim
                )
            except Exception as e:  # ImportError, arity/config errors, missing CUDA, ...
                logger.warning(
                    "PhysicsNeMo backend unavailable "
                    f"({type(e).__name__}: {e}). "
                    "Install with: pip install nvidia-physicsnemo\n"
                    "Falling back to PyG MeshGraphNet-style architecture."
                )
                # Drop any partially built submodules so the fallback owns the
                # whole state_dict (otherwise load_state_dict sees extra keys).
                for attr in ("mesh_graph_net", "node_encoder", "edge_encoder",
                             "processors", "node_decoder"):
                    if hasattr(self, attr):
                        delattr(self, attr)

        if not self._physicsnemo_available:
            self._init_fallback(input_dim, hidden_dim, latent_dim, num_layers, dropout)

        if self._physicsnemo_available:
            logger.info("Using PhysicsNeMo MeshGraphNet backend")
        else:
            logger.info("Using PyG MeshGraphNet-style fallback backend")

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

    def _probe_physicsnemo(self, input_dim: int, latent_dim: int) -> bool:
        """Run a 3-node / 2-edge forward pass to confirm the native API.

        PhysicsNeMo's ``MeshGraphNet.forward`` takes ``(node_features,
        edge_features, graph_size)`` — not PyG's ``(x, edge_index, batch)``.
        The old code passed the PyG ordering, which cannot work: the model's
        own message passing never sees ``edge_index``. Rather than guess at
        several API generations, run a tiny graph through it and check the
        output shape. A mismatch or exception means "fall back", so a
        mis-wired backend degrades to a known-good implementation instead of
        producing garbage embeddings.
        """
        try:
            dev = next(self.mesh_graph_net.parameters()).device
            n, e = 3, 2
            x = torch.zeros(n, input_dim, device=dev)
            edge_attr = torch.zeros(e, 4, device=dev)
            edge_index = torch.tensor([[0, 1], [1, 2]], device=dev)
            graph_size = torch.zeros(n, dtype=torch.long, device=dev)

            out = self._run_physicsnemo(x, edge_index, edge_attr, graph_size)
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"expected a tensor, got {type(out).__name__}")
            if tuple(out.shape) != (n, latent_dim):
                raise ValueError(
                    f"expected node embeddings of shape {(n, latent_dim)}, "
                    f"got {tuple(out.shape)}"
                )
            return True
        except Exception as e:
            logger.warning(
                "PhysicsNeMo MeshGraphNet failed its API self-check "
                f"({type(e).__name__}: {e}); falling back to the PyG "
                "implementation."
            )
            return False

    def _run_physicsnemo(self, x, edge_index, edge_attr, graph_size):
        """Invoke the native MeshGraphNet with PhysicsNeMo's argument order."""
        return self.mesh_graph_net(x, edge_attr, graph_size, edge_index=edge_index)

    def _init_fallback(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        """
        Fallback: MeshGraphNet-inspired architecture using PyTorch Geometric.

        MeshGraphNet uses:
        1. Node/edge encoders
        2. Message passing processors with residual connections
        3. Node decoder
        """
        from torch_geometric.nn import MessagePassing  # noqa: F401  (documents intent)

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
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("PhysicsNeMoWrapper requires data.x node features.")
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self._physicsnemo_available:
            if edge_attr is None:
                # The native model needs explicit edge features; synthesise
                # zeros so a caller without edge_attr still gets embeddings
                # rather than a crash deep inside the encoder.
                edge_attr = x.new_zeros(edge_index.size(1), 4)
            try:
                node_emb = self._run_physicsnemo(x, edge_index, edge_attr, batch)
            except Exception as e:
                logger.warning(
                    "PhysicsNeMo forward failed (%s: %s); switching to the PyG "
                    "fallback for the rest of this run.", type(e).__name__, e,
                )
                self._physicsnemo_available = False
                self._init_fallback(
                    self.input_dim, self.hidden_dim, self.latent_dim,
                    len(getattr(self, "processors", [])) or 6, 0.0,
                )
                node_emb = self._fallback_forward(x, edge_index, edge_attr)
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
            edge_index: Edge indices [2, E], or None for edgeless graphs.
            edge_attr: Edge features [E, hidden_dim] or None.
        """
        if edge_index is None:
            # Edgeless graph: no messages; node-only update with zero messages.
            node_update = self.node_mlp(
                torch.cat([x, torch.zeros_like(x)], dim=-1)
            )
            return self.node_norm(x + self.dropout(node_update))
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
        return self.node_norm(x + self.dropout(node_update))  # residual
