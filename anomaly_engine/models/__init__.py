"""
Model registry.

Provides a unified interface to instantiate any model by name.
"""

from typing import Any, Dict, List

from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.models.baselines import CNNClassifier, MLPClassifier
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.gat import GATClassifier, GATEncoder
from anomaly_engine.models.gcn import GCNClassifier, GCNEncoder
from anomaly_engine.models.graphsage import GraphSAGEClassifier, GraphSAGEEncoder
from anomaly_engine.models.norm import SafeBatchNorm1d, resolve_norm

# ---- Registry ----

CLASSIFIERS: Dict[str, type] = {
    "gcn": GCNClassifier,
    "graphsage": GraphSAGEClassifier,
    "gat": GATClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
}

ENCODERS: Dict[str, type] = {
    "gcn": GCNEncoder,
    "graphsage": GraphSAGEEncoder,
    "gat": GATEncoder,
    "edgeconv": EdgeConvEncoder,
}
# NOTE: "edgeconv" is encoder-only (EdgeConvEncoder); there is no EdgeConv
# classifier, so CLASSIFIERS intentionally omits it. Pair the encoder with
# GraphDecoder via get_autoencoder("edgeconv", ...) for anomaly detection.


def available_classifiers() -> List[str]:
    """Names accepted by :func:`get_classifier`."""
    return sorted(CLASSIFIERS)


def available_encoders() -> List[str]:
    """Names accepted by :func:`get_encoder` and :func:`get_autoencoder`."""
    return sorted(ENCODERS)


def get_classifier(name: str, **kwargs: Any):
    """Get classifier by name.

    Raises:
        ValueError: If ``name`` is not a registered classifier.
    """
    if name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier: {name!r}. Available: {available_classifiers()}"
        )
    return CLASSIFIERS[name](**kwargs)


def get_encoder(name: str, **kwargs: Any):
    """Get encoder by name.

    Raises:
        ValueError: If ``name`` is not a registered encoder.
    """
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder: {name!r}. Available: {available_encoders()}"
        )
    return ENCODERS[name](**kwargs)


def get_autoencoder(encoder_name: str, **kwargs: Any) -> GraphAutoencoder:
    """Build a :class:`GraphAutoencoder` with the named encoder.

    Every keyword is forwarded to the encoder. ``latent_dim``, ``hidden_dim``
    and ``input_dim`` are additionally read for the decoder so the two halves
    always agree; pass the same values to both by leaving them in ``kwargs``.

    Args:
        encoder_name: Key of :data:`ENCODERS`.
        **kwargs: Encoder keyword arguments. Recognized decoder-shared keys:
            ``input_dim`` (default 11), ``hidden_dim`` (default 64),
            ``latent_dim`` (default 32), and ``norm`` (default 'batch').

    Returns:
        A wired-up :class:`GraphAutoencoder`.

    Raises:
        ValueError: If ``encoder_name`` is unknown, or if a dimension is
            non-positive.
    """
    if encoder_name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder: {encoder_name!r}. Available: {available_encoders()}"
        )

    input_dim = kwargs.get("input_dim", 11)
    hidden_dim = kwargs.get("hidden_dim", 64)
    latent_dim = kwargs.get("latent_dim", 32)
    norm = kwargs.get("norm", "batch")

    for name, value in (
        ("input_dim", input_dim),
        ("hidden_dim", hidden_dim),
        ("latent_dim", latent_dim),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")

    encoder = get_encoder(encoder_name, **kwargs)
    decoder = GraphDecoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        norm=norm,
    )
    return GraphAutoencoder(encoder, decoder)


__all__ = [
    "GCNClassifier", "GCNEncoder",
    "GraphSAGEClassifier", "GraphSAGEEncoder",
    "GATClassifier", "GATEncoder",
    "EdgeConvEncoder",
    "MLPClassifier", "CNNClassifier",
    "GraphAutoencoder", "GraphDecoder",
    "SafeBatchNorm1d", "resolve_norm",
    "get_classifier", "get_encoder", "get_autoencoder",
    "available_classifiers", "available_encoders",
    "CLASSIFIERS", "ENCODERS",
]
