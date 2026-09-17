"""
Model registry.

Provides a unified interface to instantiate any model by name.
"""

from anomaly_engine.models.gcn import GCNClassifier, GCNEncoder
from anomaly_engine.models.graphsage import GraphSAGEClassifier, GraphSAGEEncoder
from anomaly_engine.models.gat import GATClassifier, GATEncoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.baselines import MLPClassifier, CNNClassifier
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder

# ---- Registry ----

CLASSIFIERS = {
    "gcn": GCNClassifier,
    "graphsage": GraphSAGEClassifier,
    "gat": GATClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
}

ENCODERS = {
    "gcn": GCNEncoder,
    "graphsage": GraphSAGEEncoder,
    "gat": GATEncoder,
    "edgeconv": EdgeConvEncoder,
}
# NOTE: "edgeconv" is encoder-only (EdgeConvEncoder); there is no EdgeConv
# classifier, so CLASSIFIERS intentionally omits it. Pair the encoder with
# GraphDecoder via get_autoencoder("edgeconv", ...) for anomaly detection.


def get_classifier(name: str, **kwargs):
    """Get classifier by name."""
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(CLASSIFIERS.keys())}")
    return CLASSIFIERS[name](**kwargs)


def get_encoder(name: str, **kwargs):
    """Get encoder by name."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODERS.keys())}")
    return ENCODERS[name](**kwargs)


def get_autoencoder(encoder_name: str, **kwargs):
    """Get autoencoder with specified encoder architecture.

    NOTE: all **kwargs go to the encoder; the decoder only takes
    latent_dim/hidden_dim/input_dim from the same kwargs. Pass
    classifier-only args (e.g. num_classes) to get_classifier instead.
    """
    encoder = get_encoder(encoder_name, **kwargs)
    decoder = GraphDecoder(
        latent_dim=kwargs.get("latent_dim", 32),
        hidden_dim=kwargs.get("hidden_dim", 64),
        output_dim=kwargs.get("input_dim", 11),
    )
    return GraphAutoencoder(encoder, decoder)


__all__ = [
    "GCNClassifier", "GCNEncoder",
    "GraphSAGEClassifier", "GraphSAGEEncoder",
    "GATClassifier", "GATEncoder",
    "EdgeConvEncoder",
    "MLPClassifier", "CNNClassifier",
    "GraphAutoencoder", "GraphDecoder",
    "get_classifier", "get_encoder", "get_autoencoder",
    "CLASSIFIERS", "ENCODERS",
]
