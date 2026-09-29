"""Model architecture, registry, and normalization-layer regression tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from anomaly_engine.models import (
    CLASSIFIERS,
    ENCODERS,
    CNNClassifier,
    EdgeConvEncoder,
    GCNEncoder,
    GraphAutoencoder,
    GraphDecoder,
    MLPClassifier,
    SafeBatchNorm1d,
    available_classifiers,
    available_encoders,
    get_autoencoder,
    get_classifier,
    get_encoder,
    resolve_norm,
)
from anomaly_engine.models.norm import apply_norm

# ==========================================================================
# SafeBatchNorm1d - the degenerate-batch crashes
# ==========================================================================

class TestSafeBatchNorm:
    def test_single_sample_train_mode_uses_running_stats(self):
        """A size-1 input must not raise, and must use running statistics.

        Plain nn.BatchNorm1d raises ValueError here, which turned a valid
        batch (single-node graph, single-edge batch, batch_size=1) into a
        training crash.
        """
        norm = SafeBatchNorm1d(8)
        # Pretend the layer has learned something.
        with torch.no_grad():
            norm.running_mean.fill_(2.0)
            norm.running_var.fill_(4.0)
        norm.train()

        x = torch.randn(1, 8)
        out = norm(x)
        expected = (x - 2.0) / torch.sqrt(torch.tensor(4.0 + norm.eps))
        assert out.shape == x.shape
        assert torch.allclose(out, expected, atol=1e-5)

    def test_multi_sample_train_mode_matches_torch(self):
        norm = SafeBatchNorm1d(8)
        norm.train()
        x = torch.randn(16, 8)
        reference = nn.BatchNorm1d(8)
        reference.train()
        assert torch.allclose(norm(x), reference(x), atol=1e-5)

    def test_running_stats_not_updated_by_degenerate_batch(self):
        norm = SafeBatchNorm1d(4)
        norm.train()
        before_mean = norm.running_mean.clone()
        before_count = int(norm.num_batches_tracked)
        norm(torch.randn(1, 4))
        assert torch.equal(norm.running_mean, before_mean)
        assert int(norm.num_batches_tracked) == before_count

    def test_state_dict_keys_match_batchnorm1d(self):
        """Existing checkpoints must keep loading after the swap."""
        assert sorted(SafeBatchNorm1d(8).state_dict()) == sorted(
            nn.BatchNorm1d(8).state_dict()
        )

    def test_resolve_norm_variants(self):
        assert isinstance(resolve_norm(8, "batch"), SafeBatchNorm1d)
        assert isinstance(resolve_norm(8, "layer"), nn.LayerNorm)
        assert isinstance(resolve_norm(8, "none"), nn.Identity)
        with pytest.raises(ValueError, match="Unknown norm"):
            resolve_norm(8, "banana")

    def test_apply_norm_skips_identity(self):
        x = torch.randn(1, 4)
        assert apply_norm(nn.Identity(), x) is x

    def test_apply_norm_on_plain_batchnorm_is_safe(self):
        """A raw nn.BatchNorm1d must not crash either (defence in depth)."""
        norm = nn.BatchNorm1d(4)
        norm.train()
        assert apply_norm(norm, torch.randn(1, 4)).shape == (1, 4)


# ==========================================================================
# Autoencoder: forward/backward on degenerate graphs
# ==========================================================================

class TestAutoencoderDegenerateBatches:
    """Every encoder previously crashed on at least one degenerate batch."""

    @pytest.mark.parametrize("encoder", ["gcn", "graphsage", "gat", "edgeconv"])
    def test_single_node_graph(self, encoder):
        dim = 16 if encoder == "edgeconv" else 11
        model = get_autoencoder(encoder, input_dim=dim, hidden_dim=16,
                                latent_dim=8, num_layers=2)
        model.train()
        data = Data(x=torch.randn(1, dim),
                    edge_index=torch.zeros((2, 0), dtype=torch.long))
        out = model(Batch.from_data_list([data]))
        assert out["per_graph_loss"].shape == (1,)
        out["loss"].backward()

    def test_edgeconv_single_edge_graph(self):
        """The EdgeConv MLPs normalize over the *edge* axis, so E == 1 crashed."""
        model = get_autoencoder("edgeconv", input_dim=16, hidden_dim=16,
                                latent_dim=8, num_layers=3)
        model.train()
        data = Data(x=torch.randn(2, 16), edge_index=torch.tensor([[0], [1]]))
        out = model(Batch.from_data_list([data]))
        out["loss"].backward()
        assert torch.isfinite(out["per_graph_loss"]).all()

    @pytest.mark.parametrize("cls", ["gcn", "graphsage", "gat", "mlp", "cnn"])
    def test_classifier_batch_of_one(self, cls):
        model = get_classifier(cls, input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        model.train()
        data = Data(x=torch.randn(1, 11),
                    edge_index=torch.zeros((2, 0), dtype=torch.long))
        out = model(Batch.from_data_list([data]))
        out.sum().backward()

    def test_forward_without_x_raises_clear_error(self):
        model = get_autoencoder("gcn", input_dim=11, hidden_dim=8, latent_dim=4,
                                num_layers=2)
        with pytest.raises(ValueError, match="requires data.x"):
            model(Data(edge_index=torch.zeros((2, 0), dtype=torch.long)))

    def test_forward_tolerates_missing_edge_index(self):
        model = get_autoencoder("gcn", input_dim=11, hidden_dim=8, latent_dim=4,
                                num_layers=2)
        model.eval()
        out = model(Data(x=torch.randn(3, 11)))
        assert out["per_graph_loss"].shape == (1,)

    def test_dimension_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="input_dim"):
            GraphAutoencoder(
                GCNEncoder(input_dim=11, hidden_dim=8, latent_dim=4, num_layers=2),
                GraphDecoder(latent_dim=4, hidden_dim=8, output_dim=16),
            )


# ==========================================================================
# Registry
# ==========================================================================

class TestRegistry:
    def test_known_models(self):
        assert set(available_classifiers()) == {"cnn", "gat", "gcn", "graphsage", "mlp"}
        assert set(available_encoders()) == {"edgeconv", "gat", "gcn", "graphsage"}

    def test_unknown_names_raise(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            get_classifier("nope")
        with pytest.raises(ValueError, match="Unknown encoder"):
            get_encoder("nope")
        with pytest.raises(ValueError, match="Unknown encoder"):
            get_autoencoder("nope")

    def test_get_autoencoder_propagates_dims(self):
        ae = get_autoencoder("gcn", input_dim=16, hidden_dim=24, latent_dim=12,
                             num_layers=2)
        assert ae.encoder.input_dim == 16
        assert ae.decoder.output_dim == 16
        assert ae.encoder.latent_dim == 12

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "8"])
    def test_get_autoencoder_rejects_bad_dims(self, bad):
        with pytest.raises(ValueError, match="positive int"):
            get_autoencoder("gcn", input_dim=bad)

    def test_edgeconv_absent_from_classifiers(self):
        """Documented: EdgeConv is encoder-only, paired via get_autoencoder."""
        assert "edgeconv" not in CLASSIFIERS
        assert "edgeconv" in ENCODERS


# ==========================================================================
# Baselines
# ==========================================================================

class TestMLPClassifier:
    def test_output_shape(self, graph_maker):
        model = MLPClassifier(input_dim=11, hidden_dim=16)
        data = [graph_maker(n_nodes=4, label=i % 2, seed=i) for i in range(6)]
        out = model(Batch.from_data_list(data))
        assert out.shape == (6, 2)

    def test_missing_x_raises(self):
        with pytest.raises(ValueError, match="requires data.x"):
            MLPClassifier(input_dim=11)(Data(edge_index=torch.zeros((2, 0))))


class TestCNNClassifier:
    def test_max_particles_default_covers_jetclass(self):
        """Was 50, which silently truncated most real JetClass jets (128)."""
        assert CNNClassifier(input_dim=16).max_particles == 128

    def test_rejects_bad_max_particles(self):
        with pytest.raises(ValueError, match="max_particles"):
            CNNClassifier(max_particles=0)

    def test_empty_input(self):
        model = CNNClassifier(input_dim=11, hidden_dim=16)
        out = model(Data(x=torch.zeros((0, 11)),
                         batch=torch.zeros((0,), dtype=torch.long)))
        assert out.shape == (0, 2)

    def test_disjoint_node_counts(self):
        model = CNNClassifier(input_dim=11, hidden_dim=16)
        data = Data(x=torch.randn(3, 11),
                    batch=torch.tensor([0, 0, 1], dtype=torch.long))
        assert model(data).shape == (2, 2)

    def test_padding_does_not_change_prediction(self):
        """The model's own zero-padding must not affect the prediction.

        The previous implementation used ``AdaptiveAvgPool1d(1)`` over the
        whole padded sequence, so the same graph scored differently depending
        on ``max_particles`` - short jets were diluted by their own padding.
        With masked pooling, changing the padding width is a no-op.

        Note this is about the model's *internal* padding, not about feeding in
        extra zero particles (which is genuinely a different graph).
        """
        torch.manual_seed(0)
        small = CNNClassifier(input_dim=11, hidden_dim=8, max_particles=8)
        large = CNNClassifier(input_dim=11, hidden_dim=8, max_particles=64)
        large.load_state_dict(small.state_dict())
        small.eval()
        large.eval()

        graphs = [
            Data(x=torch.randn(3, 11), batch=torch.tensor([0, 0, 0], dtype=torch.long)),
            Data(x=torch.randn(5, 11),
                 batch=torch.tensor([0] * 5, dtype=torch.long)),
        ]
        for g in graphs:
            assert torch.allclose(small(g), large(g), atol=1e-5), (
                "prediction changed with padding width - masking is broken"
            )

    def test_sorts_by_descending_pt(self):
        model = CNNClassifier(input_dim=11, hidden_dim=8)
        x = torch.zeros(3, 11)
        x[0, 5], x[1, 5], x[2, 5] = 1.0, 100.0, 50.0  # pT at col 5 (synthetic layout)
        _, mask = model._pad_and_sort(x, torch.zeros(3, dtype=torch.long), 1)
        # Highest pT must land in slot 0.
        assert mask[0, 0]
        assert mask[0, 2]
        assert not mask[0, 3]

    def test_pt_column_selection(self):
        from anomaly_engine.models.baselines import pt_column_index

        assert pt_column_index(16) == 0   # JetClass: px
        assert pt_column_index(11) == 0   # JetClass-compatible
        assert pt_column_index(3) == 0


# ==========================================================================
# Encoder configuration validation
# ==========================================================================

class TestEncoderValidation:
    def test_gat_rejects_indivisible_heads(self):
        with pytest.raises(ValueError, match="divisible by heads"):
            from anomaly_engine.models.gat import GATEncoder

            GATEncoder(hidden_dim=10, heads=4)

    def test_gat_hidden_dim_divisible(self):
        from anomaly_engine.models.gat import GATEncoder

        enc = GATEncoder(hidden_dim=32, heads=4, num_layers=2)
        assert enc(torch.randn(6, 11), torch.tensor([[0, 1], [1, 2]])).shape == (6, 32)

    @pytest.mark.parametrize("cls", [GCNEncoder, EdgeConvEncoder])
    def test_encoder_rejects_too_few_layers(self, cls):
        with pytest.raises(ValueError, match="num_layers"):
            cls(input_dim=11, hidden_dim=8, latent_dim=4, num_layers=0)

    def test_edgeconv_norm_none_disables_norm(self):
        enc = EdgeConvEncoder(input_dim=16, hidden_dim=8, latent_dim=4,
                              num_layers=2, norm="none")
        assert any(isinstance(m, nn.Identity) for m in enc.norms)

    def test_edgeconv_rejects_bad_norm(self):
        with pytest.raises(ValueError, match="Unknown norm"):
            EdgeConvEncoder(input_dim=16, hidden_dim=8, latent_dim=4,
                            num_layers=2, norm="banana")

    def test_encoder_encode_graph_without_batch(self):
        enc = GCNEncoder(input_dim=11, hidden_dim=8, latent_dim=4, num_layers=2)
        emb = enc.encode_graph(torch.randn(4, 11),
                               torch.tensor([[0, 1], [1, 2]]), None)
        assert emb.shape == (1, 4)
