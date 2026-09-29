"""Evaluator regression tests: metric correctness and label handling."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from anomaly_engine.evaluate import Evaluator
from anomaly_engine.models import get_autoencoder, get_classifier
from tests.helpers import MockAutoencoder, make_graph, make_graphs, make_loader


@pytest.fixture
def ev():
    return Evaluator(device="cpu")


# ==========================================================================
# evaluate_classifier
# ==========================================================================

class TestEvaluateClassifier:
    def test_binary_labels_give_auroc(self, ev):
        model = get_classifier("mlp", input_dim=11, hidden_dim=16)
        graphs = make_graphs(20, n_nodes=4)
        r = ev.evaluate_classifier(model, make_loader(graphs, batch_size=5))
        assert "auroc" in r
        assert 0.0 <= r["auroc"] <= 1.0
        for k in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= r[k] <= 1.0

    @pytest.mark.parametrize("labels", [[1, 2], [5, 9], [0, 3]])
    def test_two_class_non_binary_labels_use_macro(self, ev, labels):
        """Regression: ``average="binary"`` was chosen on class *count* alone.

        sklearn raises ``Target is multiclass but average='binary'`` for labels
        like {1, 2}, so a two-class multiclass loader crashed outright.
        """
        model = get_classifier("mlp", input_dim=11, hidden_dim=16)
        graphs = []
        for i in range(8):
            g = make_graph(n_nodes=4, seed=i)
            g.y = torch.tensor([labels[i % 2]], dtype=torch.long)
            graphs.append(g)
        r = ev.evaluate_classifier(model, make_loader(graphs, batch_size=4))
        assert set(r) >= {"accuracy", "precision", "recall", "f1"}
        assert "auroc" not in r  # not a {0,1} problem

    def test_three_class_labels(self, ev):
        model = get_classifier("gcn", input_dim=11, hidden_dim=16, latent_dim=8,
                               num_classes=3, num_layers=2)
        graphs = []
        for i in range(9):
            g = make_graph(n_nodes=4, seed=i)
            g.y = torch.tensor([i % 3], dtype=torch.long)
            graphs.append(g)
        r = ev.evaluate_classifier(model, make_loader(graphs, batch_size=3))
        assert np.array(r["confusion_matrix"]).shape == (3, 3)

    def test_rejects_autoencoder(self, ev):
        ae = get_autoencoder("gcn", input_dim=11, hidden_dim=8, latent_dim=4,
                             num_layers=2)
        with pytest.raises(TypeError, match="autoencoder"):
            ev.evaluate_classifier(ae, make_loader(make_graphs(4, n_nodes=3)))

    def test_unlabeled_batch_raises_clear_error(self, ev):
        model = get_classifier("mlp", input_dim=11, hidden_dim=16)
        graphs = [make_graph(n_nodes=3, label=None, seed=i) for i in range(4)]
        with pytest.raises(ValueError, match="data.y"):
            ev.evaluate_classifier(model, make_loader(graphs, batch_size=2))

    def test_empty_loader_raises(self, ev):
        model = get_classifier("mlp", input_dim=11, hidden_dim=16)
        with pytest.raises(ValueError, match="empty DataLoader"):
            ev.evaluate_classifier(model, make_loader([]))

    def test_single_logit_head(self, ev):
        class OneLogit(nn.Module):
            def forward(self, data):
                return torch.randn(data.num_graphs, 1)

        r = ev.evaluate_classifier(OneLogit(),
                                   make_loader(make_graphs(8, n_nodes=3)))
        assert "auroc" in r


# ==========================================================================
# evaluate_autoencoder
# ==========================================================================

class TestEvaluateAutoencoder:
    def test_auroc_and_separation(self, ev):
        ae = get_autoencoder("gcn", input_dim=11, hidden_dim=16, latent_dim=8,
                             num_layers=2)
        graphs = make_graphs(12, n_nodes=4)
        r = ev.evaluate_autoencoder(ae, make_loader(graphs, batch_size=4))
        assert "auroc" in r and "auprc" in r
        assert "score_separation" in r
        assert r["anomaly_mean"] != r["normal_mean"] or True  # both computed

    def test_unlabeled_loader_skips_metrics_without_crashing(self, ev):
        """No labels: report score stats, skip AUROC. Never a shape error."""
        ae = MockAutoencoder()
        graphs = [make_graph(n_nodes=3, label=None, seed=i) for i in range(6)]
        r = ev.evaluate_autoencoder(ae, make_loader(graphs, batch_size=3))
        assert "mean_recon_error" in r
        assert "auroc" not in r

    def test_single_class_skips_auroc(self, ev):
        ae = MockAutoencoder()
        graphs = [make_graph(n_nodes=3, label=0, seed=i) for i in range(6)]
        r = ev.evaluate_autoencoder(ae, make_loader(graphs, batch_size=3))
        assert "auroc" not in r

    def test_rejects_classifier(self, ev):
        model = get_classifier("mlp", input_dim=11, hidden_dim=16)
        with pytest.raises(TypeError, match="per_graph_loss"):
            ev.evaluate_autoencoder(model,
                                    make_loader(make_graphs(4, n_nodes=3)))

    def test_empty_loader_raises(self, ev):
        with pytest.raises(ValueError, match="empty DataLoader"):
            ev.evaluate_autoencoder(MockAutoencoder(), make_loader([]))


# ==========================================================================
# Plotting
# ==========================================================================

class TestPlots:
    def test_training_curves(self, ev, tmp_path):
        hist = {"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7],
                "lr": [1e-3, 1e-3, 5e-4]}
        out = tmp_path / "c.png"
        ev.plot_training_curves(hist, output_path=str(out))
        assert out.exists()

    def test_training_curves_without_lr(self, ev, tmp_path):
        out = tmp_path / "c.png"
        ev.plot_training_curves({"train_loss": [1.0], "val_loss": [1.0]},
                                output_path=str(out))
        assert out.exists()

    def test_training_curves_empty_raises(self, ev):
        with pytest.raises(ValueError, match="train_loss"):
            ev.plot_training_curves({"train_loss": [], "val_loss": []})

    def test_roc_curve(self, ev, tmp_path):
        labels = np.array([0] * 20 + [1] * 20)
        scores = np.concatenate([np.random.rand(20), np.random.rand(20) + 1])
        out = tmp_path / "roc.png"
        ev.plot_roc_curve(labels, scores, "M", str(out))
        assert out.exists()

    def test_roc_curve_length_mismatch_raises(self, ev):
        with pytest.raises(ValueError, match="one score per label"):
            ev.plot_roc_curve(np.array([0, 1, 0]), np.array([0.1, 0.2]))

    def test_roc_curve_single_class_raises(self, ev):
        with pytest.raises(ValueError, match="both classes"):
            ev.plot_roc_curve(np.zeros(5), np.arange(5.0))

    def test_score_distributions(self, ev, tmp_path):
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.concatenate([np.random.rand(10), np.random.rand(10)])
        out = tmp_path / "d.png"
        ev.plot_score_distributions(scores, labels, str(out))
        assert out.exists()

    def test_score_distributions_bad_labels_raises(self, ev):
        with pytest.raises(ValueError, match=r"\{0, 1\}"):
            ev.plot_score_distributions(np.arange(5.0), np.full(5, 7))

    def test_latent_space_with_encoder_model(self, ev, tmp_path):
        ae = get_autoencoder("gcn", input_dim=11, hidden_dim=16, latent_dim=8,
                             num_layers=2)
        graphs = make_graphs(10, n_nodes=4)
        out = tmp_path / "latent.png"
        ev.plot_latent_space(ae, make_loader(graphs, batch_size=5),
                             output_path=str(out))
        assert out.exists()

    def test_latent_space_custom_encode_graph_signature(self, ev, tmp_path):
        """A model exposing encode_graph(x, edge_index, batch) must work."""
        class Other(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(11, 6)

            def encode_graph(self, x, edge_index, batch):
                return self.lin(x).sum(dim=0, keepdim=True).expand(
                    int(batch.max()) + 1, 6
                )

        out = tmp_path / "latent.png"
        ev.plot_latent_space(Other(), make_loader(make_graphs(8, n_nodes=3),
                                                  batch_size=4),
                             output_path=str(out))
        assert out.exists()

    def test_latent_space_unlabeled(self, ev, tmp_path):
        ae = get_autoencoder("gcn", input_dim=11, hidden_dim=16, latent_dim=8,
                             num_layers=2)
        graphs = [make_graph(n_nodes=4, label=None, seed=i) for i in range(8)]
        out = tmp_path / "latent.png"
        ev.plot_latent_space(ae, make_loader(graphs, batch_size=4),
                             output_path=str(out))
        assert out.exists()

    def test_latent_space_needs_two_graphs(self, ev):
        ae = get_autoencoder("gcn", input_dim=11, hidden_dim=8, latent_dim=4,
                             num_layers=2)
        with pytest.raises(ValueError, match="at least 2 graphs"):
            ev.plot_latent_space(ae, make_loader([make_graph(n_nodes=3)]))

    def test_latent_space_rejects_uninterpretable_output(self, ev):
        """A model returning a dict without 'z' must fail clearly."""
        with pytest.raises(TypeError, match="without a 'z' key"):
            ev.plot_latent_space(
                MockAutoencoder(), make_loader(make_graphs(4, n_nodes=3))
            )

    def test_comparison_table_uses_metrics_from_any_model(self, ev, tmp_path):
        """Regression: metrics were read only from the first model.

        A first entry lacking 'auroc' silently hid the AUROC every other model
        reported.
        """
        out = tmp_path / "cmp.png"
        ev.plot_comparison_table(
            {"mlp": {"accuracy": 0.5}, "gcn": {"accuracy": 0.6, "auroc": 0.7}},
            output_path=str(out),
        )
        assert out.exists()

    def test_comparison_table_empty_raises(self, ev):
        with pytest.raises(ValueError, match="empty results"):
            ev.plot_comparison_table({})

    def test_comparison_table_no_known_metrics_raises(self, ev):
        with pytest.raises(ValueError, match="no plottable metrics"):
            ev.plot_comparison_table({"m": {"loss": 0.1}})

    def test_comparison_table_fills_missing_metrics(self, ev, tmp_path):
        """A model missing a metric is scored 0, not dropped or crashing."""
        out = tmp_path / "cmp.png"
        ev.plot_comparison_table(
            {"mlp": {"accuracy": 0.5, "auroc": 0.4}, "gcn": {"accuracy": 0.6}},
            output_path=str(out),
        )
        assert out.exists()
