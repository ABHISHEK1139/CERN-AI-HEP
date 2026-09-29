"""AnomalyScorer regression tests."""

import numpy as np
import pytest
import torch

from anomaly_engine.anomaly_scorer import AnomalyScorer
from anomaly_engine.models import get_autoencoder
from tests.helpers import MockAutoencoder, make_graph, make_graphs, make_loader


@pytest.fixture
def scorer():
    return AnomalyScorer(MockAutoencoder(), device="cpu")


SCORES = np.array([0.1, 0.5, 0.9, 1.2, 2.0, 3.0])
LABELS = np.array([0, 0, 1, 1, 1, 1])
IDS = np.arange(100, 106)


class TestScoreDataset:
    def test_scores_labels_ids_align(self, scorer):
        graphs = make_graphs(6, n_nodes=4)
        for i, g in enumerate(graphs):
            g.event_id = 100 + i
        s, l, e = scorer.score_dataset(make_loader(graphs, batch_size=2))
        assert len(s) == len(l) == len(e) == 6
        assert e.tolist() == list(range(100, 106))

    def test_event_ids_do_not_reset_per_batch(self, scorer):
        """Without a per-batch offset the ids repeated, breaking ranking lookups."""
        graphs = make_graphs(6, n_nodes=4)
        for i, g in enumerate(graphs):
            g.event_id = 100 + i
        _, _, e = scorer.score_dataset(make_loader(graphs, batch_size=2))
        assert len(set(e.tolist())) == 6, "event ids repeated across batches"

    def test_graphs_without_event_id_get_positional_ids(self, scorer):
        graphs = make_graphs(6, n_nodes=4)
        s, _, e = scorer.score_dataset(make_loader(graphs, batch_size=2))
        assert e.tolist() == list(range(6))
        assert len(e) == len(s)

    def test_unlabeled_dataset(self, scorer):
        graphs = [make_graph(n_nodes=4, label=None, seed=i) for i in range(4)]
        s, l, e = scorer.score_dataset(make_loader(graphs, batch_size=2))
        assert len(s) == 4
        assert l.size == 0

    def test_rejects_classifier(self):
        from anomaly_engine.models import get_classifier

        s = AnomalyScorer(get_classifier("mlp", input_dim=11, hidden_dim=16),
                          device="cpu")
        with pytest.raises(TypeError, match="per_graph_loss"):
            s.score_dataset(make_loader(make_graphs(4, n_nodes=3)))


class TestRankAnomalies:
    def test_highest_first(self, scorer):
        ranked = scorer.rank_anomalies(SCORES, IDS, top_k=3)
        assert [r["anomaly_score"] for r in ranked] == [3.0, 2.0, 1.2]
        assert [r["rank"] for r in ranked] == [1, 2, 3]
        assert ranked[0]["event_id"] == 105

    @pytest.mark.parametrize("top_k", [0, -5])
    def test_non_positive_top_k_returns_empty(self, scorer, top_k):
        assert scorer.rank_anomalies(SCORES, IDS, top_k=top_k) == []

    def test_top_k_larger_than_dataset(self, scorer):
        assert len(scorer.rank_anomalies(SCORES, IDS, top_k=999)) == len(SCORES)

    def test_length_mismatch_raises(self, scorer):
        with pytest.raises(ValueError, match="one event_id per score"):
            scorer.rank_anomalies(SCORES, IDS[:3], top_k=2)

    def test_ties_are_deterministic(self, scorer):
        tied = np.array([1.0, 1.0, 1.0, 1.0])
        first = [r["event_id"] for r in scorer.rank_anomalies(tied, IDS[:4], 4)]
        second = [r["event_id"] for r in scorer.rank_anomalies(tied, IDS[:4], 4)]
        assert first == second


class TestSelectThreshold:
    def test_percentile(self, scorer):
        t = scorer.select_threshold(SCORES, method="percentile", percentile=95.0)
        assert np.isclose(t, np.percentile(SCORES, 95))

    def test_sigma(self, scorer):
        t = scorer.select_threshold(SCORES, method="sigma", n_sigma=2.0)
        assert np.isclose(t, SCORES.mean() + 2 * SCORES.std())

    def test_empty_raises(self, scorer):
        with pytest.raises(ValueError, match="empty scores"):
            scorer.select_threshold(np.array([]))

    def test_unknown_method_raises(self, scorer):
        with pytest.raises(ValueError, match="Unknown method"):
            scorer.select_threshold(SCORES, method="bogus")

    @pytest.mark.parametrize("bad", [-1, 101])
    def test_percentile_out_of_range_raises(self, scorer, bad):
        with pytest.raises(ValueError, match="percentile"):
            scorer.select_threshold(SCORES, percentile=bad)

    def test_negative_sigma_raises(self, scorer):
        with pytest.raises(ValueError, match="n_sigma"):
            scorer.select_threshold(SCORES, method="sigma", n_sigma=-1)

    def test_returns_python_float(self, scorer):
        assert isinstance(scorer.select_threshold(SCORES), float)


class TestGenerateReport:
    def test_full_report(self, scorer):
        rep = scorer.generate_report(SCORES, LABELS, IDS, top_k=3)
        assert rep["n_events"] == 6
        assert set(rep["thresholds"]) == {"p95", "p99", "3sigma"}
        assert "auroc" in rep and "auprc" in rep
        assert 0.0 <= rep["auroc"] <= 1.0
        assert rep["top_k"] == 3
        assert 0.0 <= rep["top_k_precision"] <= 1.0

    def test_top_k_zero_gives_none_not_nan(self, scorer, recwarn):
        """Regression: ``.mean()`` over an empty slice returned NaN + warning.

        ``filterwarnings = error::RuntimeWarning`` in pyproject means the old
        code raised outright.
        """
        rep = scorer.generate_report(SCORES, LABELS, IDS, top_k=0)
        assert rep["top_k"] == 0
        assert rep["top_k_precision"] is None
        assert rep["n_anomalies_in_top_k"] == 0
        assert rep["top_anomalies"] == []

    def test_top_k_larger_than_dataset_is_clamped(self, scorer):
        rep = scorer.generate_report(SCORES, LABELS, IDS, top_k=1000)
        assert rep["top_k"] == len(SCORES)

    def test_single_class_omits_auroc(self, scorer):
        rep = scorer.generate_report(SCORES, np.zeros(6, dtype=int), IDS)
        assert "auroc" not in rep
        assert "top_k_precision" in rep

    def test_misaligned_labels_are_reported_not_scored(self, scorer):
        """Partial labels must not produce a bogus AUROC."""
        rep = scorer.generate_report(SCORES, LABELS[:3], IDS)
        assert "auroc" not in rep
        assert rep["n_events"] == 6

    def test_event_id_length_mismatch_raises(self, scorer):
        with pytest.raises(ValueError, match="one event_id per score"):
            scorer.generate_report(SCORES, LABELS, IDS[:2])

    def test_empty_scores_raises(self, scorer):
        with pytest.raises(ValueError, match="empty scores"):
            scorer.generate_report(np.array([]), np.array([]), np.array([]))

    def test_print_report_runs(self, scorer, capsys):
        rep = scorer.generate_report(SCORES, LABELS, IDS, top_k=2)
        scorer.print_report(rep)
        out = capsys.readouterr().out
        assert "ANOMALY DETECTION REPORT" in out
        assert "AUROC" in out

    def test_print_report_without_labels(self, scorer, capsys):
        rep = scorer.generate_report(SCORES, np.array([]), IDS)
        scorer.print_report(rep)
        assert "unavailable" in capsys.readouterr().out


class TestEndToEnd:
    def test_scorer_with_real_autoencoder(self):
        ae = get_autoencoder("gcn", input_dim=11, hidden_dim=16, latent_dim=8,
                             num_layers=2)
        scorer = AnomalyScorer(ae, device="cpu")
        graphs = make_graphs(10, n_nodes=4)
        s, l, e = scorer.score_dataset(make_loader(graphs, batch_size=5))
        assert len(s) == 10
        assert len(s) == len(l) == len(e)
        rep = scorer.generate_report(s, l, e, top_k=3)
        assert rep["n_events"] == 10
        assert 0.0 <= rep["auroc"] <= 1.0

    def test_scores_are_finite(self):
        ae = get_autoencoder("edgeconv", input_dim=16, hidden_dim=16,
                             latent_dim=8, num_layers=2)
        graphs = [make_graph(n_nodes=6, feature_dim=16, seed=i) for i in range(6)]
        scorer = AnomalyScorer(ae, device="cpu")
        s, _, _ = scorer.score_dataset(make_loader(graphs, batch_size=3))
        assert np.isfinite(s).all()
