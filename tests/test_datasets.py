"""Dataset, splitting, and graph-construction regression tests."""

import numpy as np
import pytest
import torch

from graph_builder.dataset import CollisionEventDataset
from graph_builder.graph_constructor import EventGraphConstructor
from graph_builder.splitting import (
    DEFAULT_SEED,
    SplittableDatasetMixin,
    collate_subset,
    split_indices,
    validate_ratios,
)
from tests.helpers import make_graph, make_graphs

# ==========================================================================
# Split ratio validation
# ==========================================================================

class TestValidateRatios:
    @pytest.mark.parametrize("ratios", [
        (0.7, 0.15, 0.5),
        (0.5, 0.5, 0.5),
        (0.3, 0.3, 0.3),
        (0.7, 0.15, 0.149),
    ])
    def test_sums_to_one_required(self, ratios):
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_ratios(*ratios)

    @pytest.mark.parametrize("ratios", [
        (-0.1, 0.6, 0.5),
        (0.5, -0.2, 0.7),
    ])
    def test_negative_rejected(self, ratios):
        with pytest.raises(ValueError, match=">= 0"):
            validate_ratios(*ratios)

    def test_zero_ratio_allowed(self):
        validate_ratios(0.5, 0.5, 0.0)

    @pytest.mark.parametrize("ratios", [(0.7, 0.15, 0.15), (1.0, 0.0, 0.0),
                                        (0.0, 0.0, 1.0)])
    def test_valid_combinations(self, ratios):
        validate_ratios(*ratios)

    def test_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_ratios("0.7", 0.15, 0.15)

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_ratios(True, 0.5, 0.5)

    def test_validation_survives_python_optimized_mode(self):
        """Regression: this used ``assert``, which ``python -O`` strips out.

        Under -O the guard vanished and invalid ratios silently produced
        garbage splits. Checked two ways: the guard must not be an ``assert``
        statement, and it must actually raise.
        """
        import inspect

        source = inspect.getsource(validate_ratios)
        assert "assert " not in source, (
            "validate_ratios must not use assert; it is stripped under -O"
        )
        with pytest.raises(ValueError):
            validate_ratios(0.5, 0.5, 0.5)


class TestSplitIndices:
    def test_disjoint_and_exhaustive(self):
        tr, va, te = split_indices(100, 0.7, 0.15, 0.15)
        allidx = set(tr) | set(va) | set(te)
        assert len(allidx) == 100
        assert not (set(tr) & set(va))
        assert not (set(tr) & set(te))
        assert not (set(va) & set(te))
        assert (len(tr), len(va), len(te)) == (70, 15, 15)

    def test_deterministic_for_same_seed(self):
        a = split_indices(50, 0.6, 0.2, 0.2, seed=7)
        b = split_indices(50, 0.6, 0.2, 0.2, seed=7)
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)

    def test_different_seed_differs(self):
        a = split_indices(50, 0.6, 0.2, 0.2, seed=1)
        b = split_indices(50, 0.6, 0.2, 0.2, seed=2)
        assert not np.array_equal(a[0], b[0])

    def test_does_not_disturb_global_numpy_seed(self):
        np.random.seed(1234)
        before = np.random.rand()
        np.random.seed(1234)
        split_indices(10, 0.5, 0.25, 0.25)
        assert np.random.rand() == before

    def test_zero_items(self):
        tr, va, te = split_indices(0, 0.7, 0.15, 0.15)
        assert len(tr) == len(va) == len(te) == 0


# ==========================================================================
# CollisionEventDataset
# ==========================================================================

class TestCollisionEventDataset:
    def test_explicit_graphs_win_over_stale_cache(self, tmp_path):
        """Regression: a cached ``processed/dataset.pt`` silently overrode
        whatever graphs the caller passed in.

        That made the synthetic-regeneration path in train_classifier.py and
        run_benchmark.py a no-op: new graphs were generated, then the old
        collated dataset was served instead.
        """
        CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(50))
        fresh = make_graphs(7, seed=999, label=1)
        ds = CollisionEventDataset(root=str(tmp_path), graphs=fresh)
        assert len(ds) == 7
        assert {int(ds.get(i).y) for i in range(len(ds))} == {1}

    def test_cache_used_when_no_graphs_given(self, tmp_path):
        CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(11))
        ds = CollisionEventDataset(root=str(tmp_path))
        assert len(ds) == 11

    def test_empty_graph_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty graphs list"):
            CollisionEventDataset(root=str(tmp_path), graphs=[])

    def test_missing_data_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No graphs found"):
            CollisionEventDataset(root=str(tmp_path / "empty"))

    def test_index_select_empty_raises(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(4))
        with pytest.raises(ValueError, match="empty indices"):
            ds.index_select([])

    def test_index_select_subset_is_independent(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(20))
        sub = ds.index_select([0, 1, 2])
        assert len(sub) == 3
        assert isinstance(sub, CollisionEventDataset)
        assert len(ds) == 20  # parent untouched

    def test_splits_sum_to_total_and_are_disjoint(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(40))
        tr, va, te = ds.get_splits(0.7, 0.15, 0.15)
        assert len(tr) + len(va) + len(te) == 40

        def keys(d):
            return {tuple(d.get(i).x.flatten().tolist()) for i in range(len(d))}

        kt, kv, kte = keys(tr), keys(va), keys(te)
        assert not (kt & kv)
        assert not (kt & kte)
        assert not (kv & kte)

    def test_get_loaders(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(30))
        tr, va, te = ds.get_loaders(batch_size=8)
        assert sum(len(l.dataset) for l in (tr, va, te)) == 30
        batch = next(iter(tr))
        assert batch.num_graphs == 8

    def test_get_loaders_rejects_bad_batch_size(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(10))
        with pytest.raises(ValueError, match="batch_size"):
            ds.get_loaders(batch_size=0)

    def test_get_loaders_rejects_negative_workers(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(10))
        with pytest.raises(ValueError, match="num_workers"):
            ds.get_loaders(num_workers=-1)

    def test_uses_shared_mixin(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(4))
        assert isinstance(ds, SplittableDatasetMixin)

    def test_get_stats(self, tmp_path):
        ds = CollisionEventDataset(root=str(tmp_path), graphs=make_graphs(12))
        stats = ds.get_stats()
        assert stats["n_graphs"] == 12
        assert stats["node_feature_dim"] == 11
        assert stats["label_distribution"]["normal"] == 6
        assert stats["label_distribution"]["anomaly"] == 6
        assert stats["nodes_per_graph"]["mean"] == 5.0

    def test_get_stats_handles_unlabeled_graphs(self, tmp_path):
        ds = CollisionEventDataset(
            root=str(tmp_path),
            graphs=[make_graph(n_nodes=4, label=None, seed=i) for i in range(3)],
        )
        stats = ds.get_stats()
        assert stats["n_graphs"] == 3
        assert stats["label_distribution"] == {"normal": 0, "anomaly": 0}


# ==========================================================================
# EventGraphConstructor
# ==========================================================================

def _event(eid, pts, etas=None, phis=None):
    n = len(pts)
    return {
        "event_id": eid,
        "particles": [
            {
                "node_type": i % 5,
                "pt": pts[i],
                "eta": (etas or [0.1 * i for i in range(n)])[i],
                "phi": (phis or [0.2 * i for i in range(n)])[i],
                "mass": 1.0,
                "charge": 1.0 if i % 2 else -1.0,
                "energy": pts[i],
            }
            for i in range(n)
        ],
    }


class TestEventGraphConstructor:
    def test_knn_edges_no_self_loops(self):
        c = EventGraphConstructor(strategy="knn", k=3)
        g = c.event_to_graph(_event(0, [10.0, 20.0, 30.0, 40.0, 50.0]))
        src, dst = g.edge_index
        assert not torch.any(src == dst)

    def test_knn_respects_k(self):
        c = EventGraphConstructor(strategy="knn", k=2)
        g = c.event_to_graph(_event(0, [float(i) for i in range(1, 9)]))
        # Bidirectional, at most k out-edges per node.
        n = g.num_nodes
        for i in range(n):
            assert int((g.edge_index[0] == i).sum()) <= 2 * 2

    def test_too_few_particles_returns_none(self):
        c = EventGraphConstructor(strategy="knn", k=4)
        assert c.event_to_graph(_event(0, [10.0])) is None
        assert c.event_to_graph(_event(1, [])) is None

    def test_missing_particles_raises(self):
        c = EventGraphConstructor()
        with pytest.raises(KeyError, match="particles"):
            c.event_to_graph({"event_id": 0})

    def test_unknown_strategy_raises(self):
        c = EventGraphConstructor(strategy="banana")
        with pytest.raises(ValueError, match="Unknown edge strategy"):
            c.event_to_graph(_event(0, [1.0, 2.0, 3.0]))

    def test_zero_k_raises(self):
        c = EventGraphConstructor(strategy="knn", k=0)
        with pytest.raises(ValueError, match="k must be >= 1"):
            c.event_to_graph(_event(0, [1.0, 2.0, 3.0]))

    def test_fully_connected(self):
        c = EventGraphConstructor(strategy="fully_connected", k=1)
        g = c.event_to_graph(_event(0, [1.0, 2.0, 3.0, 4.0]))
        n = g.num_nodes
        assert g.edge_index.shape[1] == n * (n - 1)

    def test_delta_r_falls_back_to_knn(self):
        c = EventGraphConstructor(strategy="delta_r", k=4, delta_r_threshold=1e-9)
        g = c.event_to_graph(_event(0, [1.0, 2.0, 3.0, 4.0]))
        assert g.edge_index.shape[1] > 0

    def test_node_feature_dim(self):
        c = EventGraphConstructor(strategy="knn", k=2)
        g = c.event_to_graph(_event(0, [1.0, 2.0, 3.0]))
        assert g.x.shape == (3, 11)

    def test_label_from_argument_and_event(self):
        c = EventGraphConstructor(strategy="knn", k=2)
        g = c.event_to_graph(_event(0, [1.0, 2.0, 3.0]), label=1)
        assert int(g.y) == 1
        ev = _event(1, [1.0, 2.0, 3.0])
        ev["is_anomaly"] = True
        assert int(c.event_to_graph(ev).y) == 1

    def test_event_id_is_int(self):
        c = EventGraphConstructor(strategy="knn", k=2)
        assert isinstance(c.event_to_graph(_event(7, [1.0, 2.0])).event_id, int)

    def test_edge_features_present(self):
        c = EventGraphConstructor(strategy="knn", k=2, include_edge_features=True)
        g = c.event_to_graph(_event(0, [1.0, 2.0, 3.0, 4.0]))
        assert g.edge_attr is not None
        assert g.edge_attr.shape[1] == 4

    def test_convert_dataset_keeps_labels_in_sync(self):
        """Skipped events must not desynchronize graphs from labels."""
        c = EventGraphConstructor(strategy="knn", k=2)
        events = [_event(0, [1.0, 2.0, 3.0]), _event(1, []), _event(2, [4.0, 5.0])]
        labels = np.array([10, 20, 30])
        graphs = c.convert_dataset(events, labels)
        assert len(graphs) == 2
        assert [int(g.y) for g in graphs] == [10, 30]

    def test_save_dataset_writes_graphs_and_labels(self, tmp_path):
        c = EventGraphConstructor(strategy="knn", k=2)
        events = [_event(0, [1.0, 2.0, 3.0]), _event(1, [4.0, 5.0, 6.0])]
        c.save_dataset(events, np.array([0, 1]), tmp_path)
        assert (tmp_path / "graphs.pt").exists()
        assert (tmp_path / "labels.pt").exists()
        labels = torch.load(tmp_path / "labels.pt", weights_only=False)
        assert labels.tolist() == [0, 1]

    def test_save_dataset_handles_all_skipped(self, tmp_path):
        c = EventGraphConstructor(strategy="knn", k=2)
        assert c.save_dataset([_event(0, [])], np.array([0]), tmp_path) == []
        assert not (tmp_path / "graphs.pt").exists()
