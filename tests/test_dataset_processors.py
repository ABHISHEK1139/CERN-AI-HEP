"""
Dataset processor tests using synthetic ROOT/HDF5 files.

``uproot.recreate`` and ``pandas.to_hdf`` can write the real formats, so the
CMS / JetClass / LHCO processing paths are exercised here without any download.
Before this module those four classes had no coverage at all, which is exactly
where the interesting bugs live (branch handling, ragged-array trimming,
k-NN edge construction, label derivation).
"""

import awkward as ak
import numpy as np
import pytest
import torch
import uproot

from graph_builder.cms_dataset import CMSDataset
from graph_builder.jetclass_dataset import (
    LABEL_BRANCHES,
    PARTICLE_FEATURES,
    JetClassDataset,
)
from graph_builder.jetclass_iterable import JetClassIterableDataset
from graph_builder.lhco_dataset import LHCO_INPUT_DIM, LHCODataset

N_CONS = 6


def _jetclass_records(n_jets, qcd_flags, n_cons=N_CONS):
    """Build ragged JetClass-shaped records."""
    rec = {}
    for name in PARTICLE_FEATURES:
        cols = []
        for _ in range(n_jets):
            cols.append([float(i + 1) for i in range(n_cons)])
        rec[name] = ak.Array(cols)
    for name in LABEL_BRANCHES:
        vals = []
        for i in range(n_jets):
            if name == "label_QCD":
                vals.append(1.0 if qcd_flags[i] else 0.0)
            else:
                vals.append(0.0 if qcd_flags[i] else 1.0)
        rec[name] = ak.Array(vals)
    return rec


@pytest.fixture
def jetclass_root(tmp_path):
    path = tmp_path / "ZJetsToNuNu_0.root"
    with uproot.recreate(path) as f:
        f["tree"] = _jetclass_records(12, [True] * 8 + [False] * 4)
    return str(path)


@pytest.fixture
def cms_root(tmp_path):
    """3 jets per event, all above the 25 GeV Jet pT cut in EventConfig."""
    path = tmp_path / "TTbar.root"
    with uproot.recreate(path) as f:
        f["Events"] = {
            "Jet_pt": ak.Array([[50.0, 60.0, 45.0], [30.0, 80.0, 40.0],
                                [70.0, 90.0, 35.0]]),
            "Jet_eta": ak.Array([[-0.5, 0.8, 0.1], [0.1, 0.2, -0.3],
                                 [0.3, 0.4, 0.5]]),
            "Jet_phi": ak.Array([[1.2, -1.0, 0.4], [0.5, 0.6, 0.7],
                                 [0.7, 0.8, 0.9]]),
            "Jet_mass": ak.Array([[5.0, 6.0, 2.0], [4.0, 3.0, 1.0],
                                  [2.0, 7.0, 3.0]]),
        }
    return str(path)


@pytest.fixture
def lhco_h5(tmp_path):
    import pandas as pd

    n = 30
    rng = np.random.RandomState(0)
    data = {}
    for tag, base in (("j1", 0.0), ("j2", 100.0)):
        data[f"px{tag}"] = rng.uniform(base, base + 50, n)
        data[f"py{tag}"] = rng.uniform(-20, 20, n)
        data[f"pz{tag}"] = rng.uniform(-20, 20, n)
        data[f"m{tag}"] = rng.uniform(0, 10, n)
        data[f"tau1{tag}"] = rng.uniform(0, 1, n)
        data[f"tau2{tag}"] = rng.uniform(0, 1, n)
        data[f"tau3{tag}"] = rng.uniform(0, 1, n)
    data["label"] = np.array([0, 1] * (n // 2))
    path = tmp_path / "events.h5"
    pd.DataFrame(data).to_hdf(str(path), key="df", mode="w")
    return str(path)


# ==========================================================================
# JetClassDataset
# ==========================================================================

class TestJetClassDataset:
    def test_builds_graphs(self, jetclass_root, tmp_path):
        ds = JetClassDataset(root=str(tmp_path / "jc"),
                             root_file_paths=[jetclass_root],
                             k_neighbors=4, tag="t")
        assert len(ds) > 0
        g = ds[0]
        assert g.x.shape[1] == 16
        assert g.edge_index.shape[0] == 2
        assert not torch.any(g.edge_index[0] == g.edge_index[1])
        assert int(g.y) in (0, 1)

    def test_qcd_maps_to_label_zero(self, jetclass_root, tmp_path):
        ds = JetClassDataset(root=str(tmp_path / "jc"),
                             root_file_paths=[jetclass_root], k_neighbors=4)
        labels = [int(ds.get(i).y) for i in range(len(ds))]
        assert set(labels) == {0, 1}
        # 8 QCD + 4 non-QCD in the fixture.
        assert labels.count(1) == 4

    def test_kneighbors_respected(self, jetclass_root, tmp_path):
        for k in (2, 4):
            ds = JetClassDataset(root=str(tmp_path / f"jc{k}"),
                                 root_file_paths=[jetclass_root],
                                 k_neighbors=k, tag=f"k{k}")
            g = ds[0]
            n = g.x.size(0)
            for i in range(n):
                assert int((g.edge_index[0] == i).sum()) <= 2 * k

    def test_cache_key_distinguishes_k(self, jetclass_root, tmp_path):
        root = str(tmp_path / "jc")
        a = JetClassDataset(root=root, root_file_paths=[jetclass_root],
                            k_neighbors=2, tag="x")
        b = JetClassDataset(root=root, root_file_paths=[jetclass_root],
                            k_neighbors=8, tag="x")
        assert a._processed_file_name != b._processed_file_name
        assert len(a) == len(b)

    def test_cache_key_distinguishes_files(self, jetclass_root, tmp_path):
        other = tmp_path / "HTo_bb.root"
        with uproot.recreate(other) as f:
            f["tree"] = _jetclass_records(6, [False] * 6)
        root = str(tmp_path / "jc")
        a = JetClassDataset(root=root, root_file_paths=[jetclass_root], tag="x")
        b = JetClassDataset(root=root, root_file_paths=[str(other)], tag="x")
        assert a._processed_file_name != b._processed_file_name
        assert len(a) != len(b)

    def test_max_particles_trimmed_keeping_highest_pt(self, jetclass_root,
                                                      tmp_path):
        ds = JetClassDataset(root=str(tmp_path / "jc"),
                             root_file_paths=[jetclass_root],
                             k_neighbors=2, max_particles=3, tag="m")
        assert ds[0].x.size(0) <= 3

    def test_splits_are_disjoint_and_deterministic(self, jetclass_root, tmp_path):
        ds = JetClassDataset(root=str(tmp_path / "jc"),
                             root_file_paths=[jetclass_root], k_neighbors=4)
        tr, va, te = ds.get_splits(0.6, 0.2, 0.2)
        assert len(tr) + len(va) + len(te) == len(ds)
        tr2, _, _ = ds.get_splits(0.6, 0.2, 0.2)
        assert len(tr) == len(tr2)

    def test_get_loaders(self, jetclass_root, tmp_path):
        ds = JetClassDataset(root=str(tmp_path / "jc"),
                             root_file_paths=[jetclass_root], k_neighbors=4)
        tr, va, te = ds.get_loaders(batch_size=4)
        assert next(iter(tr)).num_graphs > 0

    def test_index_select(self, jetclass_root, tmp_path):
        ds = JetClassDataset(root=str(tmp_path / "jc"),
                             root_file_paths=[jetclass_root], k_neighbors=4)
        sub = ds.index_select([0, 1, 2])
        assert len(sub) == 3

    @pytest.mark.parametrize("kwargs, msg", [
        ({"root_file_paths": []}, "at least one ROOT file"),
        ({"k_neighbors": 0}, "k_neighbors"),
        ({"max_particles": 1}, "max_particles"),
    ])
    def test_validation(self, jetclass_root, tmp_path, kwargs, msg):
        base = {"root": str(tmp_path / "jc"),
                "root_file_paths": [jetclass_root], "k_neighbors": 4}
        base.update(kwargs)
        with pytest.raises(ValueError, match=msg):
            JetClassDataset(**base)

    def test_missing_file_error_is_clear(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.root"):
            JetClassDataset(root=str(tmp_path / "jc"),
                            root_file_paths=[str(tmp_path / "nope.root")])

    def test_static_knn_no_self_loops(self):
        pos = torch.randn(10, 2)
        ei = JetClassDataset._build_knn_graph(pos, 3)
        assert not torch.any(ei[0] == ei[1])
        assert ei.shape[1] == 30

    def test_static_knn_degenerate(self):
        assert JetClassDataset._build_knn_graph(torch.randn(1, 2), 3).shape == (2, 0)


# ==========================================================================
# CMSDataset
# ==========================================================================

class TestCMSDataset:
    def test_two_node_graphs(self, cms_root, tmp_path):
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                        label=0)
        assert len(ds) == 3
        g = ds[0]
        assert g.x.shape == (2, 4)
        assert g.edge_index.shape == (2, 2)

    def test_label_applied(self, cms_root, tmp_path):
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                        label=1)
        assert all(int(ds.get(i).y) == 1 for i in range(len(ds)))

    def test_single_jet_events_filtered(self, tmp_path):
        path = tmp_path / "one.root"
        with uproot.recreate(path) as f:
            f["Events"] = {
                "Jet_pt": ak.Array([[50.0], [30.0, 20.0]]),
                "Jet_eta": ak.Array([[0.1], [0.2, 0.3]]),
                "Jet_phi": ak.Array([[0.1], [0.2, 0.3]]),
                "Jet_mass": ak.Array([[5.0], [4.0, 3.0]]),
            }
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=str(path),
                        label=0)
        assert len(ds) == 1

    def test_no_valid_events_raises(self, tmp_path):
        path = tmp_path / "none.root"
        with uproot.recreate(path) as f:
            f["Events"] = {
                "Jet_pt": ak.Array([[50.0]]),
                "Jet_eta": ak.Array([[0.1]]),
                "Jet_phi": ak.Array([[0.1]]),
                "Jet_mass": ak.Array([[5.0]]),
            }
        with pytest.raises(ValueError, match="No events with >= 2 jets"):
            CMSDataset(root=str(tmp_path / "cms"), root_file_path=str(path),
                       label=0)

    def test_sample_size_larger_than_file(self, cms_root, tmp_path):
        """Regression: an unbound ``indices`` variable used to raise here."""
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                        label=0, sample_size=100)
        assert len(ds) == 3

    def test_sample_size_smaller_than_file(self, cms_root, tmp_path):
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                        label=0, sample_size=2)
        assert len(ds) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="CMS ROOT not found"):
            CMSDataset(root=str(tmp_path / "cms"),
                       root_file_path=str(tmp_path / "nope.root"), label=0)

    def test_cache_key_includes_label_and_stat(self, cms_root, tmp_path):
        root = str(tmp_path / "cms")
        a = CMSDataset(root=root, root_file_path=cms_root, label=0)
        b = CMSDataset(root=root, root_file_path=cms_root, label=1)
        assert a._processed_file_name != b._processed_file_name

    @pytest.mark.parametrize("label", [2, -1, 99])
    def test_rejects_bad_label(self, cms_root, tmp_path, label):
        with pytest.raises(ValueError, match="label must be 0"):
            CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                       label=label)

    def test_rejects_bad_sample_size(self, cms_root, tmp_path):
        with pytest.raises(ValueError, match="sample_size"):
            CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                       label=0, sample_size=0)

    def test_edge_index_not_shared_between_graphs(self, cms_root, tmp_path):
        """A single shared tensor would alias across every Data object."""
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                        label=0)
        a, b = ds.get(0), ds.get(1)
        assert a.edge_index.data_ptr() != b.edge_index.data_ptr()

    def test_splits_need_enough_items(self, cms_root, tmp_path):
        """A split that would be empty must fail with an actionable message.

        Previously this surfaced as the opaque "index_select received empty
        indices", which does not explain that the dataset is simply too small.
        """
        ds = CMSDataset(root=str(tmp_path / "cms"), root_file_path=cms_root,
                        label=0)
        assert len(ds) == 3
        with pytest.raises(ValueError, match="would be empty"):
            ds.get_loaders(batch_size=2)


# ==========================================================================
# LHCODataset
# ==========================================================================

class TestLHCODataset:
    def test_builds_two_node_graphs(self, lhco_h5, tmp_path):
        ds = LHCODataset(root=str(tmp_path / "lhco"), h5_path=lhco_h5)
        assert len(ds) == 30
        g = ds[0]
        assert g.x.shape == (2, LHCO_INPUT_DIM)
        assert g.edge_index.shape == (2, 2)
        assert int(g.y) in (0, 1)

    def test_labels_preserved(self, lhco_h5, tmp_path):
        ds = LHCODataset(root=str(tmp_path / "lhco"), h5_path=lhco_h5)
        labels = {int(ds.get(i).y) for i in range(len(ds))}
        assert labels == {0, 1}

    def test_sample_size(self, lhco_h5, tmp_path):
        ds = LHCODataset(root=str(tmp_path / "lhco"), h5_path=lhco_h5,
                         sample_size=10)
        assert len(ds) == 10
        labels = [int(ds.get(i).y) for i in range(len(ds))]
        assert set(labels) == {0, 1}   # stratified, not single-class

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="LHCO H5 not found"):
            LHCODataset(root=str(tmp_path / "lhco"),
                        h5_path=str(tmp_path / "nope.h5"))

    def test_missing_columns_raise(self, tmp_path):
        import pandas as pd

        p = tmp_path / "bad.h5"
        pd.DataFrame({"label": [0, 1]}).to_hdf(str(p), key="df", mode="w")
        with pytest.raises(ValueError, match="missing expected LHCO columns"):
            LHCODataset(root=str(tmp_path / "lhco"), h5_path=str(p))

    def test_rejects_bad_sample_size(self, lhco_h5, tmp_path):
        with pytest.raises(ValueError, match="sample_size"):
            LHCODataset(root=str(tmp_path / "lhco"), h5_path=lhco_h5,
                        sample_size=0)

    def test_cache_key_includes_stat(self, lhco_h5, tmp_path):
        a = LHCODataset(root=str(tmp_path / "lhco"), h5_path=lhco_h5)
        assert "m" in a.processed_file_names[0]

    def test_splits_and_loaders(self, lhco_h5, tmp_path):
        ds = LHCODataset(root=str(tmp_path / "lhco"), h5_path=lhco_h5)
        tr, va, te = ds.get_loaders(batch_size=8)
        assert sum(len(l.dataset) for l in (tr, va, te)) == len(ds)


# ==========================================================================
# JetClassIterableDataset
# ==========================================================================

class TestJetClassIterableDataset:
    def test_streams_graphs(self, jetclass_root):
        ds = JetClassIterableDataset(root_file_paths=[jetclass_root],
                                     k_neighbors=4)
        graphs = list(ds)
        assert len(graphs) == 12
        assert graphs[0].x.shape[1] == 16
        assert not torch.any(graphs[0].edge_index[0] == graphs[0].edge_index[1])

    def test_labels_derived(self, jetclass_root):
        ds = JetClassIterableDataset(root_file_paths=[jetclass_root])
        labels = [int(g.y) for g in ds]
        assert set(labels) == {0, 1}
        assert labels.count(1) == 4

    def test_start_idx_skips(self, jetclass_root):
        full = list(JetClassIterableDataset(root_file_paths=[jetclass_root]))
        part = list(JetClassIterableDataset(root_file_paths=[jetclass_root],
                                            start_batch=5, batch_size=2))
        assert len(part) == len(full) - 10

    def test_missing_tree_is_skipped(self, tmp_path):
        bad = tmp_path / "bad.root"
        with uproot.recreate(bad) as f:
            f["Events"] = {"x": ak.Array([[1.0]])}   # no 'tree'
        ds = JetClassIterableDataset(root_file_paths=[str(bad)])
        assert list(ds) == []

    def test_unreadable_file_logs_and_continues(self, jetclass_root, tmp_path,
                                                caplog):
        missing = tmp_path / "missing.root"
        ds = JetClassIterableDataset(
            root_file_paths=[str(missing), jetclass_root]
        )
        with caplog.at_level("ERROR"):
            graphs = list(ds)
        assert len(graphs) == 12
        assert "Error reading" in caplog.text

    def test_no_self_loops_knn(self):
        pos = torch.randn(8, 2)
        ei = JetClassIterableDataset._build_knn_graph(pos, 3)
        assert not torch.any(ei[0] == ei[1])

    def test_degenerate_knn(self):
        assert JetClassIterableDataset._build_knn_graph(
            torch.randn(1, 2), 3
        ).shape == (2, 0)


# ==========================================================================
# EventLoader.load_root against a real (synthetic) file
# ==========================================================================

class TestLoadRoot:
    def test_reads_events(self, cms_root):
        from event_ingestion.loader import EventLoader

        events = EventLoader().load_root(cms_root)
        assert len(events) == 3
        assert all("particles" in e for e in events)
        assert all(e["n_particles"] >= 2 for e in events)

    def test_particles_have_required_fields(self, cms_root):
        from event_ingestion.loader import EventLoader

        events = EventLoader().load_root(cms_root)
        for e in events:
            for p in e["particles"]:
                for key in ("type", "node_type", "pt", "eta", "phi", "mass",
                            "energy"):
                    assert key in p
                assert np.isfinite(p["pt"])
                assert p["pt"] > 0

    def test_energy_derived_from_kinematics(self, cms_root):
        from event_ingestion.loader import EventLoader

        events = EventLoader().load_root(cms_root)
        for e in events:
            for p in e["particles"]:
                if p["type"] == "MET":
                    continue
                expected = np.sqrt(
                    p["pt"] ** 2 * np.cosh(p["eta"]) ** 2 + p["mass"] ** 2
                )
                assert np.isclose(p["energy"], expected, rtol=1e-6)

    def test_max_events(self, cms_root):
        from event_ingestion.loader import EventLoader

        assert len(EventLoader().load_root(cms_root, max_events=1)) == 1

    def test_unknown_tree_raises(self, cms_root):
        from event_ingestion.loader import EventLoader

        with pytest.raises(KeyError, match="not found"):
            EventLoader().load_root(cms_root, tree_name="NoSuchTree")

    def test_min_particles_filter(self, cms_root):
        from event_ingestion.config import EventConfig
        from event_ingestion.loader import EventLoader

        cfg = EventConfig(min_particles=50, max_particles=100)
        assert EventLoader(cfg).load_root(cms_root) == []

    def test_max_particles_cap_keeps_highest_pt(self, cms_root):
        from event_ingestion.config import EventConfig
        from event_ingestion.loader import EventLoader

        cfg = EventConfig(min_particles=2, max_particles=2)
        for e in EventLoader(cfg).load_root(cms_root):
            pts = sorted((p["pt"] for p in e["particles"]), reverse=True)
            assert e["n_particles"] == 2
            assert pts[0] >= pts[-1]

    def test_iter_events(self, cms_root):
        from event_ingestion.loader import EventLoader

        assert len(list(EventLoader().iter_events(cms_root))) == 3
