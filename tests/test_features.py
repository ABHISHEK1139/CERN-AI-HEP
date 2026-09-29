"""Feature extraction and event-loader tests."""

import numpy as np
import pytest
import torch

from event_ingestion.config import NODE_FEATURE_DIM, NUM_PARTICLE_TYPES
from event_ingestion.loader import EventLoader
from graph_builder.features import FeatureExtractor


def _particle(**kw):
    p = {"node_type": 0, "pt": 10.0, "eta": 0.5, "phi": 1.0, "mass": 1.0,
         "charge": 1.0, "energy": 10.0}
    p.update(kw)
    return p


class TestFeatureExtractor:
    def test_node_feature_dim(self):
        f = FeatureExtractor()
        assert f.particle_to_node_features(_particle()).shape == (
            NODE_FEATURE_DIM,
        )

    def test_onehot_encoding(self):
        f = FeatureExtractor()
        v = f.particle_to_node_features(_particle(node_type=2))
        assert v[:NUM_PARTICLE_TYPES].tolist() == [0, 0, 1, 0, 0]

    def test_out_of_range_node_type_ignored(self):
        f = FeatureExtractor()
        v = f.particle_to_node_features(_particle(node_type=99))
        assert v[:NUM_PARTICLE_TYPES].sum() == 0.0

    def test_missing_pt_raises(self):
        p = _particle()
        del p["pt"]
        with pytest.raises(KeyError, match="pt"):
            FeatureExtractor().particle_to_node_features(p)

    def test_negative_pt_does_not_produce_nan(self):
        """log1p of a negative pT would be NaN and poison the MSE loss."""
        v = FeatureExtractor().particle_to_node_features(_particle(pt=-5.0))
        assert np.isfinite(v).all()

    def test_nonfinite_inputs_sanitised(self):
        f = FeatureExtractor()
        v = f.particle_to_node_features(
            _particle(pt=float("inf"), eta=float("nan"), phi=1e30)
        )
        assert np.isfinite(v).all()

    def test_log_pt_toggle(self):
        raw = FeatureExtractor(log_pt=False).particle_to_node_features(_particle())
        logged = FeatureExtractor(log_pt=True).particle_to_node_features(_particle())
        assert not np.isclose(raw[5], logged[5])

    def test_extract_event_features_empty(self):
        out = FeatureExtractor().extract_event_features([])
        assert out.shape == (0, NODE_FEATURE_DIM)

    def test_extract_event_features_stack(self):
        f = FeatureExtractor()
        out = f.extract_event_features([_particle() for _ in range(7)])
        assert out.shape == (7, NODE_FEATURE_DIM)

    def test_delta_phi_wrapped(self):
        f = FeatureExtractor()
        # 0 and 2*pi are the same direction: dphi must be ~0, not ~2*pi.
        assert abs(f._delta_phi(0.0, 2 * np.pi)) < 1e-9
        assert abs(abs(f._delta_phi(0.0, np.pi)) - np.pi) < 1e-9

    def test_delta_phi_nonfinite(self):
        f = FeatureExtractor()
        assert f._delta_phi(float("inf"), 1.0) == 0.0
        assert f._delta_phi(float("nan"), 1.0) == 0.0

    def test_delta_phi_terminates_on_infinity(self):
        """A while-loop wrap hangs on inf: inf - 2*pi == inf."""
        f = FeatureExtractor()
        assert f._delta_phi(0.0, float("inf")) == 0.0

    def test_edge_features(self):
        f = FeatureExtractor()
        e = f.compute_edge_features(_particle(eta=0.0, phi=0.0),
                                    _particle(eta=0.0, phi=0.0))
        assert e.shape == (4,)
        assert e[0] == pytest.approx(0.0)   # dR
        assert e[3] == pytest.approx(0.0)   # relative pT

    def test_edge_features_require_pt(self):
        p = _particle()
        del p["pt"]
        with pytest.raises(KeyError, match="pt"):
            FeatureExtractor().compute_edge_features(p, _particle())

    def test_delta_r_matrix_shape_and_diagonal(self):
        f = FeatureExtractor()
        d = f.compute_delta_r_matrix([_particle(eta=float(i), phi=0.0)
                                      for i in range(4)])
        assert d.shape == (4, 4)
        assert np.allclose(np.diag(d), 0.0)
        assert np.allclose(d, d.T)

    def test_delta_r_matrix_symmetric_for_wrapped_phi(self):
        f = FeatureExtractor()
        d = f.compute_delta_r_matrix([_particle(eta=0.0, phi=0.0),
                                      _particle(eta=0.0, phi=2 * np.pi)])
        assert d[0, 1] == pytest.approx(0.0, abs=1e-6)


class TestStandardization:
    def test_fit_transform(self):
        f = FeatureExtractor(standardize=True)
        X = np.random.RandomState(0).randn(200, NODE_FEATURE_DIM)
        f.fit(X)
        Z = f.transform(X)
        assert np.abs(Z.mean(axis=0)).max() < 1e-6

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError, match="0 samples"):
            FeatureExtractor().fit(np.zeros((0, 11)))

    def test_transform_without_fit_is_identity(self):
        f = FeatureExtractor(standardize=True)
        X = np.random.RandomState(1).randn(10, NODE_FEATURE_DIM)
        assert np.allclose(f.transform(X), X)

    def test_zero_variance_columns_do_not_divide_by_zero(self):
        f = FeatureExtractor(standardize=True)
        X = np.zeros((50, NODE_FEATURE_DIM))
        f.fit(X)
        Z = f.transform(X)
        assert np.isfinite(Z).all()


class TestEventLoader:
    def test_unsupported_suffix(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("")
        with pytest.raises(ValueError, match="Unsupported file format"):
            EventLoader().load(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EventLoader().load_synthetic(tmp_path / "nope.npz")

    def test_synthetic_roundtrip(self, tmp_path):
        from event_ingestion.synthetic import SyntheticEventGenerator

        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(6, 2)
        path = gen.save(tmp_path / "e.npz", events, labels)
        loaded = EventLoader().load_synthetic(path)
        assert len(loaded) == 8
        assert len(list(EventLoader().iter_events(path))) == 8

    def test_max_events_truncation(self, tmp_path):
        from event_ingestion.synthetic import SyntheticEventGenerator

        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(10, 0)
        path = gen.save(tmp_path / "e.npz", events, labels)
        assert len(EventLoader().load(path, max_events=3)) == 3

    def test_npz_without_events_raises(self, tmp_path):
        p = tmp_path / "bad.npz"
        np.savez(p, something=1)
        with pytest.raises(KeyError, match="no 'events' array"):
            EventLoader().load_synthetic(p)

    def test_load_root_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ROOT file not found"):
            EventLoader().load_root(tmp_path / "nope.root")

    def test_config_validation_applied(self):
        from event_ingestion.config import EventConfig

        with pytest.raises(ValueError):
            EventLoader(EventConfig(min_particles=1))
