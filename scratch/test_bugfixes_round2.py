"""Round-2 regression tests: no heavy JetClass/CMS download required."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_particle_feature_order_aligned():
    from graph_builder.jetclass_dataset import PARTICLE_FEATURES as DS_FEATS
    from graph_builder.jetclass_iterable import PARTICLE_FEATURES as IT_FEATS
    from experiments.preprocess_6m import PARTICLE_FEATURES as PP_FEATS

    assert DS_FEATS == IT_FEATS, f"iterable order diverged: {IT_FEATS}"
    assert DS_FEATS == PP_FEATS, f"preprocess order diverged: {PP_FEATS}"
    # demo-sensitive columns: deta/dphi at 4,5 and charge at 10
    assert DS_FEATS[4] == "part_deta" and DS_FEATS[5] == "part_dphi"
    assert DS_FEATS[10] == "part_charge"


def test_delta_phi_handles_nonfinite():
    from graph_builder.features import FeatureExtractor

    assert FeatureExtractor._delta_phi(float("inf"), 0.0) == 0.0
    assert FeatureExtractor._delta_phi(float("nan"), 0.0) == 0.0
    assert abs(FeatureExtractor._delta_phi(3.5, -3.5)) <= np.pi


def test_feature_extractor_sanitizes_negative_log_inputs():
    from graph_builder.features import FeatureExtractor

    fe = FeatureExtractor()
    p = {"node_type": 0, "pt": 10.0, "eta": 0.5, "phi": 0.1,
         "mass": -5.0, "charge": 1.0, "energy": -2.0}
    feats = fe.particle_to_node_features(p)
    assert feats.shape == (11,)
    assert np.all(np.isfinite(feats))


def test_empty_event_features():
    from graph_builder.features import FeatureExtractor

    fe = FeatureExtractor()
    out = fe.extract_event_features([])
    assert out.shape[0] == 0


def test_graph_constructor_empty_dataset_guard():
    from graph_builder.graph_constructor import EventGraphConstructor

    cons = EventGraphConstructor(strategy="knn", k=4)
    assert cons.convert_dataset([], None) == []


def test_event_loader_tree_cycle_and_suffix():
    from event_ingestion.loader import EventLoader

    loader = EventLoader()
    # Unsupported suffix still raises ValueError (case-insensitive check)
    try:
        loader.load("some/file.TXT")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for .TXT")
    # Missing file raises FileNotFoundError, not KeyError/crash
    try:
        loader.load_synthetic("does/not/exist.npz")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_statistics_empty_and_deterministic(tmp_path):
    from event_ingestion.statistics import EventStatistics

    stats = EventStatistics()
    assert stats.compute([]) == {"n_events": 0}
    stats.plot_distributions([], output_dir=str(tmp_path / "figs"))
    # Seeded scatter must not consume global RNG
    np.random.seed(123)
    before = np.random.rand()
    np.random.seed(123)
    stats.plot_distributions(
        [{"n_particles": 1, "particles": [{"type": "Jet", "pt": 50.0}]}],
        output_dir=str(tmp_path / "figs"),
    )
    np.random.seed(123)
    assert np.random.rand() == before


def test_iterable_knn_guard():
    from graph_builder.jetclass_iterable import JetClassIterableDataset

    e = JetClassIterableDataset._build_knn_graph(torch.randn(1, 2), k=8)
    assert e.shape == (2, 0)
    e2 = JetClassIterableDataset._build_knn_graph(torch.randn(5, 2), k=4)
    assert e2.shape[0] == 2 and e2.shape[1] > 0


if __name__ == "__main__":
    test_particle_feature_order_aligned()
    test_delta_phi_handles_nonfinite()
    test_feature_extractor_sanitizes_negative_log_inputs()
    test_empty_event_features()
    test_graph_constructor_empty_dataset_guard()
    test_event_loader_tree_cycle_and_suffix()
    test_statistics_empty_and_deterministic()
    test_iterable_knn_guard()
    print("round-2 regression checks passed.")
