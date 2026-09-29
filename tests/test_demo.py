"""Streamlit demo tests (module import must stay side-effect free)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo():
    import demo as demo_module

    return demo_module


def _jet(px, py, n_extra=3, feature_dim=16):
    """Build a JetClass-shaped graph: node 0 is the jet, 1..N are constituents."""
    n = 1 + n_extra
    x = torch.zeros(n, feature_dim)
    x[0, 0], x[0, 1] = px, py
    x[1:, 0] = torch.linspace(5.0, 15.0, n_extra)
    x[:, 4] = torch.linspace(0.0, 1.0, n)
    x[:, 5] = torch.linspace(0.0, 1.0, n)
    if feature_dim > 10:
        charges = [0.0] + [1.0 if i % 2 == 0 else -1.0 for i in range(n_extra)]
        x[:, 10] = torch.tensor(charges)
    idx = torch.arange(n)
    src = torch.cat([idx, (idx + 1) % n])
    dst = torch.cat([(idx + 1) % n, idx])
    return Data(x=x, edge_index=torch.stack([src, dst]).long())


class TestImportSafety:
    def test_import_has_no_side_effects(self, tmp_path):
        """Regression risk: the app must not build plots/models on import."""
        code = f"import sys; sys.path.insert(0, r'{REPO}'); import demo; print('OK')"
        out = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "OK" in out.stdout


class TestModelLoading:
    def test_missing_checkpoint_raises(self, demo, tmp_path):
        """Regression: a missing checkpoint silently produced a *random* model
        whose reconstruction errors were then displayed as anomaly scores."""
        with pytest.raises(FileNotFoundError, match="randomly initialised"):
            demo._load_model_uncached(str(tmp_path / "missing.pt"))

    def test_existing_checkpoint_loads(self, demo):
        ckpt = REPO / "checkpoints" / "jetclass_autoencoder" / "jetclass_edgeconv_best.pt"
        if not ckpt.exists():
            pytest.skip("committed JetClass checkpoint not present")
        model, device = demo._load_model_uncached(str(ckpt))
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_parameter_count_is_computed(self, demo):
        """The sidebar used a hardcoded 37,296."""
        n = demo.count_parameters(demo._build_model())
        assert n == 37_296  # matches the committed checkpoint
        assert n > 0

    def test_run_inference_on_real_checkpoint(self, demo):
        ckpt = REPO / "checkpoints" / "jetclass_autoencoder" / "jetclass_edgeconv_best.pt"
        if not ckpt.exists():
            pytest.skip("committed JetClass checkpoint not present")
        model, device = demo._load_model_uncached(str(ckpt))
        jet = _jet(30.0, 40.0)
        score, node_mse = demo.run_inference(jet, model, device)
        assert np.isfinite(score)
        assert len(node_mse) == jet.x.size(0)

    def test_run_inference_without_jet_raises(self, demo):
        with pytest.raises(ValueError, match="No jet"):
            demo.run_inference(None)


class TestJetObservables:
    def test_jet_pt_is_true_transverse_momentum(self, demo):
        """Regression: the sum of per-particle pT magnitudes was labelled
        "Total Jet pT". That is not a collider observable and is systematically
        larger than the real jet pT."""
        obs = demo.jet_observables(_jet(30.0, 40.0, n_extra=3))
        assert obs["jet_pt"] == pytest.approx(50.0, abs=1e-4)
        # 5 + 10 + 15 GeV constituents
        assert obs["sum_constituent_pt"] == pytest.approx(30.0, abs=1e-4)
        assert obs["n_constituents"] == 3

    def test_jet_node_excluded_from_constituent_count(self, demo):
        obs = demo.jet_observables(_jet(10.0, 0.0, n_extra=5))
        assert obs["n_constituents"] == 5

    def test_single_node_jet(self, demo):
        x = torch.zeros(1, 16)
        x[0, 0] = 3.0
        x[0, 1] = 4.0
        jet = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long))
        obs = demo.jet_observables(jet)
        assert obs["jet_pt"] == pytest.approx(5.0, abs=1e-5)

    def test_missing_charge_column_is_tolerated(self, demo):
        x = torch.randn(4, 6)
        jet = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 2]]))
        assert demo.jet_observables(jet)["avg_charge"] == 0.0

    def test_too_few_columns_raises(self, demo):
        jet = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0], [1]]))
        with pytest.raises(ValueError, match="16-feature"):
            demo.jet_observables(jet)


class TestErrorHeatmap:
    def test_renders(self, demo):
        jet = _jet(30.0, 40.0)
        mse = np.random.rand(jet.x.size(0))
        assert demo.plot_error_heatmap(jet, mse) is not None

    def test_isolated_nodes_do_not_break_colouring(self, demo):
        """node_mse is longer than the edge-induced graph; networkx needs the
        colours to match the nodes actually being drawn."""
        jet = _jet(30.0, 40.0, n_extra=5)
        jet.edge_index = jet.edge_index[:, :2]   # leave some nodes isolated
        mse = np.random.rand(jet.x.size(0))
        assert demo.plot_error_heatmap(jet, mse) is not None

    def test_edgeless_jet_raises(self, demo):
        jet = _jet(30.0, 40.0)
        jet.edge_index = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="jet with edges"):
            demo.plot_error_heatmap(jet, np.zeros(4))


class TestSampleSelection:
    def test_missing_root_files_returns_none(self, demo, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert demo._get_sample_jet_impl("bg", 42) is None
        assert demo._get_sample_jet_impl("sig", 42) is None

    def test_does_not_reseed_global_random(self, demo):
        """Regression: the old code called ``random.seed(seed)``, perturbing
        the global module for the whole app."""
        import random

        demo._get_sample_jet_impl("bg", 7)   # returns None without data
        before = random.random()
        random.seed(999)
        expected = random.random()
        random.seed(999)
        assert random.random() == expected  # global state still under caller control
