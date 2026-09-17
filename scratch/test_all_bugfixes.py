"""Comprehensive regression test suite for all bug fixes."""
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_build_knn_graph_gpu_no_self_loops():
    """Verify build_knn_graph_gpu does NOT generate self-loops (i -> i) and handles batching."""
    from experiments.run_6m_ablation import build_knn_graph_gpu as knn_ablation
    from experiments.run_5epochs_edgeconv import build_knn_graph_gpu as knn_5epochs

    for knn_fn in [knn_ablation, knn_5epochs]:
        pos = torch.tensor([
            # Graph 0
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            # Graph 1
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
            [11.0, 11.0],
        ], dtype=torch.float32)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

        edge_index = knn_fn(pos, batch, k=2)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

        src, dst = edge_index[0], edge_index[1]
        # NO self loops allowed
        assert not torch.any(src == dst), f"Self-loops detected in {knn_fn.__module__}!"

        # Ensure no cross-graph edges
        assert torch.all(batch[src] == batch[dst]), "Cross-graph edges detected!"


def test_cms_dataset_process_unbound_local_error(tmp_path):
    """Verify CMSDataset does not raise UnboundLocalError when sample_size >= len(pt)."""
    from graph_builder.cms_dataset import CMSDataset
    import awkward as ak
    import uproot

    root_file = tmp_path / "test_cms.root"

    # Write a dummy ROOT file with 1 event having 2 jets
    with uproot.recreate(root_file) as f:
        f["Events"] = {
            "Jet_pt": ak.Array([[50.0, 60.0]]),
            "Jet_eta": ak.Array([[-0.5, 0.8]]),
            "Jet_phi": ak.Array([[1.2, -1.0]]),
            "Jet_mass": ak.Array([[5.0, 6.0]]),
        }

    # sample_size=10 is > len(pt) (1 event).
    # Prior to fix, this skipped `indices` assignment inside `if sample_size < len(pt)`,
    # causing UnboundLocalError on line 120.
    dataset = CMSDataset(
        root=str(tmp_path / "processed_cms"),
        root_file_path=str(root_file),
        label=1,
        sample_size=10,
    )
    assert len(dataset) == 1
    data = dataset[0]
    assert data.x.shape == (2, 4)
    assert data.edge_index.shape == (2, 2)


def test_collision_event_dataset_empty_index_select(tmp_path):
    """Verify CollisionEventDataset.index_select raises ValueError on empty index list."""
    from graph_builder.dataset import CollisionEventDataset

    g = Data(x=torch.randn(3, 11), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    ds = CollisionEventDataset(root=str(tmp_path), graphs=[g])
    assert len(ds) == 1

    with pytest.raises(ValueError, match="empty indices"):
        ds.index_select([])


def test_graph_constructor_sync_and_import(tmp_path):
    """Verify EventGraphConstructor synchronizes labels with kept graphs."""
    from graph_builder.graph_constructor import EventGraphConstructor

    constructor = EventGraphConstructor(strategy="knn", k=2)

    # 3 events: one valid (3 particles), one empty/skipped (0 particles), one valid (3 particles)
    events = [
        {
            "event_id": 0,
            "particles": [
                {"node_type": 0, "pt": 10.0, "eta": 0.1, "phi": 0.2, "mass": 0.1, "charge": 1.0, "energy": 10.0},
                {"node_type": 0, "pt": 12.0, "eta": 0.2, "phi": 0.3, "mass": 0.1, "charge": -1.0, "energy": 12.0},
                {"node_type": 0, "pt": 15.0, "eta": 0.3, "phi": 0.4, "mass": 0.1, "charge": 1.0, "energy": 15.0},
            ]
        },
        {
            "event_id": 1,
            "particles": []  # Empty event, skipped (<2 particles)
        },
        {
            "event_id": 2,
            "particles": [
                {"node_type": 0, "pt": 20.0, "eta": -0.1, "phi": -0.2, "mass": 0.1, "charge": 1.0, "energy": 20.0},
                {"node_type": 0, "pt": 22.0, "eta": -0.2, "phi": -0.3, "mass": 0.1, "charge": -1.0, "energy": 22.0},
                {"node_type": 0, "pt": 25.0, "eta": -0.3, "phi": -0.4, "mass": 0.1, "charge": 1.0, "energy": 25.0},
            ]
        }
    ]
    labels = np.array([10, 20, 30])

    out_dir = str(tmp_path / "graph_output")
    constructor.save_dataset(events, labels, out_dir)

    saved_graphs = torch.load(f"{out_dir}/graphs.pt", weights_only=False)
    saved_labels = torch.load(f"{out_dir}/labels.pt", weights_only=False)

    assert len(saved_graphs) == 2, "Only 2 non-empty events should be saved"
    assert len(saved_labels) == 2, "Labels must be synchronized with saved graphs"
    assert saved_labels.tolist() == [10, 30], "Dropped event label (20) must not be present"


def test_cnn_classifier_empty_and_disjoint_nodes():
    """Verify CNNClassifier handles zero nodes without crashing."""
    from anomaly_engine.models.baselines import CNNClassifier

    model = CNNClassifier(input_dim=11, num_classes=2, max_particles=50)
    model.eval()

    # Empty data (0 nodes)
    empty_data = Data(x=torch.zeros((0, 11)), batch=torch.zeros((0,), dtype=torch.long))

    with torch.no_grad():
        out = model(empty_data)
    assert out.shape == (0, 2)

    # Non-empty data (2 graphs, 3 nodes total)
    data = Data(
        x=torch.randn(3, 11),
        batch=torch.tensor([0, 0, 1], dtype=torch.long)
    )
    with torch.no_grad():
        out = model(data)
    assert out.shape == (2, 2)


def test_plot_latent_space_model_api_and_label_length(tmp_path):
    """Verify plot_latent_space does not crash with diverse model types or label sizes."""
    from anomaly_engine.evaluate import Evaluator
    import torch.nn as nn

    class MockEncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 8)
        def encode_graph(self, data):
            batch = getattr(data, "batch", None)
            num_graphs = (int(batch.max().item()) + 1) if (batch is not None and batch.numel() > 0) else 1
            return torch.randn(num_graphs, 8)

    model = MockEncoderModel()
    g1 = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0], [1]], dtype=torch.long), y=torch.tensor([0]))
    g2 = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[1], [0]], dtype=torch.long), y=torch.tensor([1]))
    loader = [Batch.from_data_list([g1, g2])]

    evaluator = Evaluator(device="cpu")
    out_file = str(tmp_path / "latent.png")
    evaluator.plot_latent_space(model, loader, output_path=out_file)
    assert os.path.exists(out_file)


def test_download_robust_empty_dirname(tmp_path, monkeypatch):
    """Verify download_robust does not crash when dirname is empty."""
    from scripts.download_robust import download_file

    class MockResponse:
        def raise_for_status(self): pass
        @property
        def headers(self): return {"content-length": "5"}
        def iter_content(self, block_size):
            yield b"hello"

    import requests
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

    monkeypatch.chdir(tmp_path)
    success = download_file("http://dummy.url/file.root", "file.root")
    assert success
    assert (tmp_path / "file.root").exists()


def test_rebuild_loss_curve_cli_arg(tmp_path):
    """Verify rebuild_loss_curve.py parses CLI log argument properly."""
    log_path = tmp_path / "test_run.log"
    log_path.write_text(
        "Epoch 1: 10it [00:01, 10.0it/s, loss=0.5432]\n"
        "Epoch 1: 20it [00:02, 10.0it/s, loss=0.4321]\n"
    )

    import subprocess
    cmd = [sys.executable, str(REPO_ROOT / "rebuild_loss_curve.py"), str(log_path)]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, f"Error: {result.stderr}"
    assert (tmp_path / "docs" / "loss_curve.png").exists()
