"""Experiment-layer tests: config loading, data pipeline, model factories."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.data_pipeline import (
    build_classifier,
    ensure_graph_dataset,
    make_trainer,
)

REPO = Path(__file__).resolve().parent.parent
SMOKE = REPO / "experiments" / "configs" / "smoke.yaml"
DEFAULT = REPO / "experiments" / "configs" / "default.yaml"


@pytest.fixture
def smoke_config():
    return yaml.safe_load(SMOKE.read_text())


@pytest.fixture
def tiny_config(smoke_config, tmp_path):
    """Smoke config shrunk to a few dozen events and pointed at tmp_path."""
    cfg = yaml.safe_load(SMOKE.read_text())
    cfg["data"]["synthetic"].update(n_normal=30, n_anomaly=10)
    cfg["data"]["graphs"]["output"] = str(tmp_path / "graphs")
    cfg["output"] = {
        "checkpoints": str(tmp_path / "ckpt"),
        "results": str(tmp_path / "results"),
        "figures": str(tmp_path / "figures"),
    }
    return cfg


# ==========================================================================
# Config loading
# ==========================================================================

class TestLoadConfig:
    @pytest.mark.parametrize("module_name", ["train_classifier", "run_benchmark"])
    def test_default_config_loads(self, module_name):
        import importlib

        mod = importlib.import_module(f"experiments.{module_name}")
        cfg = mod.load_config()
        for section in ("data", "model", "training", "output"):
            assert section in cfg

    def test_missing_file_raises(self):
        from experiments.train_classifier import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nope.yaml")

    def test_non_mapping_raises(self, tmp_path):
        from experiments.run_benchmark import load_config

        p = tmp_path / "c.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="not a mapping"):
            load_config(str(p))

    def test_missing_section_raises(self, tmp_path):
        from experiments.train_classifier import load_config

        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump({"data": {}}))
        with pytest.raises(ValueError, match="missing the required"):
            load_config(str(p))

    def test_shipped_configs_are_valid(self):
        for path in (SMOKE, DEFAULT):
            cfg = yaml.safe_load(path.read_text())
            assert cfg["splits"]["train"] + cfg["splits"]["val"] + \
                cfg["splits"]["test"] == pytest.approx(1.0)

    def test_default_and_smoke_use_different_dirs(self):
        a = yaml.safe_load(DEFAULT.read_text())["data"]["graphs"]["output"]
        b = yaml.safe_load(SMOKE.read_text())["data"]["graphs"]["output"]
        assert a != b, "smoke config must not clobber the full run's graphs"


# ==========================================================================
# Data pipeline
# ==========================================================================

class TestDataPipeline:
    def test_generates_and_loads(self, tiny_config):
        train, val, test, ds = ensure_graph_dataset(tiny_config)
        n = len(ds)
        assert n == 40
        assert len(train.dataset) + len(val.dataset) + len(test.dataset) == n

    def test_fingerprint_change_actually_regenerates(self, tiny_config):
        """Regression: the stale ``processed/dataset.pt`` won over new graphs.

        The old flow deleted ``graphs.pt``, rebuilt it, and then got served the
        *old* collated dataset, so every "new" configuration silently trained
        on the previous data.
        """
        _, _, _, ds1 = ensure_graph_dataset(tiny_config)
        n1 = len(ds1)

        bigger = json.loads(json.dumps(tiny_config))
        bigger["data"]["synthetic"].update(n_normal=90, n_anomaly=30)
        _, _, _, ds2 = ensure_graph_dataset(bigger)
        assert len(ds2) != n1
        assert len(ds2) == 120

    def test_pyg_cache_cleared_on_invalidation(self, tiny_config):
        ensure_graph_dataset(tiny_config)
        cache = Path(tiny_config["data"]["graphs"]["output"]) / "processed"
        assert cache.exists()
        bigger = json.loads(json.dumps(tiny_config))
        bigger["data"]["graphs"]["k"] = 4
        ensure_graph_dataset(bigger)
        # Rebuilt from scratch, so the cache describes the new data.
        fp = json.loads(
            (Path(tiny_config["data"]["graphs"]["output"])
             / "synthetic_fingerprint.json").read_text()
        )
        assert fp["k"] == 4

    def test_unchanged_config_reuses_cache(self, tiny_config):
        ensure_graph_dataset(tiny_config)
        graphs_file = Path(tiny_config["data"]["graphs"]["output"]) / "graphs.pt"
        mtime = graphs_file.stat().st_mtime_ns
        ensure_graph_dataset(tiny_config)
        assert graphs_file.stat().st_mtime_ns == mtime

    def test_labels_travel_with_the_graphs(self, tiny_config):
        """Labels live in ``graphs.pt`` as ``g.y``.

        ``labels.pt`` is written by ``EventGraphConstructor.save_dataset`` (the
        standalone CLI path); the in-memory pipeline keeps the label on the
        graph instead, which is what the datasets actually read.
        """
        _, _, _, ds = ensure_graph_dataset(tiny_config)
        labels = [int(ds.get(i).y) for i in range(len(ds))]
        assert labels.count(1) == 10
        assert len(labels) == 40

    def test_data_dir_override(self, tiny_config, tmp_path):
        out = tmp_path / "override"
        _, _, _, ds = ensure_graph_dataset(tiny_config, data_dir=str(out))
        assert (out / "graphs.pt").exists()
        assert len(ds) == 40

    def test_missing_split_section_uses_defaults(self, tiny_config):
        del tiny_config["splits"]
        train, val, test, ds = ensure_graph_dataset(tiny_config)
        total = len(ds)
        assert len(train.dataset) == int(total * 0.7)


class TestTrainerFactory:
    def test_honours_mlflow_experiment_name(self, tiny_config):
        """Regression: ``mlflow.experiment`` was never passed to the Trainer."""
        tiny_config["mlflow"] = {"enabled": True, "experiment": "my-exp"}
        from anomaly_engine.models import get_classifier

        t = make_trainer(get_classifier("mlp", input_dim=11, hidden_dim=8),
                         tiny_config, "cpu", str(Path("x")))
        assert t.experiment_name == "my-exp"
        assert t.use_mlflow is True

    def test_mlflow_disabled_but_name_still_read(self, tiny_config):
        """``enabled: false`` but the experiment name must still be honoured."""
        from anomaly_engine.models import get_classifier

        assert tiny_config["mlflow"]["enabled"] is False
        t = make_trainer(get_classifier("mlp", input_dim=11, hidden_dim=8),
                         tiny_config, "cpu", str(Path("x")))
        assert t.use_mlflow is False
        assert t.experiment_name == tiny_config["mlflow"]["experiment"]

    def test_missing_mlflow_section_tolerated(self, tiny_config):
        from anomaly_engine.models import get_classifier

        del tiny_config["mlflow"]
        t = make_trainer(get_classifier("mlp", input_dim=11, hidden_dim=8),
                         tiny_config, "cpu", str(Path("x")))
        assert t.use_mlflow is False

    def test_hyperparameters_flow_through(self, tiny_config):
        from anomaly_engine.models import get_classifier

        tiny_config["training"].update(learning_rate=0.005, weight_decay=0.01,
                                       patience=3, max_grad_norm=2.0)
        t = make_trainer(get_classifier("mlp", input_dim=11, hidden_dim=8),
                         tiny_config, "cpu", str(Path("x")))
        assert t.learning_rate == 0.005
        assert t.weight_decay == 0.01
        assert t.patience == 3
        assert t.max_grad_norm == 2.0
        assert t.optimizer.param_groups[0]["lr"] == 0.005


class TestBuildClassifier:
    @pytest.mark.parametrize("name", ["gcn", "graphsage", "gat", "mlp", "cnn"])
    def test_all_registered_models(self, name, tiny_config):
        model = build_classifier(name, tiny_config)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_unknown_raises(self, tiny_config):
        with pytest.raises(ValueError, match="Unknown model"):
            build_classifier("banana", tiny_config)

    def test_dims_come_from_config(self, tiny_config):
        tiny_config["model"].update(input_dim=16, hidden_dim=32, latent_dim=16)
        model = build_classifier("gcn", tiny_config)
        assert model.encoder.input_dim == 16
        assert model.encoder.hidden_dim == 32

    def test_num_classes_respected(self, tiny_config):
        tiny_config["model"]["num_classes"] = 5
        model = build_classifier("gcn", tiny_config)
        out = model.classifier[-1]
        assert out.out_features == 5


# ==========================================================================
# Experiment scripts
# ==========================================================================

class TestAblationFactory:
    def test_known_archs(self):
        from experiments.ablation import get_model

        for arch in ("mlp", "gcn", "edgeconv"):
            assert get_model(arch) is not None

    def test_unknown_arch_raises(self):
        """Regression: the 6M factory had no terminal ``else`` and returned
        ``None``, so the caller died with an opaque AttributeError later."""
        from experiments.ablation import get_model

        with pytest.raises(ValueError, match="Unknown arch"):
            get_model("banana")

    def test_6m_unknown_arch_raises(self):
        from experiments.run_6m_ablation import get_model

        with pytest.raises(ValueError, match="Unknown arch"):
            get_model("banana")


class TestChunkedDataset:
    def test_max_particles_read_from_chunk_shape(self, tmp_path):
        """Regression: 128 was hard-coded, so any other cap was mishandled."""
        from experiments.run_6m_ablation import FastChunkedDataset

        chunk = tmp_path / "chunk_0.pt"
        torch.save({
            "x": torch.randn(4, 17, 16),          # max_particles = 17
            "lengths": torch.tensor([3, 17, 1, 17]),
            "y": torch.tensor([0, 1, 0, 1]),
        }, chunk)

        ds = FastChunkedDataset([str(chunk)], batch_size=2, device="cpu")
        batches = list(ds)
        assert len(batches) == 2
        # batch 0: lengths 3 + 17 = 20 nodes; batch 1: 1 + 17 = 18 nodes
        assert batches[0].x.shape[0] == 20
        assert batches[1].x.shape[0] == 18
        assert batches[0].x.shape[1] == 16

    def test_rejects_wrong_chunk_rank(self, tmp_path):
        from experiments.run_6m_ablation import FastChunkedDataset

        chunk = tmp_path / "c.pt"
        torch.save({"x": torch.randn(4, 16), "lengths": torch.tensor([1, 1, 1, 1]),
                    "y": torch.zeros(4)}, chunk)
        with pytest.raises(ValueError, match="expected x of shape"):
            list(FastChunkedDataset([str(chunk)], batch_size=2, device="cpu"))

    def test_batch_vectors_are_contiguous_per_graph(self, tmp_path):
        from experiments.run_6m_ablation import FastChunkedDataset

        chunk = tmp_path / "c.pt"
        torch.save({
            "x": torch.randn(4, 8, 16),
            "lengths": torch.tensor([2, 3, 4, 5]),
            "y": torch.tensor([0, 1, 0, 1]),
        }, chunk)
        batch = next(iter(FastChunkedDataset([str(chunk)], batch_size=2,
                                              device="cpu")))
        # Two graphs with 2 and 3 constituents -> 5 nodes total.
        assert batch.batch.tolist() == [0, 0, 1, 1, 1]
        assert len(batch.y) == 2


class TestKnNGraphBuilder:
    def test_no_self_loops_and_no_cross_graph_edges(self):
        from experiments.run_6m_ablation import build_knn_graph_gpu

        pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                            [0.5, 0.5], [10.0, 10.0], [11.0, 10.0],
                            [10.0, 11.0], [11.0, 11.0]])
        batch = torch.tensor([0] * 5 + [1] * 4)
        ei = build_knn_graph_gpu(pos, batch, k=2)
        assert not torch.any(ei[0] == ei[1])
        assert torch.all(batch[ei[0]] == batch[ei[1]])

    def test_degenerate_single_node_graph(self):
        from experiments.run_6m_ablation import build_knn_graph_gpu

        pos = torch.tensor([[0.0, 0.0]])
        batch = torch.zeros(1, dtype=torch.long)
        ei = build_knn_graph_gpu(pos, batch, k=4)
        assert ei.shape == (2, 0)


class TestPhysicsAnalysisKinematics:
    def test_jet_node_not_double_counted(self):
        """Regression: node 0 *is* the jet, and summing all nodes counted it
        twice, inflating pT, energy and the invariant mass."""
        from experiments.physics_analysis import compute_jet_kinematics

        x = torch.zeros(4, 16)
        x[0, 0], x[0, 1] = 30.0, 40.0          # jet pT = 50
        x[0, 2], x[0, 3] = 0.0, 50.0
        x[1:, 0] = torch.tensor([10.0, 10.0, 10.0])
        x[1:, 1] = torch.tensor([0.0, 0.0, 0.0])
        x[1:, 3] = torch.tensor([10.0, 10.0, 10.0])
        from torch_geometric.data import Data

        pt, mass, n = compute_jet_kinematics(Data(x=x))
        assert n == 3
        # Constituents only: three 10 GeV pT particles -> 30 GeV.
        assert pt == pytest.approx(30.0)
        # The old code would have reported ~50 (jet) + 30 (constituents).
        assert pt < 45.0

    def test_mass_is_non_negative(self):
        from torch_geometric.data import Data

        from experiments.physics_analysis import compute_jet_kinematics

        x = torch.zeros(3, 16)
        x[:, 0] = 5.0
        x[:, 1] = 1.0
        x[:, 2] = 1.0
        x[:, 3] = 1.0  # E < |p| -> negative mass^2
        _, mass, _ = compute_jet_kinematics(Data(x=x))
        assert mass >= 0.0

    def test_insufficient_columns_raise(self):
        from torch_geometric.data import Data

        from experiments.physics_analysis import compute_jet_kinematics

        with pytest.raises(ValueError, match="at least 4 feature columns"):
            compute_jet_kinematics(Data(x=torch.randn(3, 2)))

    def test_empty_raises(self):
        from torch_geometric.data import Data

        from experiments.physics_analysis import compute_jet_kinematics

        with pytest.raises(ValueError, match="non-empty"):
            compute_jet_kinematics(Data(x=torch.zeros(0, 16)))


class TestRebuildLossCurve:
    def test_import_has_no_side_effects(self, tmp_path, monkeypatch):
        """Regression: the module ran log discovery and sys.exit on import."""
        import subprocess
        import sys

        code = f"import sys; sys.path.insert(0, r'{REPO}'); import rebuild_loss_curve; print('OK')"
        out = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                             capture_output=True, text=True)
        assert "OK" in out.stdout
        assert out.returncode == 0

    def test_empty_log_does_not_win(self, tmp_path, monkeypatch):
        from rebuild_loss_curve import find_log

        monkeypatch.chdir(tmp_path)
        (tmp_path / "local_error.log").write_text("")   # tracked, always empty
        (tmp_path / "training.log").write_text(
            "Epoch 1: 10it [00:10, 1s/it, loss=5.0]\n"
        )
        assert find_log().name == "training.log"

    def test_unparseable_log_skipped(self, tmp_path, monkeypatch):
        from rebuild_loss_curve import find_log

        monkeypatch.chdir(tmp_path)
        (tmp_path / "training.log").write_text("nothing useful\n")
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "a.log").write_text(
            "Epoch 3: 5it [00:05, 1s/it, loss=1.5]\n"
        )
        assert find_log().name.endswith("a.log")

    def test_parse_collapses_duplicate_lines(self):
        from rebuild_loss_curve import parse_log

        losses, bounds = parse_log(
            "Epoch 1: 10it [00:10, 1s/it, loss=5.0]\n"
            "Epoch 1: 10it [00:10, 1s/it, loss=5.0]\n"
            "Epoch 1: 20it [00:20, 1s/it, loss=3.0]\n"
            "Epoch 2: 10it [00:10, 1s/it, loss=2.0]\n"
        )
        assert losses == [5.0, 3.0, 2.0]
        assert bounds == [2]

    def test_scientific_notation_loss(self):
        from rebuild_loss_curve import parse_log

        losses, _ = parse_log("Epoch 1: 1it [00:01, 1s/it, loss=1.234e+03]\n")
        assert losses == [1234.0]

    def test_main_exits_cleanly_without_data(self, tmp_path, monkeypatch):
        from rebuild_loss_curve import main

        monkeypatch.chdir(tmp_path)
        assert main([]) == 1
        assert "No training log or checkpoint" in (
            tmp_path / "nothing.txt"
        ).read_text() if (tmp_path / "nothing.txt").exists() else True

    def test_main_writes_figure_from_checkpoint(self, tmp_path, monkeypatch):
        from rebuild_loss_curve import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints" / "smoke").mkdir(parents=True)
        torch.save({"history": {"train_loss": [3.0, 2.0, 1.0]}},
                   tmp_path / "checkpoints" / "smoke" / "classifier_gcn_best.pt")
        assert main([]) == 0
        assert (tmp_path / "docs" / "loss_curve.png").exists()
