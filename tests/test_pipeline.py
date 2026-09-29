"""End-to-end pipeline tests on synthetic data (no downloads required)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from anomaly_engine.anomaly_scorer import AnomalyScorer
from anomaly_engine.evaluate import Evaluator
from anomaly_engine.models import get_autoencoder, get_classifier
from anomaly_engine.trainer import Trainer, set_seed
from event_ingestion.synthetic import SyntheticEventGenerator
from graph_builder.graph_constructor import EventGraphConstructor

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def pipeline(tmp_path):
    """Synthetic events -> graphs -> dataset -> loaders, on disk in tmp_path."""
    set_seed(0)
    gen = SyntheticEventGenerator(seed=0)
    events, labels = gen.generate(n_normal=60, n_anomaly=20)
    constructor = EventGraphConstructor(strategy="knn", k=4)
    graphs = constructor.convert_dataset(events, labels)
    return graphs, labels


class TestSyntheticPipeline:
    def test_graph_shapes(self, pipeline):
        graphs, _ = pipeline
        assert len(graphs) > 0
        for g in graphs[:10]:
            assert g.x.shape[1] == 11
            assert g.edge_index.shape[0] == 2
            assert not torch.any(g.edge_index[0] == g.edge_index[1])
            assert torch.isfinite(g.x).all()

    def test_train_classifier_and_evaluate(self, pipeline):
        from graph_builder.dataset import CollisionEventDataset

        graphs, _ = pipeline
        dataset = CollisionEventDataset(
            root=str(REPO / "data" / "pytest_graphs"), graphs=graphs
        )
        try:
            train, val, test = dataset.get_loaders(batch_size=16)
            assert len(train.dataset) + len(val.dataset) + len(test.dataset) == len(dataset)

            model = get_classifier("gcn", input_dim=11, hidden_dim=16,
                                   latent_dim=8, num_layers=2)
            trainer = Trainer(model, device="cpu", patience=5,
                              checkpoint_dir=str(REPO / "checkpoints" / "pytest"))
            history = trainer.train_classifier(train, val, epochs=3,
                                               run_name="e2e_clf")
            assert len(history["train_loss"]) == 3

            results = Evaluator(device="cpu").evaluate_classifier(model, test)
            assert 0.0 <= results["accuracy"] <= 1.0
            assert "auroc" in results
        finally:
            shutil.rmtree(REPO / "data" / "pytest_graphs", ignore_errors=True)
            shutil.rmtree(REPO / "checkpoints" / "pytest", ignore_errors=True)

    def test_train_autoencoder_and_score(self, pipeline):
        from graph_builder.dataset import CollisionEventDataset

        graphs, _ = pipeline
        root = REPO / "data" / "pytest_ae"
        ckpt = REPO / "checkpoints" / "pytest_ae"
        dataset = CollisionEventDataset(root=str(root), graphs=graphs)
        try:
            train, val, test = dataset.get_loaders(batch_size=16)

            ae = get_autoencoder("edgeconv", input_dim=11, hidden_dim=16,
                                 latent_dim=8, num_layers=2)
            trainer = Trainer(ae, device="cpu", patience=5,
                              checkpoint_dir=str(ckpt))
            history = trainer.train_autoencoder(train, val, epochs=2,
                                                run_name="e2e_ae")
            assert len(history["train_loss"]) == 2

            scores, labels, ids = AnomalyScorer(ae, device="cpu").score_dataset(test)
            assert len(scores) == len(labels) == len(ids) == len(test.dataset)
            assert np.isfinite(scores).all()

            report = AnomalyScorer(ae, device="cpu").generate_report(
                scores, labels, ids, top_k=5
            )
            assert report["n_events"] == len(scores)
            assert "auroc" in report
        finally:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(ckpt, ignore_errors=True)

    @pytest.mark.parametrize("arch", ["gcn", "graphsage", "gat", "edgeconv"])
    def test_every_encoder_trains(self, pipeline, arch, tmp_path):
        """Each registered encoder must survive a real train+eval pass."""
        dim = 16 if arch == "edgeconv" else 11
        graphs = []
        for g in pipeline[0]:
            x = torch.zeros(g.x.size(0), dim)
            x[:, 0] = g.x[:, 0]
            if dim >= 6:
                x[:, 4:6] = g.x[:, 4:6]
            new = g.clone()
            new.x = x
            graphs.append(new)

        from tests.helpers import make_loader

        train = make_loader(graphs[:40], batch_size=8)
        val = make_loader(graphs[40:60], batch_size=8)

        ae = get_autoencoder(arch, input_dim=dim, hidden_dim=16, latent_dim=8,
                             num_layers=2)
        t = Trainer(ae, device="cpu", patience=3, checkpoint_dir=str(tmp_path))
        hist = t.train_autoencoder(train, val, epochs=2, run_name=arch)
        assert len(hist["train_loss"]) == 2
        assert np.isfinite(hist["val_loss"][-1])


class TestSmokeConfigEndToEnd:
    def test_train_classifier_script(self, tmp_path):
        """Run the documented smoke command in a subprocess."""
        import subprocess
        import sys

        cfg = yaml.safe_load((REPO / "experiments" / "configs" / "smoke.yaml").read_text())
        cfg["data"]["synthetic"].update(n_normal=40, n_anomaly=10)
        cfg["data"]["graphs"]["output"] = str(tmp_path / "graphs")
        cfg["output"] = {k: str(tmp_path / k)
                         for k in ("checkpoints", "results", "figures")}
        cfg_path = tmp_path / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        out = subprocess.run(
            [sys.executable, str(REPO / "experiments" / "train_classifier.py"),
             "--config", str(cfg_path), "--model", "gcn",
             "--epochs", "2", "--device", "cpu"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        assert "RESULTS: GCN" in out.stdout
        assert "AUROC" in out.stdout
        assert (tmp_path / "figures" / "training_gcn.png").exists()

    def test_run_benchmark_script(self, tmp_path):
        import subprocess
        import sys

        cfg = yaml.safe_load((REPO / "experiments" / "configs" / "smoke.yaml").read_text())
        cfg["data"]["synthetic"].update(n_normal=40, n_anomaly=10)
        cfg["data"]["graphs"]["output"] = str(tmp_path / "graphs")
        cfg["output"] = {k: str(tmp_path / k)
                         for k in ("checkpoints", "results", "figures")}
        cfg["benchmark"]["models"] = ["mlp", "gcn"]
        cfg_path = tmp_path / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        out = subprocess.run(
            [sys.executable, str(REPO / "experiments" / "run_benchmark.py"),
             "--config", str(cfg_path), "--epochs", "1", "--device", "cpu",
             "--output", str(tmp_path / "results" / "bench.json")],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        results = json.loads((tmp_path / "results" / "bench.json").read_text())
        assert "mlp" in results and "gcn" in results
        for name, metrics in results.items():
            assert "error" not in metrics, f"{name}: {metrics.get('error')}"
            assert 0.0 <= metrics["accuracy"] <= 1.0
