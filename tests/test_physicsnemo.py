"""PhysicsNeMo integration tests."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from physicsnemo_integration import (
    MeshGraphNetLayer,
    PhysicsNeMoBenchmark,
    PhysicsNeMoWrapper,
)
from tests.helpers import make_graph, make_graphs, make_loader


def _graph(n=5, feature_dim=11, edges=True, edge_attr=True, seed=0):
    g = make_graph(n_nodes=n, feature_dim=feature_dim, seed=seed)
    if not edges:
        g.edge_index = torch.zeros((2, 0), dtype=torch.long)
    if edge_attr:
        g.edge_attr = torch.randn(g.edge_index.shape[1], 4)
    return g


class TestMeshGraphNetLayer:
    def test_with_edges(self):
        layer = MeshGraphNetLayer(8)
        out = layer(torch.randn(4, 8), torch.tensor([[0, 1], [1, 2]]),
                    torch.randn(2, 8))
        assert out.shape == (4, 8)

    def test_without_edge_attr(self):
        layer = MeshGraphNetLayer(8)
        out = layer(torch.randn(4, 8), torch.tensor([[0, 1], [1, 2]]), None)
        assert out.shape == (4, 8)

    def test_edgeless_graph(self):
        layer = MeshGraphNetLayer(8)
        out = layer(torch.randn(3, 8), None, None)
        assert out.shape == (3, 8)

    def test_backward(self):
        layer = MeshGraphNetLayer(8)
        layer(torch.randn(4, 8), torch.tensor([[0, 1], [1, 2]]),
              torch.randn(2, 8)).sum().backward()

    def test_single_edge(self):
        layer = MeshGraphNetLayer(8)
        out = layer(torch.randn(2, 8), torch.tensor([[0], [1]]), torch.randn(1, 8))
        assert out.shape == (2, 8)


class TestPhysicsNeMoWrapper:
    def test_falls_back_when_unavailable(self):
        """The native backend is not installed here, so the fallback is used."""
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        assert m._physicsnemo_available is False
        assert hasattr(m, "node_encoder") and hasattr(m, "processors")

    def test_no_partial_submodules_left_behind(self):
        """A failed native init must not leave orphan params in the state dict."""
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2, use_physicsnemo=True)
        assert not hasattr(m, "mesh_graph_net")

    def test_forward(self):
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        batch = Batch.from_data_list([_graph(), _graph(seed=1)])
        assert m(batch).shape == (2, 2)

    def test_forward_without_edge_attr(self):
        """Previously crashed: the native path needs explicit edge features."""
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        batch = Batch.from_data_list([_graph(edge_attr=False)])
        assert m(batch).shape == (1, 2)

    def test_forward_edgeless(self):
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        batch = Batch.from_data_list([_graph(edges=False, edge_attr=False)])
        assert m(batch).shape == (1, 2)

    def test_missing_x_raises(self):
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        with pytest.raises(ValueError, match="requires data.x"):
            m(Data(edge_index=torch.zeros((2, 0), dtype=torch.long)))

    def test_backward(self):
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        m(Batch.from_data_list([_graph(), _graph(seed=1)])).sum().backward()

    def test_predict_helpers(self):
        m = PhysicsNeMoWrapper(input_dim=11, hidden_dim=16, latent_dim=8,
                               num_layers=2)
        batch = Batch.from_data_list([_graph()])
        assert m.predict(batch).shape == (1,)
        probs = m.predict_proba(batch)
        assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)

    def test_self_check_rejects_wrong_shape(self, monkeypatch):
        """A mis-wired native backend must fall back, not emit garbage."""
        m = PhysicsNeMoWrapper.__new__(PhysicsNeMoWrapper)
        torch.nn.Module.__init__(m)
        m._physicsnemo_available = False

        class Bogus(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(2))

            def forward(self, *a, **k):
                # Wrong arity entirely.
                raise TypeError("unexpected keyword argument 'graph_size'")

        m.mesh_graph_net = Bogus()
        assert m._probe_physicsnemo(11, 8) is False

    def test_self_check_rejects_bad_shape(self):
        m = PhysicsNeMoWrapper.__new__(PhysicsNeMoWrapper)
        torch.nn.Module.__init__(m)
        m._physicsnemo_available = False

        class WrongShape(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(2))

            def forward(self, x, edge_attr, graph_size, edge_index=None):
                return torch.zeros(3, 5)  # expected (3, 8)

        m.mesh_graph_net = WrongShape()
        assert m._probe_physicsnemo(11, 8) is False


class TestPhysicsNeMoBenchmark:
    def test_unknown_model_raises(self):
        bench = PhysicsNeMoBenchmark(input_dim=11, hidden_dim=16, latent_dim=8,
                                    device="cpu")
        train = make_loader(make_graphs(8, n_nodes=4), batch_size=4)
        val = make_loader(make_graphs(4, n_nodes=4, seed=50), batch_size=4)
        with pytest.raises(ValueError, match="Unknown model"):
            bench._train_and_evaluate("banana", train, val, val, 1)

    def test_benchmark_records_errors_and_continues(self):
        """A failing model must not abort the whole sweep."""
        bench = PhysicsNeMoBenchmark(input_dim=11, hidden_dim=16, latent_dim=8,
                                    device="cpu")
        train, val, test = (make_loader(make_graphs(8, n_nodes=4), batch_size=4)
                            for _ in range(3))
        results = bench.run_benchmark(train, val, test,
                                      models_to_test=["mlp", "banana"], epochs=1)
        assert "mlp" in results and "error" not in results["mlp"]
        assert "error" in results["banana"]
        assert "ValueError" in results["banana"]["error"]

    def test_results_serializable(self, tmp_path):
        import json

        bench = PhysicsNeMoBenchmark(input_dim=11, hidden_dim=16, latent_dim=8,
                                    device="cpu")
        bench.results = {"mlp": {"accuracy": 0.5, "n_parameters": 10,
                                 "confusion_matrix": [[1, 0], [0, 1]]}}
        out = tmp_path / "r.json"
        bench.save_results(str(out))
        assert json.loads(out.read_text())["mlp"]["accuracy"] == 0.5

    def test_print_comparison_handles_empty(self, capsys):
        PhysicsNeMoBenchmark(device="cpu").print_comparison()
        assert "No results yet" in capsys.readouterr().out

    def test_print_comparison_handles_errors(self, capsys):
        bench = PhysicsNeMoBenchmark(device="cpu")
        bench.results = {"mlp": {"accuracy": 0.9, "auroc": 0.8, "f1": 0.8,
                                 "precision": 0.9, "recall": 0.8,
                                 "n_parameters": 100, "training_time_sec": 1.0},
                         "bad": {"error": "ValueError: boom"}}
        bench.print_comparison()
        out = capsys.readouterr().out
        assert "ERROR" in out and "0.9000" in out
