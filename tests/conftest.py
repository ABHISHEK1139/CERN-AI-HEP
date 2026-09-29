"""
Shared pytest fixtures.

Also makes the repository root importable so tests can
``import anomaly_engine`` without an editable install.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Graph builders
# --------------------------------------------------------------------------

def make_graph(
    n_nodes: int = 5,
    feature_dim: int = 11,
    label: int = 0,
    seed: int = 0,
) -> Data:
    """A small ring-connected graph with a ring adjacency (no self-loops)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, feature_dim, generator=g)
    idx = torch.arange(n_nodes)
    src = torch.cat([idx, (idx + 1) % n_nodes])
    dst = torch.cat([(idx + 1) % n_nodes, idx])
    return Data(
        x=x,
        edge_index=torch.stack([src, dst]).long(),
        y=torch.tensor([label], dtype=torch.long),
    )


def make_graphs(n: int = 20, **kwargs) -> list:
    """``n`` graphs with alternating labels 0/1."""
    return [make_graph(label=i % 2, seed=i, **kwargs) for i in range(n)]


@pytest.fixture
def small_graph() -> Data:
    return make_graph()


@pytest.fixture
def graph_list():
    return make_graphs(20)


@pytest.fixture
def graph_maker():
    """Expose :func:`make_graph` as a fixture.

    Helpers live here rather than in an importable module because a top-level
    ``tests`` package name can be shadowed by an unrelated distribution in
    site-packages. Fixtures sidestep the naming collision entirely.
    """
    return make_graph


@pytest.fixture
def graphs_maker():
    return make_graphs


@pytest.fixture
def graph_loader(graph_list):
    from torch_geometric.loader import DataLoader

    return DataLoader(graph_list, batch_size=4, shuffle=False)


@pytest.fixture
def tiny_loader():
    from torch_geometric.loader import DataLoader

    return DataLoader(make_graphs(8, n_nodes=5), batch_size=4, shuffle=False)


@pytest.fixture
def tmp_checkpoints(tmp_path) -> Path:
    return tmp_path / "checkpoints"


@pytest.fixture(autouse=True)
def _deterministic():
    """Keep every test reproducible regardless of execution order."""
    torch.manual_seed(0)
    yield
