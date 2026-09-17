import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch


def test_cms_dataset_importable():
    """CMSDataset module must be importable (uproot/awkward installed)."""
    try:
        from graph_builder.cms_dataset import CMSDataset  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Optional CMS deps missing: {e}")


def test_cms_dataset_requires_data():
    """Loading the real CMS ROOT file requires data; skip gracefully if absent."""
    try:
        from graph_builder.cms_dataset import CMSDataset
    except ImportError as e:
        pytest.skip(f"Optional CMS deps missing: {e}")
    root_file = Path("data/cms/higgs/GluGluToHToTauTau.root")
    if not root_file.exists():
        pytest.skip("CMS Higgs ROOT file not present (expected without heavy download)")

    dataset = CMSDataset(
        root="data/cms/graphs",
        root_file_path=str(root_file),
        label=1,
        sample_size=2,
    )
    assert len(dataset) > 0
    graph = dataset[0]
    assert graph.x is not None and graph.x.dim() == 2 and graph.x.shape == (2, 4)
    assert graph.edge_index is not None and graph.edge_index.shape == (2, 2)
    assert graph.y is not None and int(graph.y.flatten()[0].item()) == 1
    assert torch.isfinite(graph.x).all()


if __name__ == "__main__":
    test_cms_dataset_importable()
    print("CMS import check passed (data test may skip).")
