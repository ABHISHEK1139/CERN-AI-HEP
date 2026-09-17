import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


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
    assert graph.x is not None
    assert graph.edge_index is not None
    assert graph.y is not None


if __name__ == "__main__":
    test_cms_dataset_importable()
    print("CMS import check passed (data test may skip).")
