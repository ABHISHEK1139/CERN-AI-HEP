"""Demo-jet smoke test (does not require heavy JetClass download)."""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_demo_module_importable():
    try:
        import demo  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Demo optional deps missing: {e}")


def test_get_sample_jet_handles_missing_data():
    """get_sample_jet should return None (not crash) when demo ROOT files are absent."""
    try:
        from demo import get_sample_jet
    except ImportError as e:
        pytest.skip(f"Demo optional deps missing: {e}")

    bg_files = sorted(Path("data/jetclass/val_5M").glob("ZJetsToNuNu_*.root"))
    sig_files = sorted(Path("data/jetclass/val_5M").glob("HTo*.root"))
    if not bg_files or not sig_files:
        # Function must degrade gracefully.
        assert get_sample_jet("bg", 42) is None
        assert get_sample_jet("sig", 42) is None
        pytest.skip("Demo ROOT files absent (expected); graceful-None path verified")
    jet_bg = get_sample_jet("bg", 42)
    jet_sig = get_sample_jet("sig", 42)
    assert jet_bg is not None and jet_sig is not None
    # Different jets may have different node counts; compare safely.
    same_shape = jet_bg.x.shape == jet_sig.x.shape
    assert not (same_shape and bool(torch.allclose(jet_bg.x, jet_sig.x)))


if __name__ == "__main__":
    test_demo_module_importable()
    test_get_sample_jet_handles_missing_data()
    print("demo checks passed.")
