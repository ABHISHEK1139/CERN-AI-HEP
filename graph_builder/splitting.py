"""
Shared train/val/test splitting for every :class:`InMemoryDataset` in this package.

``CollisionEventDataset``, ``JetClassDataset``, ``CMSDataset`` and
``LHCODataset`` all need the same four things: validated split ratios, a
deterministic permutation, a re-collated index subset, and three DataLoaders.
That logic used to be copy-pasted into each class, which meant every fix had to
be applied four times. It lives here once instead.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from torch_geometric.loader import DataLoader

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_geometric.data import InMemoryDataset

logger = logging.getLogger(__name__)

DEFAULT_SPLITS = (0.7, 0.15, 0.15)
DEFAULT_SEED = 42


def validate_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> None:
    """Validate split ratios.

    Raises:
        ValueError: If any ratio is negative or the three do not sum to 1.
            This is a real check rather than an ``assert``: assertions vanish
            under ``python -O``, silently producing garbage splits in
            optimized production runs.
    """
    for name, value in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number, got {value!r}.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint train/val/test index arrays for ``n`` items.

    The permutation is drawn from an isolated ``RandomState`` so that splitting
    never disturbs the global NumPy seed used by data generation.
    """
    validate_ratios(train_ratio, val_ratio, test_ratio)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")

    indices = np.random.RandomState(seed).permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def collate_subset(dataset: "InMemoryDataset", indices: list[int]):
    """Materialize ``indices`` into a new, re-collated dataset instance.

    Bypasses ``__init__`` (which would re-run processing) and rebuilds the
    in-memory store, so the subset shares no tensors with the parent and can be
    shuffled independently.
    """
    indices = [int(i) for i in indices]
    graphs = [dataset.get(i) for i in indices]
    if not graphs:
        raise ValueError("index_select received empty indices.")
    subset = dataset.__class__.__new__(dataset.__class__)
    subset.transform = dataset.transform
    subset.pre_transform = dataset.pre_transform
    subset.pre_filter = getattr(dataset, "pre_filter", None)
    subset._indices = None
    try:
        subset.data, subset.slices = dataset.collate(graphs)
    except TypeError:
        # InMemoryDataset.collate is an instance method in older PyG but a
        # staticmethod in newer releases; support both.
        subset.data, subset.slices = dataset.__class__.collate(graphs)
    subset._data_list = None
    return subset


class SplittableDatasetMixin:
    """Adds ``get_splits`` / ``index_select`` / ``get_loaders`` to a dataset."""

    def index_select(self, indices: list[int]):
        """Return a new dataset holding only ``indices``."""
        return collate_subset(self, indices)

    def get_splits(
        self,
        train_ratio: float = DEFAULT_SPLITS[0],
        val_ratio: float = DEFAULT_SPLITS[1],
        test_ratio: float = DEFAULT_SPLITS[2],
        seed: int = DEFAULT_SEED,
    ):
        """Split the dataset into disjoint train/val/test subsets.

        Raises:
            ValueError: If the ratios are invalid, or a split would be empty.
                An empty split is reported here with the dataset size and
                ratios rather than surfacing later as the opaque
                "index_select received empty indices".
        """
        validate_ratios(train_ratio, val_ratio, test_ratio)
        n = len(self)
        train_idx, val_idx, test_idx = split_indices(
            n, train_ratio, val_ratio, test_ratio, seed
        )
        for name, idx in (("train", train_idx), ("val", val_idx),
                          ("test", test_idx)):
            if len(idx) == 0:
                raise ValueError(
                    f"The {name} split would be empty: {n} total item(s) with "
                    f"ratios train={train_ratio}, val={val_ratio}, "
                    f"test={test_ratio}. Use more data or larger ratios for "
                    f"that split."
                )
        return (
            self.index_select(train_idx.tolist()),
            self.index_select(val_idx.tolist()),
            self.index_select(test_idx.tolist()),
        )

    def get_loaders(
        self,
        batch_size: int = 32,
        train_ratio: float = DEFAULT_SPLITS[0],
        val_ratio: float = DEFAULT_SPLITS[1],
        test_ratio: float = DEFAULT_SPLITS[2],
        seed: int = DEFAULT_SEED,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build the three DataLoaders, warning about any empty split."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")

        train_ds, val_ds, test_ds = self.get_splits(
            train_ratio, val_ratio, test_ratio, seed
        )

        loaders = (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        )
        logger.info(
            "%s DataLoaders: train=%d, val=%d, test=%d, batch_size=%d",
            type(self).__name__, len(train_ds), len(val_ds), len(test_ds), batch_size,
        )
        return loaders
