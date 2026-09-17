"""Fast chunked-dataset smoke check (no heavy data required)."""
import torch
from glob import glob
from torch_geometric.utils import scatter


class FastChunkedDataset:
    def __init__(self, chunk_files, batch_size=2048, device=None):
        self.chunk_files = chunk_files
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __iter__(self):
        for fpath in self.chunk_files:
            # Load chunk
            chunk = torch.load(fpath, map_location=self.device, weights_only=True)
            x_all = chunk['x'].to(self.device)
            lengths = chunk['lengths'].to(self.device)
            y_all = chunk['y'].to(self.device)

            num_jets = x_all.size(0)
            for start_idx in range(0, num_jets, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_jets)
                B = end_idx - start_idx

                x_batch = x_all[start_idx:end_idx]
                lengths_batch = torch.clamp(lengths[start_idx:end_idx], max=128)
                y_batch = y_all[start_idx:end_idx]

                # Filter out padding and construct the flat batch
                mask = torch.arange(128, device=self.device).unsqueeze(0) < lengths_batch.unsqueeze(1)
                x_collated = x_batch[mask]
                batch_idx_tensor = torch.arange(B, device=self.device).unsqueeze(1).expand(B, 128)[mask]

                class BatchData:
                    def __init__(self, x, batch, y):
                        self.x = x
                        self.batch = batch
                        self.y = y

                yield BatchData(x=x_collated, batch=batch_idx_tensor, y=y_batch)


def test_fast_chunked_iter_skips_without_data():
    """Without processed chunks there is nothing to iterate; must not crash."""
    import pytest

    chunk_files = sorted(glob("data/jetclass/processed_chunks/chunk_*.pt"))
    if not chunk_files:
        pytest.skip("No processed chunks (expected without heavy JetClass download)")
    dataset = FastChunkedDataset(chunk_files[:1], batch_size=2048)
    for i, data in enumerate(dataset):
        assert data.x.shape[0] > 0
        assert data.batch.shape[0] == data.x.shape[0]
        if i >= 1:
            break


def test_scatter_smoke():
    """Sanity check that torch_geometric scatter works in this env."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    batch = torch.tensor([0, 0, 1, 1])
    out = scatter(x, batch, reduce="mean")
    assert out.tolist() == [1.5, 3.5]


if __name__ == "__main__":
    test_fast_chunked_iter_skips_without_data()
    test_scatter_smoke()
    print("fast checks passed.")
