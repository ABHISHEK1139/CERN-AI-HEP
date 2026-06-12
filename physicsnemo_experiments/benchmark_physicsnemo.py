"""
PhysicsNeMo (NVIDIA Modulus) vs PyTorch Geometric Benchmark
============================================================
Compares inference speed and GPU memory between:
  1. PyTorch Geometric EdgeConv encoder + MLP decoder (our production model)
  2. NVIDIA Modulus FullyConnected network as a drop-in decoder replacement

This demonstrates that our PyG graph construction pipeline is fully compatible
with NVIDIA's scientific AI framework, enabling future multi-GPU scaling on
national lab clusters (DGX, Perlmutter, etc.).

Note: Modulus's graph-native models (MeshGraphNet, GraphCast) require DGL,
which is Linux-only. On Windows, we benchmark using Modulus's FullyConnected
layers as the decoder component, proving framework interoperability.
"""

import os
import sys
import time
import torch
import logging

os.environ.setdefault("HOME", os.environ.get("USERPROFILE", "."))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from torch_geometric.loader import DataLoader
from graph_builder.jetclass_dataset import JetClassDataset
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_gpu_memory_mb():
    """Returns current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_pyg(sample_graphs, device, n_passes=100):
    """Benchmark the full PyG EdgeConv Autoencoder pipeline."""
    encoder = EdgeConvEncoder(input_dim=16, hidden_dim=64, latent_dim=32, num_layers=3)
    decoder = GraphDecoder(latent_dim=32, hidden_dim=64, output_dim=16)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder).to(device)
    model.eval()

    # Load best checkpoint if available
    ckpt_path = Path("checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt")
    if ckpt_path.exists():
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
        )
        logger.info("Loaded trained weights for PyG benchmark.")

    # Warmup
    with torch.no_grad():
        for g in sample_graphs[:5]:
            _ = model(g.to(device))

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_before = get_gpu_memory_mb()

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n_passes):
            g = sample_graphs[i % len(sample_graphs)].to(device)
            _ = model(g)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    mem_after = get_gpu_memory_mb()
    return elapsed, max(mem_after, mem_before)


def benchmark_modulus(sample_graphs, device, n_passes=100):
    """Benchmark using NVIDIA Modulus FullyConnected as decoder."""
    from modulus.models.mlp import FullyConnected
    from torch_geometric.nn import global_mean_pool

    # Use same PyG encoder (graph construction is framework-agnostic)
    encoder = EdgeConvEncoder(input_dim=16, hidden_dim=64, latent_dim=32, num_layers=3).to(device)

    # Replace our custom MLP decoder with NVIDIA Modulus FullyConnected
    modulus_decoder = FullyConnected(
        in_features=32,
        out_features=16,
        num_layers=3,
        layer_size=64,
    ).to(device)

    encoder.eval()
    modulus_decoder.eval()

    # Warmup
    with torch.no_grad():
        for g in sample_graphs[:5]:
            g = g.to(device)
            z = encoder(g.x, g.edge_index, g.batch)
            z_pooled = global_mean_pool(z, g.batch)
            _ = modulus_decoder(z_pooled)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_before = get_gpu_memory_mb()

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n_passes):
            g = sample_graphs[i % len(sample_graphs)].to(device)
            z = encoder(g.x, g.edge_index, g.batch)
            z_pooled = global_mean_pool(z, g.batch)
            _ = modulus_decoder(z_pooled)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    mem_after = get_gpu_memory_mb()
    return elapsed, max(mem_after, mem_before)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    # Load validation graphs
    val_file = "data/jetclass/val_5M/ZJetsToNuNu_120.root"
    if not Path(val_file).exists():
        logger.error(f"Validation file not found: {val_file}")
        return

    dataset = JetClassDataset(
        root="data/jetclass/graphs",
        root_file_paths=[val_file],
        k_neighbors=8,
        sample_size=500,
        tag="nemo_bench",
    )
    sample_graphs = [dataset[i] for i in range(min(50, len(dataset)))]
    logger.info(f"Loaded {len(sample_graphs)} graphs for benchmarking.")

    N_PASSES = 200

    # --- PyG Benchmark ---
    logger.info("=" * 60)
    logger.info("BENCHMARK 1: PyTorch Geometric (Full EdgeConv Autoencoder)")
    logger.info("=" * 60)
    pyg_time, pyg_mem = benchmark_pyg(sample_graphs, device, N_PASSES)
    logger.info(f"  Time ({N_PASSES} passes): {pyg_time:.4f}s")
    logger.info(f"  Per-pass:              {pyg_time/N_PASSES*1000:.2f}ms")
    logger.info(f"  GPU Memory:            {pyg_mem:.1f} MB")

    # --- Modulus Benchmark ---
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("=" * 60)
    logger.info("BENCHMARK 2: NVIDIA Modulus (PyG Encoder + Modulus Decoder)")
    logger.info("=" * 60)
    try:
        nemo_time, nemo_mem = benchmark_modulus(sample_graphs, device, N_PASSES)
        logger.info(f"  Time ({N_PASSES} passes): {nemo_time:.4f}s")
        logger.info(f"  Per-pass:              {nemo_time/N_PASSES*1000:.2f}ms")
        logger.info(f"  GPU Memory:            {nemo_mem:.1f} MB")
    except Exception as e:
        import traceback
        logger.error(f"  Modulus benchmark failed: {e}")
        traceback.print_exc()
        nemo_time, nemo_mem = None, None

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'PyG':>12} {'Modulus':>12}")
    logger.info(f"{'-'*25} {'-'*12} {'-'*12}")
    logger.info(f"{'Time ('+str(N_PASSES)+' passes)':<25} {pyg_time:>10.4f}s {(f'{nemo_time:.4f}s' if nemo_time else 'N/A'):>12}")
    logger.info(f"{'Per-pass latency':<25} {pyg_time/N_PASSES*1000:>9.2f}ms {(f'{nemo_time/N_PASSES*1000:.2f}ms' if nemo_time else 'N/A'):>12}")
    logger.info(f"{'GPU Memory':<25} {pyg_mem:>9.1f}MB {(f'{nemo_mem:.1f}MB' if nemo_mem else 'N/A'):>12}")


if __name__ == "__main__":
    main()
