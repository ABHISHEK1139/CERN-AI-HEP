# Default: the official PyTorch CUDA runtime image (torch 2.1.0 + cu121).
# The base image already satisfies `torch>=2.0` from pyproject.toml, so an
# unpinned `pip install torch` here would replace the CUDA build with a
# CPU/default wheel. Do not add one.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="CERN-AI-HEP"
LABEL org.opencontainers.image.description="GNN anomaly detection for LHC collision events (JetClass, CMS NanoAOD, LHCO 2020)."
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="Abhishek <ak612520208365@gmail.com>"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# libgl/libglib are needed by matplotlib; git is used by the downloader.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch-scatter / torch-sparse are OPTIONAL accelerators. They need a C++
# toolchain and a wheel matching the exact torch/CUDA build, so they are not
# installed by default. Every model in this project runs without them; PyG
# falls back to native torch ops.
#
# To enable them, uncomment after pinning to this image's torch version:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         g++ cmake python3-dev && rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir \
#         torch-scatter torch-sparse \
#         -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Dependencies first so the layer is cached across source changes.
COPY pyproject.toml README.md LICENSE ./
COPY anomaly_engine ./anomaly_engine
COPY event_ingestion ./event_ingestion
COPY graph_builder ./graph_builder
COPY physicsnemo_integration ./physicsnemo_integration

RUN pip install --upgrade pip && pip install .

# Everything else (experiments, demo, configs, scripts).
COPY . .

# Streamlit demo port:
#   docker run -p 8501:8501 <image> streamlit run demo.py --server.address=0.0.0.0
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import anomaly_engine, graph_builder, event_ingestion" || exit 1

# Default CMD: the CPU smoke benchmark (synthetic data, no downloads needed).
# The large JetClass run needs data/jetclass/*.root, which .dockerignore
# excludes; mount a volume for that:
#   docker run -v ./data:/app/data <image> \
#       python experiments/train_jetclass.py --large --arch edgeconv
CMD ["python", "experiments/run_benchmark.py", \
     "--config", "experiments/configs/smoke.yaml", \
     "--models", "gcn", "mlp", \
     "--epochs", "2", \
     "--device", "cpu"]
