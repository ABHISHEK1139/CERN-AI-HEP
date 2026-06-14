# =============================================================================
# CERN-AI Docker Environment
# =============================================================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Abhishek <ak612520208365@gmail.com>"
LABEL description="GNN Anomaly Detection for LHC Events"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch-geometric extensions (CUDA-aware)
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy project
COPY . .

# Install project in editable mode
RUN pip install -e .

# Default: run the fully verified ablation study baseline
CMD ["python", "experiments/run_6m_ablation.py"]
