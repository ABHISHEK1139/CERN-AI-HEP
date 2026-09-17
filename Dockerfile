FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# NOTE: requirements.txt asks for torch>=2.0, which pip treats as already
# satisfied by the base image's torch 2.1.0 — it will NOT reinstall/overwrite
# the CUDA build. Do not add an unpinned `pip install torch` step here.

LABEL maintainer="Abhishek <ak612520208365@gmail.com>"
LABEL description="Graph Neural Network Anomaly Detection for High Energy Physics Collision Events"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install C++ extensions first from pre-compiled PyG wheels
# NOTE: torch-scatter / torch-sparse are optional accelerators. They are
# commented out because they need an exact torch/CUDA wheel match and a
# compiler; PyG falls back to native torch ops without them.
# RUN pip install --no-cache-dir \
#     torch-scatter \
#     torch-sparse \
#     -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

# Streamlit demo port (run with: docker run -p 8501:8501 <image> streamlit run demo.py)
EXPOSE 8501

# Default CMD runs the CPU smoke benchmark (synthetic data, no downloads).
# The large JetClass run needs data/jetclass/*.root which is excluded from
# the image by .dockerignore; run it explicitly with a mounted data volume:
#   docker run -v ./data:/app/data <image> python experiments/train_jetclass.py --large ...
CMD ["python", "experiments/run_benchmark.py", "--config", "experiments/configs/smoke.yaml", "--models", "gcn", "mlp", "--epochs", "2", "--device", "cpu"]
