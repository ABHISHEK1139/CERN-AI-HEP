FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Abhishek <ak612520208365@gmail.com>"
LABEL description="Graph Neural Network Anomaly Detection for High Energy Physics Collision Events"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

COPY . .

RUN pip install .

CMD ["python", "experiments/run_6m_ablation.py"]
