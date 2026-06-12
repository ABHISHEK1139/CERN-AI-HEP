# =============================================================================
# CERN-AI Makefile
# =============================================================================

.PHONY: help install data synthetic graphs train-classifier train-autoencoder benchmark clean

help:
	@echo "CERN-AI: GNN Anomaly Detection for LHC Events"
	@echo ""
	@echo "  make install           Install dependencies"
	@echo "  make download          Download CMS Open Data sample"
	@echo "  make synthetic         Generate synthetic dataset"
	@echo "  make graphs            Build event graphs"
	@echo "  make train-classifier  Train GNN classifier"
	@echo "  make train-autoencoder Train graph autoencoder"
	@echo "  make benchmark         Run full benchmark"
	@echo "  make clean             Remove generated data and models"

install:
	pip install -r requirements.txt
	pip install -e .

download:
	python -c "from event_ingestion.downloader import CMSDataDownloader; CMSDataDownloader().download()"

synthetic:
	python -m event_ingestion.synthetic --n-normal 10000 --n-anomaly 1000 --output data/synthetic/

graphs:
	python -m graph_builder.graph_constructor --input data/synthetic/ --output data/graphs/ --strategy knn --k 8

train-classifier:
	python experiments/train_classifier.py --model gcn --data data/graphs/ --epochs 100

train-autoencoder:
	python experiments/train_autoencoder.py --encoder gcn --data data/graphs/ --epochs 100

benchmark:
	python experiments/run_benchmark.py --data data/graphs/

clean:
	rm -rf data/ models/ checkpoints/ mlruns/ outputs/
