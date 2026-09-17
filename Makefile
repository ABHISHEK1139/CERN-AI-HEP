# =============================================================================
# CERN-AI Makefile
# =============================================================================

.PHONY: help install download synthetic graphs train-classifier train-autoencoder benchmark clean

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
	@echo "  make clean             Remove generated caches (asks first; never touches data/ or checkpoints/)"

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
	python experiments/train_autoencoder.py --epochs 100 --batch-size 256

benchmark:
	python experiments/run_benchmark.py --data data/graphs/

clean:
	@echo "This removes generated caches and results (data/ and checkpoints/ are left alone)."
	@read -p "Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -rf reports/ results/ mlruns/ outputs/ models/ .pytest_cache/; \
		find . -name "__pycache__" -type d -prune -exec rm -rf {} +; \
	else \
		echo "Aborted."; \
	fi
