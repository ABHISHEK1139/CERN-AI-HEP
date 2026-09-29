# =============================================================================
# CERN-AI Makefile  (POSIX shell: use Git Bash / WSL on Windows)
# =============================================================================

.PHONY: help install install-dev lint format test coverage smoke classifier \
        benchmark clean clean-cache

PYTHON ?= python

help:
	@echo "CERN-AI: GNN Anomaly Detection for LHC Events"
	@echo ""
	@echo "  make install          Install the project and its dependencies"
	@echo "  make install-dev      Install with test/lint tooling (dev extra)"
	@echo "  make lint             Run ruff"
	@echo "  make test             Run the test suite"
	@echo "  make coverage         Run tests with a coverage report"
	@echo "  make smoke            End-to-end synthetic run (no downloads, ~1 min)"
	@echo "  make classifier       Train the GCN classifier on the default config"
	@echo "  make benchmark        Run the full multi-architecture benchmark"
	@echo "  make clean            Remove generated caches (asks first; never touches data/ or checkpoints/)"
	@echo ""
	@echo "Data acquisition (large downloads, run deliberately):"
	@echo "  make download         Download a CMS Open Data sample"
	@echo "  make synthetic        Generate the synthetic event dataset"
	@echo "  make graphs           Build event graphs from synthetic events"
	@echo "  make train-autoencoder Train the LHCO autoencoder (needs the H5 file)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term-missing

# Fast, self-contained verification: synthetic data only, CPU, two epochs.
smoke:
	$(PYTHON) experiments/train_classifier.py \
		--config experiments/configs/smoke.yaml --model gcn --epochs 2 --device cpu
	$(PYTHON) experiments/run_benchmark.py \
		--config experiments/configs/smoke.yaml \
		--models mlp cnn gcn --epochs 1 --device cpu \
		--output results/smoke/benchmark.json

# --- data acquisition -------------------------------------------------------

download:
	$(PYTHON) -m event_ingestion.downloader

synthetic:
	$(PYTHON) -m event_ingestion.synthetic --n-normal 10000 --n-anomaly 1000 --output data/synthetic/

graphs:
	$(PYTHON) -m graph_builder.graph_constructor --input data/synthetic/ --output data/graphs/ --strategy knn --k 8

# --- training ---------------------------------------------------------------

classifier:
	$(PYTHON) experiments/train_classifier.py --config experiments/configs/default.yaml --model gcn

train-autoencoder:
	$(PYTHON) experiments/train_autoencoder.py --epochs 100 --batch-size 256

benchmark:
	$(PYTHON) experiments/run_benchmark.py --config experiments/configs/default.yaml

# --- housekeeping -----------------------------------------------------------

clean:
	@echo "This removes generated caches and results (data/ and checkpoints/ are left alone)."
	@read -p "Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -rf reports/ results/ mlruns/ outputs/ models/ .pytest_cache/ .ruff_cache/ coverage.xml .coverage; \
		find . -name "__pycache__" -type d -prune -exec rm -rf {} +; \
	else \
		echo "Aborted."; \
	fi

# No prompt, for CI and scripts. Still never touches data/ or checkpoints/.
clean-cache:
	rm -rf reports/ results/ mlruns/ outputs/ models/ .pytest_cache/ .ruff_cache/ coverage.xml .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
