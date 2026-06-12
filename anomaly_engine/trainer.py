"""
Training pipeline with MLflow integration.

Supports both:
- Supervised training (classification)
- Unsupervised training (autoencoder)

Features:
- MLflow experiment tracking
- Early stopping
- Learning rate scheduling
- Checkpoint saving
- Gradient clipping
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Unified training pipeline for classifiers and autoencoders."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        use_mlflow: bool = False,
        experiment_name: str = "cern-ai",
    ):
        """
        Args:
            model: PyTorch model to train.
            device: Device string ('auto', 'cuda', 'cpu').
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight.
            patience: Early stopping patience (epochs).
            max_grad_norm: Maximum gradient norm for clipping.
            checkpoint_dir: Directory for model checkpoints.
            use_mlflow: Whether to log to MLflow.
            experiment_name: MLflow experiment name.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        # MLflow
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self._mlflow_run = None

        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        logger.info(f"Trainer initialized: device={self.device}, lr={learning_rate}")

    def _init_mlflow(self, run_name: str, params: Dict[str, Any]):
        """Initialize MLflow tracking."""
        if not self.use_mlflow:
            return
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}. Continuing without tracking.")
            self.use_mlflow = False

    def _log_mlflow(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow."""
        if not self.use_mlflow:
            return
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def _end_mlflow(self):
        """End MLflow run."""
        if self.use_mlflow and self._mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

    # ----------------------------------------------------------------
    # Classification Training
    # ----------------------------------------------------------------

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        run_name: str = "classifier",
    ) -> Dict[str, Any]:
        """
        Train a graph classifier.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum epochs.
            run_name: Name for this training run.

        Returns:
            Training history dict.
        """
        criterion = nn.CrossEntropyLoss()

        params = {
            "model": self.model.__class__.__name__,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": epochs,
            "mode": "classification",
        }
        self._init_mlflow(run_name, params)

        logger.info(f"Training classifier for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self._train_epoch_classifier(train_loader, criterion)

            # Validate
            val_loss, val_acc = self._eval_epoch_classifier(val_loader, criterion)

            # Scheduler
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            self._log_mlflow(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "train_acc": train_acc, "val_acc": val_acc, "lr": lr},
                step=epoch,
            )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs}: "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} lr={lr:.2e}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(f"{run_name}_best.pt")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self._end_mlflow()
        self._load_checkpoint(f"{run_name}_best.pt")

        return self.history

    def _train_epoch_classifier(self, loader, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = criterion(logits, data.y.squeeze())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs
            pred = logits.argmax(dim=-1)
            correct += (pred == data.y.squeeze()).sum().item()
            total += data.num_graphs

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch_classifier(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            logits = self.model(data)
            loss = criterion(logits, data.y.squeeze())

            total_loss += loss.item() * data.num_graphs
            pred = logits.argmax(dim=-1)
            correct += (pred == data.y.squeeze()).sum().item()
            total += data.num_graphs

        return total_loss / total, correct / total

    # ----------------------------------------------------------------
    # Autoencoder Training
    # ----------------------------------------------------------------

    def train_autoencoder(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        run_name: str = "autoencoder",
    ) -> Dict[str, Any]:
        """
        Train a graph autoencoder (unsupervised).

        Args:
            train_loader: Training DataLoader (normal events only for best results).
            val_loader: Validation DataLoader.
            epochs: Maximum epochs.
            run_name: Name for this training run.

        Returns:
            Training history dict.
        """
        params = {
            "model": "GraphAutoencoder",
            "encoder": self.model.encoder.__class__.__name__,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": epochs,
            "mode": "autoencoder",
        }
        self._init_mlflow(run_name, params)

        logger.info(f"Training autoencoder for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch_autoencoder(train_loader)
            val_loss = self._eval_epoch_autoencoder(val_loader)

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            self._log_mlflow(
                {"train_loss": train_loss, "val_loss": val_loss, "lr": lr},
                step=epoch,
            )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs}: "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.2e}"
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(f"{run_name}_best.pt")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self._end_mlflow()
        self._load_checkpoint(f"{run_name}_best.pt")

        return self.history

    def _train_epoch_autoencoder(self, loader):
        self.model.train()
        total_loss = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            result = self.model(data)
            loss = result["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total += data.num_graphs

        return total_loss / total

    @torch.no_grad()
    def _eval_epoch_autoencoder(self, loader):
        self.model.eval()
        total_loss = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            result = self.model(data)
            total_loss += result["loss"].item() * data.num_graphs
            total += data.num_graphs

        return total_loss / total

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)

    def _load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from {path}")
