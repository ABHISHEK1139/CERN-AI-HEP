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
- Intra-epoch checkpointing for large streaming runs
"""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the training path can touch.

    Args:
        seed: Non-negative integer seed.
        deterministic: When True, request deterministic cuDNN kernels so that
            repeated runs on the same hardware reproduce bit-for-bit.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}.")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device='{device}' requested but CUDA is not available on this host."
            )

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
        self.history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

        logger.info("Trainer initialized: device=%s, lr=%s", self.device, learning_rate)

    def _init_mlflow(self, run_name: str, params: dict[str, Any]):
        """Initialize MLflow tracking."""
        if not self.use_mlflow:
            return
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(params)
        except Exception as e:
            logger.warning("MLflow init failed: %s. Continuing without tracking.", e)
            self.use_mlflow = False

    def _log_mlflow(self, metrics: dict[str, float], step: int):
        """Log metrics to MLflow."""
        if not self.use_mlflow:
            return
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.debug("MLflow metric logging failed: %s", e)

    def _end_mlflow(self):
        """End MLflow run."""
        if self.use_mlflow and self._mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                logger.debug("MLflow end_run failed: %s", e)
            finally:
                self._mlflow_run = None

    # ----------------------------------------------------------------
    # Classification Training
    # ----------------------------------------------------------------

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        run_name: str = "classifier",
        resume: bool = False,
    ) -> dict[str, Any]:
        """
        Train a graph classifier.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum epochs.
            run_name: Name for this training run.
            resume: Whether to resume from the latest checkpoint.

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

        start_epoch = 0
        if resume:
            start_epoch = self._load_checkpoint(f"{run_name}_latest.pt", restore_history=False)
            if start_epoch:
                logger.info("Resuming classifier training from epoch %d", start_epoch + 1)

        for epoch in range(start_epoch + 1, epochs + 1):
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
                    "Epoch %3d/%d: train_loss=%.4f train_acc=%.3f "
                    "val_loss=%.4f val_acc=%.3f lr=%.2e",
                    epoch, epochs, train_loss, train_acc, val_loss, val_acc, lr,
                )

            # Always save latest for resumable training, *before* the
            # early-stop check, so a stopped run is resumable from the epoch it
            # actually completed. (Saving after the break would discard the
            # final completed epoch and silently redo it on --resume.)
            self._save_checkpoint(f"{run_name}_latest.pt", epoch)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(f"{run_name}_best.pt", epoch)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self._end_mlflow()
        # Restore the best weights. restore_history=False keeps the full
        # in-memory curve: reloading the best checkpoint used to overwrite
        # self.history with that checkpoint's (shorter) history, silently
        # truncating the returned loss curve at the best epoch.
        self._load_checkpoint(f"{run_name}_best.pt", restore_history=False)

        return self.history

    @staticmethod
    def _require_labels(data):
        """Return flattened long labels or raise a clear error for unlabeled batches."""
        y = getattr(data, "y", None)
        if y is None:
            raise ValueError(
                "Classification training requires data.y labels, but this batch "
                "has y=None. Use train_autoencoder() for unsupervised graphs."
            )
        return y.view(-1).long()

    def _train_epoch_classifier(self, loader, criterion):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(data)
            target = self._require_labels(data)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += data.num_graphs

        if total == 0:
            raise ValueError("Classifier training received an empty DataLoader.")
        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch_classifier(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data in loader:
            data = data.to(self.device)
            logits = self.model(data)
            target = self._require_labels(data)
            loss = criterion(logits, target)

            total_loss += loss.item() * data.num_graphs
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += data.num_graphs

        if total == 0:
            raise ValueError("Classifier evaluation received an empty DataLoader.")
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
        resume: bool = False,
        save_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Train a graph autoencoder (unsupervised).

        Args:
            train_loader: Training DataLoader (normal events only for best results).
            val_loader: Validation DataLoader.
            epochs: Maximum epochs.
            run_name: Name for this training run.
            resume: Whether to resume from the latest checkpoint.

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

        start_epoch = 0
        start_batch = 0
        if resume:
            start_epoch, start_batch = self._load_checkpoint_with_batch(
                f"{run_name}_latest.pt", restore_history=True
            )
            if start_batch > 0:
                logger.info("Resuming within epoch %d at batch %d", start_epoch, start_batch)
                # Tell IterableDataset to skip
                if hasattr(train_loader.dataset, 'start_idx'):
                    bs = getattr(train_loader, 'batch_size', 1) or 1
                    train_loader.dataset.start_idx = start_batch * bs

        initial_epoch = start_epoch if start_batch > 0 else start_epoch + 1

        for epoch in range(initial_epoch, epochs + 1):
            train_loss = self._train_epoch_autoencoder(
                train_loader,
                epoch=epoch,
                run_name=run_name,
                save_steps=save_steps,
                start_batch=start_batch if (start_batch > 0 and epoch == initial_epoch) else 0
            )
            # Reset start_batch and start_idx after first epoch
            start_batch = 0
            if hasattr(train_loader.dataset, 'start_idx'):
                train_loader.dataset.start_idx = 0

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
                    "Epoch %3d/%d: train_loss=%.6f val_loss=%.6f lr=%.2e",
                    epoch, epochs, train_loss, val_loss, lr,
                )

            # Save latest before the early-stop check (see train_classifier).
            self._save_checkpoint(f"{run_name}_latest.pt", epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(f"{run_name}_best.pt", epoch)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self._end_mlflow()
        # restore_history=False: keep the full curve collected this run.
        self._load_checkpoint(f"{run_name}_best.pt", restore_history=False)

        return self.history

    def _train_epoch_autoencoder(
        self,
        loader,
        epoch: int = 0,
        run_name: str = "",
        save_steps: int | None = None,
        start_batch: int = 0,
    ):
        self.model.train()
        total_loss = 0.0
        total = 0

        # Create progress bar if save_steps is used (likely a large dataset)
        pbar = None
        if save_steps is not None:
            try:
                total_batches = len(loader)
            except TypeError:
                total_batches = None
            pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}")

        skip_to = max(0, start_batch)
        for batch_idx, data in enumerate(loader):
            # Skip batches already consumed by a resumed map-style dataset.
            # IterableDatasets cannot be re-indexed, so they are advanced via
            # dataset.start_idx instead and must not be skipped here.
            if (
                batch_idx < skip_to
                and not isinstance(getattr(loader, "dataset", None), IterableDataset)
            ):
                if pbar is not None:
                    pbar.update(1)
                continue

            data = data.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            result = self.model(data)
            if not isinstance(result, dict) or "loss" not in result:
                raise TypeError(
                    "Autoencoder training needs a model returning dict(loss=...). "
                    f"Got {type(result).__name__}."
                )
            loss = result["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            batch_graphs = data.num_graphs
            total_loss += loss.item() * batch_graphs
            total += batch_graphs

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": total_loss / total})

            # Intra-epoch checkpointing
            if save_steps and (batch_idx + 1) % save_steps == 0:
                logger.info("Saving intra-epoch checkpoint at batch %d...", batch_idx + 1)
                self._save_checkpoint(f"{run_name}_latest.pt", epoch, batch_idx + 1)

        if pbar is not None:
            pbar.close()

        if total == 0:
            raise ValueError(
                "Autoencoder training received an empty DataLoader "
                f"(epoch {epoch}, start_batch {start_batch})."
            )
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

        if total == 0:
            raise ValueError("Autoencoder validation received an empty DataLoader.")
        return total_loss / total

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_checkpoint(self, filename: str, epoch: int = 0, batch_idx: int = 0):
        path = self.checkpoint_dir / filename
        # Copy the history: torch.save would otherwise serialize a live
        # reference, and a later in-place append could leak into the file.
        torch.save({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": {k: list(v) for k, v in self.history.items()},
            "epochs_without_improvement": self.epochs_without_improvement,
        }, path)

    def _load_checkpoint(self, filename: str, restore_history: bool = True) -> int:
        epoch, _ = self._load_checkpoint_with_batch(
            filename, restore_history=restore_history
        )
        return epoch

    def _load_checkpoint_with_batch(
        self, filename: str, restore_history: bool = True
    ) -> tuple[int, int]:
        """Load a checkpoint if present.

        Args:
            filename: Checkpoint filename inside :attr:`checkpoint_dir`.
            restore_history: When False, the checkpoint's ``history`` is
                ignored and the in-memory history is left untouched. Pass False
                when reloading weights at the end of training, otherwise the
                full loss curve is silently truncated to the best epoch's.

        Returns:
            ``(epoch, batch_idx)`` from the checkpoint, or ``(0, 0)`` when the
            file does not exist.
        """
        path = self.checkpoint_dir / filename
        if not path.exists():
            logger.info("Checkpoint %s not found; starting fresh.", path)
            return 0, 0

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to read checkpoint {path}: {exc}") from exc

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint {path} is malformed: expected a dict with "
                "'model_state_dict'."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if restore_history:
            self.history = checkpoint.get(
                "history", {"train_loss": [], "val_loss": [], "lr": []}
            )
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        epoch = checkpoint.get("epoch", 0)
        batch_idx = checkpoint.get("batch_idx", 0)
        logger.info("Loaded checkpoint from %s (epoch %d, batch %d)", path, epoch, batch_idx)
        return epoch, batch_idx
