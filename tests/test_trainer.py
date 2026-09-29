"""Trainer regression tests: history integrity, checkpointing, resume."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from anomaly_engine.models import get_autoencoder, get_classifier
from anomaly_engine.trainer import Trainer, set_seed
from tests.helpers import make_graph, make_graphs


@pytest.fixture
def classifier():
    return get_classifier("mlp", input_dim=11, hidden_dim=16)


@pytest.fixture
def autoencoder():
    return get_autoencoder("gcn", input_dim=11, hidden_dim=16, latent_dim=8,
                           num_layers=2)


@pytest.fixture
def loaders():
    return (
        DataLoader(make_graphs(16, n_nodes=5), batch_size=4, shuffle=False),
        DataLoader(make_graphs(8, n_nodes=5, seed=100), batch_size=4, shuffle=False),
    )


def force_val_losses(trainer, values):
    """Drive validation loss deterministically, ignoring the data.

    Lets a test place the best epoch at a known position without depending on
    optimisation dynamics.
    """
    it = iter(values)
    trainer._train_epoch_classifier = lambda loader, crit: (0.1, 0.5)
    trainer._eval_epoch_classifier = lambda loader, crit: (next(it), 0.5)


# ==========================================================================
# History integrity - the truncation bug
# ==========================================================================

class TestHistoryIntegrity:
    def test_returned_history_is_not_truncated_at_best_epoch(self, classifier,
                                                             loaders, tmp_path):
        """The returned loss curve must contain every epoch that ran.

        Regression: training finished by reloading the best checkpoint, which
        also overwrote ``self.history`` with that checkpoint's shorter history.
        A 10-epoch run where the best epoch was 2 returned 2 entries, so
        ``plot_training_curves`` and any downstream analysis silently lost 80%
        of the data.
        """
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        # Best at epoch 2, then monotonically worse.
        force_val_losses(t, [2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        history = t.train_classifier(train_loader, val_loader, epochs=10,
                                     run_name="t")

        assert len(history["train_loss"]) == 10
        assert len(history["val_loss"]) == 10
        assert history["val_loss"] == [2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                       9.0, 10.0]

    def test_history_survives_for_autoencoder_too(self, autoencoder, loaders,
                                                  tmp_path):
        train_loader, val_loader = loaders
        t = Trainer(autoencoder, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        it = iter([5.0, 1.0, 4.0, 3.0])
        t._train_epoch_autoencoder = lambda *a, **k: 0.2
        t._eval_epoch_autoencoder = lambda loader: next(it)

        history = t.train_autoencoder(train_loader, val_loader, epochs=4,
                                      run_name="ae")
        assert len(history["train_loss"]) == 4

    def test_restore_history_false_keeps_in_memory_curve(self, classifier,
                                                        loaders, tmp_path):
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        force_val_losses(t, [3.0, 1.0, 2.0])
        t.train_classifier(train_loader, val_loader, epochs=3, run_name="h")
        full = list(t.history["val_loss"])

        t._load_checkpoint("h_best.pt", restore_history=False)
        assert t.history["val_loss"] == full

        t._load_checkpoint("h_best.pt", restore_history=True)
        # best.pt is written after the winning epoch is logged, so it holds
        # epochs 1..best - still shorter than the full 3-epoch run.
        assert len(t.history["val_loss"]) == 2
        assert len(full) == 3

    def test_training_curves_plot_from_returned_history(self, classifier,
                                                       loaders, tmp_path):
        """The regression that motivated the fix: plotting must not raise."""
        from anomaly_engine.evaluate import Evaluator

        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        force_val_losses(t, [4.0, 2.0, 3.0, 5.0, 6.0])
        history = t.train_classifier(train_loader, val_loader, epochs=5,
                                     run_name="p")
        out = tmp_path / "curve.png"
        Evaluator(device="cpu").plot_training_curves(history, output_path=str(out))
        assert out.exists()


# ==========================================================================
# Checkpoint / resume integrity
# ==========================================================================

class TestCheckpointing:
    def test_latest_is_last_completed_epoch_on_early_stop(self, classifier,
                                                          loaders, tmp_path):
        """An early-stopped run must remain resumable from where it stopped.

        Saving ``_latest`` after the early-stop ``break`` discarded the final
        completed epoch, so ``--resume`` silently redid it.
        """
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=2,
                    checkpoint_dir=str(tmp_path))
        force_val_losses(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        t.train_classifier(train_loader, val_loader, epochs=8, run_name="e")

        best = torch.load(tmp_path / "e_best.pt", map_location="cpu",
                          weights_only=False)
        latest = torch.load(tmp_path / "e_latest.pt", map_location="cpu",
                            weights_only=False)
        assert latest["epoch"] >= best["epoch"]
        assert latest["epoch"] == 3  # patience=2 -> stops after epoch 3

    def test_resume_starts_after_completed_epoch(self, classifier, loaders,
                                                 tmp_path):
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        force_val_losses(t, [1.0, 2.0])
        t.train_classifier(train_loader, val_loader, epochs=2, run_name="r")

        t2 = Trainer(get_classifier("mlp", input_dim=11, hidden_dim=16),
                     device="cpu", patience=100, checkpoint_dir=str(tmp_path))
        started = []
        t2._train_epoch_classifier = lambda loader, crit: (started.append(1), (0.1, 0.5))[1]
        t2._eval_epoch_classifier = lambda loader, crit: (0.5, 0.5)
        t2.train_classifier(train_loader, val_loader, epochs=5, run_name="r",
                            resume=True)
        assert len(started) == 3  # epochs 3, 4, 5

    def test_resume_without_checkpoint_starts_from_scratch(self, classifier,
                                                          loaders, tmp_path):
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        assert t._load_checkpoint("missing.pt") == 0

    def test_history_snapshot_is_a_copy(self, classifier, loaders, tmp_path):
        """A checkpoint must not alias the live history list."""
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        t._train_epoch_classifier = lambda loader, crit: (0.1, 0.5)
        t._eval_epoch_classifier = lambda loader, crit: (0.2, 0.5)
        t.train_classifier(train_loader, val_loader, epochs=1, run_name="c")

        ckpt = torch.load(tmp_path / "c_latest.pt", map_location="cpu",
                          weights_only=False)
        n_saved = len(ckpt["history"]["train_loss"])
        t.history["train_loss"].append(999.0)
        reloaded = torch.load(tmp_path / "c_latest.pt", map_location="cpu",
                              weights_only=False)
        assert len(reloaded["history"]["train_loss"]) == n_saved

    def test_malformed_checkpoint_raises(self, tmp_path):
        model = get_classifier("mlp", input_dim=11, hidden_dim=16)
        t = Trainer(model, device="cpu", checkpoint_dir=str(tmp_path))
        (tmp_path / "junk.pt").write_bytes(b"not a torch file")
        with pytest.raises(RuntimeError, match="Failed to read checkpoint"):
            t._load_checkpoint("junk.pt")

    def test_checkpoint_without_state_dict_raises(self, classifier, tmp_path):
        t = Trainer(classifier, device="cpu", checkpoint_dir=str(tmp_path))
        torch.save({"epoch": 1}, tmp_path / "nostate.pt")
        with pytest.raises(ValueError, match="malformed"):
            t._load_checkpoint("nostate.pt")

    def test_intra_epoch_checkpoint_records_batch(self, autoencoder, loaders,
                                                 tmp_path):
        train_loader, val_loader = loaders
        t = Trainer(autoencoder, device="cpu", patience=100,
                    checkpoint_dir=str(tmp_path))
        it = iter([1.0, 2.0])
        t._eval_epoch_autoencoder = lambda loader: next(it)
        t.train_autoencoder(train_loader, val_loader, epochs=2, run_name="s",
                            save_steps=2)
        ckpt = torch.load(tmp_path / "s_latest.pt", map_location="cpu",
                          weights_only=False)
        assert ckpt["batch_idx"] == 0  # final save is end-of-epoch


# ==========================================================================
# Input validation and misc
# ==========================================================================

class TestTrainerValidation:
    @pytest.mark.parametrize("kw, msg", [
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": -1}, "learning_rate"),
        ({"patience": 0}, "patience"),
        ({"max_grad_norm": 0}, "max_grad_norm"),
    ])
    def test_rejects_bad_hyperparameters(self, classifier, tmp_path, kw, msg):
        with pytest.raises(ValueError, match=msg):
            Trainer(classifier, device="cpu", checkpoint_dir=str(tmp_path), **kw)

    def test_cuda_request_without_cuda_raises(self, classifier, tmp_path):
        if torch.cuda.is_available():
            pytest.skip("CUDA present, cannot test the unavailable-device path")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            Trainer(classifier, device="cuda", checkpoint_dir=str(tmp_path))

    def test_empty_loader_raises(self, classifier, tmp_path):
        t = Trainer(classifier, device="cpu", checkpoint_dir=str(tmp_path))
        with pytest.raises(ValueError, match="empty DataLoader"):
            t._train_epoch_classifier(DataLoader([], batch_size=2), nn.CrossEntropyLoss())

    def test_unlabeled_data_raises_clear_error(self, classifier, tmp_path):
        """Classification needs labels; the error must say so."""
        from torch_geometric.data import Data

        t = Trainer(classifier, device="cpu", checkpoint_dir=str(tmp_path))
        unlabeled = DataLoader([Data(x=torch.randn(3, 11),
                                     edge_index=torch.tensor([[0], [1]]))],
                               batch_size=2)
        with pytest.raises(ValueError, match="requires data.y"):
            t._train_epoch_classifier(unlabeled, nn.CrossEntropyLoss())

    def test_autoencoder_mode_rejects_classifier_output(self, classifier,
                                                        loaders, tmp_path):
        train_loader, val_loader = loaders
        t = Trainer(classifier, device="cpu", checkpoint_dir=str(tmp_path))
        with pytest.raises(TypeError, match="dict\\(loss="):
            t._train_epoch_autoencoder(train_loader)

    def test_checkpoint_dir_is_created(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "ckpt"
        Trainer(get_classifier("mlp", input_dim=11, hidden_dim=8),
                device="cpu", checkpoint_dir=str(target))
        assert target.is_dir()


class TestSetSeed:
    def test_seed_rejects_negative(self):
        with pytest.raises(ValueError, match="seed"):
            set_seed(-1)

    def test_seed_is_reproducible(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        assert torch.allclose(a, torch.randn(5))

    def test_trains_deterministically(self, loaders, tmp_path):
        train_loader, val_loader = loaders

        def run(seed):
            set_seed(seed)
            model = get_classifier("mlp", input_dim=11, hidden_dim=16)
            t = Trainer(model, device="cpu", patience=100,
                        checkpoint_dir=str(tmp_path / str(seed)), weight_decay=0.0)
            t.train_classifier(train_loader, val_loader, epochs=2, run_name="d")
            return t.history["train_loss"]

        assert run(7) == run(7)
