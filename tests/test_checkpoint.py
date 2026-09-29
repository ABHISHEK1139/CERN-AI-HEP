"""Strict checkpoint loading tests.

The scripts these cover used to silently keep randomly initialised weights when
a checkpoint was absent, then publish the resulting scores and figures as if
they were real results.
"""

import pytest
import torch
import torch.nn as nn

from anomaly_engine.checkpoint import (
    MissingCheckpointError,
    load_checkpoint,
    load_state_dict,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x):
        return self.lin(x)


def _write(path, model, extra=None):
    payload = {"model_state_dict": model.state_dict(), "epoch": 7}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


class TestLoadCheckpoint:
    def test_missing_raises_by_default(self, tmp_path):
        model = Tiny()
        with pytest.raises(MissingCheckpointError, match="randomly initialised"):
            load_checkpoint(tmp_path / "nope.pt", model=model)

    def test_missing_message_is_actionable(self, tmp_path):
        with pytest.raises(MissingCheckpointError) as exc:
            load_checkpoint(tmp_path / "nope.pt", model=Tiny())
        msg = str(exc.value)
        assert "nope.pt" in msg          # names the file
        assert "train" in msg.lower()   # says what to do
        assert "randomly initialised" in msg  # says what it refuses to do

    def test_optional_missing_warns_and_returns_empty(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            result = load_checkpoint(tmp_path / "nope.pt", model=Tiny(),
                                     require=False)
        assert result == {}
        assert "randomly initialised" in caplog.text

    def test_loads_weights(self, tmp_path):
        src, dst = Tiny(), Tiny()
        _write(tmp_path / "c.pt", src)
        load_state_dict(tmp_path / "c.pt", dst)
        assert torch.allclose(src.lin.weight, dst.lin.weight)

    def test_returns_full_payload(self, tmp_path):
        model = Tiny()
        payload = _write(tmp_path / "c.pt", model, extra={"history": {"train_loss": [1.0]}})
        ckpt = load_checkpoint(payload, model=model)
        assert ckpt["epoch"] == 7
        assert ckpt["history"]["train_loss"] == [1.0]

    def test_shape_mismatch_raises(self, tmp_path):
        other = nn.Linear(9, 5)
        _write(tmp_path / "c.pt", Tiny())
        with pytest.raises(RuntimeError):
            load_state_dict(tmp_path / "c.pt", other)

    def test_missing_key_raises(self, tmp_path):
        torch.save({"epoch": 1}, tmp_path / "c.pt")
        with pytest.raises(ValueError, match="no 'model_state_dict' key"):
            load_checkpoint(tmp_path / "c.pt", model=Tiny())

    def test_non_dict_raises(self, tmp_path):
        p = tmp_path / "c.pt"
        torch.save([1, 2, 3], p)
        with pytest.raises(ValueError, match="expected a dict"):
            load_checkpoint(p, model=Tiny())

    def test_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "c.pt"
        p.write_bytes(b"definitely not a checkpoint")
        with pytest.raises(ValueError, match="Could not read checkpoint"):
            load_checkpoint(p, model=Tiny())

    def test_non_strict_reports_missing(self, tmp_path, caplog):
        _write(tmp_path / "c.pt", Tiny())
        other = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2))
        with caplog.at_level("WARNING"):
            load_checkpoint(tmp_path / "c.pt", model=other, strict=False)
        assert "strict=False" in caplog.text

    def test_custom_key(self, tmp_path):
        model = Tiny()
        torch.save({"weights": model.state_dict()}, tmp_path / "c.pt")
        load_state_dict(tmp_path / "c.pt", model, key="weights")
        assert model.lin.weight is not None

    def test_map_location(self, tmp_path):
        model = Tiny()
        _write(tmp_path / "c.pt", model)
        load_state_dict(tmp_path / "c.pt", model, device="cpu")
        assert model.lin.weight.device.type == "cpu"
