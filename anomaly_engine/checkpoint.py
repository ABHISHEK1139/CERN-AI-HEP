"""
Strict checkpoint loading.

Several scripts used to do this::

    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    model.eval()

which silently left the model at **random initialisation** when the checkpoint
was absent, and then published ROC curves, t-SNE plots and anomaly scores
computed from noise as if they were research results. The helpers here make a
missing or mismatched checkpoint an explicit, early failure.
"""

import logging
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]  # noqa: UP007 - py3.9 compat in type aliases


class MissingCheckpointError(FileNotFoundError):
    """Raised when a checkpoint that a result depends on does not exist."""


def load_checkpoint(
    path: PathLike,
    model: nn.Module | None = None,
    device: str | torch.device = "cpu",
    key: str = "model_state_dict",
    strict: bool = True,
    require: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, failing loudly when it is required but missing.

    Args:
        path: Checkpoint file.
        model: If given, load ``checkpoint[key]`` into it.
        device: Device to map the checkpoint onto.
        key: Key holding the state dict.
        strict: Require an exact parameter-name/shape match.
        require: Raise :class:`MissingCheckpointError` if ``path`` is absent.
            Set False only for genuinely optional checkpoints, and log a
            warning in that case.

    Returns:
        The full checkpoint dict.

    Raises:
        MissingCheckpointError: If the file is absent and ``require`` is True.
        ValueError: If the file exists but is not a dict, or lacks ``key``.
    """
    p = Path(path)
    if not p.exists():
        msg = (
            f"Checkpoint not found: {p}. Train the model first, or pass an "
            f"explicit --checkpoint path. Refusing to continue with randomly "
            f"initialised weights, which would make any reported metric meaningless."
        )
        if require:
            raise MissingCheckpointError(msg)
        logger.warning("%s Continuing with random weights.", msg)
        return {}

    try:
        ckpt = torch.load(p, map_location=device, weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not read checkpoint {p}: {exc}") from exc

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Checkpoint {p} is a {type(ckpt).__name__}, expected a dict."
        )
    if key not in ckpt:
        raise ValueError(
            f"Checkpoint {p} has no {key!r} key. Available: {sorted(ckpt)}"
        )

    if model is not None:
        missing, unexpected = model.load_state_dict(ckpt[key], strict=strict)
        if not strict and (missing or unexpected):
            logger.warning(
                "Loaded %s with strict=False. Missing: %s Unexpected: %s",
                p, sorted(missing), sorted(unexpected),
            )
    return ckpt


def load_state_dict(
    path: PathLike,
    model: nn.Module,
    device: str | torch.device = "cpu",
    key: str = "model_state_dict",
) -> nn.Module:
    """Load a checkpoint into ``model`` and return the model."""
    load_checkpoint(path, model=model, device=device, key=key, strict=True)
    model.eval()
    return model
