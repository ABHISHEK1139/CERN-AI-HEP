"""Normalization helpers for the GNN models.

The central piece is :class:`SafeBatchNorm1d`: a drop-in ``nn.BatchNorm1d``
that does not crash on degenerate (single-sample) batches. Plain
``nn.BatchNorm1d`` raises ``ValueError`` when the batch size is 1 in training
mode — a routine occurrence here (single-node graphs, single-edge batches) —
so the safe variant falls back to the running statistics for those batches
without updating them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that tolerates batch-size-1 inputs in training mode.

    For a degenerate batch the layer normalizes with the running statistics
    (like eval mode) instead of raising, and leaves ``running_mean``,
    ``running_var`` and ``num_batches_tracked`` untouched. Buffers, parameters
    and ``state_dict`` keys are identical to ``nn.BatchNorm1d``, so existing
    checkpoints keep loading.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            # Degenerate batch: use running statistics, do not update them.
            out = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            if self.affine:
                out = out * self.weight + self.bias
            return out
        return super().forward(x)


def resolve_norm(num_features: int, name: str) -> nn.Module:
    """Factory for normalization layers.

    Args:
        num_features: feature dimension to normalize.
        name: one of ``"batch"`` (:class:`SafeBatchNorm1d`), ``"layer"``
            (``nn.LayerNorm``) or ``"none"`` (``nn.Identity``).

    Raises:
        ValueError: for an unknown norm name.
    """
    if name == "batch":
        return SafeBatchNorm1d(num_features)
    if name == "layer":
        return nn.LayerNorm(num_features)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name!r} (expected 'batch', 'layer' or 'none')")


def apply_norm(norm: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
    """Apply a normalization module, skipping no-ops and unsafe cases.

    - ``None``/``nn.Identity`` → returns ``x`` unchanged.
    - A plain ``nn.BatchNorm1d`` (not :class:`SafeBatchNorm1d`) on a
      batch-size-1 training input → running-statistics fallback instead of
      the ``ValueError`` that plain batch norm raises (defence in depth).
    - Otherwise → ``norm(x)``.
    """
    if norm is None or isinstance(norm, nn.Identity):
        return x
    if (
        isinstance(norm, nn.BatchNorm1d)
        and not isinstance(norm, SafeBatchNorm1d)
        and norm.training
        and x.size(0) == 1
    ):
        out = (x - norm.running_mean) / torch.sqrt(norm.running_var + norm.eps)
        if norm.affine:
            out = out * norm.weight + norm.bias
        return out
    return norm(x)
