"""FlickEncoder: 1D-CNN -> GRU -> 64-dim embedding + regression heads.

Input is the ``(B, 3, N_SAMPLES)`` normalized velocity curve from
``dataset.py``; the convolutional front-end extracts local shape features
(acceleration ramps, correction wiggles), the GRU integrates them along
the flick's time base, and the embedding feeds three small heads that
regress ``TRAIN_HEADS`` = (overshoot fraction, correction count,
log-duration). The default configuration is ~1.4M parameters — big enough
to be worth the GPU, small enough to train in minutes on a few thousand
flicks.

This is the ONE ``kovadapt.ml`` submodule that needs torch at import time
(it subclasses ``nn.Module``); per the package contract it is only ever
imported lazily behind an ``ML_AVAILABLE``/try-except guard.
"""

from __future__ import annotations

import torch
from torch import nn

from .dataset import TRAIN_HEADS

#: Constructor kwargs of the default (checkpoint-shipped) configuration.
DEFAULT_CONFIG: dict = {
    "in_channels": 3,
    "conv_channels": (64, 128, 256),
    "gru_hidden": 320,
    "gru_layers": 2,
    "embed_dim": 64,
}


class FlickEncoder(nn.Module):
    """Curve -> (predictions over TRAIN_HEADS, embedding)."""

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        gru_hidden: int = 320,
        gru_layers: int = 2,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        # Kept for checkpoint round-trip: FlickEncoder(**ckpt["config"]).
        self.config: dict = {
            "in_channels": in_channels,
            "conv_channels": tuple(conv_channels),
            "gru_hidden": gru_hidden,
            "gru_layers": gru_layers,
            "embed_dim": embed_dim,
        }
        blocks: list[nn.Module] = []
        prev = in_channels
        for ch in conv_channels:  # each block halves the time axis: 64 -> 8
            blocks += [
                nn.Conv1d(prev, ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            prev = ch
        self.conv = nn.Sequential(*blocks)
        self.gru = nn.GRU(prev, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.embed = nn.Linear(gru_hidden, embed_dim)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1)
                )
                for name in TRAIN_HEADS
            }
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` curves -> ``(B, embed_dim)`` embeddings."""
        h = self.conv(x)                      # (B, C', T')
        _, hn = self.gru(h.transpose(1, 2))   # hn: (layers, B, H)
        return self.embed(hn[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> ``(pred, embedding)``; ``pred`` is ``(B, len(TRAIN_HEADS))``
        in ``TRAIN_HEADS`` order, in standardized target units (training
        standardizes targets; infer.py de-normalizes with the checkpoint's
        mean/std)."""
        z = self.encode(x)
        pred = torch.cat([self.heads[name](z) for name in TRAIN_HEADS], dim=1)
        return pred, z


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
