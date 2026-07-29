"""Checkpoint loading + per-flick scoring for the FlickEncoder.

The scorer is the read side of the neural workstream: load the checkpoint
written by ``train.py`` (if any) and score a run's flicks against it.

Score semantics: the model learns the *population-normal* relation between
a flick's curve shape and its outcome (overshoot, corrections, duration).
Per flick we report the signed, standardized prediction residual per head
(``residual[h] = (actual - predicted) / train_std``) and collapse the
badness-aligned residuals into one scalar:

    quality = -(z_overshoot + z_corrections + z_log_duration) / 3

clipped to [-3, 3]. Positive quality = the flick came out *better* than
its own movement shape predicts (cleaner than usual for that kind of
motion); negative = worse. The 64-dim embedding rides along for
downstream clustering/visualization.

Heads whose training target was constant carry no z scale at all and are
dropped from both the residuals and the mean (see ``_MIN_TRAIN_STD``).

Everything degrades gracefully (CLAUDE.md contract): ``load_scorer``
returns ``None`` when torch is missing, the checkpoint does not exist, or
it fails to load — callers need no torch-awareness beyond a None check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..analysis.movement import Flick, segment_flicks
from ..telemetry.trace import MouseTrace
from .dataset import flick_curves
from .train import CHECKPOINT_NAME

if TYPE_CHECKING:  # torch is optional at runtime ([ml] extra); types only here
    from .model import FlickEncoder

#: train.py standardizes targets with ``std.clamp_min(1e-6)``, so a head that
#: was *constant* across the train split (every flick undershot -> overshoot
#: 0.0, every flick clean -> corrections 0) ships a std sitting on that floor.
#: Dividing a real difference by it yields ~1e5 z-units, which pins every
#: flick's quality at the -3 clip and writes the nonsense into the report
#: JSON. Such a head has no calibrated scale, so it is scored as no signal
#: rather than as catastrophe. NaN stds (a one-sample split) fail this test
#: too, which is the behaviour we want.
_MIN_TRAIN_STD = 1e-5


@dataclass
class FlickScore:
    t_click: float
    quality: float                                  # [-3, 3], + = better than predicted
    pred: dict[str, float] = field(default_factory=dict)      # de-normalized, per head
    actual: dict[str, float] = field(default_factory=dict)
    residual: dict[str, float] = field(default_factory=dict)  # z-units, + = worse
    embedding: list[float] = field(default_factory=list)      # 64-dim


class FlickScorer:
    """A loaded FlickEncoder checkpoint, ready to score traces."""

    def __init__(
        self,
        model: FlickEncoder,
        mean: np.ndarray,
        std: np.ndarray,
        heads: list[str],
        n_samples: int,
        rate: float,
    ) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        # Heads with real training variance — the only ones a residual means
        # anything for (see _MIN_TRAIN_STD).
        self.scored = np.asarray(std, dtype=np.float64) > _MIN_TRAIN_STD
        self.heads = heads
        self.n_samples = n_samples
        self.rate = rate

    def score(
        self, trace: MouseTrace, flicks: list[Flick] | None = None
    ) -> list[FlickScore]:
        """Score every scoreable flick of ``trace`` (pass ``flicks`` to
        reuse an existing segmentation). Flicks whose curve is degenerate
        are skipped, so the list can be shorter than the segmentation."""
        import torch

        if flicks is None:
            flicks = segment_flicks(trace, rate=self.rate)
        x, kept_idx = flick_curves(trace, flicks, rate=self.rate, n_samples=self.n_samples)
        if x.shape[0] == 0:
            return []
        self.model.eval()
        with torch.no_grad():
            pred_z, emb = self.model(torch.from_numpy(x))
        pred = pred_z.numpy() * self.std + self.mean          # de-normalized
        emb = emb.numpy()

        # Degenerate heads divide by the training clamp, so neutralize them.
        denom = np.where(self.scored, self.std, 1.0)
        out: list[FlickScore] = []
        for row, i in enumerate(kept_idx):
            f = flicks[i]
            actual = np.array(
                [f.overshoot, float(f.corrections), np.log(max(f.duration, 1e-3))],
                dtype=np.float64,
            )
            # + = worse than predicted; unscored heads report no residual and
            # are left out of the mean instead of drowning it.
            z = np.where(self.scored, (actual - pred[row]) / denom, 0.0)
            live = z[self.scored]
            quality = float(np.clip(-live.mean(), -3.0, 3.0)) if live.size else 0.0
            out.append(
                FlickScore(
                    t_click=f.t_click,
                    quality=quality,
                    pred={h: float(pred[row][j]) for j, h in enumerate(self.heads)},
                    actual={h: float(actual[j]) for j, h in enumerate(self.heads)},
                    residual={h: float(z[j]) for j, h in enumerate(self.heads)},
                    embedding=[float(v) for v in emb[row]],
                )
            )
        return out


def summarize(scores: list[FlickScore]) -> dict:
    """Compact, JSON-ready digest of a run's flick scores.

    This is what the watcher stamps into ``RunReport.ml`` — the report JSON
    must stay small and serializable, so per-flick embeddings/residuals are
    reduced to run-level aggregates plus the best/worst flick anchors
    (``t_click`` epoch times, cross-referencable with notable moments and
    clips via the shared time base)."""
    if not scores:
        return {}
    qs = np.array([s.quality for s in scores], dtype=np.float64)
    worst = min(scores, key=lambda s: s.quality)
    best = max(scores, key=lambda s: s.quality)
    return {
        "n_scored": len(scores),
        "mean_quality": round(float(qs.mean()), 4),
        "p10_quality": round(float(np.percentile(qs, 10)), 4),
        "worst_quality": round(worst.quality, 4),
        "worst_t_click": worst.t_click,
        "best_quality": round(best.quality, 4),
        "best_t_click": best.t_click,
        "mean_residual": {
            h: round(float(np.mean([s.residual.get(h, 0.0) for s in scores])), 4)
            for h in scores[0].residual
        },
    }


def load_scorer(profile_dir: Path | str) -> FlickScorer | None:
    """Load ``<profile_dir>/ml/flick_encoder.pt`` -> scorer, or ``None``
    when torch is missing, the checkpoint does not exist, or it is
    unreadable/corrupt (never raises)."""
    try:
        import torch
    except Exception:
        return None
    path = Path(profile_dir) / "ml" / CHECKPOINT_NAME
    if not path.is_file():
        return None
    try:
        from .model import FlickEncoder

        ckpt = torch.load(path, map_location="cpu")
        model = FlickEncoder(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return FlickScorer(
            model=model,
            mean=ckpt["target_mean"].numpy().astype(np.float64),
            std=ckpt["target_std"].numpy().astype(np.float64),
            heads=list(ckpt["heads"]),
            n_samples=int(ckpt["n_samples"]),
            rate=float(ckpt["rate"]),
        )
    except Exception:
        return None
