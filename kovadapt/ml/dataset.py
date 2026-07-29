"""Flick dataset over the stored telemetry trace library.

Each sample is one aimed movement (a ``Flick`` from analysis/movement.py),
re-derived from its trace as a fixed-length, scale-free velocity curve:

  x  ``(3, N_SAMPLES)`` float32 — the flick resampled onto its own
     [onset, click] time base:
       0  |v| / peak speed — the speed profile, amplitude/duration-normalized
       1  vx / |v|         — unit direction, aim convention (+y up)
       2  vy / |v|
  y  ``(4,)`` float32 regression targets (``TARGET_NAMES`` order):
     overshoot fraction, correction count, log duration (s), log amplitude
     (counts). The model's heads train on the first three (``TRAIN_HEADS``);
     amplitude is carried along for analysis/conditioning.

Normalizing amplitude and duration out of ``x`` forces the model to judge
curve *shape* — acceleration profile, curvature, correction wiggles — which
is what transfers across sensitivities and scenario scales.

Split: a deterministic per-sample hash (crc32 of
``"<scenario slug>/<trace stem>:<flick index>"``) sends ~20% of samples to
validation, so a given trace library always yields the byte-identical split
across runs, machines and torch versions.

Torch is imported lazily (CLAUDE.md contract): the numpy layer
(``extract_samples``, ``flick_curves``, ``iter_trace_files``,
``is_val_sample``) works on a core install; ``build_dataset`` returns
``None`` when torch is missing.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..analysis.movement import Flick, _smooth, segment_flicks
from ..telemetry.trace import MouseTrace, ResampleCache

if TYPE_CHECKING:  # torch is optional at runtime ([ml] extra); types only here
    import torch

N_SAMPLES = 64
RATE = 500.0  # analysis grid rate (Hz), matches segment_flicks' default

TARGET_NAMES = ("overshoot", "corrections", "log_duration", "log_amplitude")
TRAIN_HEADS = TARGET_NAMES[:3]

_VAL_PERCENT = 20


def is_val_sample(trace_id: str, flick_idx: int) -> bool:
    """Deterministic ~20% validation membership for one flick sample."""
    key = f"{trace_id}:{flick_idx}".encode()
    return zlib.crc32(key) % 100 < _VAL_PERCENT


def iter_trace_files(traces_root: Path | str) -> list[Path]:
    """All ``*.npz`` traces under ``<profile_dir>/traces``, deterministic order."""
    root = Path(traces_root)
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.npz"), key=lambda p: str(p).lower())


def flick_curves(
    trace: MouseTrace,
    flicks: list[Flick],
    *,
    rate: float = RATE,
    n_samples: int = N_SAMPLES,
    grid: ResampleCache | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Fixed-length normalized velocity curves for ``flicks``.

    Returns ``(x, kept)``: ``x`` is ``(m, 3, n_samples)`` float32 and
    ``kept`` the indices into ``flicks`` that produced a row (flicks whose
    grid slice is degenerate — under 4 samples or zero peak — are dropped).
    Uses the same smoothing as segment_flicks so onset/click indices line
    up with the features the segmenter saw.
    """
    tg, vx, vy = (grid if grid is not None else trace).resample(rate)
    if tg.size == 0 or not flicks:
        return np.zeros((0, 3, n_samples), dtype=np.float32), []
    vy = -vy  # aim convention: +y up (Y-axis contract in CLAUDE.md)
    vxs, vys = _smooth(vx, rate), _smooth(vy, rate)
    speed = np.hypot(vxs, vys)

    dst = np.linspace(0.0, 1.0, n_samples)
    rows: list[np.ndarray] = []
    kept: list[int] = []
    n = tg.size
    for i, f in enumerate(flicks):
        i0 = int(np.searchsorted(tg, f.t_onset))
        i1 = min(int(np.searchsorted(tg, f.t_click)), n)
        seg_s = speed[i0:i1]
        if seg_s.size < 4:
            continue
        pk = float(seg_s.max())
        if pk <= 0:
            continue
        seg_x, seg_y = vxs[i0:i1], vys[i0:i1]
        safe = np.maximum(seg_s, 1e-9)
        src = np.linspace(0.0, 1.0, seg_s.size)
        rows.append(np.stack([
            np.interp(dst, src, seg_s / pk),
            np.interp(dst, src, seg_x / safe),
            np.interp(dst, src, seg_y / safe),
        ]).astype(np.float32))
        kept.append(i)
    if not rows:
        return np.zeros((0, 3, n_samples), dtype=np.float32), []
    return np.stack(rows), kept


def flick_targets(flicks: list[Flick]) -> np.ndarray:
    """``(n, 4)`` float32 targets in ``TARGET_NAMES`` order."""
    if not flicks:
        return np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
    return np.array(
        [
            [
                f.overshoot,
                float(f.corrections),
                np.log(max(f.duration, 1e-3)),
                np.log(max(f.amplitude, 1.0)),
            ]
            for f in flicks
        ],
        dtype=np.float32,
    )


def extract_samples(
    trace: MouseTrace,
    *,
    rate: float = RATE,
    n_samples: int = N_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, list[Flick]]:
    """Segment ``trace`` and return ``(x, y, kept_flicks)``.

    ``x`` is ``(n, 3, n_samples)`` float32 curves, ``y`` the matching
    ``(n, 4)`` float32 targets. Pure numpy — importable and runnable
    without torch.
    """
    grid = ResampleCache(trace)
    flicks = segment_flicks(trace, rate=rate, grid=grid)
    x, kept_idx = flick_curves(trace, flicks, rate=rate, n_samples=n_samples, grid=grid)
    kept = [flicks[i] for i in kept_idx]
    return x, flick_targets(kept), kept


@dataclass
class FlickDataset:
    """Train/val tensors plus provenance counts."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    n_traces: int
    n_flicks: int


def build_dataset(
    traces_root: Path | str,
    *,
    rate: float = RATE,
    n_samples: int = N_SAMPLES,
) -> FlickDataset | None:
    """Walk the trace library and assemble the flick dataset.

    Returns ``None`` when torch is not installed. Corrupt/unreadable trace
    files are skipped silently (same policy as analysis/skill.py's report
    loader). Empty splits come back as zero-length tensors — callers decide
    their own minimum-data policy.
    """
    try:
        import torch  # noqa: F811 — runtime import of the TYPE_CHECKING name
    except Exception:
        return None

    xs: dict[bool, list[np.ndarray]] = {False: [], True: []}
    ys: dict[bool, list[np.ndarray]] = {False: [], True: []}
    n_traces = n_flicks = 0
    for p in iter_trace_files(traces_root):
        try:
            tr = MouseTrace.load(p)
        except Exception:
            continue
        x, y, _kept = extract_samples(tr, rate=rate, n_samples=n_samples)
        if x.shape[0] == 0:
            continue
        n_traces += 1
        trace_id = f"{p.parent.name}/{p.stem}"
        for i in range(x.shape[0]):
            val = is_val_sample(trace_id, i)
            xs[val].append(x[i])
            ys[val].append(y[i])
            n_flicks += 1

    def tensor(rows: list[np.ndarray], shape: tuple[int, ...]) -> "torch.Tensor":
        if not rows:
            return torch.zeros(shape, dtype=torch.float32)
        return torch.from_numpy(np.stack(rows))

    return FlickDataset(
        x_train=tensor(xs[False], (0, 3, n_samples)),
        y_train=tensor(ys[False], (0, len(TARGET_NAMES))),
        x_val=tensor(xs[True], (0, 3, n_samples)),
        y_val=tensor(ys[True], (0, len(TARGET_NAMES))),
        n_traces=n_traces,
        n_flicks=n_flicks,
    )
