"""Mouse telemetry traces: storage, slicing, resampling.

A trace is the raw record of relative mouse motion (Raw Input counts) plus
left-button click times. Everything downstream (flick segmentation, bias,
heatmaps, replays) is derived from this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MouseTrace:
    """Timestamped relative mouse motion.

    t       epoch seconds, monotonic non-decreasing, one per motion packet
    dx, dy  raw counts per packet (mouse coordinates: +x right, +y down)
    clicks  epoch seconds of left-button presses
    """

    t: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    dx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    dy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    clicks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return self.t.size

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0

    # ------------------------------------------------------------------ io
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, dx=self.dx, dy=self.dy, clicks=self.clicks)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "MouseTrace":
        z = np.load(path)
        return cls(t=z["t"], dx=z["dx"], dy=z["dy"], clicks=z["clicks"])

    # -------------------------------------------------------------- slicing
    def window(self, t0: float, t1: float) -> "MouseTrace":
        """Sub-trace covering [t0, t1] (epoch seconds)."""
        i0, i1 = np.searchsorted(self.t, [t0, t1])
        c0, c1 = np.searchsorted(self.clicks, [t0, t1])
        return MouseTrace(
            t=self.t[i0:i1].copy(),
            dx=self.dx[i0:i1].copy(),
            dy=self.dy[i0:i1].copy(),
            clicks=self.clicks[c0:c1].copy(),
        )

    # ------------------------------------------------------------ resample
    def resample(self, rate: float = 500.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform-grid velocities.

        Returns (tg, vx, vy): tg is the uniform time grid (epoch s) at `rate`
        Hz; vx/vy are counts-per-second obtained by binning packet deltas into
        grid cells. Empty grid cells = zero velocity (mouse at rest).
        """
        if self.t.size < 2:
            return (np.empty(0), np.empty(0), np.empty(0))
        t0, t1 = self.t[0], self.t[-1]
        n = max(int(np.ceil((t1 - t0) * rate)), 1)
        tg = t0 + np.arange(n) / rate
        idx = np.clip(((self.t - t0) * rate).astype(np.int64), 0, n - 1)
        vx = np.bincount(idx, weights=self.dx, minlength=n) * rate
        vy = np.bincount(idx, weights=self.dy, minlength=n) * rate
        return tg, vx, vy

    def path(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cumulative crosshair displacement in counts (y flipped so +y = up,
        i.e. screen/aim convention for plotting)."""
        return self.t, np.cumsum(self.dx, dtype=np.float64), -np.cumsum(self.dy, dtype=np.float64)


class TraceStore:
    """Per-run trace files under <profile_dir>/traces/<scenario slug>/."""

    def __init__(self, profile_dir: Path) -> None:
        self.root = Path(profile_dir) / "traces"

    def path_for(self, scenario: str, started_iso: str) -> Path:
        import re

        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", scenario)
        ts = started_iso.replace(":", "-")
        return self.root / slug / f"{ts}.npz"

    def save(self, trace: MouseTrace, scenario: str, started_iso: str) -> Path:
        return trace.save(self.path_for(scenario, started_iso))

    def load(self, scenario: str, started_iso: str) -> MouseTrace | None:
        p = self.path_for(scenario, started_iso)
        return MouseTrace.load(p) if p.is_file() else None
