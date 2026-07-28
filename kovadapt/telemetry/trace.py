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

    t          epoch seconds, monotonic non-decreasing, one per motion packet
    dx, dy     raw counts per packet (mouse coordinates: +x right, +y down)
    clicks     epoch seconds of left-button presses
    clicks_up  epoch seconds of left-button releases (may be empty on traces
               recorded before v0.3 — treat click-hold metrics as unavailable)
    """

    t: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    dx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    dy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    clicks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    clicks_up: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return self.t.size

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0

    # ------------------------------------------------------------------ io
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, dx=self.dx, dy=self.dy,
                            clicks=self.clicks, clicks_up=self.clicks_up)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "MouseTrace":
        z = np.load(path)
        return cls(
            t=z["t"], dx=z["dx"], dy=z["dy"], clicks=z["clicks"],
            clicks_up=z["clicks_up"] if "clicks_up" in z.files
            else np.empty(0, dtype=np.float64),
        )

    # -------------------------------------------------------------- slicing
    def window(self, t0: float, t1: float) -> "MouseTrace":
        """Sub-trace covering [t0, t1] (epoch seconds)."""
        i0, i1 = np.searchsorted(self.t, [t0, t1])
        c0, c1 = np.searchsorted(self.clicks, [t0, t1])
        u0, u1 = np.searchsorted(self.clicks_up, [t0, t1])
        return MouseTrace(
            t=self.t[i0:i1].copy(),
            dx=self.dx[i0:i1].copy(),
            dy=self.dy[i0:i1].copy(),
            clicks=self.clicks[c0:c1].copy(),
            clicks_up=self.clicks_up[u0:u1].copy(),
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

    # --------------------------------------------------------- input health
    def input_health(self) -> dict:
        """Polling/sensor quality metrics from inter-packet timing.

        Raw Input only delivers packets on motion, so only intervals during
        continuous movement estimate the true polling cadence: cadence and
        jitter keep gaps under 30 ms, while the hitch percentile keeps gaps
        under 200 ms — a stall in that range is a pipeline hiccup mid-motion,
        anything longer is the hand at rest, not the sensor.

          polling_hz_est   1 / median moving interval (0 when undetermined)
          jitter_ms        IQR of those intervals — timing consistency; big
                           values mean USB/timer contention (the stutter the
                           optimizer exists to fix)
          gap_ms_p99       99th percentile interval under 200 ms — worst-case
                           hitches during movement
          click_hold_ms    median press->release time (0 without clicks_up)
        """
        out = {"polling_hz_est": 0.0, "jitter_ms": 0.0, "gap_ms_p99": 0.0,
               "click_hold_ms": 0.0}
        if self.t.size >= 100:
            dt = np.diff(self.t)
            moving = dt[(dt > 0) & (dt < 0.030)]
            if moving.size >= 50:
                med = float(np.median(moving))
                q1, q3 = np.percentile(moving, [25, 75])
                out["polling_hz_est"] = round(1.0 / med, 0) if med > 0 else 0.0
                out["jitter_ms"] = round(float(q3 - q1) * 1000.0, 3)
            active = dt[(dt > 0) & (dt < 0.200)]
            if active.size >= 50:
                out["gap_ms_p99"] = round(float(np.percentile(active, 99)) * 1000.0, 3)
        if self.clicks.size and self.clicks_up.size:
            # pair each press with the first release after it
            idx = np.searchsorted(self.clicks_up, self.clicks)
            ok = idx < self.clicks_up.size
            holds = self.clicks_up[idx[ok]] - self.clicks[ok]
            holds = holds[(holds > 0) & (holds < 1.0)]
            if holds.size:
                out["click_hold_ms"] = round(float(np.median(holds)) * 1000.0, 1)
        return out


class ResampleCache:
    """Memoized `MouseTrace.resample` grids shared across analysis passes.

    Duck-types the `resample` method, so analysis code accepts either a bare
    trace or a cache. Each analysis step used to re-bin every packet
    independently; at 4-8 kHz polling over multi-minute runs those per-packet
    passes dominate `build_report`, so the cache computes each grid once —
    and when a grid at exactly twice the requested rate is already cached,
    the coarser grid is *derived* from it by merging adjacent bin pairs
    instead of re-binning the packets.

    The derivation is bitwise identical to `trace.resample(rate)`:

    - Doubling is exact in binary floating point (`fl(2x) == 2*fl(x)`), so
      each packet's bin index at rate r is its index at 2r floor-divided by
      2, and the grid times satisfy `tg_r[j] == tg_2r[2j]`.
    - Bin velocities are integer count sums times the rate — exactly
      representable in float64 — so `(v_2r[2j] + v_2r[2j+1]) / 2 == v_r[j]`
      exactly (test-pinned in tests/test_telemetry.py).

    Cached arrays are shared: callers must not mutate what `resample`
    returns (the analysis code never does — it derives new arrays).
    """

    def __init__(self, trace: MouseTrace) -> None:
        self.trace = trace
        self._grids: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def resample(self, rate: float = 500.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        got = self._grids.get(rate)
        if got is None:
            got = self._derive_half(rate)
            if got is None:
                got = self.trace.resample(rate)
            self._grids[rate] = got
        return got

    def _derive_half(self, rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        parent = self._grids.get(rate * 2.0)
        if parent is None:
            return None
        tg2, vx2, vy2 = parent
        if tg2.size == 0:
            return parent  # <2-packet trace: resample() is empty at any rate
        n = (tg2.size + 1) // 2  # ceil(n_2r / 2) == n_r (see docstring)

        def merge(v: np.ndarray) -> np.ndarray:
            if v.size < 2 * n:  # odd parent length: last bin pairs with zero
                v = np.concatenate([v, np.zeros(2 * n - v.size)])
            return (v[0::2] + v[1::2]) / 2.0

        return tg2[::2].copy(), merge(vx2), merge(vy2)


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
