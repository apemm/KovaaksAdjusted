"""Typed models for parsed KovaaK's stats files."""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass(slots=True)
class KillEvent:
    index: int
    timestamp: str          # wall-clock "HH:MM:SS.mmm"
    t: float                # seconds since first kill
    bot: str
    weapon: str
    ttk: float              # seconds
    shots: int
    hits: int
    accuracy: float
    cheated: bool
    overshots: int


@dataclass(slots=True)
class Run:
    """One completed scenario run (one stats CSV)."""

    scenario: str
    started: datetime
    kills: list[KillEvent] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)
    source_file: str = ""

    # -- summary accessors -------------------------------------------------
    def _f(self, key: str, default: float = 0.0) -> float:
        """A summary number, or `default` when the file does not carry one.

        The isfinite check is load-bearing, not defensive tidiness:
        `float("nan")` and `float("inf")` do NOT raise ValueError, so a
        summary cell reading "nan" walked straight through this into `score`
        and `accuracy`, from there into the profile EWMAs — where NaN
        propagates permanently — and into the charts, where int(nan) raises
        inside paintEvent and Qt kills the PROCESS. The per-kill columns are
        guarded in parser._finite; this is the other half of that path, and
        the half the summary block actually uses.
        """
        try:
            value = float(self.summary.get(key, default))
        except ValueError:
            return default
        return value if math.isfinite(value) else default

    @property
    def score(self) -> float:
        return self._f("Score:")

    @property
    def kill_count(self) -> int:
        return int(self._f("Kills:"))

    @property
    def hit_count(self) -> int:
        return int(self._f("Hit Count:"))

    @property
    def miss_count(self) -> int:
        return int(self._f("Miss Count:"))

    @property
    def accuracy(self) -> float:
        h, m = self.hit_count, self.miss_count
        return h / (h + m) if (h + m) else 0.0

    @property
    def avg_ttk(self) -> float:
        return self._f("Avg TTK:")

    @property
    def avg_target_scale(self) -> float:
        return self._f("Avg Target Scale:", 1.0)

    # -- vectorized per-kill views -----------------------------------------
    def ttk_array(self) -> np.ndarray:
        return np.array([k.ttk for k in self.kills], dtype=np.float64)

    def accuracy_array(self) -> np.ndarray:
        return np.array([k.accuracy for k in self.kills], dtype=np.float64)

    def interkill_intervals(self) -> np.ndarray:
        """Seconds between consecutive kills — the pace signal."""
        t = np.array([k.t for k in self.kills], dtype=np.float64)
        return np.diff(t) if t.size > 1 else np.empty(0)

    def kills_per_second(self) -> float:
        if len(self.kills) < 2:
            return 0.0
        span = self.kills[-1].t - self.kills[0].t
        return (len(self.kills) - 1) / span if span > 0 else 0.0
