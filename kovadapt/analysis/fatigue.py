"""Session fatigue detection: flick-quality decay across a watch session.

Aim quality degrades measurably before players notice it subjectively —
overshoot rates climb and flicks slow down. The tracker keeps one composite
"badness" number per run and fits a robust (Theil-Sen) trend over the
session; a sustained upward badness trend flags fatigue.

Session-scoped by design: the tracker lives inside one SessionWatcher.watch()
and starts fresh each session, so overnight recovery never bleeds into the
trend. Only runs with real telemetry (enough flicks) contribute.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Badness composite weights: overshoot rate is the primary fatigue signal,
# flick duration the secondary (seconds, so 0.5 weights ~100ms of slowdown
# like ~5pp of overshoot rate).
_DURATION_WEIGHT = 0.5
_MIN_FLICKS_PER_RUN = 8
# Normalized session-total badness increase that maps to fatigue score 1.0.
_FULL_FATIGUE_RISE = 0.30
_DECLINING_SCORE = 0.4
_FATIGUED_SCORE = 0.8


@dataclass
class FatigueState:
    level: str = "fresh"     # fresh | declining | fatigued
    score: float = 0.0       # 0 (fresh) .. 1 (fully fatigued)
    trend: float = 0.0       # badness slope per run, normalized by session median
    runs: int = 0            # runs with usable telemetry this session
    message: str = ""        # plain-language suggestion ("" when fresh)

    def as_dict(self) -> dict:
        return asdict(self)


def theil_sen_slope(y: np.ndarray) -> float:
    """Median of pairwise slopes over the run index — robust to one bad run
    in a small series.

    Shared estimator: the session fatigue tracker fits it within a session,
    and analysis/skill.py fits it across sessions of saved run reports."""
    n = y.size
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    return float(np.median((y[j] - y[i]) / (j - i)))


class SessionFatigueTracker:
    """Feed one RunReport per run; read back the current FatigueState."""

    def __init__(self, min_runs: int = 5, sensitivity: float = 1.0) -> None:
        self.min_runs = max(2, int(min_runs))
        self.sensitivity = max(0.1, float(sensitivity))
        self._badness: list[float] = []
        self.state = FatigueState()

    def add_run(self, n_flicks: int, overshoot_rate: float, mean_flick_ms: float) -> FatigueState:
        if n_flicks >= _MIN_FLICKS_PER_RUN:
            self._badness.append(
                float(overshoot_rate) + _DURATION_WEIGHT * float(mean_flick_ms) / 1000.0
            )
        self.state = self._evaluate()
        return self.state

    def _evaluate(self) -> FatigueState:
        n = len(self._badness)
        if n < self.min_runs:
            return FatigueState(runs=n)
        y = np.array(self._badness)
        scale = float(np.median(y)) or 1.0
        slope = theil_sen_slope(y) / scale         # relative badness change per run
        rise = slope * (n - 1)                     # total relative change this session
        score = float(np.clip(rise * self.sensitivity / _FULL_FATIGUE_RISE, 0.0, 1.0))
        if score >= _FATIGUED_SCORE:
            level, message = "fatigued", (
                "Flick quality has dropped steadily this session — "
                "a 10-15 minute break will likely gain you more than grinding on."
            )
        elif score >= _DECLINING_SCORE:
            level, message = "declining", (
                "Flick quality is trending down — consider a short break soon."
            )
        else:
            level, message = "fresh", ""
        return FatigueState(level=level, score=score, trend=slope, runs=n, message=message)
