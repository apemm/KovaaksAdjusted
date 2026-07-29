"""Cross-session skill tracking over the saved run-report history.

The research base (analysis/kb.py) is explicit about what long-term
improvement looks like: it shows up as SPEED, not accuracy — kills/s and
flick speed trend up over weeks while accuracy stays governed inside its
band (p-speed-is-growth-axis, Listman et al. 2021); a falling Fitts slope
(ms of flick time per bit of distance) across sessions is genuine motor
improvement even under a score plateau (dx-fitts-progress); and scores are
luck-noisy, so robust trends beat PBs (p-averages-not-highscores). This
module turns the RunReport JSONs persisted under ``<profile_dir>/reports/``
into exactly those trends. It states evidence only; the insight cards built
on top of it (analysis/insights.py) carry the citations.

Loading is deliberately light — stdlib json plus a filename sort, no numpy —
because report files are small and the GUI may call it on every refresh.
The trend fits reuse the shared Theil-Sen estimator from analysis/fatigue.py
(median of pairwise slopes: one hot or cold run barely moves it).

Value convention: ``0.0`` in a report means "not measured" for the flick
metrics (``fitts_slope_ms`` documents this explicitly; the others default to
0.0 when there is no telemetry), so every fit uses only nonzero, finite
observations and reports "insufficient data" below ``MIN_OBSERVATIONS``.

Pure leaf module: reads report JSON, never imports adapt/scenario/gui.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..profile.player import _slug
from .fatigue import theil_sen_slope

# Numeric report fields kept per entry (plus started_iso + scenario).
NUMERIC_FIELDS: tuple[str, ...] = (
    "score", "accuracy", "kps", "overshoot_rate",
    "mean_flick_ms", "mean_corrections", "fitts_slope_ms",
)

# Below this many nonzero observations a metric's trend is "insufficient
# data" — with fewer points a robust slope is still one streak away from
# flipping sign.
MIN_OBSERVATIONS = 8

# metric -> (polarity, loose-threshold). polarity is the sign of rel_change
# that counts as improvement: +1 rising = better (kps, accuracy, score),
# -1 falling = better (fitts_slope_ms, mean_flick_ms, overshoot_rate).
# The threshold is the total relative change across the fitted window needed
# to leave "flat": 5% for the speed metrics (the axis growth actually shows
# on), 10% for accuracy (band-governed by design — small drifts inside the
# band are the controller working, not skill change), 10% for overshoot_rate
# (a rate on a small base, so relative changes are noisy) and score
# (luck-noisy by doctrine). All thresholds are kovadapt's own calibration.
_METRICS: dict[str, tuple[int, float]] = {
    "fitts_slope_ms": (-1, 0.05),
    "mean_flick_ms": (-1, 0.05),
    "kps": (+1, 0.05),
    "overshoot_rate": (-1, 0.10),
    "accuracy": (+1, 0.10),
    "score": (+1, 0.10),
}

INSUFFICIENT = "insufficient data"


def load_report_history(
    profile_dir: Path, scenario: str | None = None, limit: int = 400
) -> list[dict]:
    """Slim entries from ``<profile_dir>/reports/*/*.json``, oldest first.

    When ``scenario`` is given only its slug directory is read (same slug
    convention as profile/player.py). The newest ``limit`` files are kept,
    judged by filename — report filenames are ISO timestamps with ``:`` →
    ``-``, so lexicographic order is chronological. Each entry keeps only
    ``scenario``, ``started_iso`` and ``NUMERIC_FIELDS`` (missing values
    default to ""/0.0); unreadable or corrupt files are skipped silently.
    """
    root = Path(profile_dir) / "reports"
    if scenario is not None:
        dirs = [root / _slug(scenario)]
    else:
        try:
            dirs = sorted(d for d in root.iterdir() if d.is_dir())
        except OSError:
            return []
    files: list[Path] = []
    for d in dirs:
        try:
            files.extend(f for f in d.iterdir() if f.suffix == ".json" and f.is_file())
        except OSError:
            continue
    files.sort(key=lambda p: p.name)
    files = files[-limit:] if limit > 0 else []

    out: list[dict] = []
    for f in files:
        try:
            raw = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        entry: dict = {
            "scenario": str(raw.get("scenario", "")),
            "started_iso": str(raw.get("started_iso", "")),
        }
        for key in NUMERIC_FIELDS:
            entry[key] = _num(raw.get(key))
        out.append(entry)
    return out


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class MetricTrend:
    metric: str
    classification: str      # "improving" | "flat" | "declining" | INSUFFICIENT
    n: int = 0               # nonzero observations the fit used
    slope: float = 0.0       # Theil-Sen slope, metric units per run
    rel_change: float = 0.0  # signed total relative change across the window


def _fit_metric(metric: str, values: list[float]) -> MetricTrend:
    polarity, loose = _METRICS[metric]
    xs = [v for v in values if math.isfinite(v) and v != 0.0]
    n = len(xs)
    if n < MIN_OBSERVATIONS:
        return MetricTrend(metric=metric, classification=INSUFFICIENT, n=n)
    y = np.asarray(xs, dtype=np.float64)
    slope = theil_sen_slope(y)
    baseline = abs(float(np.median(y))) or 1.0
    rel = slope * (n - 1) / baseline
    gain = rel * polarity
    if gain >= loose:
        cls = "improving"
    elif gain <= -loose:
        cls = "declining"
    else:
        cls = "flat"
    return MetricTrend(metric=metric, classification=cls, n=n,
                       slope=float(slope), rel_change=float(rel))


@dataclass(frozen=True)
class SkillTrends:
    """Robust cross-session trends, overall and per scenario."""

    n_runs: int = 0
    overall: dict = field(default_factory=dict)       # metric -> MetricTrend
    per_scenario: dict = field(default_factory=dict)  # scenario -> {metric -> MetricTrend}

    def summary(self) -> str:
        """2-4 plain evidence-first sentences for a GUI label. No citations
        here — the insight cards carry those."""
        o = self.overall
        if not o:
            return ("No saved run reports yet — cross-session trends appear "
                    "once runs are processed and their reports are on disk.")
        fitts, kps = o["fitts_slope_ms"], o["kps"]
        flick, acc, over = o["mean_flick_ms"], o["accuracy"], o["overshoot_rate"]
        if all(t.classification == INSUFFICIENT for t in o.values()):
            return (f"Only {self.n_runs} saved runs with usable data so far — at "
                    f"least {MIN_OBSERVATIONS} nonzero observations per metric are "
                    "needed before a cross-session trend is readable.")

        s: list[str] = []
        if fitts.classification == "improving":
            head = (f"Across {self.n_runs} runs: flick time per bit of distance "
                    f"fell {_pct(fitts)} — genuine motor improvement")
            if acc.classification == "flat":
                head += " — while accuracy held in band, exactly the expected signature."
            elif acc.classification == INSUFFICIENT:
                head += "."
            else:
                head += f", while accuracy {_verb(acc)}."
            s.append(head)
        elif fitts.classification == "declining":
            s.append(f"Across {self.n_runs} runs: flick time per bit of distance "
                     f"rose {_pct(fitts)} — flicks are getting slower for their "
                     "difficulty.")
        elif fitts.classification == "flat":
            s.append(f"Across {self.n_runs} runs: flick time per bit of distance "
                     "held flat.")
        elif kps.classification != INSUFFICIENT:
            s.append(f"Across {self.n_runs} runs: kill pace {_verb(kps)} — speed "
                     "is where long-term growth shows first.")
        else:
            s.append(f"Across {self.n_runs} runs: the speed metrics do not have "
                     f"{MIN_OBSERVATIONS}+ nonzero observations yet for a robust "
                     "trend.")

        span: list[str] = []
        if fitts.classification != INSUFFICIENT and kps.classification != INSUFFICIENT:
            span.append(f"kill pace {_verb(kps)}")
        if flick.classification != INSUFFICIENT:
            span.append(f"mean flick time {_verb(flick)}")
        if span:
            s.append(f"Over the same span, {' and '.join(span)}.")
        if over.classification == "declining":   # the rate is rising
            s.append(f"Overshoot rate rose {_pct(over)} across sessions — worth "
                     "attention before it compounds.")
        elif over.classification == "improving":
            s.append(f"Overshoot rate fell {_pct(over)} across sessions.")
        if len(s) < 2:
            s.append("Trends are robust fits over run order, so single hot or "
                     "cold runs barely move them.")
        return " ".join(s[:4])


def _pct(t: MetricTrend) -> str:
    return f"{abs(t.rel_change):.0%}"


def _verb(t: MetricTrend) -> str:
    if t.classification == "flat":
        return "held flat"
    return ("rose " if t.rel_change > 0 else "fell ") + _pct(t)


def fit_skill(entries: list[dict]) -> SkillTrends:
    """Fit every metric's trend over run sequence, overall and per scenario.

    ``entries`` is the (chronological) output of ``load_report_history``.
    Zero values are "not measured" and never enter a fit — in particular
    ``fitts_slope_ms == 0.0`` entries are excluded, per its contract in
    analysis/report.py.
    """
    overall = {
        m: _fit_metric(m, [_num(e.get(m)) for e in entries]) for m in _METRICS
    }
    per_scenario: dict[str, dict] = {}
    for scen in sorted({e.get("scenario", "") for e in entries if e.get("scenario")}):
        sub = [e for e in entries if e.get("scenario") == scen]
        per_scenario[scen] = {
            m: _fit_metric(m, [_num(e.get(m)) for e in sub]) for m in _METRICS
        }
    return SkillTrends(n_runs=len(entries), overall=overall, per_scenario=per_scenario)
