"""Post-run report: everything the analysis window renders, serializable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..stats.models import Run
from ..telemetry.trace import MouseTrace
from .movement import segment_flicks, directional_bias, region_deficits, movement_heatmap
from .notable import NotableMoment, find_notable_moments


def run_time_window(run: Run) -> tuple[float, float] | None:
    """Epoch (start, end) of the run, reconstructed from the stats file's
    wall-clock kill timestamps anchored to the run's date."""
    start_str = run.summary.get("Challenge Start:")
    if not run.kills:
        return None
    day = datetime(run.started.year, run.started.month, run.started.day)

    def to_epoch(hhmmss: str) -> float:
        h, m, s = hhmmss.split(":")
        return (day + timedelta(hours=int(h), minutes=int(m), seconds=float(s))).timestamp()

    if start_str:
        t0 = to_epoch(start_str)
    else:
        # Older stats files lack "Challenge Start:" — reconstruct from the
        # first kill (its TTK covers the time since that target appeared).
        t0 = to_epoch(run.kills[0].timestamp) - max(run.kills[0].ttk, 0.0) - 1.0
    t1 = to_epoch(run.kills[-1].timestamp) + 2.0
    if t1 < t0:  # crossed midnight
        t1 += 86400.0
    return t0, t1


@dataclass
class RunReport:
    scenario: str
    started_iso: str
    # stats-derived
    score: float
    accuracy: float
    avg_ttk: float
    kills: int
    kps: float
    # telemetry-derived (empty/zero when no trace was captured)
    n_flicks: int = 0
    bias: dict = field(default_factory=dict)
    region_deficits: dict = field(default_factory=dict)
    notable: list[dict] = field(default_factory=list)
    total_travel_counts: float = 0.0
    mean_flick_ms: float = 0.0
    overshoot_rate: float = 0.0
    summary_text: str = ""
    trace_file: str = ""
    clip_files: dict = field(default_factory=dict)   # notable idx -> mp4 path

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RunReport":
        return cls(**json.loads(Path(path).read_text()))


def _summary_text(rep: "RunReport", flicks_exist: bool) -> str:
    lines = [f"Accuracy {rep.accuracy:.0%}, {rep.kills} kills at {rep.kps:.2f}/s."]
    if not flicks_exist:
        lines.append("No mouse telemetry for this run — start the recorder for movement analysis.")
        return " ".join(lines)
    b = rep.bias.get("bias_score", 0.0)
    if abs(b) > 0.15:
        weak = "left" if b > 0 else "right"
        lines.append(
            f"Your {weak} side is measurably weaker "
            f"({rep.bias[weak]['overshoot']:.0%} overshoot vs "
            f"{rep.bias['right' if weak == 'left' else 'left']['overshoot']:.0%}) — "
            f"spawns will shift {weak}."
        )
    else:
        lines.append("Left/right flicks are balanced this run.")
    if rep.overshoot_rate > 0.25:
        lines.append(f"{rep.overshoot_rate:.0%} of flicks overshot — consider a slight sens decrease or larger targets; the engine will compensate.")
    if rep.mean_flick_ms > 0:
        lines.append(f"Mean flick {rep.mean_flick_ms:.0f}ms.")
    return " ".join(lines)


def build_report(run: Run, trace: MouseTrace | None) -> tuple[RunReport, list, np.ndarray | None]:
    """-> (report, flicks, heatmap). Flicks/heatmap returned separately for
    the GUI (not serialized into JSON)."""
    rep = RunReport(
        scenario=run.scenario,
        started_iso=run.started.isoformat(),
        score=run.score,
        accuracy=run.accuracy,
        avg_ttk=run.avg_ttk,
        kills=run.kill_count,
        kps=run.kills_per_second(),
    )
    flicks: list = []
    heat = None
    if trace is not None and len(trace) > 10:
        win = run_time_window(run)
        rt = trace.window(*win) if win else trace
        flicks = segment_flicks(rt)
        rep.n_flicks = len(flicks)
        rep.bias = directional_bias(flicks)
        rep.region_deficits = region_deficits(flicks)
        rep.notable = [asdict(m) for m in find_notable_moments(flicks)]
        rep.total_travel_counts = float(np.hypot(rt.dx.astype(np.float64), rt.dy.astype(np.float64)).sum())
        if flicks:
            rep.mean_flick_ms = float(np.mean([f.duration for f in flicks]) * 1000)
            rep.overshoot_rate = float(np.mean([f.overshoot > 0.1 for f in flicks]))
        heat, _, _ = movement_heatmap(rt)
    rep.summary_text = _summary_text(rep, bool(flicks))
    return rep, flicks, heat
