"""Post-run report: everything the analysis window renders, serializable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..stats.models import Run
from ..telemetry.trace import MouseTrace, ResampleCache
from .movement import segment_flicks, directional_bias, region_deficits, movement_heatmap
from .notable import find_notable_moments


# ---- input health: the one gate on every microstructure claim -------------
# Lives HERE, in the module that owns RunReport, because analysis/insights.py
# and analysis/sens.py both import report — so this is the only place all
# three (and the GUI) can share. It previously existed as three separate
# inline copies, and every surface that forgot one told the user something
# the other surfaces refused to say about the same run.
JITTER_BAD_MS = 2.0
POLLING_LOW_HZ = 490.0       # below any competitive polling class


def input_degraded(rep) -> bool:
    """True when this run's input timing is too noisy to read flick
    microstructure from — overshoot rates, correction counts, directional
    bias, per-flick moments. Tolerates a missing, empty or None
    `input_health` rather than raising on an old or partial report."""
    health = getattr(rep, "input_health", None) or {}
    jitter = float(health.get("jitter_ms", 0.0) or 0.0)
    polling = float(health.get("polling_hz_est", 0.0) or 0.0)
    return jitter > JITTER_BAD_MS or (0.0 < polling < POLLING_LOW_HZ)


def run_time_window(run: Run) -> tuple[float, float] | None:
    """Epoch (start, end) of the run, reconstructed from the stats file's
    wall-clock kill timestamps.

    run.started (the filename timestamp) is the challenge END, so times are
    anchored to that date; a wall-clock time later in the day than the end
    time happened before midnight and belongs to the previous day
    (midnight-spanning runs)."""
    start_str = run.summary.get("Challenge Start:")
    if not run.kills:
        return None
    day = datetime(run.started.year, run.started.month, run.started.day)
    end_epoch = run.started.timestamp()

    def to_epoch(hhmmss: str) -> float:
        h, m, s = hhmmss.split(":")
        t = (day + timedelta(hours=int(h), minutes=int(m), seconds=float(s))).timestamp()
        # The filename timestamp truncates milliseconds, so a same-day time
        # can nominally exceed end_epoch by <1s; anything further ahead of
        # the run's end must be pre-midnight -> previous day.
        if t > end_epoch + 5.0:
            t -= 86400.0
        return t

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
    mean_corrections: float = 0.0    # corrective submovements per flick
    fitts_slope_ms: float = 0.0      # ms of flick time per bit of distance (0 = not enough flicks)
    summary_text: str = ""
    trace_file: str = ""
    clip_files: dict = field(default_factory=dict)   # notable idx -> mp4 path
    fatigue: dict = field(default_factory=dict)      # session FatigueState snapshot
    input_health: dict = field(default_factory=dict)  # polling/jitter/click-hold
    # Neural flick-score digest (ml/infer.py:summarize), stamped by the
    # watcher when a trained checkpoint exists. analysis/ itself never
    # imports kovadapt.ml — it stays a pure leaf; ml is built ON analysis.
    ml: dict = field(default_factory=dict)

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
    # Everything below is flick microstructure, so it answers to the same
    # input-health gate the Coach and the KPI strip use. Ungated, this header
    # printed a confident "40% of flicks overshot — consider a slight sens
    # decrease" directly above a tile reading "noisy-input" and a Coach card
    # saying microstructure diagnoses were suppressed for that very run.
    if input_degraded(rep):
        ih = rep.input_health or {}
        jitter = float(ih.get("jitter_ms", 0.0) or 0.0)
        polling = float(ih.get("polling_hz_est", 0.0) or 0.0)
        detail = f"timing jitter {jitter:.1f}ms" if jitter > JITTER_BAD_MS else ""
        if 0.0 < polling < POLLING_LOW_HZ:
            detail = (detail + ", " if detail else "") + f"polling ~{polling:.0f}Hz"
        lines.append(
            f"Input timing is too noisy to read flick microstructure from "
            f"({detail}) — overshoot and bias findings are withheld for this "
            "run; run the Optimizer checkup (background apps or USB "
            "contention).")
        if rep.mean_flick_ms > 0:
            lines.append(f"Mean flick {rep.mean_flick_ms:.0f}ms.")
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
    ih = rep.input_health or {}
    # `or 0.0`: a report carrying a null polling value used to raise TypeError
    # here rather than simply skipping the note.
    polling = float(ih.get("polling_hz_est", 0.0) or 0.0)
    if polling >= 125:
        note = f"Mouse polling ~{polling:.0f}Hz"
        if float(ih.get("jitter_ms", 0.0) or 0.0) > 1.0:
            note += (f", timing jitter {float(ih['jitter_ms']):.1f}ms — high; run "
                     "the Optimizer checkup (background apps or USB contention)")
        lines.append(note + ".")
    return " ".join(lines)


def build_report(
    run: Run,
    trace: MouseTrace | None,
    *,
    region_cols: int = 3,
    region_rows: int = 3,
) -> tuple[RunReport, list, np.ndarray | None]:
    """-> (report, flicks, heatmap). Flicks/heatmap returned separately for
    the GUI (not serialized into JSON).

    region_cols/region_rows must match Settings.region_cols/region_rows so
    the report's region_deficits keys line up with the bandit's region grid
    (the r{row}c{col} cross-module contract); defaults preserve the 3x3
    behavior for callers without Settings."""
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
        # One shared resample for every analysis pass: segment_flicks bins the
        # packets once (500 Hz); movement_heatmap's 250 Hz grid is then derived
        # from that cache instead of re-binning the whole packet stream.
        grid = ResampleCache(rt)
        flicks = segment_flicks(rt, grid=grid)
        rep.n_flicks = len(flicks)
        rep.bias = directional_bias(flicks)
        rep.region_deficits = region_deficits(flicks, cols=region_cols, rows=region_rows)
        rep.notable = [asdict(m) for m in find_notable_moments(flicks)]
        rep.total_travel_counts = float(np.hypot(rt.dx.astype(np.float64), rt.dy.astype(np.float64)).sum())
        rep.input_health = rt.input_health()
        if flicks:
            rep.mean_flick_ms = float(np.mean([f.duration for f in flicks]) * 1000)
            rep.overshoot_rate = float(np.mean([f.overshoot > 0.1 for f in flicks]))
            rep.mean_corrections = float(np.mean([f.corrections for f in flicks]))
            # Within-run Fitts fit: movement time vs log2 distance. The slope
            # (ms/bit) falling across sessions is motor improvement even when
            # scores plateau (see analysis/insights.py: dx-fitts-progress).
            amps = np.array([f.amplitude for f in flicks], dtype=np.float64)
            durs = np.array([f.duration for f in flicks], dtype=np.float64) * 1000.0
            ok = amps > 1.0
            if int(ok.sum()) >= 8:
                x = np.log2(1.0 + amps[ok])
                rep.fitts_slope_ms = float(np.polyfit(x, durs[ok], 1)[0])
        heat, _, _ = movement_heatmap(rt, grid=grid)
    rep.summary_text = _summary_text(rep, bool(flicks))
    return rep, flicks, heat
