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
# Below this, the SAMPLING PERIOD starts destroying the features themselves.
# Measured rather than assumed: synthetic flicks with known geometry, sampled
# at 1000 / 500 / 250 / 125 / 62 Hz and put through segment_flicks. Overshoot
# came back 0.319 / 0.319 / 0.320 / 0.318 and corrections 2.00 / 2.00 / 2.00 /
# 2.00 — indistinguishable from 1000 Hz all the way down to 125. At 62 Hz both
# break: overshoot 0.349 (+9.4%) and corrections 3.10 (+55%), because a
# corrective submovement lasts ~25-50ms and a 16ms period leaves 2-3 samples
# to see it with.
#
# It was 490 — "below any competitive polling class", which is a judgement
# about hardware tier, not about whether the measurement survives. A 125 Hz
# mouse is the USB default and enormously common, and that threshold withheld
# every overshoot, correction, bias and moment claim on the page from anyone
# using one. All five real runs on this machine were suppressed by it while
# their jitter measured 0.54-0.93ms, which is clean.
POLLING_LOW_HZ = 100.0


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
    # A run needs an ANCHOR, not a kill. "Challenge Start:" is one, and the
    # filename timestamp is the challenge END — together they bound the run
    # without a single kill row. Bailing on `not run.kills` threw that away
    # for every invincible-target scenario: nothing dies, so the CSV reports
    # none, and 162 of the 398 real stats files here banked NO telemetry at
    # all. Those are precisely the tracking scenarios whose flick data is
    # worth having, and the recording was being discarded seconds after it
    # was made.
    if not run.kills and not start_str:
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
    # With kills, the last one plus follow-through; without, the challenge
    # end from the filename, which is what run.started already is.
    t1 = (to_epoch(run.kills[-1].timestamp) + 2.0) if run.kills else end_epoch
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
        # NAME THE ACTUAL CAUSE. These two have different causes and
        # different fixes, and the message gave one answer for both: it told
        # a 125 Hz mouse to go check for background apps. Jitter IS
        # contention — something is delaying packets that did arrive. A low
        # report rate is a device or driver setting, and no amount of closing
        # Chrome will change it.
        if jitter > JITTER_BAD_MS:
            lines.append(
                f"Input timing is too noisy to read flick microstructure from "
                f"(jitter {jitter:.1f}ms between packets) — overshoot and bias "
                "findings are withheld for this run; something is delaying "
                "mouse input, so run the Optimizer checkup for background "
                "apps or USB contention.")
        else:
            lines.append(
                f"Your mouse is reporting at about {polling:.0f}Hz, which is "
                f"under the {POLLING_LOW_HZ:.0f}Hz this analysis needs to see "
                "individual corrective submovements — overshoot and bias "
                "findings are withheld for this run. This is a device or "
                "driver setting, not background load: raise the polling rate "
                "in your mouse software.")
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
