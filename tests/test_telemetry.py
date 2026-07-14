"""Telemetry + movement-analysis tests on synthetic traces with known flicks."""

from __future__ import annotations

import numpy as np
import pytest

from kovadapt.analysis.movement import (
    directional_bias,
    movement_heatmap,
    region_deficits,
    segment_flicks,
)
from kovadapt.analysis.notable import find_notable_moments
from kovadapt.analysis.report import RunReport, build_report, run_time_window
from kovadapt.profile.player import PlayerProfile
from kovadapt.stats.parser import parse_stats_csv
from kovadapt.telemetry.trace import MouseTrace, TraceStore

RATE = 1000.0  # synthetic packet rate (Hz)


def _int_deltas(total: float, n: int) -> np.ndarray:
    """n integer deltas summing to round(total), bell-weighted (sin^2)."""
    w = np.sin(np.linspace(0.0, np.pi, n)) ** 2
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    cum = np.round(np.cumsum(w * total))
    return np.diff(np.concatenate([[0.0], cum])).astype(np.int32)


class TraceBuilder:
    """Builds packet streams: rest gaps, aimed movements, clicks."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t: list[float] = []
        self.dx: list[int] = []
        self.dy: list[int] = []
        self.clicks: list[float] = []
        self.now = t0

    def rest(self, dur: float = 0.5) -> "TraceBuilder":
        self.now += dur
        return self

    def move(self, dx_total: float, dy_total: float, dur: float = 0.15) -> "TraceBuilder":
        n = max(int(dur * RATE), 6)
        ts = self.now + np.arange(1, n + 1) / RATE
        self.t.extend(ts.tolist())
        self.dx.extend(_int_deltas(dx_total, n).tolist())
        self.dy.extend(_int_deltas(dy_total, n).tolist())
        self.now = float(ts[-1])
        return self

    def click(self) -> "TraceBuilder":
        self.now += 0.005
        self.clicks.append(self.now)
        return self

    def flick(self, dx: float, dy: float, dur: float = 0.15,
              overshoot: float = 0.0) -> "TraceBuilder":
        """Rest -> move (optionally past the target and back) -> click.
        Note raw mouse convention: +dy = down."""
        self.rest()
        if overshoot > 0:
            self.move(dx * (1 + overshoot), dy * (1 + overshoot), dur * 0.8)
            self.move(-dx * overshoot, -dy * overshoot, dur * 0.2)
        else:
            self.move(dx, dy, dur)
        return self.click()

    def build(self) -> MouseTrace:
        return MouseTrace(
            t=np.asarray(self.t, dtype=np.float64),
            dx=np.asarray(self.dx, dtype=np.int32),
            dy=np.asarray(self.dy, dtype=np.int32),
            clicks=np.asarray(self.clicks, dtype=np.float64),
        )


# ---------------------------------------------------------------- MouseTrace
def test_trace_roundtrip_and_window(tmp_path):
    tr = TraceBuilder().flick(200, 0).flick(-150, 30).build()
    p = tr.save(tmp_path / "t.npz")
    tr2 = MouseTrace.load(p)
    assert np.array_equal(tr.t, tr2.t)
    assert np.array_equal(tr.dx, tr2.dx)
    assert np.array_equal(tr.clicks, tr2.clicks)

    mid = float(tr.clicks[0]) + 0.01
    w = tr.window(tr.t[0], mid)
    assert len(w) < len(tr) and w.clicks.size == 1


def test_resample_preserves_displacement():
    tr = TraceBuilder().flick(300, -120).build()
    tg, vx, vy = tr.resample(500.0)
    assert tg.size > 0
    assert vx.sum() / 500.0 == pytest.approx(tr.dx.sum(), abs=2)
    assert vy.sum() / 500.0 == pytest.approx(tr.dy.sum(), abs=2)


def test_trace_store(tmp_path):
    tr = TraceBuilder().flick(100, 0).build()
    store = TraceStore(tmp_path)
    store.save(tr, "My Task [Adaptive]", "2026-07-13T10:00:00")
    got = store.load("My Task [Adaptive]", "2026-07-13T10:00:00")
    assert got is not None and len(got) == len(tr)
    assert store.load("Other", "2026-07-13T10:00:00") is None


# ---------------------------------------------------------- flick segmentation
def test_segment_basic_flick():
    tr = TraceBuilder().flick(200, 0).build()
    flicks = segment_flicks(tr)
    assert len(flicks) == 1
    f = flicks[0]
    assert f.amplitude == pytest.approx(200, rel=0.15)
    assert abs(f.angle) < 0.15                 # rightward
    assert f.overshoot < 0.05
    assert 0.05 < f.duration < 0.5


def test_segment_detects_overshoot():
    tr = TraceBuilder().flick(200, 0, overshoot=0.2).build()
    flicks = segment_flicks(tr)
    assert len(flicks) == 1
    assert flicks[0].overshoot == pytest.approx(0.2, abs=0.08)


def test_up_flick_angle_convention():
    # raw dy negative = up; analysis flips to +y up => angle ~ +pi/2
    tr = TraceBuilder().flick(0, -180).build()
    flicks = segment_flicks(tr)
    assert len(flicks) == 1
    assert flicks[0].angle == pytest.approx(np.pi / 2, abs=0.15)
    assert flicks[0].horizontal == "vertical"


def test_min_amplitude_filters_micro_movements():
    tr = TraceBuilder().flick(8, 0).build()   # below min_amplitude=15
    assert segment_flicks(tr) == []


# ----------------------------------------------------------------- aggregates
def _biased_trace(n: int = 5) -> MouseTrace:
    b = TraceBuilder()
    for _ in range(n):
        b.flick(220, 0)                        # clean right
        b.flick(-220, 0, overshoot=0.30)       # sloppy left
    return b.build()


def test_directional_bias_flags_weak_left():
    flicks = segment_flicks(_biased_trace())
    bias = directional_bias(flicks)
    assert bias["left"]["n"] >= 3 and bias["right"]["n"] >= 3
    assert bias["bias_score"] > 0.2            # positive => left weaker
    assert bias["left"]["overshoot"] > bias["right"]["overshoot"]


def test_region_deficits_weak_side_scores_higher():
    flicks = segment_flicks(_biased_trace())
    defs = region_deficits(flicks)
    left_key, right_key = "r1c0", "r1c2"       # 3x3 grid, horizontal flicks
    assert left_key in defs and right_key in defs
    assert defs[left_key] > defs[right_key]


def test_movement_heatmap_shape():
    heat, xe, ye = movement_heatmap(_biased_trace(), bins=32)
    assert heat.shape == (32, 32) and heat.sum() > 0
    assert xe.size == 33


# ------------------------------------------------------------ notable moments
def test_notable_moments_kinds_and_bounds():
    b = TraceBuilder()
    b.flick(200, 0, overshoot=0.35)            # bad overshoot
    for _ in range(4):
        b.flick(180, 40)                       # clean fodder
    moments = find_notable_moments(segment_flicks(b.build()))
    kinds = {m.kind for m in moments}
    assert "overshoot" in kinds and "clean_flick" in kinds
    for m in moments:
        assert m.t_start < m.t_end
        assert 0.0 <= m.severity <= 1.0
        assert m.text


# ----------------------------------------------------------------- run report
def test_report_without_trace(fixtures, tmp_path):
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    rep, flicks, heat = build_report(run, None)
    assert rep.n_flicks == 0 and flicks == [] and heat is None
    assert "telemetry" in rep.summary_text
    p = rep.save(tmp_path / "rep.json")
    assert RunReport.load(p).scenario == run.scenario


def test_report_with_synthetic_trace_in_run_window(fixtures, tmp_path):
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    win = run_time_window(run)
    assert win is not None and win[1] > win[0]
    b = TraceBuilder(t0=win[0] + 0.2)
    for _ in range(4):
        b.flick(200, 0)
        b.flick(-200, 0, overshoot=0.25)
    rep, flicks, heat = build_report(run, b.build())
    assert rep.n_flicks == len(flicks) >= 6
    assert rep.bias["bias_score"] > 0
    assert rep.region_deficits
    assert rep.overshoot_rate > 0.2
    assert rep.mean_flick_ms > 0
    assert heat is not None
    rep2 = RunReport.load(rep.save(tmp_path / "rep.json"))
    assert rep2.n_flicks == rep.n_flicks and rep2.notable == rep.notable


# ------------------------------------------------- telemetry -> bandit bridge
def test_credit_observed_regions_moves_posteriors():
    prof = PlayerProfile(scenario="X")
    prof.credit_observed_regions({"r1c0": 2.0, "r1c2": -1.5}, weight=0.6)
    assert prof.regions["r1c0"].mean > 0 > prof.regions["r1c2"].mean
    assert prof.regions["r1c0"].n == 1
