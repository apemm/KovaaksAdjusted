"""Telemetry + movement-analysis tests on synthetic traces with known flicks."""

from __future__ import annotations

from datetime import datetime

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
from kovadapt.telemetry.trace import MouseTrace, ResampleCache, TraceStore

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
        self.clicks_up: list[float] = []
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

    def click(self, hold: float = 0.06) -> "TraceBuilder":
        self.now += 0.005
        self.clicks.append(self.now)
        self.clicks_up.append(self.now + hold)
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
            clicks_up=np.asarray(self.clicks_up, dtype=np.float64),
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


def test_build_report_threads_region_grid_dims(fixtures):
    """region_deficits keys must line up with the bandit's Settings-sized
    grid (r{row}c{col} contract); default stays 3x3 for legacy callers."""
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    win = run_time_window(run)
    b = TraceBuilder(t0=win[0] + 0.2)
    for _ in range(4):
        b.flick(200, 0)
        b.flick(-200, 0, overshoot=0.25)
    trace = b.build()

    rep_default, flicks, _ = build_report(run, trace)
    assert rep_default.region_deficits == region_deficits(flicks)  # 3x3 default

    rep51, _, _ = build_report(run, trace, region_cols=5, region_rows=1)
    assert rep51.region_deficits == region_deficits(flicks, cols=5, rows=1)
    assert rep51.region_deficits and rep51.region_deficits != rep_default.region_deficits
    for key in rep51.region_deficits:  # every key valid inside the 5x1 grid
        r, c = key[1:].split("c")
        assert int(r) == 0 and 0 <= int(c) <= 4


# --------------------------------------------------------- run_time_window
def _stats_csv_text(kills: list[tuple[str, str]], start: str | None = None) -> str:
    rows = "\n".join(
        f"{i},{ts},target,BB Gun,{ttk}s,1,1,1.000000,1.0,1.0,1.0,0,0"
        for i, (ts, ttk) in enumerate(kills, start=1)
    )
    text = (
        "Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,"
        "Damage Done,Damage Possible,Efficiency,Cheated,OverShots\n"
        f"{rows}\n\n"
        f"Kills:,{len(kills)}\n"
    )
    if start is not None:
        text += f"Challenge Start:,{start}\n"
    return text


def test_run_time_window_same_day_anchoring(tmp_path):
    p = tmp_path / "day test - Challenge - 2026.05.27-20.25.38 Stats.csv"
    p.write_text(_stats_csv_text(
        [("20:24:39.372", "0.100000"), ("20:24:42.809", "0.200000")],
        start="20:24:38.500",
    ), encoding="utf-8")
    t0, t1 = run_time_window(parse_stats_csv(p))
    assert t0 == datetime(2026, 5, 27, 20, 24, 38, 500000).timestamp()
    assert t1 == datetime(2026, 5, 27, 20, 24, 42, 809000).timestamp() + 2.0


def test_run_time_window_spanning_midnight(tmp_path):
    # Filename timestamp is the challenge END: a run started 23:59:30 and
    # ended 00:00:30 must anchor pre-midnight clocks to the PREVIOUS day.
    p = tmp_path / "mid test - Challenge - 2026.05.28-00.00.30 Stats.csv"
    p.write_text(_stats_csv_text(
        [("23:59:35.000", "0.100000"),
         ("23:59:50.000", "0.200000"),
         ("00:00:28.000", "0.150000")],
        start="23:59:30.000",
    ), encoding="utf-8")
    t0, t1 = run_time_window(parse_stats_csv(p))
    assert t0 == datetime(2026, 5, 27, 23, 59, 30).timestamp()
    assert t1 == datetime(2026, 5, 28, 0, 0, 28).timestamp() + 2.0
    assert 0 < t1 - t0 < 120  # a ~60s run, not a day-inverted window


def test_run_time_window_spanning_midnight_without_challenge_start(tmp_path):
    # Older stats files reconstruct t0 from the first kill; the same
    # previous-day anchoring must apply to that fallback path.
    p = tmp_path / "mid test - Challenge - 2026.05.28-00.00.15 Stats.csv"
    p.write_text(_stats_csv_text(
        [("23:59:45.000", "0.500000"), ("00:00:10.000", "0.200000")],
    ), encoding="utf-8")
    t0, t1 = run_time_window(parse_stats_csv(p))
    assert t0 == datetime(2026, 5, 27, 23, 59, 45).timestamp() - 0.5 - 1.0
    assert t1 == datetime(2026, 5, 28, 0, 0, 10).timestamp() + 2.0
    assert 0 < t1 - t0 < 120


# ------------------------------------------------- telemetry -> bandit bridge
def test_credit_observed_regions_moves_posteriors():
    prof = PlayerProfile(scenario="X")
    prof.credit_observed_regions({"r1c0": 2.0, "r1c2": -1.5}, weight=0.6)
    assert prof.regions["r1c0"].mean > 0 > prof.regions["r1c2"].mean
    assert prof.regions["r1c0"].n == 1


# ------------------------------------------------------ v0.3: input health
def test_input_health_metrics():
    b = TraceBuilder()
    for _ in range(10):
        b.flick(300, 0, dur=0.2)     # 1 kHz packets while moving
    trace = b.build()
    ih = trace.input_health()
    assert 900 <= ih["polling_hz_est"] <= 1100      # synthetic 1 kHz stream
    assert ih["jitter_ms"] < 1.0                    # perfectly regular grid
    assert ih["click_hold_ms"] == 60.0              # TraceBuilder default hold


def test_input_health_degrades_gracefully():
    assert MouseTrace().input_health()["polling_hz_est"] == 0.0
    b = TraceBuilder()
    b.flick(50, 0, dur=0.02)         # far under 100 packets
    ih = b.build().input_health()
    assert ih["polling_hz_est"] == 0.0
    assert ih["click_hold_ms"] > 0   # click pairing still works


def test_trace_npz_backward_compatible(tmp_path):
    """Pre-v0.3 traces (no clicks_up key) must still load."""
    b = TraceBuilder()
    b.flick(100, 0)
    tr = b.build()
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, t=tr.t, dx=tr.dx, dy=tr.dy, clicks=tr.clicks)
    loaded = MouseTrace.load(legacy)
    assert loaded.clicks_up.size == 0
    assert loaded.input_health()["click_hold_ms"] == 0.0
    # and the new format round-trips clicks_up
    p2 = tr.save(tmp_path / "new.npz")
    assert MouseTrace.load(p2).clicks_up.size == tr.clicks_up.size


def test_window_slices_clicks_up():
    b = TraceBuilder()
    for _ in range(6):
        b.flick(120, 0)
    tr = b.build()
    mid = float(tr.clicks[2])
    w = tr.window(tr.t[0], mid + 0.01)
    assert w.clicks_up.size <= tr.clicks_up.size
    assert (w.clicks_up <= mid + 0.01 + 1e-9).all()


# --------------------------------------------------------- v0.3: readiness
def test_readiness_progression():
    prof = PlayerProfile(scenario="r")
    r0 = prof.readiness(9)
    assert r0["score"] < 0.1 and "calibrating" in r0["message"]
    assert r0["stage"] == "cold start"
    assert len(r0["detail"]) == 3

    # v0.4 raised the ceiling: full calibration needs BASELINE_RUNS runs and
    # REGION_OBS observations per arm — a training week, not a warm-up.
    prof.run_count = PlayerProfile.BASELINE_RUNS
    prof.bias_obs = PlayerProfile.BIAS_RUNS      # measurements, not runs
    prof.ewma_bias = 0.2
    for k in [f"r{r}c{c}" for r in range(3) for c in range(3)]:
        post = prof.region(k)
        for _ in range(PlayerProfile.REGION_OBS):
            post.update(0.1)
    r1 = prof.readiness(9)
    assert r1["score"] == 1.0
    assert r1["stage"] == "dialed in"
    assert "dialed in" in r1["message"]
    assert r0["score"] < r1["score"]


def test_readiness_partial_regions():
    prof = PlayerProfile(scenario="r")
    prof.run_count = 10
    for k in ("r0c0", "r1c1"):
        for _ in range(PlayerProfile.REGION_OBS):
            prof.region(k).update(0.1)
    r = prof.readiness(9)
    assert 0.2 < r["score"] < 1.0
    assert "7 wall regions still need evidence" in r["message"]
    assert "regions 2/9 mapped" in r["detail"][1]


def test_readiness_below_old_ceiling_is_not_full():
    """10 runs + 2 observations per arm — the old v0.3 'fully calibrated'
    state — must no longer read as done."""
    prof = PlayerProfile(scenario="r")
    prof.run_count = 10
    prof.bias_obs = PlayerProfile.BIAS_RUNS
    prof.ewma_bias = 0.2
    for k in [f"r{r}c{c}" for r in range(3) for c in range(3)]:
        post = prof.region(k)
        post.update(0.1)
        post.update(0.1)
    r = prof.readiness(9)
    assert r["score"] < 0.8
    assert r["stage"] != "dialed in"


def test_readiness_bias_counts_measurements_not_runs():
    """The bias term used to be dead code that read as evidence.

        1.0 if (run_count >= BIAS_RUNS and abs(ewma_bias) > 0.0) else
        min(run_count / BIAS_RUNS, 1.0)

    is identically `min(run_count / BIAS_RUNS, 1.0)` — once run_count reaches
    BIAS_RUNS the else branch is already 1.0, so the ewma_bias conjunct could
    not change the answer at any input. A fully-run profile that had never
    produced a single directional-bias measurement reported "bias evidence
    collected", 100%, "dialed in — adaptation is running on settled
    evidence". `kovadapt replay` reaches that state on real data, because it
    never supplies a bias measurement at all.
    """
    prof = PlayerProfile(scenario="r")
    prof.run_count = PlayerProfile.BASELINE_RUNS
    for k in [f"r{r}c{c}" for r in range(3) for c in range(3)]:
        for _ in range(PlayerProfile.REGION_OBS):
            prof.region(k).update(0.1)

    r = prof.readiness(9)                       # bias_obs 0, ewma_bias 0.0
    assert r["bias"] == 0.0
    assert r["score"] < 1.0 and r["stage"] != "dialed in"
    assert "bias evidence 0/8 measurements" in r["detail"][2]
    assert "directional-bias measurement" in r["message"], r["message"]

    # rate-independence: runs alone must never buy the component
    prof.run_count = 100
    prof.observe_bias(0.3)
    prof.observe_bias(0.3)
    assert prof.readiness(9)["bias"] == 2 / PlayerProfile.BIAS_RUNS

    for _ in range(PlayerProfile.BIAS_RUNS):
        prof.observe_bias(0.3)
    r2 = prof.readiness(9)
    assert r2["bias"] == 1.0
    assert "bias evidence collected" in r2["detail"][2]
    assert r2["stage"] == "dialed in"


def test_readiness_bias_legacy_profiles_keep_their_credit(tmp_path):
    """Profiles written before bias_obs existed carry only the EWMA. Losing
    their credit would be a silent regression on every existing install.

    The credit is stamped by `load` — the path every real profile takes —
    rather than granted inside readiness(). Doing it in readiness() made the
    score go BACKWARDS the moment such a profile took its first real
    measurement: the allowance stopped applying the instant bias_obs became
    1, so 100% "dialed in" fell to 87% "calibrating".
    """
    import json

    d = tmp_path / "prof"
    (d / "profiles").mkdir(parents=True)
    legacy = {"scenario": "r", "run_count": 30, "ewma_bias": 0.4}
    PlayerProfile.path_for("r", d).write_text(json.dumps(legacy))

    prof = PlayerProfile.load("r", d)
    assert prof.bias_obs == PlayerProfile.BIAS_RUNS
    assert prof.readiness(9)["bias"] == 1.0

    # ...and it does not fall when a real measurement finally lands
    before = prof.readiness(9)["score"]
    prof.observe_bias(0.3)
    assert prof.readiness(9)["bias"] == 1.0
    assert prof.readiness(9)["score"] >= before


# ------------------------------------------------ v0.3.x hardening regressions
def test_smooth_matches_reference_recurrence():
    """_smooth was vectorized (truncated exponential FIR); it must stay
    numerically equivalent to the exact causal-EMA recurrence."""
    from kovadapt.analysis.movement import _smooth

    rng = np.random.default_rng(7)
    for n in (1, 2, 40, 5000):
        v = rng.normal(0.0, 1000.0, n)
        a = 1.0 - np.exp(-1.0 / (500.0 * 8.0 / 1000.0))
        ref = np.empty(n)
        acc = v[0]
        for i, x in enumerate(v):
            acc += a * (x - acc)
            ref[i] = acc
        got = _smooth(v, 500.0)
        assert np.max(np.abs(ref - got)) <= 1e-6 + 1e-8 * np.max(np.abs(ref))


def test_gap_p99_sees_mid_movement_stalls():
    """Stalls of 30-200 ms during movement must surface in gap_ms_p99
    (the old <30 ms window structurally hid every hitch it documented)."""
    b = TraceBuilder()
    for _ in range(30):
        b.move(30, 0, dur=0.02)
        b.rest(0.05)                  # 50 ms stall mid-motion
        b.move(30, 0, dur=0.02)
    b.click()
    ih = b.build().input_health()
    assert 900 <= ih["polling_hz_est"] <= 1100   # cadence still from <30 ms gaps
    assert ih["jitter_ms"] < 1.0
    assert ih["gap_ms_p99"] > 30.0               # the stalls are now visible


# ------------------------------------------------ v0.4: shared resample grid
def test_resample_cache_bitwise_identical_to_direct():
    """build_report shares one ResampleCache across analysis passes; the
    250 Hz grid is derived from the cached 500 Hz grid by bin merging and
    must stay BITWISE identical to trace.resample(250) — same dtypes, same
    bits. Checked on a TraceBuilder trace and on irregular random timing."""
    rng = np.random.default_rng(11)
    irregular = MouseTrace(
        t=1000.0 + np.sort(rng.uniform(0.0, 30.0, 20000)),
        dx=rng.integers(-127, 128, 20000).astype(np.int32),
        dy=rng.integers(-127, 128, 20000).astype(np.int32),
        clicks=np.sort(rng.uniform(1000.0, 1030.0, 10)),
    )
    for tr in (_biased_trace(), irregular):
        cache = ResampleCache(tr)
        for rate in (500.0, 250.0):     # 500 warms the cache; 250 is derived
            got, ref = cache.resample(rate), tr.resample(rate)
            for a, b in zip(got, ref):
                assert a.dtype == b.dtype
                assert np.array_equal(a, b)
        assert cache.resample(500.0) is cache.resample(500.0)  # memoized


def test_grid_sharing_matches_standalone_analysis():
    """segment_flicks/movement_heatmap with a shared grid must equal the
    bare-trace call exactly (the GUI still calls them without a grid)."""
    tr = _biased_trace(6)
    cache = ResampleCache(tr)
    assert segment_flicks(tr, grid=cache) == segment_flicks(tr)
    for a, b in zip(movement_heatmap(tr, grid=cache), movement_heatmap(tr)):
        assert np.array_equal(a, b)


# ------------------------------------------- v0.4: recorder memory ceiling
def test_buffers_retention_bounds_growth(monkeypatch):
    """With retention_s set, old chunks and clicks are dropped on chunk
    rollover, but at least the retention window is always fully covered."""
    import kovadapt.telemetry.raw_input as ri

    monkeypatch.setattr(ri, "_CHUNK", 100)
    buf = ri._Buffers(retention_s=1.0)
    t0 = 1000.0
    for i in range(1000):                 # 10 s of packets at 100 Hz
        ts = t0 + i * 0.01
        buf.add(ts, 1, 1)
        if i % 10 == 0:
            buf.clicks.append(ts)
            buf.clicks_up.append(ts + 0.005)
    tr = buf.to_trace()
    newest = t0 + 999 * 0.01
    assert tr.t[-1] == newest
    assert len(tr) < 1000                            # old chunks dropped
    assert len(tr) <= 3 * ri._CHUNK                  # bounded
    assert newest - tr.t[0] >= 1.0                   # window fully covered
    assert tr.clicks.size < 100 and tr.clicks_up.size < 100
    assert tr.clicks[0] >= tr.t[0] - 0.01            # clicks pruned in step

    unbounded = ri._Buffers()                        # default: keep everything
    for i in range(1000):
        unbounded.add(t0 + i * 0.01, 1, 1)
    assert len(unbounded.to_trace()) == 1000


def test_buffers_windowed_snapshot_matches_full(monkeypatch):
    """Chunk-granular to_trace(t0, t1) + exact window() must equal the
    full-session concatenation + window()."""
    import kovadapt.telemetry.raw_input as ri

    monkeypatch.setattr(ri, "_CHUNK", 50)
    buf = ri._Buffers()
    t0 = 1000.0
    for i in range(500):
        buf.add(t0 + i * 0.001, 1, -1)
    buf.clicks.extend([t0 + 0.1, t0 + 0.3])
    buf.clicks_up.extend([t0 + 0.16, t0 + 0.36])

    lo, hi = t0 + 0.101, t0 + 0.399
    full = buf.to_trace().window(lo, hi)
    fast = buf.to_trace(lo, hi)
    assert fast.t.size < buf.to_trace().t.size   # actually pre-trimmed
    fast = fast.window(lo, hi)
    assert np.array_equal(full.t, fast.t)
    assert np.array_equal(full.dx, fast.dx)
    assert np.array_equal(full.dy, fast.dy)
    assert np.array_equal(full.clicks, fast.clicks)
    assert np.array_equal(full.clicks_up, fast.clicks_up)


def test_a_corrupt_profile_is_quarantined_instead_of_bricking_the_app(tmp_path):
    """Settings.load has always set a broken settings.json aside and booted on
    defaults. PlayerProfile.load did not, so one truncated or zero-byte
    profile JSON raised out of every path that loads one — including
    MainWindow construction — and the app would not start at all.

    Losing one scenario's learning is recoverable (`kovadapt replay` rebuilds
    it from the stats history). Losing the app is not.
    """
    import json

    d = tmp_path / "state"
    (d / "profiles").mkdir(parents=True)
    name = "Wall Task [Adaptive]"
    path = PlayerProfile.path_for(name, d)

    for broken in (b"", b"{ truncated", b"[]", b"null", b'{"regions": 3}'):
        path.write_bytes(broken)
        prof = PlayerProfile.load(name, d)          # must not raise
        assert prof.scenario == name and prof.run_count == 0
        assert not path.exists(), "the corrupt file was left in place"
        assert path.with_suffix(".json.bad").is_file()
        path.with_suffix(".json.bad").unlink()

    # a healthy profile is untouched, BOM tolerated
    good = PlayerProfile(scenario=name)
    good.run_count = 7
    good.region("r1c1").update(0.3)
    good.save(d)
    raw = path.read_text(encoding="utf-8")
    path.write_text("\ufeff" + raw, encoding="utf-8")
    back = PlayerProfile.load(name, d)
    assert back.run_count == 7
    assert back.region("r1c1").n == 1
    assert isinstance(json.loads(path.read_text(encoding="utf-8-sig")), dict)
