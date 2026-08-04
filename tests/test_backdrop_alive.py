"""The backdrop eye as a LIVE status light (offscreen QPA).

tests/test_backdrop.py pins the eye's byte-identical 8 s loop. This file pins
what `Backdrop.set_session()` adds on top and, just as importantly, what it
must NOT disturb:

* nothing set -> bit-for-bit the old behaviour (resting pupil, loop blink,
  and the periodicity guarantee test_backdrop.py asserts);
* accuracy -> pupil diameter, through ascii_art's own pupil_dim curve, still
  reading as an eye at both extremes and in every palette;
* fatigue -> blink rate, blink duration and pupillary unrest (hippus), never
  the tonic pupil size;
* watching -> blink suppression and a lifted catchlight;
* the whole thing gated on gui/motion.py, with a timer that cannot tick while
  the window it paints is off screen.

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtCore import QEvent, QPointF  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from kovadapt.config import Settings  # noqa: E402
from kovadapt.gui import ascii_art, motion, theme  # noqa: E402
from kovadapt.gui import backdrop as bkd  # noqa: E402
from kovadapt.gui.backdrop import Backdrop  # noqa: E402

DT = bkd._DT


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Same guard as test_gui_smoke: nothing here may ever write the
    developer's real ~/.kovadapt."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def restore_palette():
    """Palette switches are process-global; put it back for the next test."""
    saved = theme.current()
    yield
    theme._current = saved


def _settings(tmp_path, **kw):
    """A Settings that can never reach the real home or the real game dir.

    config.py evaluates profile_dir as a CLASS-level default at import time,
    so monkeypatching Path.home afterwards does NOT move it — it has to be
    passed explicitly. kovaaks_root likewise, or __post_init__ goes probing.
    """
    return Settings(profile_dir=str(tmp_path / "prof"),
                    kovaaks_root=str(tmp_path / "game"), **kw)


def _backdrop(settings=None, size=(900, 700)):
    """A shown window + an ensured backdrop with its timer parked.

    The app paints before the first tick, and _tick() only recomposes when a
    frame pixmap exists, so _ensure() has to happen first or nothing animates.
    """
    win = QWidget()
    win.resize(*size)
    win.show()
    bd = Backdrop(win, settings)
    bd._timer.stop()
    bd._ensure()
    return win, bd


def _ticks(bd, glyph_frames):
    for _ in range(2 * glyph_frames):
        bd._tick()


def _bits(bd):
    return bytes(bd._frame.toImage().constBits())


def _settled(bd, **session):
    """Drive the pupil to its target. The ease is deliberately slower than
    motion.AMBIENT_MIN, so this needs many steps, not one call — but it needs
    the ease, not composed frames, so it drives _advance_pupil directly
    (exactly what _tick calls) and keeps the suite fast."""
    bd.set_session(**session)
    for _ in range(120):
        bd._advance_pupil()
    return bd._pupil


def _iris(bd):
    return [(i, c) for i, (_pt, _st, c) in enumerate(bd._live)
            if c.role == "iris"]


def _core_and_ring(bd, radius):
    """Mean HSV value of the iris inside the pupil vs out on the fibers."""
    pens = bd._pen_list(0)
    core = [pens[i].valueF() for i, c in _iris(bd) if c.rad < radius * 0.6]
    ring = [pens[i].valueF() for i, c in _iris(bd) if 0.60 < c.rad < 0.95]
    assert core and ring
    return statistics.mean(core), statistics.mean(ring)


# ------------------------------------------------------- nothing set at all
def test_untouched_backdrop_is_exactly_the_old_behaviour(qapp):
    """No session state: identity pupil factors, the loop blink, and the
    perfect loop test_backdrop.py pins."""
    win, bd = _backdrop()
    assert bd._session is False
    assert bd._fs is None and bd._fv is None      # the untouched colour path
    assert bd._rng is None
    assert bd._pupil == pytest.approx(ascii_art._PUPIL)
    # the blink is ascii_art's own periodic bump, not a stochastic track
    for phase in (0.0, 2.5, ascii_art.BLINK_PHASE + ascii_art.BLINK_LEN / 2):
        bd._loop = phase
        assert bd._blink_k() == pytest.approx(ascii_art.blink_amount(phase))
    bd._loop = 0.0
    bd._render_frame()
    frame0 = _bits(bd)
    bd._loop = ascii_art.LOOP_T
    bd._render_frame()
    assert _bits(bd) == frame0
    win.deleteLater()


def test_empty_set_session_is_a_no_op(qapp):
    """set_session() with nothing knowable must put the eye back on its
    deterministic loop rather than half-engage a live track."""
    win, bd = _backdrop()
    before = _bits(bd)
    bd.set_session()
    assert bd._session is False and bd._rng is None and bd._fs is None
    assert _bits(bd) == before
    # ...and it is reversible: bind, then clear
    bd.set_session(accuracy=0.9, watching=True)
    assert bd._session is True and bd._rng is not None
    bd.set_session()
    assert bd._session is False and bd._rng is None and bd._fs is None
    bd._render_frame()
    assert _bits(bd) == before
    win.deleteLater()


def test_session_state_is_a_snapshot_and_is_clamped(qapp):
    win, bd = _backdrop()
    bd.set_session(accuracy=1.7, fatigue=-3.0, watching=True)
    assert bd._acc == 1.0 and bd._fatigue == 0.0 and bd._watching is True
    bd.set_session(accuracy=0.5)          # omitted values mean "not known"
    assert bd._fatigue is None and bd._watching is False
    win.deleteLater()


# ------------------------------------------------------------------- pupil
def test_pupil_dilates_when_struggling_and_focuses_when_precise(qapp):
    win, bd = _backdrop()
    wide = _settled(bd, accuracy=0.20, watching=True)
    win2, bd2 = _backdrop()
    focus = _settled(bd2, accuracy=0.97, watching=True)
    assert wide > ascii_art._PUPIL > focus
    assert wide == pytest.approx(bkd._PUPIL_WIDE, abs=2e-3)
    assert focus == pytest.approx(bkd._PUPIL_FOCUS, abs=2e-3)
    # monotone across the band, no inversion anywhere
    seen = [_settled(bd, accuracy=acc)
            for acc in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    assert all(a >= b - 1e-9 for a, b in zip(seen, seen[1:]))
    win.deleteLater()
    win2.deleteLater()


def test_pupil_moves_the_rendered_frame_not_just_a_number(qapp):
    """A number that never reaches a pixel is not a status light."""
    win, bd = _backdrop()
    rest = _bits(bd)
    _settled(bd, accuracy=0.15)
    bd._loop = 0.0
    bd._render_frame()
    assert _bits(bd) != rest
    win.deleteLater()


@pytest.mark.parametrize("palette", [
    dict(dark=True, accent="indigo"),
    dict(dark=False, accent="mint"),
    dict(dark=True, accent="indigo", rgb=True),
])
def test_eye_still_reads_at_both_pupil_extremes(qapp, restore_palette, palette):
    """Both extremes must still look like an eye: a dark core inside a lit
    ring of fibers. Too small and there is no pupil, too wide and the iris is
    a hole — these bounds were picked by rendering them and looking, and this
    is the assertion that keeps them honest."""
    theme._current = theme.build_palette(**palette)
    win, bd = _backdrop()
    for acc, target in ((0.15, bkd._PUPIL_WIDE), (0.99, bkd._PUPIL_FOCUS)):
        r = _settled(bd, accuracy=acc)
        assert r == pytest.approx(target, abs=2e-3)
        core, ring = _core_and_ring(bd, r)
        assert core < 0.35, f"{palette} acc={acc}: no dark pupil ({core:.2f})"
        assert ring > 0.50, f"{palette} acc={acc}: iris unlit ({ring:.2f})"
        assert ring > core * 2.0
    win.deleteLater()


def test_pupil_settles_slower_than_the_ambient_floor(qapp):
    """Ambient motion never resolves faster than motion.AMBIENT_MIN; a pupil
    that snapped would read as a UI state flip, not as an eye adjusting."""
    assert bkd._PUPIL_SETTLE * 1000.0 >= motion.AMBIENT_MIN
    win, bd = _backdrop()
    bd.set_session(accuracy=1.0)
    start = bd._pupil
    goal = bd._pupil_goal()
    assert start != goal
    _ticks(bd, 1)
    assert 0.0 < (start - bd._pupil) / (start - goal) < 0.25
    _ticks(bd, 4)                          # ~0.4 s in: still on its way
    assert (start - bd._pupil) / (start - goal) < 0.75
    win.deleteLater()


def test_fatigue_shakes_the_pupil_but_never_sizes_it(qapp):
    """Pupillary unrest grows with sleepiness (the pupillographic sleepiness
    test); tonic diameter does not, so fatigue must not move the target."""
    win, bd = _backdrop()
    calm = _settled(bd, accuracy=0.7)
    win2, bd2 = _backdrop()
    tired = _settled(bd2, accuracy=0.7, fatigue=1.0)
    assert calm == pytest.approx(tired, abs=1e-6)      # same size

    def swing(b):
        seen = [b._pupil_now() for i in range(65)
                for _ in (b.__dict__.__setitem__(
                    "_clock", i * bkd._HIPPUS_T / 32.0),)]
        return max(seen) - min(seen)

    assert swing(bd) == pytest.approx(0.0, abs=1e-9)   # fatigue unknown: still
    assert swing(bd2) > 2.0 * bkd._HIPPUS_REST         # tired: unsteady
    assert swing(bd2) <= 2.0 * bkd._HIPPUS_TIRED + 1e-9
    # and slow: the oscillation is ambient, so its period respects the floor
    assert bkd._HIPPUS_T * 1000.0 >= motion.AMBIENT_MIN
    win.deleteLater()
    win2.deleteLater()


# ------------------------------------------------------------------- blink
def _blink_track(bd, minutes=8.0):
    """Walk the session clock and collect (start, duration) of every blink the
    track schedules, plus the closure seen on each frame. No frames composed.

    The measured gap is start-to-start minus the blink itself, i.e. the time
    the eye is OPEN — which is the quantity the 4-19 s band describes. It can
    overshoot the drawn value by up to one tick, because the reschedule lands
    on the first tick at or after the previous blink ended."""
    starts, durs, closures = [], [], []
    prev = None
    for _ in range(int(minutes * 60.0 / DT)):
        bd._clock += DT
        bd._advance_blink()
        if bd._blink_at != prev:
            if prev is not None:
                starts.append(bd._blink_at)
                durs.append(bd._blink_len)
            prev = bd._blink_at
        closures.append(bd._blink_k())
    gaps = [b - a - d for a, b, d in zip(starts, starts[1:], durs)]
    return starts, durs, gaps, closures


def test_blink_track_matches_real_ocular_statistics(qapp):
    """Every interval inside the observed 4-19 s band, every duration inside
    the observed 120-300 ms one, and a rate in the range a person staring at a
    monitor actually blinks at."""
    win, bd = _backdrop()
    bd.set_session(accuracy=0.7, watching=True)
    starts, durs, gaps, closures = _blink_track(bd, minutes=8.0)
    assert len(starts) > 20
    rate = len(starts) / 8.0
    assert 4.0 <= rate <= 20.0, f"{rate:.1f} blinks/min is not human"
    assert min(gaps) >= bkd._GAP_MIN - 1e-6
    assert max(gaps) <= bkd._GAP_MAX + DT + 1e-6
    assert all(0.12 <= d <= 0.30 for d in durs), (min(durs), max(durs))
    # a refractory point process, not a metronome
    assert statistics.stdev(gaps) > 1.0
    # a blink is an EVENT: the eye is open the overwhelming majority of time
    shut = sum(1 for k in closures if k > 0.0) / len(closures)
    assert 0.005 < shut < 0.12
    assert max(closures) >= 0.9              # it does fully close
    # every interval is also comfortably an ambient duration
    assert bkd._GAP_MIN * 1000.0 >= motion.AMBIENT_MIN
    win.deleteLater()


def test_fatigue_raises_the_blink_rate_and_watching_suppresses_it(qapp):
    rates, durations = {}, {}
    for name, kw in (("idle", dict(accuracy=0.7)),
                     ("watching", dict(accuracy=0.7, watching=True)),
                     ("tired", dict(accuracy=0.7, fatigue=1.0))):
        win, bd = _backdrop()
        bd.set_session(**kw)
        starts, durs, _gaps, _k = _blink_track(bd, minutes=8.0)
        rates[name] = len(starts) / 8.0
        durations[name] = statistics.mean(durs)
        win.deleteLater()
    # a demanding visual task suppresses blinking; fatigue raises the rate
    assert rates["watching"] < rates["idle"] < rates["tired"]
    # and drowsy blinks last longer — both ends inside the 120-300 ms band
    assert durations["tired"] > durations["idle"] * 1.15
    assert durations["idle"] == pytest.approx(motion.SLOW / 1000.0, abs=0.02)
    assert durations["tired"] == pytest.approx(bkd._DUR_BAND[1], abs=0.02)


def test_every_blink_spans_enough_glyph_frames_to_read_as_one(qapp):
    """Found by rendering it: at motion.GLYPH_HZ a 150 ms blink is only TWO
    frames, so the eye goes from wide open to fully shut in one step and reads
    as a dropped frame rather than a lid sweeping. Three is the minimum that
    shows a close, a closure and a reopen — which is why the rested duration
    is motion.SLOW and not the shorter BASE rung."""
    glyph_dt = motion.GLYPH_MS / 1000.0
    for fat in (0.0, 0.25, 0.5, 0.75, 1.0):
        win, bd = _backdrop()
        bd.set_session(accuracy=0.7, fatigue=fat)
        _starts, durs, _gaps, _k = _blink_track(bd, minutes=3.0)
        assert durs
        for d in durs:
            assert bkd._DUR_BAND[0] <= d <= bkd._DUR_BAND[1]
            # worst-case phase alignment still leaves three closed frames
            assert int(d / glyph_dt) >= 3, f"fatigue={fat}: {d * 1000:.0f} ms"
        win.deleteLater()


def test_blink_track_is_reproducible_from_its_fixed_seed(qapp):
    """The stochastic track replaces the loop's byte-identity guarantee with
    this one: the same state history always yields the same blinks."""
    def track():
        win, bd = _backdrop()
        bd.set_session(accuracy=0.6, fatigue=0.3, watching=True)
        starts, durs, _g, _k = _blink_track(bd, minutes=4.0)
        win.deleteLater()
        return [(round(s, 9), round(d, 9)) for s, d in zip(starts, durs)]

    a, b = track(), track()
    assert a == b and len(a) > 10


def test_live_blink_is_asymmetric_and_ignores_the_loop_clock(qapp):
    """Real blinks close fast and reopen slowly (Trutoiu et al.), unlike the
    symmetric sin^2 fallback — and once live, the closure is a function of the
    session clock, not of the 8 s loop phase."""
    win, bd = _backdrop()
    bd.set_session(watching=True)
    bd._blink_at, bd._blink_len = 10.0, 0.20

    def k_at(u):
        bd._clock = 10.0 + u * 0.20
        return bd._blink_k()

    assert k_at(-0.1) == 0.0 and k_at(1.1) == 0.0
    assert k_at(bkd._CLOSE + bkd._HOLD / 2) == pytest.approx(1.0)
    us = [i / 400.0 for i in range(401)]
    half = [u for u in us if k_at(u) >= 0.5]
    # half-closed on the way down arrives sooner than half-open on the way up
    assert min(half) < 1.0 - max(half)
    assert bkd._CLOSE < 1.0 - bkd._CLOSE - bkd._HOLD
    # the loop phase no longer decides anything
    bd._clock = 10.0 + 0.5 * 0.20
    a = bd._blink_k()
    bd._loop = ascii_art.BLINK_PHASE          # peak of the FALLBACK blink
    assert bd._blink_k() == a
    win.deleteLater()


def test_blink_plan_never_asks_for_a_zero_closure(qapp):
    """Regression: quantizing a tiny closure DOWN to zero made
    blink_lid_paths return None and the first frame of every blink crash."""
    win, bd = _backdrop()
    for k in (1e-9, 1e-4, 0.002, 0.01, 0.5, 1.0):
        wedges, groups = bd._blink_plan(k)
        assert not wedges.isEmpty() and groups
        assert all(0.0 <= a <= 1.0 and pts and sts for a, pts, sts in groups)
    win.deleteLater()


def test_a_whole_blink_renders_and_the_eye_comes_back(qapp):
    """Drive a real blink through _render_frame: the lids must actually change
    pixels frame by frame, and the open eye must return unchanged."""
    win, bd = _backdrop()
    bd.set_session(watching=True)
    open_bits = _bits(bd)
    bd._blink_at, bd._blink_len = bd._clock + 0.05, 0.20
    seen = set()
    for _ in range(12):
        bd._clock += DT
        bd._render_frame()
        seen.add(_bits(bd))
    assert len(seen) > 3
    bd._clock = bd._blink_at + bd._blink_len + 1.0
    bd._render_frame()
    assert _bits(bd) == open_bits
    win.deleteLater()


# ---------------------------------------------------------------- watching
def test_watching_calms_the_wander(qapp):
    """Watching used to ALSO lift the glint catchlights. With the glint role
    removed from the art there is nothing to lift, and the lift was not moved
    onto the iris — so watching must now leave the colours untouched and show
    up only in the wander and the blink cadence. Pinned in both directions,
    because a lift that silently returned would be a claim about the session
    that this layer is explicitly not allowed to make."""
    win, bd = _backdrop()
    assert all(c.role == "iris" for _p, _s, c in bd._live)
    bd.set_session(accuracy=0.7)
    off = [c.alphaF() for c in bd._pen_list(0)]
    bd.set_session(accuracy=0.7, watching=True)
    on = [c.alphaF() for c in bd._pen_list(0)]
    assert off == on, "watching changed the pens; it has no colour left to move"
    assert bkd._WATCH_WANDER < 1.0            # a fixating eye drifts less
    win.deleteLater()


# ------------------------------------------------------------------ motion
@pytest.mark.parametrize("level", ["reduced", "off"])
def test_ambient_off_freezes_the_eye_at_its_end_state(qapp, tmp_path, level):
    s = _settings(tmp_path, motion=level)
    win, bd = _backdrop(s)
    resting = _bits(bd)
    _ticks(bd, 200)
    assert _bits(bd) == resting             # nothing animated
    assert bd._loop == 0.0                  # ...and it sits at the end state
    assert bd._off == [QPointF(0, 0)] * 3   # no parallax drift either
    assert bd._blink_k() == 0.0             # an open eye, never mid-blink
    win.deleteLater()


def test_ambient_off_still_shows_the_pupil_it_was_told_about(qapp, tmp_path):
    """A dilation is information, not decoration: it survives motion being
    off, it just arrives instantly instead of easing."""
    s = _settings(tmp_path, motion="off")
    win, bd = _backdrop(s)
    before = _bits(bd)
    bd.set_session(accuracy=0.10)
    assert bd._pupil == pytest.approx(bkd._PUPIL_WIDE, abs=1e-9)
    assert _bits(bd) != before
    win.deleteLater()


def test_motion_setting_is_read_at_use_time(qapp, tmp_path):
    """The dial is user-editable while the app runs, so nothing may cache it —
    flipping the field has to take effect on the very next tick."""
    s = _settings(tmp_path, motion="off")
    win, bd = _backdrop(s)
    frozen = _bits(bd)
    _ticks(bd, 40)
    assert _bits(bd) == frozen
    s.motion = "full"
    _ticks(bd, 40)
    assert _bits(bd) != frozen
    s.motion = "reduced"
    _ticks(bd, 2)
    settled = _bits(bd)
    _ticks(bd, 40)
    assert _bits(bd) == settled
    win.deleteLater()


def test_resuming_ambient_redraws_the_blink_under_the_live_intensity(qapp,
                                                                    tmp_path):
    """With motion off a scheduled blink has zero length; turning motion back
    on must redraw it, or the next blink runs at a length the intensity scale
    produced rather than one that reads at GLYPH_HZ."""
    s = _settings(tmp_path, motion="off")
    win, bd = _backdrop(s)
    bd.set_session(accuracy=0.7, watching=True)
    _ticks(bd, 2)
    assert bd._blink_len == 0.0
    s.motion = "full"
    _ticks(bd, 2)
    assert bd._blink_len >= motion.SLOW / 1000.0 * bkd._BLINK_JITTER[0]
    win.deleteLater()


def test_settings_may_be_omitted_and_the_window_can_supply_them(qapp, tmp_path):
    """The backdrop is built by the window, so it falls back to the window's
    own Settings rather than reaching for a global — and None stays valid."""
    win, bd = _backdrop(None)
    assert bd._cfg() is None
    assert motion.ambient(bd._cfg()) is True     # None means FULL
    win.deleteLater()

    holder = QWidget()
    holder.resize(900, 700)
    holder.s = _settings(tmp_path, motion="off")
    bd2 = Backdrop(holder)
    assert bd2._cfg() is holder.s
    assert motion.ambient(bd2._cfg()) is False
    holder.deleteLater()


def test_glyph_work_runs_at_the_glyph_rate(qapp):
    """Character animation belongs at motion.GLYPH_HZ; the parallax pixmaps
    are continuous transform motion and get double that."""
    assert bkd._TICK_MS * 2 == motion.GLYPH_MS
    assert bkd._STEPS == round(ascii_art.LOOP_T * motion.GLYPH_HZ)
    win, bd = _backdrop()
    real, calls = bd._render_frame, []
    bd._render_frame = lambda: (calls.append(1), real())[1]
    for _ in range(60):
        bd._tick()
    assert len(calls) == 30              # exactly half the ticks compose glyphs
    del bd._render_frame
    win.deleteLater()


# -------------------------------------------------------------- timer life
def test_timer_never_ticks_while_the_window_is_off_screen(qapp):
    win = QWidget()
    win.resize(800, 600)
    bd = Backdrop(win)
    assert not bd._timer.isActive()         # never shown: never started
    win.show()
    qapp.processEvents()
    assert bd._timer.isActive()
    win.hide()
    qapp.processEvents()
    assert not bd._timer.isActive()
    win.show()
    qapp.processEvents()
    assert bd._timer.isActive()
    win.deleteLater()


def test_teardown_of_the_window_does_not_crash(qapp):
    """The show/hide relay used to hold the Backdrop strongly, closing a
    reference cycle — and the cyclic collector then delivered the dying
    window's Hide event into a half-cleared Backdrop."""
    win, bd = _backdrop()
    win.hide()
    win.deleteLater()
    del win, bd
    gc.collect()
    qapp.processEvents()
    gc.collect()


# --------------------------------------------------------- cost of a frame
def test_frame_caches_absorb_the_per_cell_work(qapp):
    """The per-frame cost is only bounded because these tables fill and then
    stop being rebuilt. Measured on this machine: 3.37 ms -> 1.07 ms per glyph
    frame, peak blink 7.7 ms -> 3.1 ms. Assert the mechanism, not the
    milliseconds, so the pin does not depend on the host's speed."""
    win, bd = _backdrop()
    _ticks(bd, bkd._STEPS + 8)              # one loop: the step index cannot
    #                                         skip, so every phase is visited
    assert len(bd._rgba) == bkd._STEPS      # one colour table per glyph phase
    assert len(bd._blink_cache) == bkd._BLINK_STEPS   # every closure prefetched
    rgba_ids = {k: id(v) for k, v in bd._rgba.items()}
    lids = dict(bd._blink_cache)
    _ticks(bd, 40)                          # warm frames build nothing
    assert {k: id(v) for k, v in bd._rgba.items()} == rgba_ids
    assert set(bd._blink_cache) == set(lids)
    assert all(bd._blink_cache[k] is v for k, v in lids.items())
    assert len(bd._pen_list(7)) == len(bd._live)
    win.deleteLater()


def test_theme_change_drops_every_colour_cache(qapp, restore_palette):
    """The colour tables are keyed on the palette implicitly, so a theme
    switch has to invalidate them or the eye keeps the old accent forever."""
    win, bd = _backdrop()
    _ticks(bd, 40)
    assert bd._rgba
    theme._current = theme.build_palette(dark=False, accent="rose")
    bd.notify_theme()
    assert bd._frame is None
    bd._ensure()
    assert len(bd._rgba) == 1               # rebuilt from scratch
    # dropped too; _ensure's own first frame has already prefetched one plan
    assert len(bd._blink_cache) <= 1
    win.deleteLater()


def test_backdrop_paints_over_a_window_with_live_state(qapp):
    """End to end: a full paint with session state bound must not raise and
    must actually put ink on the canvas."""
    win, bd = _backdrop(size=(1000, 800))
    bd.set_session(accuracy=0.33, fatigue=0.6, watching=True)
    _ticks(bd, 12)
    canvas = QPixmap(1000, 800)
    canvas.fill(QColor("#101116"))
    flat = QPixmap(1000, 800)
    flat.fill(QColor("#101116"))
    p = QPainter(canvas)
    bd.paint(p)
    p.end()
    assert canvas.toImage() != flat.toImage()
    win.deleteLater()


def test_the_backdrop_stops_while_you_are_in_the_game(qapp):
    """This app exists to sit behind a game you are playing, and a window you
    alt-tabbed away from is still `isVisible()` to Qt — so the ambient layer
    went on animating a full-window parallax at 21Hz the whole time you were
    in Valorant. Measured on this machine: 26% of a core focused, and the
    same 26% backgrounded, with 275 QLabel repaints a second cascading off
    one backdrop tick. Backgrounded now costs 2%.

    Driven by the DEACTIVATION event rather than isActiveWindow(), which
    reports False for a window that has never been activated at all — every
    headless context — and would freeze the backdrop exactly where it is
    measured.
    """
    win, bd = _backdrop()
    win.show()
    qapp.processEvents()
    assert bd._timer.isActive()

    qapp.sendEvent(win, QEvent(QEvent.Type.WindowDeactivate))
    qapp.processEvents()
    assert not bd._timer.isActive(), (
        "the backdrop keeps animating while the user is in another window")

    qapp.sendEvent(win, QEvent(QEvent.Type.WindowActivate))
    qapp.processEvents()
    assert bd._timer.isActive(), "it never comes back when you tab in"
    win.deleteLater()


def test_motion_off_actually_stops_the_frames(qapp):
    """`off` used to leave the timer running at 30Hz with a no-op tick —
    cheaper than full, but "off" should mean the frames stop, not that they
    cost less."""
    from kovadapt.config import Settings

    win = QWidget()
    win.resize(800, 600)
    bd = Backdrop(win, Settings(motion="off", telemetry_enabled=False))
    win.show()
    qapp.processEvents()
    assert not bd._timer.isActive(), "motion=off still runs the ambient timer"
    win.deleteLater()


def test_changing_the_motion_setting_takes_effect_immediately(qapp):
    """`_sync_timer` reads motion.ambient(), but only ran on show / hide /
    activation — so a settings change did nothing until the next alt-tab.
    `config_view.settings_changed` was declared AND emitted with zero
    connected receivers.

    Turning motion off left a 30Hz timer running; turning it on left the
    backdrop frozen, which is the worse direction because it looks broken
    rather than merely costly.
    """
    from kovadapt.config import Settings

    s = Settings(motion="full", telemetry_enabled=False)
    win = QWidget()
    win.resize(800, 600)
    bd = Backdrop(win, s)
    win.show()
    qapp.processEvents()
    assert bd._timer.isActive()

    s.motion = "off"
    bd.motion_changed()
    assert not bd._timer.isActive(), "turning motion off left the timer running"

    s.motion = "full"
    bd.motion_changed()
    assert bd._timer.isActive(), "turning motion back on left the backdrop frozen"
    win.deleteLater()
