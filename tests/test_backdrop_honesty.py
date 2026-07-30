"""The backdrop must not describe itself as something a user cannot see, and
its durations must be true in WALL time (offscreen QPA).

tests/test_backdrop.py pins the 8 s loop; tests/test_backdrop_alive.py pins
what set_session() adds in CLOCK time. Neither noticed that the clock itself
ran at 0.707x real time, because both drive `_tick()` synchronously — so the
numbers those files assert were correct in a unit that never reached the
screen. Every pin here was written from a measurement or a render:

* the session clock advances at wall rate under a real event loop, which is
  what makes "a 240 ms blink" and "a 4 s interval floor" mean anything;
* the timer asks precisely, because a coarse 33 ms request came back every
  46.9 ms on this machine;
* a blink is legible AT THE RATE IT IS SAMPLED — measured by counting the
  frames the composition rule actually composes, at every sub-tick
  alignment, not by trusting a duration;
* the tables that make a frame cheap are keyed on what they actually depend
  on, so a window drag does not rebuild 340 ms of colour work;
* and the composited contrast of the whole signal is measured, so the
  docstring cannot drift back to calling a 1.0:1 difference a status light.

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtCore import QPointF, QTimer, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from kovadapt.gui import ascii_art, color, motion, theme  # noqa: E402
from kovadapt.gui import backdrop as bkd  # noqa: E402
from kovadapt.gui.backdrop import Backdrop  # noqa: E402

DT = bkd._DT


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Same guard as every other GUI file: nothing here may ever write the
    developer's real ~/.kovadapt."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def restore_palette():
    saved = theme.current()
    yield
    theme._current = saved


def _backdrop(size=(900, 700), settings=None):
    win = QWidget()
    win.resize(*size)
    win.show()
    bd = Backdrop(win, settings)
    bd._timer.stop()
    bd._ensure()
    return win, bd


# --------------------------------------------------------------- the clock
def test_the_session_clock_advances_at_wall_rate(qapp):
    """The pin the whole file exists for. Every duration in backdrop.py is in
    seconds of `_clock`, and `_clock` used to advance by the timer's NOMINAL
    interval — so when Qt delivered a 33 ms coarse timer every 46.9 ms, the
    eye ran at 0.707x and a "240 ms" blink drew for 340 ms. Measured against
    a real event loop, not a synchronous tick loop, because a synchronous
    loop is exactly what hid this."""
    win = QWidget()
    win.resize(900, 700)
    win.show()
    bd = Backdrop(win)
    bd._ensure()
    bd.set_session(accuracy=0.6, fatigue=0.3, watching=True)

    # Sampled AT tick boundaries rather than at two wall-clock marks: with
    # ~36 ticks in a second, an endpoint landing mid-interval is worth 2.7% on
    # its own. The first few are skipped because the very first tick credits
    # the gap since the timer started — real time, but startup's, capped at
    # _DT_MAX — and this is a claim about the steady rate.
    samples: list[tuple[float, float]] = []
    real = bd._tick
    bd._tick = lambda: (real(), samples.append((time.perf_counter(),
                                                bd._clock)))[0]
    QTimer.singleShot(1300, qapp.quit)
    qapp.exec()
    bd._timer.stop()
    del bd._tick

    assert len(samples) > 25, len(samples)
    (t0, c0), (t1, c1) = samples[3], samples[-1]
    wall, clock = t1 - t0, c1 - c0
    ratio = clock / wall
    assert 0.98 <= ratio <= 1.02, (
        f"the eye runs at {ratio:.3f}x real time: {clock:.2f} s of session "
        f"clock in {wall:.2f} s of wall clock")
    # ...and the durations the docstrings cite therefore mean what they say
    assert bkd._DUR_BAND[0] <= motion.SLOW / 1000.0 * ratio <= bkd._DUR_BAND[1]
    assert 60.0 / (bkd._GAP_MIN / ratio) <= 16.0     # the claimed 15/min cap
    win.deleteLater()


def test_the_timer_asks_precisely(qapp):
    """Half of the fix above: at Qt's default CoarseTimer this 33 ms request
    was delivered every 46.9 ms (median, measured) — 21.3 Hz for an animation
    that documents itself as running at twice motion.GLYPH_HZ."""
    win = QWidget()
    bd = Backdrop(win)
    assert bd._timer.timerType() == Qt.TimerType.PreciseTimer
    assert bd._timer.interval() == bkd._TICK_MS
    win.deleteLater()


def test_a_tick_is_worth_the_time_it_measures(qapp):
    """_tick_dt takes the measured gap and caps it, so a stall costs the eye
    frames instead of fast-forwarding it through a blink."""
    win = QWidget()
    bd = Backdrop(win)
    bd._timer.stop()
    bd._timed = True            # what _on_timeout does around a real tick
    bd._el.invalidate()

    assert bd._tick_dt() == pytest.approx(DT)      # nothing to measure yet
    time.sleep(0.06)
    measured = bd._tick_dt()
    assert measured == pytest.approx(0.06, abs=0.02)
    time.sleep(0.022)                              # an early tick is taken at
    early = bd._tick_dt()                          # face value, NOT rounded up
    assert early == pytest.approx(0.022, abs=0.008)
    time.sleep(0.30)
    assert bd._tick_dt() == pytest.approx(bkd._DT_MAX)    # capped
    win.deleteLater()


def test_only_the_timer_gets_to_claim_elapsed_time(qapp):
    """A tick that did not come from the timer cannot have measured anything —
    and guessing from the size of the gap is what ran the clock 4% fast, since
    Qt delivers a compressed timeout 0.9 ms after the first one."""
    win = QWidget()
    bd = Backdrop(win)
    bd._timer.stop()
    assert bd._timed is False
    time.sleep(0.05)
    assert bd._tick_dt() == pytest.approx(DT)     # a direct call: one frame
    seen = []
    bd._tick = lambda: seen.append(bd._timed)
    bd._on_timeout()
    assert seen == [True]                          # ...the timer's is timed
    assert bd._timed is False                      # ...and it does not stick
    win.deleteLater()


def test_the_cursor_follow_no_longer_depends_on_the_tick_rate(qapp):
    """The parallax follow was a flat per-tick lerp, so its SPEED was a
    function of how often Qt happened to call back. One long tick must land
    where the same time in short ticks does."""
    a = 1.0 - np.exp(-DT / bkd._FOLLOW_TAU)
    assert a == pytest.approx(0.08, abs=1e-9)      # unchanged at the nominal
    one_long = 1.0 - np.exp(-(4 * DT) / bkd._FOLLOW_TAU)
    four_short = 1.0 - (1.0 - a) ** 4
    assert one_long == pytest.approx(four_short, abs=1e-9)


def test_the_pupil_settle_is_measured_in_seconds_not_frames(qapp):
    """_advance_pupil eases by the interval the caller measured, so the
    settle takes _PUPIL_SETTLE of real time at any frame rate."""
    win, bd = _backdrop()
    bd.set_session(accuracy=1.0)
    start = bd._pupil
    for _ in range(15):
        bd._advance_pupil(bkd._GLYPH_DT)
    fast = bd._pupil
    bd._pupil = start
    bd._advance_pupil(15 * bkd._GLYPH_DT)
    assert bd._pupil == pytest.approx(fast, abs=0.004)
    win.deleteLater()


# --------------------------------------------------------------- the blink
def _strip(bd, dur, offset, ticks=24):
    """The closures the user is SHOWN: drive the real _tick and read the
    closure on every frame it decides to compose. Nothing is rendered — the
    thing under test is the composition rule, and rendering 20 alignments
    would cost the suite a second."""
    bd._tick_dt = lambda: DT              # a precise, unloaded host
    bd._clock = 100.0
    bd._blink_at = bd._clock + offset
    bd._blink_len = dur
    bd._since = 0.0
    seen: list[float] = []
    bd._render_frame = lambda: seen.append(bd._blink_k())
    for _ in range(ticks):
        bd._tick()
    del bd._render_frame, bd._tick_dt
    return [k for k in seen if k > 0.0]


def _sweep_alignments(bd, dur, steps=12):
    """Worst case over every sub-tick phase a blink can start on."""
    worst = None
    for i in range(steps):
        ks = _strip(bd, dur, i / steps * bkd._GLYPH_DT)
        rising = sum(1 for j, k in enumerate(ks)
                     if k < 1.0 and (j == 0 or ks[j - 1] < k))
        jump = max(abs(b - a) for a, b in zip([0.0] + ks, ks + [0.0]))
        row = (len(ks), rising, max(ks), -jump)
        if worst is None or row < worst[0]:
            worst = (row, ks)
    (n, rising, peak, neg_jump), ks = worst
    return n, rising, peak, -neg_jump, ks


# What the track can actually schedule. A blink only exists while ambient
# motion runs, which is FULL intensity only, so motion.ms cannot shrink one:
# the drawn range is motion.SLOW..300 ms with _BLINK_JITTER on top. The
# 120 ms end of _DUR_BAND is a clamp, never a duration — worth knowing,
# because 120 ms is 3.6 ticks and NO sample rate this timer can offer makes
# a lid sweep out of it.
_DRAWN = (motion.SLOW / 1000.0 * bkd._BLINK_JITTER[0],
          motion.SLOW / 1000.0,
          bkd._DUR_BAND[1])


def test_the_shortest_blink_the_app_can_draw_is_worth_sampling(qapp):
    assert bkd._DUR_BAND[0] / DT < 4.0          # unreachable, and unsweepable
    assert min(_DRAWN) / DT >= 6.0              # what actually gets drawn
    assert motion.ambient(type("S", (), {"motion": "reduced"})()) is False


def test_every_blink_reads_as_a_lid_sweeping_at_the_rate_it_is_sampled(qapp):
    """Found by rendering the strip, which is the only way to find it: an
    82 ms close inside a 240 ms blink is barely one glyph frame, so sampling
    the lids at motion.GLYPH_HZ gave (worst alignment) three lit frames of
    0.90, 0.90, 0.23 — open to nine-tenths shut in one step, never a full
    closure, indistinguishable from a dropped frame.

    A lid is a moving silhouette rather than a character ramp crossing
    thresholds, so it is sampled at the TICK rate instead. Pin the outcome,
    at every alignment and every duration the track can draw."""
    win, bd = _backdrop()
    bd.set_session(watching=True)
    for dur in _DRAWN:
        n, rising, peak, jump, ks = _sweep_alignments(bd, dur)
        why = f"{dur * 1000:.0f} ms blink, worst alignment: {ks}"
        assert n >= 5, why                  # enough frames to be a sweep
        assert rising >= 2, why             # ...at least two on the way down
        assert peak >= 0.99, why            # ...and it really does close
        assert jump <= 0.62, why            # ...without a step that big
    win.deleteLater()


def test_sampling_the_lids_at_the_glyph_rate_is_what_broke_it(qapp):
    """The reason the rule above exists, kept alive as an assertion: at the
    glyph rate these same durations cannot show a sweep, so nobody can
    "simplify" _lids_moving away without this failing."""
    win, bd = _backdrop()
    bd.set_session(watching=True)
    lids = bd._lids_moving
    bd._lids_moving = lambda: False         # i.e. compose every other tick
    for dur in _DRAWN:
        n, rising, _peak, jump, ks = _sweep_alignments(bd, dur)
        assert n <= 4 or rising < 2 or jump > 0.62, (
            f"{dur * 1000:.0f} ms at the glyph rate: {ks}")
    bd._lids_moving = lids
    win.deleteLater()


def test_the_lid_rate_costs_nothing_while_the_eye_is_open(qapp):
    """The extra frames are for the blink only: an open eye still composes at
    motion.GLYPH_HZ, and a frozen eye composes not at all."""
    win, bd = _backdrop()
    bd.set_session(watching=True)
    bd._blink_at, bd._blink_len = 1e9, 0.24     # no blink in sight
    bd._tick_dt = lambda: DT
    calls = []
    real = bd._render_frame
    bd._render_frame = lambda: (calls.append(1), real())[1]
    for _ in range(40):
        bd._tick()
    del bd._render_frame, bd._tick_dt
    assert len(calls) == 20                      # exactly every other tick
    assert bd._lids_moving() is False
    win.deleteLater()


def test_a_frozen_eye_never_claims_a_lid_is_moving(qapp, tmp_path):
    from kovadapt.config import Settings

    s = Settings(profile_dir=str(tmp_path / "p"),
                 kovaaks_root=str(tmp_path / "g"), motion="reduced")
    win, bd = _backdrop(settings=s)
    bd.set_session(watching=True)
    bd._blink_at, bd._blink_len = bd._clock, 0.24
    assert bd._lids_moving() is False
    assert bd._blink_k() == 0.0
    win.deleteLater()


# ------------------------------------------------------------ cost of tables
def test_a_resize_keeps_the_colour_table_it_paid_for(qapp, restore_palette):
    """The table is a pure function of the palette: loop_cell_color reads the
    stencil, which is a fixed COLS x ROWS grid, so pixel size cannot change a
    single entry. It was being cleared on every _ensure() anyway — 340 ms of
    rebuild per delivered resize event, i.e. dozens of times across one
    window drag. The lid plans DO hold pixel geometry and must still go."""
    win, bd = _backdrop(size=(1000, 800))
    for step in range(6):
        bd._rgba_at(step)
    bd._blink_plan(0.5)
    ids = {k: id(v) for k, v in bd._rgba.items()}
    assert len(ids) >= 6

    win.resize(940, 760)
    bd.notify_resize()
    bd._ensure()
    assert {k: id(v) for k, v in bd._rgba.items()} == ids   # same arrays
    assert bd._blink_cache == {}                            # pixel geometry
    assert bd._warm == 1

    theme._current = theme.build_palette(dark=False, accent="rose")
    bd.notify_theme()
    bd._ensure()
    assert len(bd._rgba) == 1        # a palette change DOES invalidate it
    win.deleteLater()


def test_the_first_paint_prefetches_nothing(qapp):
    """Prefetch rides the animation's own frames. It used to hang off
    _render_frame, which put a 1.5 ms lid build on the first paint — inside
    MainWindow's construction — and on every set_session() and theme
    switch."""
    win = QWidget()
    win.resize(900, 700)
    win.show()
    bd = Backdrop(win)
    bd._timer.stop()
    canvas = QPixmap(900, 700)
    p = QPainter(canvas)
    bd.paint(p)                       # the launch-critical first paint
    p.end()
    assert bd._blink_cache == {}
    assert len(bd._rgba) == 1         # only the phase it had to draw

    bd.set_session(accuracy=0.4, watching=True)
    assert bd._blink_cache == {}

    bd._tick_dt = lambda: DT
    for _ in range(80):
        bd._tick()                    # ...and the animation does the warming
    del bd._tick_dt
    assert len(bd._blink_cache) >= 20
    win.deleteLater()


def test_the_colour_table_is_bit_identical_to_loop_cell_color(qapp):
    """The table is built through getRgbF() and one asarray now (2.99 ->
    2.77 ms a step). Faster is worthless if it moves a pixel."""
    win, bd = _backdrop()
    for step in (0, 37, bkd._STEPS - 1):
        phase = step * (ascii_art.LOOP_T / bkd._STEPS)
        want = np.empty((len(bd._live), 4))
        for i, (_pt, _st, cell) in enumerate(bd._live):
            col = ascii_art.loop_cell_color(
                cell, phase, is_dark=bd._is_dark, iris_hue=bd._iris_hue,
                ink=bd._ink)
            want[i] = (col.redF(), col.greenF(), col.blueF(), col.alphaF())
        assert np.array_equal(bd._rgba_at(step), want)
    win.deleteLater()


# ---------------------------------------------------- what a user can see
def _composited(bd, pal):
    """paint()'s own composite — the eye at its real opacity over the app
    background — as an RGB float array, plus where the eye landed."""
    w, h = bd._win.width(), bd._win.height()
    canvas = QPixmap(w, h)
    canvas.fill(QColor(pal.bg))
    p = QPainter(canvas)
    p.setOpacity(0.16 if pal.is_dark else 0.12)
    ex = w - bd._eye_w - w * 0.04
    ey = (h - bd._eye_h) * 0.42
    p.drawPixmap(QPointF(ex, ey), bd._frame)
    p.end()
    img = canvas.toImage().convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(img.constBits(), dtype=np.uint8)
    raw = raw.reshape(img.height(), img.bytesPerLine() // 4, 4)
    return raw[:, :img.width(), :3][:, :, ::-1] / 255.0, ex, ey


def _band(arr, bd, ex, ey, r0, r1):
    """Mean composited RGB over the iris annulus r0..r1 — every pixel in it,
    which is what a viewer integrates."""
    cw = bd._frame.width() / ascii_art.COLS
    ch = bd._frame.height() / ascii_art.ROWS
    half = ascii_art.COLS / 2.0
    cx = ex + ((ascii_art.COLS - 1) / 2.0 + 0.5) * cw
    cy = ey + (ascii_art._CY * half / ascii_art._ASPECT
               + (ascii_art.ROWS - 1) / 2.0 + 0.5) * ch
    rx = ascii_art._RI * half * cw
    ry = ascii_art._RI * half / ascii_art._ASPECT * ch
    y0, y1 = int(cy - ry) - 2, int(cy + ry) + 3
    x0, x1 = int(cx - rx) - 2, int(cx + rx) + 3
    ys, xs = np.mgrid[y0:y1, x0:x1]
    rad = np.hypot((xs - cx) / rx, (ys - cy) / ry)
    px = arr[y0:y1, x0:x1][(rad >= r0) & (rad < r1)]
    assert len(px)
    return np.mean(px, axis=0)


def _ratio(a, b):
    def lum(rgb):
        return color.relative_luminance(
            "#" + "".join(f"{int(round(float(c) * 255)):02x}" for c in rgb))

    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _settle(bd, **session):
    bd.set_session(**session)
    for _ in range(60):
        bd._advance_pupil(0.1)
    bd._render_frame()


@pytest.mark.parametrize("palette", [dict(dark=True, accent="indigo"),
                                     dict(dark=False, accent="mint")])
def test_the_eye_is_not_a_readout_and_must_not_say_it_is(qapp, restore_palette,
                                                        palette):
    """The honest one. Composited at the opacity paint() uses, the pupil
    binding measures ~1.0:1 between a 20% run and a 99% one — WCAG's floor
    for a UI element is 3.0:1, so a diameter behind content cannot carry a
    value and the docstring must not pretend otherwise.

    Two-sided on purpose. If someone later makes the signal genuinely
    perceptible, THIS assertion fails, and the fix is to rewrite the
    docstring — not to loosen the number."""
    theme._current = theme.build_palette(**palette)
    pal = theme.current()
    win, bd = _backdrop(size=(1400, 900))
    bg = np.array([QColor(pal.bg).redF(), QColor(pal.bg).greenF(),
                   QColor(pal.bg).blueF()])

    cores = {}
    for acc in (0.20, 0.99):
        _settle(bd, accuracy=acc)
        arr, ex, ey = _composited(bd, pal)
        cores[acc] = [_band(arr, bd, ex, ey, lo, hi)
                      for lo, hi in ((0.0, 0.10), (0.10, 0.22), (0.22, 0.34))]
    best = max(_ratio(a, b) for a, b in zip(cores[0.20], cores[0.99]))
    assert best < 1.5, (
        f"the accuracy binding now reaches {best:.2f}:1 — if that is real, "
        f"say so in backdrop.py's docstring instead of calling it subtle")

    bd.set_session()
    bd._render_frame()
    arr, ex, ey = _composited(bd, pal)
    iris = _band(arr, bd, ex, ey, 0.55, 0.95)
    core = _band(arr, bd, ex, ey, 0.0, 0.18)
    assert _ratio(iris, bg) < 1.5      # the eye itself is barely there
    assert _ratio(iris, core) < 1.5    # the pupil has no edge to see
    win.deleteLater()


def _mean_lum(a):
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return float(np.mean(0.2126 * lin[..., 0] + 0.7152 * lin[..., 1]
                         + 0.0722 * lin[..., 2]))


def test_the_blink_is_the_biggest_thing_the_layer_does(qapp, restore_palette):
    """Why the timing work went into the blink and not the pupil, as a
    number: composited, a blink modulates the eye's whole footprint several
    times more than the difference between a 20% and a 99% pupil. Both are
    ~1% or less — neither is a signal — but the ORDERING is why one of them
    got the careful sampling."""
    theme._current = theme.build_palette(dark=True, accent="indigo")
    pal = theme.current()
    win, bd = _backdrop(size=(1400, 900))

    def footprint():
        arr, ex, ey = _composited(bd, pal)
        x0, y0 = int(ex), int(ey)
        return _mean_lum(arr[y0:y0 + bd._frame.height(),
                             x0:x0 + bd._frame.width()])

    bd.set_session(watching=True)
    bd._blink_at, bd._blink_len = bd._clock + 1e6, 0.24
    bd._render_frame()
    open_l = footprint()
    bd._blink_at = bd._clock - 0.24 * (bkd._CLOSE + bkd._HOLD / 2)
    bd._render_frame()
    assert bd._blink_k() == pytest.approx(1.0)
    blink = abs(footprint() - open_l) / open_l

    pupil = []
    for acc in (0.20, 0.99):
        bd._blink_at = bd._clock + 1e6
        _settle(bd, accuracy=acc, watching=True)
        pupil.append(footprint())
    pupil = abs(pupil[0] - pupil[1]) / max(pupil)

    assert blink > 3.0 * pupil, (
        f"blink modulates {blink * 100:.2f}%, pupil {pupil * 100:.2f}% — the "
        f"docstring's ordering no longer holds")
    assert blink < 0.05      # ...and it is still a 1% change, not a signal
    win.deleteLater()


def test_the_docstring_makes_no_claim_the_render_cannot_keep(qapp):
    """It called itself "the status light for a live watch session" while
    measuring 1.0:1. Language that promises a reading is banned here; the
    Dashboard hero is where numbers live."""
    doc = bkd.__doc__ or ""
    low = doc.lower()
    for phrase in ("status light", "is a readout", "reports it", "report it",
                   "reading off", "displays the", "so the user can see"):
        assert phrase not in low, f"backdrop.py claims to be a {phrase!r}"
    assert "not as a readout" in low
    assert "ambient" in low
    # and the two numbers a reader has to trust are the composited ones
    assert ":1" in doc and "3.0:1" in doc


def test_glyph_work_is_timed_rather_than_counted(qapp):
    """Composition is due when a glyph PERIOD has elapsed, not on every
    second callback, so the character animation cannot be slowed down by the
    event loop being late — it degrades to fewer, later frames instead."""
    win, bd = _backdrop()
    calls = []
    real = bd._render_frame
    bd._render_frame = lambda: (calls.append(bd._since), real())[1]
    bd._tick_dt = lambda: bkd._DT_MAX          # every tick runs 4x long
    for _ in range(10):
        bd._tick()
    del bd._render_frame, bd._tick_dt
    assert len(calls) == 10                    # every late tick composes
    assert all(c >= bkd._GLYPH_DT - 1e-9 for c in calls)
    # ...and the loop clock moved by the time that really passed
    assert bd._loop == pytest.approx(10 * bkd._DT_MAX, abs=1e-9)
    win.deleteLater()


def test_blink_statistics_are_wall_statistics(qapp):
    """test_backdrop_alive pins the track's intervals and durations against
    real ocular numbers in units of `_clock`. That is only a claim about the
    screen because `_clock` is wall time — so pin the arithmetic that links
    them, at the intensity scale a user can actually pick."""
    win, bd = _backdrop()
    bd.set_session(accuracy=0.7)
    durs = []
    prev = None
    for _ in range(int(240.0 / DT)):
        bd._clock += DT
        bd._advance_blink()
        if bd._blink_at != prev:
            if prev is not None:
                durs.append(bd._blink_len)
            prev = bd._blink_at
    assert len(durs) > 15
    assert min(durs) >= bkd._DUR_BAND[0] and max(durs) <= bkd._DUR_BAND[1]
    # the floor really is the rate cap the comment claims
    assert 60.0 / bkd._GAP_MIN == pytest.approx(15.0)
    # every drawn blink is long enough for the tick rate to sweep a lid
    assert min(durs) / DT >= 3.0
    assert statistics.mean(durs) == pytest.approx(motion.SLOW / 1000.0,
                                                  abs=0.03)
    win.deleteLater()
