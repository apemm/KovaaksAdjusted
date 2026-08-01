"""The opening's pacing contract (gui/logo.py).

logo.py cannot move ascii_art's beat constants — the live backdrop shares
that module — so it retimes the splash with a warped clock. These tests pin
the properties that make the warp safe: it is monotonic, every named beat
still lands inside the visible window in the authored order, the pupil
finishes resolving well before the show ends (PUPIL_T1 falling past the end
would leave the eye with no pupil at all), the first beat arrives almost
immediately, and any key or click ends the opening.
"""

from __future__ import annotations

import math
import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Nothing here saves, but logo pulls in theme/config through ascii_art —
    the suite once corrupted the developer's real settings.json, so every GUI
    test file re-homes Path.home() before touching anything Qt-adjacent."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def logo(qapp):
    from kovadapt.gui import logo as mod

    return mod


@pytest.fixture()
def art(qapp):
    from kovadapt.gui import ascii_art as mod

    return mod


def _wall_of(logo, choreo: float) -> float:
    """Invert the warp by bisection (it is monotonic, so this is well posed)."""
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if logo._choreo_time(mid) < choreo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


class _StubField:
    """Stands in for _EyeField so the widget tests don't pay a 381x181
    stencil build (~0.3 s) just to assert scheduling."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self.last_t: float | None = None

    def prepare(self, cell_w: float, cell_h: float) -> None:
        self.cell = (cell_w, cell_h)

    def paint(self, p, t: float) -> None:
        self.last_t = t


@pytest.fixture()
def splash(logo, monkeypatch):
    monkeypatch.setattr(logo, "_EyeField", _StubField)
    s = logo.SplashScreen()
    yield s
    s.close()


# ------------------------------------------------------------------- warp
def test_warp_is_monotonic_and_continuous(logo):
    prev = -1.0
    w = 0.0
    while w <= 9.0:
        c = logo._choreo_time(w)
        assert c > prev, f"warp went backwards at wall {w}"
        prev = c
        w += 0.002
    for wall, choreo in logo._WARP:          # knots hit exactly => continuous
        assert logo._choreo_time(wall) == pytest.approx(choreo, abs=1e-9)
    assert logo._choreo_time(-1.0) == 0.0
    last_w, last_c = logo._WARP[-1]
    assert logo._choreo_time(last_w + 2.0) == pytest.approx(last_c + 2.0)


def test_total_runtime_lands_in_the_target_band(logo, splash):
    """~5 s of wall clock, not the 8.4 s the opening used to run."""
    fade_s = math.ceil(1.0 / 0.12) * splash._timer.interval() / 1000.0
    total = logo.SplashScreen.MIN_SECONDS + fade_s
    assert 4.5 <= total <= 5.5, total
    assert logo._WARP[-1][0] < logo.SplashScreen.MIN_SECONDS


def test_every_beat_still_fires_in_order_inside_the_window(logo, art):
    """No beat deleted, none reordered, none pushed past the fade."""
    beats = [
        ("heartbeat", logo._HEART_T0),
        ("spark", art._SPARK),
        ("pupil starts", art.PUPIL_T0),
        ("pupil resolved", art.PUPIL_T1),
        ("blink", art.BLINK_T),
        ("blink shut", art.BLINK_T + art.BLINK_CLOSE),
        ("blink reopened", logo._BLINK_END),
        ("crosshair strikes", art.RETICLE_T),
        ("crosshair steady", art.RETICLE_T + art.RETICLE_LEN),
        ("breathe", art.BREATHE_T),
        ("settled", art.TOTAL),
    ]
    walls = [_wall_of(logo, c) for _, c in beats]
    assert walls == sorted(walls), dict(zip([n for n, _ in beats], walls))
    for (name, _), wall in zip(beats, walls):
        assert 0.0 < wall < logo.SplashScreen.MIN_SECONDS, f"{name} at {wall}"
    assert logo._choreo_time(logo.SplashScreen.MIN_SECONDS) > art.TOTAL


def test_no_beat_is_crushed_below_a_couple_of_frames(logo, art):
    """Recompressed, not deleted: each beat still spans several frames at the
    splash's ~30 fps, so it reads as motion rather than a cut."""
    spans = {
        "ignition -> pupil": (art._SPARK, art.PUPIL_T0),
        "fill": (art.PUPIL_T0, art.BLINK_T),
        "blink": (art.BLINK_T, logo._BLINK_END),
        "crosshair strike": (art.RETICLE_T, art.RETICLE_T + art.RETICLE_LEN),
    }
    for name, (c0, c1) in spans.items():
        seconds = _wall_of(logo, c1) - _wall_of(logo, c0)
        assert seconds > 4 * 0.033, f"{name} collapsed to {seconds:.3f}s"


def test_the_crosshair_flickers_rather_than_fades(art):
    """A fade is monotonic; a flicker is not. That is the whole difference
    Arjun asked for, so it is the thing worth pinning: the level must go DOWN
    again after going up, at least twice, and must reach full dark after
    first lighting — a tube that catches, drops out, and catches again."""
    t0, t1 = art.RETICLE_T, art.RETICLE_T + art.RETICLE_LEN
    xs = [t0 + i * 0.005 for i in range(int((t1 - t0) / 0.005) + 1)]
    lv = [art.reticle_flicker(x) for x in xs]

    assert art.reticle_flicker(t0 - 0.01) == 0.0, "lit before its cue"
    assert lv[0] > 0.0, "the strike must begin lit — it does not ramp in"

    drops = sum(1 for a, b in zip(lv, lv[1:]) if b < a - 1e-9)
    assert drops >= 2, f"only {drops} downward steps — that is a fade"
    lit_then_dark = [b for a, b in zip(lv, lv[1:]) if a > 0.4 and b == 0.0]
    assert lit_then_dark, "never fully drops out; a flicker must cut to black"
    assert max(lv) > 1.0, "no overshoot at the moment it holds"


def test_the_crosshair_holds_steady_once_struck(art):
    """It settles ON and stays on — an opening that ends mid-flicker would
    read as a fault rather than a finish."""
    end = art.RETICLE_T + art.RETICLE_LEN
    for t in (end, end + 0.5, end + 5.0, 60.0):
        assert art.reticle_flicker(t) == 1.0, t


def test_no_flicker_step_is_shorter_than_two_frames(art):
    """The splash paints at 33 ms. A step shorter than that gets sampled at
    an arbitrary phase and some steps would simply never be drawn — which
    looks like dropped frames, not like a strike. Two frames is the floor.

    This is also why the crosshair sits after logo._WARP's last knot: at the
    old 0.8x compression these steps would each lose a fifth of their length.
    """
    ats = [at for at, _ in art._FLICKER]
    gaps = [b - a for a, b in zip(ats, ats[1:])]
    assert gaps, "the flicker table has no steps"
    assert min(gaps) >= 2 * 0.033 - 1e-9, f"shortest step {min(gaps):.3f}s"


def test_the_whole_crosshair_shares_one_level(art):
    """It strikes whole. If reticle cells had per-cell timings the shape
    would appear to be drawn in, which is exactly what was asked against."""
    ret = [c for c in art.stencil() if c.role in ("reticle", "hub")]
    assert len(ret) > 20, "no crosshair in the stencil"
    for t in (art.RETICLE_T + 0.02, art.RETICLE_T + 0.21,
              art.RETICLE_T + 0.55, art.RETICLE_T + 2.0):
        levels = {round(art.led_state(c, t), 9) for c in ret}
        assert len(levels) == 1, f"crosshair not uniform at t={t}: {levels}"


def test_the_eye_has_no_highlight_cells_anywhere(art):
    """The glint role is gone from the art entirely — one iris, everywhere.

    Pinned at both resolutions because the two used to disagree: the splash
    asked for an intact iris plus overlay glints while every static render
    got the highlights subtracted out of the fiber detail.
    """
    for cols, rows in ((art.COLS, art.ROWS), (255, 121)):
        roles = {c.role for c in art.stencil(cols, rows)}
        assert "glint" not in roles, f"{cols}x{rows} still emits glint cells"
        assert "iris" in roles and "reticle" in roles


def test_pupil_resolves_before_the_blink_and_well_before_the_end(logo, art):
    """PUPIL_T1 landing after the show would finish the eye with no pupil."""
    t1 = _wall_of(logo, art.PUPIL_T1)
    assert t1 < _wall_of(logo, art.BLINK_T)
    assert t1 < logo.SplashScreen.MIN_SECONDS - 2.0
    assert art.pupil_mix(logo._choreo_time(t1 + 0.01)) == 1.0
    assert art.pupil_mix(
        logo._choreo_time(logo.SplashScreen.MIN_SECONDS)) == 1.0
    # ...but not before the ignition, or the spark would be swallowed
    assert art.pupil_mix(logo._choreo_time(_wall_of(logo, art._SPARK))) == 0.0


# ------------------------------------------------------------- first beat
def test_heartbeat_reads_almost_immediately(logo):
    """First glow inside ~0.2 s of wall clock, first pulse peaking ~0.35 s —
    the opening used to be a black slab until the spark at 1.15 s."""
    step = 0.002
    samples = [(i * step, logo._heartbeat(logo._choreo_time(i * step)))
               for i in range(int(1.2 / step))]
    lit = [w for w, v in samples if v > 0.02]     # _EyeField's visible cutoff
    assert lit and lit[0] < 0.20, lit[:1]
    peak_w, peak_v = max(samples, key=lambda p: p[1])
    assert 0.28 <= peak_w <= 0.45, peak_w
    assert peak_v > 0.25          # brighter than a hint, or it reads as black


def test_heartbeat_is_silent_outside_its_window(logo, art):
    assert logo._heartbeat(0.0) == 0.0
    assert logo._heartbeat(logo._HEART_T0) == 0.0
    assert logo._heartbeat(art._SPARK + 0.6) == 0.0
    assert logo._heartbeat(art.BLINK_T) == 0.0


# ------------------------------------------------------------------- skip
def test_skip_snaps_to_the_settled_pose_and_fades_fast(logo, splash):
    splash.start()
    splash._tick()
    assert splash._t < 1.0                      # still early in the show
    splash.skip()
    assert splash._skipped
    assert splash._t >= splash.MIN_SECONDS
    assert logo._choreo_time(splash._t) > logo._WARP[-1][1]

    revealed = []
    splash.finish(lambda: revealed.append(True))
    for _ in range(6):
        splash._tick()
        if revealed:
            break
    assert revealed, "skip must reveal the app within a few frames"


def test_skip_before_finish_holds_until_the_boot_worker_lands(logo, splash):
    """finish() has not been called yet: there is nothing to reveal, so the
    settled eye holds rather than the splash closing on a blank screen."""
    splash.start()
    splash.skip()
    for _ in range(10):
        splash._tick()
    assert splash.isVisible()
    assert splash._fade == 1.0
    revealed = []
    splash.finish(lambda: revealed.append(True))
    for _ in range(6):
        splash._tick()
    assert revealed


def test_no_skip_means_no_early_reveal(logo, splash):
    splash.start()
    revealed = []
    splash.finish(lambda: revealed.append(True))
    for _ in range(5):
        splash._tick()
    assert not revealed
    assert splash._fade == 1.0


def test_key_press_and_click_both_skip(qapp, logo, monkeypatch):
    """The widget's own handlers, with the app-wide filter taken out of the
    way so this pins the focused-splash path rather than the safety net."""
    monkeypatch.setattr(logo, "_EyeField", _StubField)
    events = (
        lambda: QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier),
        lambda: QMouseEvent(QEvent.MouseButtonPress, QPointF(4.0, 4.0),
                            QPointF(4.0, 4.0), Qt.LeftButton, Qt.LeftButton,
                            Qt.NoModifier),
    )
    for make in events:
        s = logo.SplashScreen()
        s.start()
        qapp.removeEventFilter(s)
        s._filtering = False
        assert not s._skipped
        qapp.sendEvent(s, make())
        assert s._skipped
        s.close()


def test_app_event_filter_catches_keys_aimed_elsewhere(qapp, logo, monkeypatch):
    """A splash-flagged window may never be activated, so the key press can
    land on another object entirely — the app-wide filter is the safety net."""
    monkeypatch.setattr(logo, "_EyeField", _StubField)
    s = logo.SplashScreen()
    s.start()
    other = QWidget()
    qapp.sendEvent(other, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                    Qt.NoModifier))
    assert s._skipped
    s.close()
    # closing must unhook the filter, or a dead splash keeps seeing every event
    assert not s._filtering
    other.deleteLater()


def test_skip_is_idempotent_and_safe_before_start(logo, splash):
    splash.skip()
    splash.skip()
    assert splash._skipped
    assert splash._t >= splash.MIN_SECONDS


# ------------------------------------------------------------ stencil tier
@pytest.mark.parametrize("screen", [(2560, 1440), (3840, 2160)])
def test_dense_tier_on_1440p_and_4k(logo, splash, screen):
    """Both stages pick 381x181 — the tier the frame-cost numbers in the
    module docstring were measured at."""
    w, h = (int(v * 0.94) for v in screen)
    splash._configure(w, h)
    assert (splash._field.cols, splash._field.rows) == (381, 181)
    assert splash._eye_rect.height() / 181 >= 5.8   # the ~3 px glyph floor
    assert splash._eye_font.pixelSize() >= 3


def test_small_stage_falls_back(logo, splash):
    splash._configure(1280, 800)
    assert splash._field.rows in (67, 95, 121)


# --------------------------------------------------------------- the paint
def test_field_paints_every_phase_without_error(qapp, logo, art):
    """A cheap tier, but the real paint path — heartbeat, ignition, pupil
    resolve, blink (lid glyphs), gleam and settle."""
    from PySide6.QtGui import QPainter, QPixmap

    field = logo._EyeField(141, 67)
    field.prepare(4.0, 8.0)
    pm = QPixmap(600, 540)
    p = QPainter(pm)
    p.setFont(art._mono())
    try:
        for wall in (0.0, 0.2, 0.4, 1.0, 1.6, 2.2, 2.6, 3.0, 3.2, 3.5,
                     3.9, 4.2, 4.7, 5.0):
            field.paint(p, logo._choreo_time(wall))
    finally:
        p.end()
    # the lid origin grid is prebuilt, so no blink frame pays for it
    assert len(field._all_pts) == 141 * 67


def test_paint_event_runs_on_the_choreography_clock(qapp, logo, splash):
    """paintEvent must hand the field _ct, not the wall clock: the wordmark
    and tagline are cued against choreography times."""
    from PySide6.QtGui import QPixmap

    splash._configure(400, 300)
    splash._t = 4.0
    splash._ct = logo._choreo_time(4.0)
    assert splash._ct > splash._t          # the warp is running ahead by now
    splash.render(QPixmap(400, 300))
    assert splash._field.last_t == pytest.approx(splash._ct)
    assert splash._frame_ms > 0.0
