"""The rebuilt in-game session overlay (gui/overlay.py).

Pins the correctness fix first: the accuracy spark no longer normalises to
its own min/max, so a session drifting a point renders as a drift. Then the
surface contracts around it — a frozen reference, mono numerals, background
alpha instead of window opacity, no timer on a hidden widget, and the
click-through / (-1, -1) position rules the card has always had.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.analysis.report import RunReport  # noqa: E402
from kovadapt.config import Settings  # noqa: E402
from kovadapt.gui import theme  # noqa: E402
from kovadapt.gui.overlay import _MIN_HALF_SPAN, _BaselineSpark, OverlayWindow  # noqa: E402
from kovadapt.profile.player import PlayerProfile  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Settings.save() defaults to Path.home()/.kovadapt/settings.json and the
    overlay saves on drag — never let that reach the developer's real file."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def settings(tmp_path):
    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    (root / "Saved" / "SaveGames" / "Scenarios").mkdir(parents=True)
    return Settings(
        kovaaks_root=str(root),
        profile_dir=str(tmp_path / "prof"),
        telemetry_enabled=False,
        onboarding_done=True,
    )


def _report(accuracy: float, score: float = 400.0, fatigue: dict | None = None,
            input_health: dict | None = None) -> RunReport:
    return RunReport(
        scenario="Beta 1wall Click", started_iso="2026-07-28T10:00:00",
        score=score, accuracy=accuracy, avg_ttk=0.9, kills=30, kps=1.4,
        fatigue=fatigue or {}, input_health=input_health or {},
        summary_text="30 kills.")


def _profile(ewma: float = 0.61, score: float = 400.0, scale: float = 1.0,
             movement: float = 0.15) -> PlayerProfile:
    p = PlayerProfile(scenario="Beta 1wall Click")
    p.ewma_accuracy, p.ewma_score = ewma, score
    p.target_scale, p.movement = scale, movement
    p.run_count = 10
    return p


def _spark(qapp) -> _BaselineSpark:
    sp = _BaselineSpark()
    sp.resize(280, sp.height())          # a real lattice, not the 100px default
    return sp


# --------------------------------------------------------------- the fix
def test_flat_session_renders_flat(qapp):
    """61% -> 62% is a drift, and must paint as one. Min-max normalisation
    sent the lowest run to the floor and the highest to the ceiling whatever
    their magnitudes, which manufactured a climb out of noise."""
    sp = _spark(qapp)
    sp.set_reference(0.61, "base")
    sp.set_values([0.610, 0.612, 0.615, 0.618, 0.620])
    assert sp.span() == pytest.approx(_MIN_HALF_SPAN)       # the floor held
    levels = sp.levels()
    assert len(levels) == 5
    assert max(abs(x) for x in levels) < 0.15
    # under min-max these same five values spanned the full chart
    assert max(levels) - min(levels) < 0.2


def test_two_runs_a_point_apart_do_not_span_the_chart(qapp):
    sp = _spark(qapp)
    sp.set_reference(0.61, "base")
    sp.set_values([0.61, 0.62])
    lo, hi = sp.levels()
    assert abs(hi - lo) < 0.15              # min-max gave exactly 0.0 and 1.0


def test_a_real_swing_still_opens_the_scale(qapp):
    """The floor is a floor, not a clamp: a run that genuinely leaves the
    band grows the span so it stays on the chart."""
    sp = _spark(qapp)
    sp.set_reference(0.60, "base")
    sp.set_values([0.60, 0.45, 0.78])
    assert sp.span() == pytest.approx(0.18)                 # widest deviation
    levels = sp.levels()
    assert levels[0] == pytest.approx(0.0)
    assert levels[1] == pytest.approx(-0.15 / 0.18, abs=1e-6)
    assert levels[2] == pytest.approx(1.0)


def test_empty_and_referenceless_states_are_inert(qapp):
    sp = _spark(qapp)
    assert sp.levels() == []
    assert sp.span() == pytest.approx(_MIN_HALF_SPAN)
    sp.set_values([0.6, 0.7])               # values but no reference yet
    assert sp.levels() == []


# ------------------------------------------------------- the reference line
def test_reference_is_frozen_for_the_session(qapp, settings):
    """The profile handed to on_report has already folded the run in, so a
    live EWMA walks with the session and hides real movement."""
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.on_report(_report(0.61), _profile(ewma=0.610))
    ov.on_report(_report(0.70), _profile(ewma=0.650))
    ov.on_report(_report(0.72), _profile(ewma=0.690))
    value, kind = ov.spark.reference()
    assert kind == "base"
    assert value == pytest.approx(0.610)
    assert max(ov.spark.levels()) > 0.5     # the climb is visible, not absorbed
    ov.close()


def test_session_average_fallback_is_labelled(qapp, settings):
    """No profile -> the card anchors to the session mean and SAYS so rather
    than presenting a made-up number as your baseline."""
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.on_report(_report(0.50), None)
    ov.on_report(_report(0.60), None)
    value, kind = ov.spark.reference()
    assert kind == "avg"
    assert value == pytest.approx(0.55)
    ov.close()


def test_start_session_clears_the_previous_one(qapp, settings):
    ov = OverlayWindow(settings)
    ov.start_session("A [Adaptive]")
    ov.on_report(_report(0.61), _profile(ewma=0.61))
    ov.start_session("B [Adaptive]")
    assert ov.scenario.text() == "B [Adaptive]"
    assert ov.spark.reference() == (None, "base")
    assert ov.spark.levels() == []
    assert ov.deck.row_text("acc") == "—"
    assert ov.status.text() == "watching"
    ov.close()


# --------------------------------------------------------------- the deck
def test_deck_reports_the_run_against_the_frozen_baseline(qapp, settings):
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.on_report(
        _report(0.640, score=512.0,
                input_health={"polling_hz_est": 998.0, "jitter_ms": 0.4}),
        _profile(ewma=0.610, score=500.0, scale=1.25, movement=0.30))
    assert ov.deck.row_text("acc") == "64.0% +3.0pp"    # the delta is stated
    assert ov.deck.row_text("score") == "512 +12"
    assert ov.deck.row_text("size") == "1.25x"
    assert ov.deck.row_text("move") == "0.30"
    assert ov.deck.row_text("input") == "998Hz ±0.4ms"
    assert ov.status.text() == "watching · 1 run"
    ov.close()


def test_fatigue_row_waits_for_the_tracker(qapp, settings):
    """FatigueState defaults to level 'fresh' before it has enough runs —
    printing that is reporting a verdict the tracker has not reached."""
    settings.fatigue_min_runs = 5
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.on_report(_report(0.61, fatigue={"level": "fresh", "score": 0.0, "runs": 2}),
                 _profile())
    assert ov.deck.row_text("fatigue") == "2/5 runs"
    ov.on_report(_report(0.55, fatigue={"level": "declining", "score": 0.4, "runs": 6}),
                 _profile())
    assert ov.deck.row_text("fatigue") == "declining"
    ov.close()


def test_missing_telemetry_says_so(qapp, settings):
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.on_report(_report(0.61), _profile())
    assert ov.deck.row_text("input") == "no telemetry"
    ov.close()


def test_values_are_grid_snapped_mono(qapp, settings):
    """theme.mono only, so digits keep their column as they update. The
    labels carry it in their STYLESHEET — setFont loses to the app-wide
    'Segoe UI' rule."""
    ov = OverlayWindow(settings)
    for font in (ov.deck.value_font(), ov.deck.caption_font()):
        assert font.family() == theme.mono_family()
        assert font.pixelSize() in theme.CELL_SIZES
    for label in (ov.status, ov.title):
        assert theme.mono_family() in label.styleSheet()
        assert "font-size: 12px" in label.styleSheet()
    ov.close()


# ------------------------------------------------------------- translucency
def test_opacity_is_background_alpha_not_window_opacity(qapp, settings):
    """setWindowOpacity fades the glyphs too — that is the muddiness."""
    ov = OverlayWindow(settings)
    assert ov.windowOpacity() == pytest.approx(1.0)
    opaque = ov.panel_alpha()
    ov.set_opacity(0.4)
    assert settings.overlay_opacity == pytest.approx(0.4)
    assert ov.windowOpacity() == pytest.approx(1.0)
    assert ov.panel_alpha() < opaque
    ov.show_overlay()
    assert ov.windowOpacity() == pytest.approx(1.0)   # show must not re-add it
    ov.close()


# ------------------------------------------------------------------ frames
def test_no_timer_while_hidden_or_between_sessions(qapp, settings):
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.on_report(_report(0.61), _profile(ewma=0.61))
    assert not ov.spark._anim.isActive()          # never shown: never ticking
    ov.show_overlay()
    assert ov.spark._anim.isActive()
    ov.hide()
    assert not ov.spark._anim.isActive()
    ov.show_overlay()
    assert ov.spark._anim.isActive()
    ov.stop_session()
    assert not ov.spark._anim.isActive()          # idle card earns no frames
    ov.close()


def test_no_timer_before_the_first_run(qapp, settings):
    ov = OverlayWindow(settings)
    ov.start_session("Beta 1wall Click [Adaptive]")
    ov.show_overlay()
    assert not ov.spark._anim.isActive()          # nothing to animate yet
    ov.close()


# --------------------------------------------------------------- contracts
def test_clickthrough_survives_the_unlock_round_trip(qapp, settings):
    ov = OverlayWindow(settings)
    assert bool(ov.windowFlags() & Qt.WindowTransparentForInput)
    ov.set_unlocked(True)
    assert not bool(ov.windowFlags() & Qt.WindowTransparentForInput)
    assert not ov.hint.isHidden() or not ov.isVisible()
    ov.set_unlocked(False)
    assert bool(ov.windowFlags() & Qt.WindowTransparentForInput)
    assert bool(ov.windowFlags() & Qt.FramelessWindowHint)
    ov.close()


def test_negative_saved_position_is_honoured(qapp, settings):
    """(-1, -1) is the ONLY unset sentinel: a monitor left of or above the
    primary gives legitimately negative coordinates."""
    ov = OverlayWindow(settings)
    settings.overlay_x, settings.overlay_y = -5, -5
    ov._place()
    assert (ov.pos().x(), ov.pos().y()) == (-5, -5)
    settings.overlay_x = settings.overlay_y = -1
    ov._place()
    assert (ov.pos().x(), ov.pos().y()) != (-1, -1)
    ov.close()


# ------------------------------------------------------------------- paint
def _render(widget) -> None:
    widget.adjustSize()
    pm = QPixmap(max(widget.width(), 1), max(widget.height(), 1))
    pm.fill(Qt.transparent)
    widget.render(pm)


def test_card_paints_empty_and_full_in_both_themes(qapp, settings):
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    ov = OverlayWindow(settings)
    ov.resize(ov.width(), 320)
    for mode in ("dark", "light"):
        themes.set_mode(mode)
        ov.restyle()
        _render(ov)                                   # empty state
        ov.start_session("Beta 1wall Click [Adaptive]")
        for acc in (0.58, 0.61, 0.63, 0.60, 0.71):
            ov.on_report(
                _report(acc, score=400 + acc * 100,
                        fatigue={"level": "declining", "score": 0.4, "runs": 9},
                        input_health={"polling_hz_est": 998.0, "jitter_ms": 1.8}),
                _profile(ewma=0.61, scale=0.9, movement=0.42))
        _render(ov)                                   # populated state
    ov.close()
    themes.set_mode("dark")


def test_spark_paints_its_empty_state(qapp):
    sp = _spark(qapp)
    _render(sp)
    sp.set_reference(0.61, "base")
    sp.set_values([0.61])                             # a single run: no polyline
    _render(sp)
