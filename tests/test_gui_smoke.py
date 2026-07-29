"""GUI smoke tests (offscreen QPA): the app constructs, themes switch, the
overlay and onboarding pieces exist. Skipped wholesale without PySide6."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import Settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Settings.save() defaults to Path.home()/.kovadapt/settings.json — these
    tests exercise theme/hint/guide code that saves, and must never write the
    developer's real settings file (that actually happened once)."""
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


def _expected_sections() -> list[str]:
    """5 base sections, +'How it learns' once gui/ml_page.py exists."""
    names = ["Dashboard", "Scenarios", "Analysis", "Adaptability", "Optimizer"]
    try:
        import kovadapt.gui.ml_page  # noqa: F401
    except ImportError:
        return names
    return names + ["How it learns"]


def test_main_window_constructs_and_closes(qapp, settings):
    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    expected = _expected_sections()
    assert len(expected) in (5, 6)
    assert win.space.count() == len(expected)
    assert win.space.names() == expected
    # one nav link per section, in order — these are the scroll targets
    assert [b.text() for b in win.nav.links()] == expected
    assert win.dashboard.worker is None
    win.close()


def test_editorial_column_caps_content_width(qapp, settings):
    """Every section flows in one centered column: content capped near
    ~950px at the 1360 default window, whitespace rails both sides."""
    from PySide6.QtTest import QTest

    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    win.resize(1360, 900)
    win.show()
    QTest.qWait(50)
    for i in range(win.space.count()):
        sec = win.space.section_at(i)
        page = sec.page
        assert page.width() <= 1000                       # column cap
        x = page.mapTo(sec, page.rect().topLeft()).x()
        left, right = x, sec.width() - (x + page.width())
        assert left >= 80 and right >= 80                 # backdrop rails
        assert abs(left - right) <= 60                    # centered
    win.close()


def test_shell_sections_lay_out_and_nav_scrolls(qapp, settings):
    from PySide6.QtTest import QTest

    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    win.resize(1100, 720)
    win.show()
    QTest.qWait(50)          # let the min-height relayout settle
    # every section is laid out at least one viewport tall
    viewport_h = win.space.viewport().height()
    assert viewport_h > 0
    for i in range(win.space.count()):
        assert win.space.section_at(i).height() >= viewport_h
    # a nav jump lands the section's top at the scroll position
    bar = win.space.verticalScrollBar()
    assert bar.value() == 0
    win.space.scroll_to(3, animated=False)
    assert bar.value() == min(win.space.section_at(3).y(), bar.maximum())
    assert bar.value() > 0
    assert win.space.current_index() == 3
    win.close()


def test_browser_play_scrolls_home_and_reaches_dashboard(qapp, settings):
    from PySide6.QtTest import QTest

    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    win.resize(1100, 720)
    win.show()
    QTest.qWait(50)
    win.space.scroll_to(1, animated=False)          # start at Scenarios
    assert win.space.current_index() == 1
    played: list[str] = []
    win.dashboard.play_scenario = played.append     # never launch the game
    win.browser.play_requested.emit("Alpha Track Long")
    assert played == ["Alpha Track Long"]           # signal reached dashboard
    QTest.qWait(600)                                # smooth-scroll home plays out
    assert win.space.current_index() == 0
    win.close()


def test_report_badges_analysis_nav_link(qapp, settings):
    from PySide6.QtTest import QTest

    from kovadapt.analysis.report import RunReport
    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    win.resize(1100, 720)
    win.show()
    QTest.qWait(50)
    rep = RunReport(
        scenario="Beta 1wall Click", started_iso="2026-07-28T10:00:00",
        score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
        summary_text="30 kills at 61% accuracy.")
    idx = win.space.index_of(win.analysis)
    win.dashboard.report_ready.emit(rep)            # full _on_report path
    assert win.nav.links()[idx].text() == "Analysis •"
    win.space.scroll_to(idx, animated=False)        # into view clears the dot
    assert win.space.current_index() == idx
    assert win.nav.links()[idx].text() == "Analysis"
    win.close()


def test_theme_switch_relayers_the_app(qapp, settings):
    from kovadapt.gui import theme
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    themes.set_mode("light")
    assert not theme.current().is_dark
    assert theme.LIGHT.bg in qapp.styleSheet()
    assert settings.theme == "light"        # persisted choice
    themes.set_mode("dark")
    assert theme.current().is_dark
    assert theme.DARK.bg in qapp.styleSheet()


def test_overlay_lifecycle(qapp, settings):
    from kovadapt.gui.overlay import OverlayWindow

    ov = OverlayWindow(settings)
    ov.start_session("Foo [Adaptive]")
    assert ov.scenario.text() == "Foo [Adaptive]"
    ov.set_opacity(0.5)
    assert settings.overlay_opacity == 0.5
    ov.set_unlocked(True)
    ov.set_unlocked(False)
    ov.stop_session()
    ov.close()


def test_scenario_browser_lists_and_filters(qapp, settings):
    from kovadapt.gui.browser import ScenarioBrowser

    (settings.scenarios_dir / "Alpha Track Long.sce").write_text("[Scenario]\n")
    (settings.scenarios_dir / "Beta 1wall Click.sce").write_text("[Scenario]\n")
    (settings.scenarios_dir / "Beta 1wall Click [Adaptive].sce").write_text("[Scenario]\n")
    b = ScenarioBrowser(settings)
    assert b.table.rowCount() == 2          # adaptive variant folds into base
    names = {b.table.item(i, 0).data(0x0100) for i in range(2)}  # Qt.UserRole
    assert names == {"Alpha Track Long", "Beta 1wall Click"}
    b.search.setText("beta")
    visible = [i for i in range(2) if not b.table.isRowHidden(i)]
    assert len(visible) == 1
    b.search.setText("")
    fired: list[str] = []
    b.play_requested.connect(fired.append)
    b.table.selectRow(0)
    b._emit_play()
    assert len(fired) == 1


def test_config_sensitivity_group_computes_cm360(qapp, settings):
    """Mouse & sensitivity group: DPI/sens spins drive the live cm/360
    readout (KovaaK's yaw 0.022°/count) and save onto Settings."""
    from kovadapt.gui.config_view import ConfigView

    view = ConfigView(settings)
    view.dpi.setValue(800)
    view.sens.setValue(1.0)
    expected = 2.54 * 360.0 / (800 * 1.0 * 0.022)
    assert f"{expected:.1f}" in view.cm360.text()
    view.sens.setValue(2.0)                    # live: either spin updates it
    expected = 2.54 * 360.0 / (800 * 2.0 * 0.022)
    assert f"{expected:.1f}" in view.cm360.text()
    view._save()                               # wired like every other field
    assert getattr(settings, "mouse_dpi", None) == 800
    assert getattr(settings, "game_sens", None) == 2.0
    view.deleteLater()


def test_hint_bars_tuck_away(qapp, settings):
    from kovadapt.gui.onboarding import HintBar, set_hints_visible

    bar = HintBar(settings, "hello")
    set_hints_visible(settings, False)
    assert settings.show_hints is False
    assert bar.isHidden()
    set_hints_visible(settings, True)
    assert settings.show_hints is True


def test_welcome_dialog_completion_marks_done(qapp, settings):
    from kovadapt.gui.onboarding import WelcomeDialog

    settings.onboarding_done = False
    dlg = WelcomeDialog(settings)
    assert dlg.pages.count() >= 3
    assert not dlg.again.isChecked()     # finishing must not re-show by default
    for _ in range(dlg.pages.count() - 1):
        dlg._next()
    assert dlg.next_btn.text() == "Get started"
    dlg._next()                          # accept() on the last page
    assert settings.onboarding_done is True
    dlg.deleteLater()


def test_welcome_dialog_dismissed_early_shows_again(qapp, settings):
    from kovadapt.gui.onboarding import WelcomeDialog

    settings.onboarding_done = False
    dlg = WelcomeDialog(settings)
    dlg.reject()                         # closed without finishing
    assert settings.onboarding_done is False
    dlg.deleteLater()


def test_analysis_view_captions_toggles_and_clip_story(qapp, settings):
    import pyqtgraph as pg

    from kovadapt.analysis.report import RunReport
    from kovadapt.gui.analysis_view import AnalysisView
    from kovadapt.gui.viz import AsciiBars, AsciiHeatmap, AsciiTrend

    view = AnalysisView(settings)
    rep = RunReport(
        scenario="Beta 1wall Click", started_iso="2026-07-28T10:00:00",
        score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
        notable=[{"kind": "overshoot", "text": "worst overshoot",
                  "t_start": 1.0, "t_end": 2.0}],
        summary_text="30 kills at 61% accuracy.")
    view.show_report(rep)                # no trace: heatmap/replay stay empty

    # the charts are ASCII-art widgets (gui/viz.py); pyqtgraph survives ONLY
    # as the TrajectoryReplay canvas
    assert isinstance(view.bias_bars, AsciiBars)
    assert isinstance(view.heat_map, AsciiHeatmap)
    assert isinstance(view.trend_spark, AsciiTrend)
    assert view.findChildren(pg.PlotWidget) == [view.replay.plot]
    assert view.trend_w.isHidden()       # no profile history yet -> no sparkline

    # captions: plain-language, dim, word-wrapped, non-empty
    for cap in (view.bias_caption, view.heat_caption, view.trend_caption):
        assert cap.text()
        assert cap.wordWrap()
        assert cap.property("dim") is True

    # replay layer toggles flip visibility of the existing plot items
    r = view.replay
    for box, items in ((r.toggle_path, (r._full,)),
                       (r.toggle_flicks, (r._good, r._bad)),
                       (r.toggle_shots, (r._shots,))):
        assert box.isChecked()           # all layers default ON
        box.setChecked(False)
        assert all(not it.isVisible() for it in items)
        box.setChecked(True)
        assert all(it.isVisible() for it in items)

    # clips are disabled in these Settings: the dead button explains itself,
    # and the same one-liner appears as the inline dim hint under it
    assert not view.clip_btn.isEnabled()
    assert "Capture video clips" in view.clip_btn.toolTip()
    assert not view.clip_hint.isHidden()
    assert view.clip_hint.text() == view.clip_btn.toolTip()
    view.deleteLater()
