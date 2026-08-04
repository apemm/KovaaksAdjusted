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
    """The section list, in order, with the two optional pages folded in
    exactly where app.py inserts them: 'What changed' sits between Analysis
    and Adaptability (it reads per TASK, where Analysis reads per run), and
    'How it learns' is always last. Both appear only once their module
    exists, so this stays true while either is authored separately."""
    names = ["Dashboard", "Scenarios", "Analysis"]
    try:
        import kovadapt.gui.changes_view  # noqa: F401
        names.append("What changed")
    except ImportError:
        pass
    names += ["Adaptability", "Optimizer"]
    try:
        import kovadapt.gui.ml_page  # noqa: F401
        names.append("How it learns")
    except ImportError:
        pass
    return names


def test_main_window_constructs_and_closes(qapp, settings):
    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    expected = _expected_sections()
    assert 5 <= len(expected) <= 7
    assert win.space.count() == len(expected)
    assert win.space.names() == expected
    # one nav link per section, in order — these are the scroll targets
    assert [b.text() for b in win.nav.links()] == expected
    assert win.dashboard.worker is None
    win.close()
    win.deleteLater()   # close() only hides it


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
        # Each section caps at the measure its CONTENT needs (shell.py
        # COLUMN_WIDTHS): prose narrow enough to read, charts and tables wide
        # enough to be legible. One global 950 made prose run ~146 characters
        # a line while squeezing the charts.
        assert page.width() <= sec.max_width + 50         # column cap
        x = page.mapTo(sec, page.rect().topLeft()).x()
        left, right = x, sec.width() - (x + page.width())
        assert left >= 20 and right >= 20                 # backdrop rails
        assert abs(left - right) <= 60                    # centered
    win.close()
    win.deleteLater()   # close() only hides it


def test_sections_use_their_content_measure(qapp, settings):
    """Prose must stay narrower than the data sections."""
    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.shell import COLUMN_WIDTHS
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    widths = {win.space.names()[i]: win.space.section_at(i).max_width
              for i in range(win.space.count())}
    assert widths["Analysis"] == COLUMN_WIDTHS["wide"]
    assert widths["Dashboard"] == COLUMN_WIDTHS["default"]
    if "How it learns" in widths:
        assert widths["How it learns"] == COLUMN_WIDTHS["prose"]
        assert widths["How it learns"] < widths["Analysis"]
    win.close()
    win.deleteLater()   # close() only hides it


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
    win.deleteLater()   # close() only hides it


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
    win.deleteLater()   # close() only hides it


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
    win.deleteLater()   # close() only hides it


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


def test_a_large_library_gets_a_table_taller_than_five_rows(qapp, settings):
    """_fit_table_height set only a MAXIMUM, so it could only ever shrink.

    A QVBoxLayout gives a non-stretched child its sizeHint, and
    QTableWidget.sizeHint() is Qt's content-independent QSize(256, 192), so
    raising the maximum above 192 did nothing at all. Measured at 80
    scenarios: 192px tall, 5 rows painted, 531px of empty page underneath —
    a porthole over the whole library, while the method's own docstring
    described a fit that never happened.
    """
    from kovadapt.gui.browser import ScenarioBrowser

    for i in range(80):
        (settings.scenarios_dir / f"Scenario {i:03d}.sce").write_text("[Scenario]\n")
    b = ScenarioBrowser(settings)
    assert b.table.rowCount() == 80
    assert b.table.height() >= 600, (
        f"table is {b.table.height()}px for 80 scenarios")
    # 14 at the 620px ceiling on this row height; the defect painted 5
    rows_shown = b.table.viewport().height() // b.table.rowHeight(0)
    assert rows_shown >= 12, f"only {rows_shown} rows visible"

    # ...and it still shrinks to fit a short list rather than leaving a
    # ruled box full of nothing, which is what the ceiling was for.
    b.search.setText("Scenario 007")
    b._fit_table_height()
    assert b.table.height() <= 260, b.table.height()
    b.deleteLater()


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


def test_theme_switch_does_not_resurrect_a_previous_runs_coach_cards(qapp, settings):
    """restyle() refills the Coach box from a cached (rep, profile) pair.

    The early-return path hid the box but left that cache populated, so
    after opening a report with no usable profile, ANY theme or accent
    change (or the OS scheme flipping while mode=='auto') re-rendered the
    PREVIOUS run's insights underneath the current run's header — stale
    coaching presented as live analysis.
    """
    from kovadapt.analysis.report import RunReport
    from kovadapt.gui.analysis_view import AnalysisView
    from kovadapt.profile.player import PlayerProfile

    def report(name: str) -> RunReport:
        return RunReport(
            scenario=name, started_iso="2026-07-28T10:00:00",
            score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
            summary_text=f"{name}: 30 kills at 61% accuracy.")

    view = AnalysisView(settings)

    # 1) a run WITH a profile that has history -> cards render, cache fills
    prof = PlayerProfile(scenario="First Run")
    prof.run_count = 5
    prof.history = [{"accuracy": 0.6, "kps": 1.2}] * 5
    view.show_report(report("First Run"), profile=prof)
    assert view._last_insights is not None

    # 2) a run whose profile carries no history -> the box must stay empty
    view.show_report(report("Second Run"), profile=PlayerProfile(scenario="Second Run"))
    assert view.coach_box.isHidden()

    # 3) a theme switch must not bring run #1's cards back
    view.restyle()
    assert view.coach_box.isHidden(), "theme switch resurrected stale Coach cards"
    assert view.coach_lay.count() == 0


def test_no_section_view_leaves_its_page_margin_to_qt(qapp, settings):
    """Every section's top-level layout states its horizontal margins.

    None of the seven did, so each inherited Qt's ~9px default while the
    section's own H1, its divider rule and every panel sit flush to
    shell._Section's column — which gave one screen three different left
    edges, with bare page text the only thing indented and lined up with
    nothing. The column IS the measure; panels pad their own contents.
    """
    from kovadapt.gui.analysis_view import AnalysisView
    from kovadapt.gui.browser import ScenarioBrowser
    from kovadapt.gui.config_view import ConfigView
    from kovadapt.gui.dashboard import Dashboard
    from kovadapt.gui.optimizer_view import OptimizerView

    (settings.scenarios_dir / "Alpha.sce").write_text("[Scenario]\n")
    views = [Dashboard(settings), ScenarioBrowser(settings),
             AnalysisView(settings), ConfigView(settings),
             OptimizerView(settings)]
    try:
        for v in views:
            m = v.layout().contentsMargins()
            assert (m.left(), m.right()) == (0, 0), (
                f"{type(v).__name__} indents its page by "
                f"({m.left()}, {m.right()}) — it will not line up with the "
                f"section header, the divider or any panel")
    finally:
        for v in views:
            if hasattr(v, "shutdown"):
                v.shutdown()
            v.deleteLater()


def test_filtering_the_selection_off_screen_disarms_the_actions(qapp, settings):
    """Typing a filter that hides the selected row left Play, Start adapting
    and Generate all ENABLED for a scenario no longer on screen, with the
    detail line still describing it.

    Qt does drop a hidden row out of selectedItems() (measured — I had
    assumed the opposite and written a guard for it), so `selected()` goes
    empty on its own. What was missing is that nothing RE-ASKED: `_rebuild`
    calls `_selection_changed`, the typing and archetype-filter path did not.
    """
    from kovadapt.gui.browser import ScenarioBrowser

    for n in ("Alpha Track Long", "Beta 1wall Click", "Gamma Switch"):
        (settings.scenarios_dir / f"{n}.sce").write_text("[Scenario]\n")
    b = ScenarioBrowser(settings)
    b.table.selectRow(0)
    assert b.selected()
    assert b.play_btn.isEnabled()

    b.search.setText("zzzzzz")                 # hides every row
    assert all(b.table.isRowHidden(r) for r in range(b.table.rowCount()))
    assert b.selected() == "", "a hidden row is still reported as selected"
    for btn in (b.play_btn, b.watch_btn, b.gen_btn):
        assert not btn.isEnabled(), "armed for something invisible"
    assert b.detail.text() == "select a scenario"

    b.search.setText("")                       # and it comes back
    assert b.selected()
    assert b.play_btn.isEnabled()
    b.deleteLater()


def test_the_motion_setting_reaches_the_backdrop_through_the_signal(qapp, settings):
    """`config_view.settings_changed` was declared AND emitted with zero
    connected receivers, so changing motion in Settings did nothing until the
    next alt-tab — and turning it ON left the backdrop frozen, which looks
    broken rather than merely costly.

    Driven through the SIGNAL, not by calling the backdrop directly: the
    defect was the missing connection, and a test that calls the slot itself
    cannot see it.
    """
    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    settings.motion = "full"
    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    win.show()
    qapp.processEvents()
    assert win.backdrop._timer.isActive()

    settings.motion = "off"
    win.config.settings_changed.emit(settings)
    qapp.processEvents()
    assert not win.backdrop._timer.isActive(), (
        "changing motion in Settings did not reach the backdrop")

    settings.motion = "full"
    win.config.settings_changed.emit(settings)
    qapp.processEvents()
    assert win.backdrop._timer.isActive(), (
        "turning motion back on left the backdrop frozen")
    win.close()
    win.deleteLater()
