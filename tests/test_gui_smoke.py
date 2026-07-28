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


def test_main_window_constructs_and_closes(qapp, settings):
    from kovadapt.gui.app import MainWindow
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    win = MainWindow(settings, themes)
    assert win._tabs.count() == 5
    assert [win._tabs.tabText(i) for i in range(5)] == [
        "Dashboard", "Scenarios", "Analysis", "Adaptability", "Optimizer"]
    assert win.dashboard.worker is None
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
    from kovadapt.analysis.report import RunReport
    from kovadapt.gui.analysis_view import AnalysisView

    view = AnalysisView(settings)
    rep = RunReport(
        scenario="Beta 1wall Click", started_iso="2026-07-28T10:00:00",
        score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
        notable=[{"kind": "overshoot", "text": "worst overshoot",
                  "t_start": 1.0, "t_end": 2.0}],
        summary_text="30 kills at 61% accuracy.")
    view.show_report(rep)                # no trace: heatmap/replay stay empty

    # captions: plain-language, dim, word-wrapped, non-empty
    for cap in (view.bias_caption, view.heat_caption):
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
