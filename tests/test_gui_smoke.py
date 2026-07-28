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
    assert win._tabs.count() == 4
    assert [win._tabs.tabText(i) for i in range(4)] == [
        "Dashboard", "Analysis", "Adaptability", "Optimizer"]
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
