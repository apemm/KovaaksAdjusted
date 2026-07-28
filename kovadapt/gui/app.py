"""kovadapt GUI: the KovaaK's hub.

    Dashboard    play adaptive tasks, launch the game, live session + overlay
    Analysis     post-run report: bias, heatmap, notable moments, replays/clips
    Adaptability full configuration surface
    Optimizer    free Process Lasso basics + tuning checklist

Themes follow the Windows light/dark setting ("auto") or can be pinned; the
first launch opens a short startup guide, and dismissible TIP bars carry the
instructions after that. Run with `kovadapt gui` (pip install kovadapt[gui]).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QPushButton,
    QTabWidget,
    QWidget,
)

from ..config import Settings
from .analysis_view import AnalysisView
from .browser import ScenarioBrowser
from .config_view import ConfigView
from .dashboard import Dashboard
from .onboarding import WelcomeDialog, set_hints_visible
from .optimizer_view import OptimizerView
from .theme import ThemeManager


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, themes: ThemeManager) -> None:
        super().__init__()
        self.s = settings
        self.themes = themes
        self.setWindowTitle("kovadapt — adaptive KovaaK's")
        self.resize(1140, 760)

        tabs = QTabWidget()
        self.dashboard = Dashboard(settings)
        self.browser = ScenarioBrowser(settings)
        self.analysis = AnalysisView(settings)
        self.config = ConfigView(settings)
        self.optimizer = OptimizerView(settings)
        tabs.addTab(self.dashboard, "Dashboard")
        tabs.addTab(self.browser, "Scenarios")
        tabs.addTab(self.analysis, "Analysis")
        tabs.addTab(self.config, "Adaptability")
        tabs.addTab(self.optimizer, "Optimizer")
        self.setCentralWidget(tabs)
        self._tabs = tabs
        tabs.setCornerWidget(self._corner(), Qt.TopRightCorner)

        # browser actions land on the dashboard (session owner)
        self.browser.play_requested.connect(self._browser_play)
        self.browser.watch_requested.connect(self._browser_watch)

        # new run report -> refresh analysis tab and flag it
        self.dashboard.report_ready.connect(self._on_report)
        tabs.currentChanged.connect(self._clear_unread)
        themes.changed.connect(self._restyle)

        sb = self.statusBar()
        sb.showMessage(
            f"KovaaK's: {settings.kovaaks_root or 'NOT FOUND — set KOVAAKS_ROOT'}"
        )

    # ----------------------------------------------------------- corner bar
    def _corner(self) -> QWidget:
        self.theme_pick = QComboBox()
        self.theme_pick.addItems(["Auto theme", "Dark", "Light"])
        self.theme_pick.setToolTip("Auto follows the Windows light/dark setting")
        self.theme_pick.setCurrentIndex(
            {"auto": 0, "dark": 1, "light": 2}.get(self.themes.mode, 0))
        self.theme_pick.currentIndexChanged.connect(
            lambda i: self.themes.set_mode(("auto", "dark", "light")[i]))

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(30)
        help_btn.setToolTip("Guide, hints, and your data")
        menu = QMenu(help_btn)
        menu.addAction("Startup guide…", self._show_guide)
        self._hints_action = menu.addAction("Show hints")
        self._hints_action.setCheckable(True)
        self._hints_action.setChecked(self.s.show_hints)
        self._hints_action.toggled.connect(
            lambda on: set_hints_visible(self.s, on))
        # A TIP bar's × also hides hints; resync so the first menu click
        # after that actually re-enables them instead of re-hiding.
        menu.aboutToShow.connect(self._sync_hints_action)
        menu.addSeparator()
        menu.addAction("Open data folder", self._open_data_dir)
        help_btn.setMenu(menu)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 2, 6, 4)
        lay.setSpacing(6)
        lay.addWidget(self.theme_pick)
        lay.addWidget(help_btn)
        return w

    def _sync_hints_action(self) -> None:
        self._hints_action.blockSignals(True)
        self._hints_action.setChecked(self.s.show_hints)
        self._hints_action.blockSignals(False)

    def _show_guide(self) -> None:
        WelcomeDialog(self.s, self).exec()
        self._sync_hints_action()

    def _open_data_dir(self) -> None:
        p = Path(self.s.profile_dir)
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _browser_play(self, name: str) -> None:
        self._tabs.setCurrentWidget(self.dashboard)
        self.dashboard.play_scenario(name)

    def _browser_watch(self, name: str) -> None:
        self._tabs.setCurrentWidget(self.dashboard)
        self.dashboard.watch_scenario(name)

    # ------------------------------------------------------------------
    def _restyle(self, pal) -> None:
        for view in (self.dashboard, self.browser, self.analysis, self.optimizer):
            view.restyle(pal)

    def _on_report(self, rep) -> None:
        self.analysis.show_report(rep, profile=self.dashboard.last_profile)
        self.optimizer.note_report(rep)
        idx = self._tabs.indexOf(self.analysis)
        self._tabs.setTabText(idx, "Analysis •")

    def _clear_unread(self, i: int) -> None:
        idx = self._tabs.indexOf(self.analysis)
        if i == idx:
            self._tabs.setTabText(idx, "Analysis")

    def closeEvent(self, event) -> None:  # stop worker threads cleanly
        w = self.dashboard.worker
        if w is not None:
            w.stop()
            # A QThread destroyed while running is a fatal abort in Qt 6;
            # stop latency is ~1s, so this wait practically always succeeds.
            if not w.wait(10000):
                # Post-run processing can legitimately exceed it (clip
                # encoding on a slow disk). Losing one run's adaptation
                # beats aborting the whole process at exit — the profile
                # save is atomic, so no file is left torn.
                w.terminate()
                w.wait(2000)
        self.dashboard.shutdown()  # overlay window
        self.optimizer.shutdown()  # optimizer window + in-flight scan QThread
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    settings = Settings.load()
    themes = ThemeManager(app, settings)
    win = MainWindow(settings, themes)
    win.show()
    if not settings.onboarding_done:
        QTimer.singleShot(150, lambda: WelcomeDialog(settings, win).exec())
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
