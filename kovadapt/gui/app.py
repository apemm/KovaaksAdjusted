"""kovadapt GUI: the KovaaK's hub.

    Dashboard    play adaptive tasks, launch the game, live session + overlay
    Scenarios    every installed scenario, training state, one click to play
    Analysis     post-run report: bias, heatmap, notable moments, replays/clips
    Adaptability full configuration surface
    Optimizer    free Process Lasso basics + tuning checklist

One continuous scrollable page-space (gui/shell.py) instead of tabs: the
sections stack over the parallax backdrop and the slim top nav bar
smooth-scrolls between them. Themes follow the Windows light/dark setting
("auto") or can be pinned; the first launch opens a short startup guide, and
dismissible TIP bars carry the instructions after that. Run with
`kovadapt gui` (pip install kovadapt[gui]).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtGui import QPainter

from ..config import Settings
from . import logo, transition
from .backdrop import Backdrop
from .analysis_view import AnalysisView
from .browser import ScenarioBrowser
from .config_view import ConfigView
from .dashboard import Dashboard
from .onboarding import WelcomeDialog, set_hints_visible
from .optimizer_view import OptimizerView
from .shell import NavBar, PageSpace
from .theme import ACCENTS, ThemeManager


class _ThemeCombo(QComboBox):
    """Theme picker whose 'Gamer' entry wears its own cycling RGB letters."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._advance)
        self.currentIndexChanged.connect(lambda _i: self._sync_timer())

    def _sync_timer(self) -> None:
        if self.currentData() == "rgb":
            self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def _advance(self) -> None:
        from PySide6.QtGui import QBrush, QColor

        self._phase += 1
        # the popup entry cycles too
        idx = self.findData("rgb")
        if idx >= 0:
            self.setItemData(idx, QBrush(QColor.fromHsvF(
                (self._phase * 0.045) % 1.0, 0.85, 1.0)), Qt.ForegroundRole)
        self.update()

    def paintEvent(self, event) -> None:
        if self.currentData() != "rgb":
            super().paintEvent(event)
            return
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QStyle, QStyleOptionComboBox, QStylePainter

        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = ""
        sp = QStylePainter(self)
        sp.drawComplexControl(QStyle.CC_ComboBox, opt)
        rect = self.style().subControlRect(
            QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)
        x = rect.x() + 6
        for i, chq in enumerate("Gamer"):
            sp.setPen(QColor.fromHsvF(
                ((self._phase * 0.045) + i * 0.14) % 1.0, 0.85, 1.0))
            sp.drawText(x, rect.y(), rect.width(), rect.height(),
                        Qt.AlignVCenter, chq)
            x += sp.fontMetrics().horizontalAdvance(chq)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, themes: ThemeManager) -> None:
        super().__init__()
        self.s = settings
        self.themes = themes
        self.setWindowTitle("kovadapt — adaptive KovaaK's")
        self.resize(1300, 860)

        self.dashboard = Dashboard(settings)
        self.browser = ScenarioBrowser(settings)
        self.analysis = AnalysisView(settings)
        self.config = ConfigView(settings)
        self.optimizer = OptimizerView(settings)
        # one continuous scroll of transparent sections over the backdrop
        self.space = PageSpace()
        for name, page in (("Dashboard", self.dashboard),
                           ("Scenarios", self.browser),
                           ("Analysis", self.analysis),
                           ("Adaptability", self.config),
                           ("Optimizer", self.optimizer)):
            self.space.add_section(name, page)
        self.nav = NavBar(self.space, corner=self._corner())

        central = QWidget()
        central.setObjectName("tabPage")   # transparent: backdrop shows through
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self.nav)
        col.addWidget(self.space, 1)
        self.setCentralWidget(central)
        self.backdrop = Backdrop(self)

        # Ctrl+1..5 jump straight to a section
        for i in range(self.space.count()):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda i=i: self.space.scroll_to(i))

        # browser actions land on the dashboard (session owner)
        self.browser.play_requested.connect(self._browser_play)
        self.browser.watch_requested.connect(self._browser_watch)

        # new run report -> refresh analysis section and badge its nav link
        self.dashboard.report_ready.connect(self._on_report)
        self.space.current_changed.connect(self._section_changed)
        themes.changed.connect(self._restyle)

        sb = self.statusBar()
        sb.showMessage(
            f"KovaaK's: {settings.kovaaks_root or 'NOT FOUND — set KOVAAKS_ROOT'}"
        )

    # ----------------------------------------------------------- corner bar
    def _corner(self) -> QWidget:
        self.theme_pick = _ThemeCombo()
        for label, mode in (("Auto theme", "auto"), ("Light", "light"),
                            ("Dark", "dark"), ("Midnight", "midnight"),
                            ("Gamer", "rgb")):
            self.theme_pick.addItem(label, mode)
        self.theme_pick.setToolTip(
            "Auto follows Windows · Midnight is near-black · Gamer is "
            "midnight with cycling RGB (and a certain cat)")
        self.theme_pick.setCurrentIndex(
            {"auto": 0, "light": 1, "dark": 2, "midnight": 3, "rgb": 4}
            .get(self.themes.mode, 0))
        self.theme_pick.currentIndexChanged.connect(self._pick_mode)

        self.accent_pick = QComboBox()
        for key in ACCENTS:
            self.accent_pick.addItem(key.capitalize(), key)
        self.accent_pick.setToolTip("Accent color")
        idx = list(ACCENTS).index(self.s.accent) if self.s.accent in ACCENTS else 0
        self.accent_pick.setCurrentIndex(idx)
        self.accent_pick.currentIndexChanged.connect(self._pick_accent)

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
        lay.addWidget(self.accent_pick)
        lay.addWidget(help_btn)
        return w

    def _pick_mode(self, i: int) -> None:
        transition.ascii_wipe(self)   # capture the old look, then restyle
        self.themes.set_mode(self.theme_pick.itemData(i))

    def _pick_accent(self, i: int) -> None:
        transition.ascii_wipe(self)
        self.themes.set_accent(self.accent_pick.itemData(i))

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

    def set_trends(self, trends) -> None:
        """Cross-session skill model from the boot worker (may be None)."""
        if trends is None:
            return
        self.analysis.set_trends(trends)
        summary = getattr(trends, "summary", lambda: "")()
        if summary:
            self.dashboard.append_log(f"skill model: {summary}")

    def _browser_play(self, name: str) -> None:
        self.space.scroll_to(self.space.index_of(self.dashboard))
        self.dashboard.play_scenario(name)

    def _browser_watch(self, name: str) -> None:
        self.space.scroll_to(self.space.index_of(self.dashboard))
        self.dashboard.watch_scenario(name)

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        self.backdrop.paint(p)
        p.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "backdrop"):
            self.backdrop.notify_resize()

    def _restyle(self, pal) -> None:
        for view in (self.dashboard, self.browser, self.analysis, self.optimizer):
            view.restyle(pal)
        self.nav.restyle(pal)
        self.backdrop.notify_theme()

    def _on_report(self, rep) -> None:
        self.analysis.show_report(rep, profile=self.dashboard.last_profile)
        self.optimizer.note_report(rep)
        idx = self.space.index_of(self.analysis)
        if self.space.current_index() != idx:   # unread only if not in view
            self.nav.set_badge(idx, True)

    def _section_changed(self, i: int) -> None:
        if i == self.space.index_of(self.analysis):
            self.nav.set_badge(i, False)        # scrolled into view: seen

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
    app.setWindowIcon(logo.make_icon())

    splash = logo.SplashScreen()   # the ASCII eye wakes up while we work
    splash.start()
    app.processEvents()
    win = MainWindow(settings, themes)

    from .boot import BootWorker

    boot = BootWorker(settings, parent=win)
    boot.status.connect(splash.set_status)
    boot.trends_ready.connect(win.set_trends)
    boot.start()

    def reveal() -> None:
        win.show()
        if not settings.onboarding_done:
            QTimer.singleShot(150, lambda: WelcomeDialog(settings, win).exec())

    splash.finish(reveal)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
