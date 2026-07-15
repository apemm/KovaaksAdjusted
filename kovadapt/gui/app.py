"""kovadapt GUI: dark, simple, four tabs.

    Dashboard    pick scenario, start/stop the adaptation loop, live log
    Analysis     post-run report: bias, heatmap, notable moments, replays/clips
    Adaptability full configuration surface
    Optimizer    free Process Lasso basics + tuning checklist

Run with `kovadapt gui` (requires: pip install kovadapt[gui]).
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from ..config import Settings
from .analysis_view import AnalysisView
from .config_view import ConfigView
from .dashboard import Dashboard
from .optimizer_view import OptimizerView
from .theme import apply_theme


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.setWindowTitle("kovadapt — adaptive KovaaK's")
        self.resize(1100, 720)

        tabs = QTabWidget()
        self.dashboard = Dashboard(settings)
        self.analysis = AnalysisView()
        self.config = ConfigView(settings)
        self.optimizer = OptimizerView(settings)
        tabs.addTab(self.dashboard, "Dashboard")
        tabs.addTab(self.analysis, "Analysis")
        tabs.addTab(self.config, "Adaptability")
        tabs.addTab(self.optimizer, "Optimizer")
        self.setCentralWidget(tabs)
        self._tabs = tabs

        # new run report -> refresh analysis tab and flag it
        self.dashboard.report_ready.connect(self._on_report)
        tabs.currentChanged.connect(self._clear_unread)

        sb = self.statusBar()
        sb.showMessage(
            f"KovaaK's: {settings.kovaaks_root or 'NOT FOUND — set KOVAAKS_ROOT'}"
        )

    def _on_report(self, rep) -> None:
        self.analysis.show_report(rep)
        idx = self._tabs.indexOf(self.analysis)
        self._tabs.setTabText(idx, "Analysis •")

    def _clear_unread(self, i: int) -> None:
        idx = self._tabs.indexOf(self.analysis)
        if i == idx:
            self._tabs.setTabText(idx, "Analysis")

    def closeEvent(self, event) -> None:  # stop watcher thread cleanly
        w = self.dashboard.worker
        if w is not None:
            w.stop()
            w.wait(3000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow(Settings.load())
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
