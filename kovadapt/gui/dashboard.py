"""Dashboard tab: pick a scenario, start/stop the adaptation loop, live log."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from .theme import ACCENT
from .workers import WatcherWorker


class Dashboard(QWidget):
    report_ready = Signal(object)   # re-emitted RunReport for the analysis tab

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.worker: WatcherWorker | None = None

        # scenario picker + controls
        self.scenario = QComboBox()
        self.scenario.setEditable(True)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_scenarios)
        self.start_btn = QPushButton("Start adapting")
        self.start_btn.setProperty("accent", True)
        self.start_btn.clicked.connect(self.toggle)

        top = QHBoxLayout()
        top.addWidget(QLabel("Scenario:"))
        top.addWidget(self.scenario, 1)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.start_btn)

        # live profile stats
        self.stat_labels: dict[str, QLabel] = {}
        stats_box = QGroupBox("Learned profile")
        grid = QGridLayout(stats_box)
        for i, (key, cap) in enumerate([
            ("runs", "Runs"), ("accuracy", "Accuracy EWMA"),
            ("scale", "Target scale"), ("movement", "Movement"),
            ("focus", "Focus region"), ("pace", "Pace (kills/s)"),
            ("archetype", "Archetype"), ("fatigue", "Session fatigue"),
        ]):
            grid.addWidget(QLabel(cap), i // 3, (i % 3) * 2)
            lab = QLabel("—")
            lab.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
            grid.addWidget(lab, i // 3, (i % 3) * 2 + 1)
            self.stat_labels[key] = lab

        # calibration readiness: how much baseline data adaptation still wants
        self.readiness = QProgressBar()
        self.readiness.setRange(0, 100)
        self.readiness.setValue(0)
        self.readiness.setTextVisible(True)
        self.readiness.setFormat("calibration %p%")
        self.readiness_msg = QLabel("play runs to calibrate the adaptive model")
        self.readiness_msg.setProperty("dim", True)
        self.readiness_msg.setWordWrap(True)
        grid.addWidget(self.readiness, 3, 0, 1, 3)
        grid.addWidget(self.readiness_msg, 3, 3, 1, 3)

        # accuracy history sparkline
        self.trend = pg.PlotWidget(title="Accuracy per run")
        self.trend.setMaximumHeight(160)
        self.trend.showGrid(y=True, alpha=0.2)
        self.trend_curve = self.trend.plot([], [], pen=pg.mkPen(ACCENT, width=2),
                                           symbol="o", symbolSize=5,
                                           symbolBrush=ACCENT, symbolPen=None)

        # log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(stats_box)
        lay.addWidget(self.trend)
        lay.addWidget(self.log, 1)

        self.refresh_scenarios()

    # ------------------------------------------------------------------
    def refresh_scenarios(self) -> None:
        cur = self.scenario.currentText()
        self.scenario.clear()
        if self.s.scenarios_dir.is_dir():
            names = sorted(
                p.stem for p in self.s.scenarios_dir.glob("*.sce")
                if not p.stem.endswith(ADAPTIVE_SUFFIX)
            )
            self.scenario.addItems(names)
        if cur:
            self.scenario.setCurrentText(cur)

    def toggle(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Stopping…")
            return
        name = self.scenario.currentText().strip()
        if not name:
            self.append_log("pick a scenario first")
            return
        self.worker = WatcherWorker(self.s, name, parent=self)
        self.worker.message.connect(self.append_log)
        self.worker.report_ready.connect(self._on_report)
        self.worker.stopped.connect(self._on_stopped)
        self.worker.start()
        self.start_btn.setText("Stop")
        self.scenario.setEnabled(False)
        self.refresh_profile(name)

    def _on_stopped(self) -> None:
        self.worker = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start adapting")
        self.scenario.setEnabled(True)
        self.append_log("stopped")

    def _on_report(self, rep) -> None:
        self.refresh_profile(self.scenario.currentText().strip())
        fat = getattr(rep, "fatigue", None) or {}
        if fat.get("runs", 0) > 0:
            self.stat_labels["fatigue"].setText(
                f"{fat.get('level', 'fresh')} ({fat.get('score', 0.0):.0%})"
            )
        self.report_ready.emit(rep)

    # ------------------------------------------------------------------
    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def refresh_profile(self, base_name: str) -> None:
        prof = PlayerProfile.load(base_name + ADAPTIVE_SUFFIX, self.s.profile_path)
        sl = self.stat_labels
        ready = prof.readiness(self.s.region_cols * self.s.region_rows)
        self.readiness.setValue(int(ready["score"] * 100))
        self.readiness_msg.setText(ready["message"])
        if prof.run_count == 0:
            for lab in sl.values():
                lab.setText("—")
            self.trend_curve.setData([], [])
            return
        sl["runs"].setText(str(prof.run_count))
        sl["accuracy"].setText(f"{prof.ewma_accuracy:.1%}")
        sl["scale"].setText(f"{prof.target_scale:.2f}x")
        sl["movement"].setText(f"{prof.movement:.2f}")
        sl["focus"].setText(prof.last_focus or "—")
        sl["pace"].setText(f"{prof.ewma_kps:.2f}")
        sl["archetype"].setText(prof.archetype or "—")
        accs = [h.get("accuracy", 0.0) for h in prof.history[-60:]]
        self.trend_curve.setData(list(range(len(accs))), accs)
