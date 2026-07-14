"""Post-run analysis tab: summary, bias, heatmap, notable moments with
trajectory replays and (when captured) video clips — side by side."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..analysis.movement import movement_heatmap, segment_flicks
from ..analysis.report import RunReport
from ..telemetry.trace import MouseTrace
from .replay import TrajectoryReplay
from .theme import ACCENT, BAD, GOOD

_KIND_COLOR = {"overshoot": BAD, "hesitation": BAD, "slow_flick": "#d9a44f",
               "clean_flick": GOOD}


class AnalysisView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.report: RunReport | None = None
        self.trace: MouseTrace | None = None

        # header
        self.title = QLabel("No run analyzed yet")
        self.title.setProperty("headline", True)
        self.summary = QLabel("Finish a run while watching (or open a saved report).")
        self.summary.setWordWrap(True)
        open_btn = QPushButton("Open report…")
        open_btn.clicked.connect(self._open_dialog)

        head = QHBoxLayout()
        head_col = QVBoxLayout()
        head_col.addWidget(self.title)
        head_col.addWidget(self.summary)
        head.addLayout(head_col, 1)
        head.addWidget(open_btn, 0, Qt.AlignTop)

        # left column: bias bars + movement heatmap
        self.bias_plot = pg.PlotWidget(title="Flick quality by direction (lower = better)")
        self.bias_plot.setMaximumHeight(180)
        self.heat_plot = pg.PlotWidget(title="Aim travel around engagements")
        self.heat_plot.setAspectLocked(True)
        self.heat_img = pg.ImageItem()
        self.heat_plot.addItem(self.heat_img)
        cmap = pg.colormap.get("inferno")
        self.heat_img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        left = QVBoxLayout()
        left.addWidget(self.bias_plot)
        left.addWidget(self.heat_plot, 1)
        left_w = QWidget()
        left_w.setLayout(left)

        # right column: notable moments + replay
        self.moments = QListWidget()
        self.moments.currentRowChanged.connect(self._select_moment)
        self.clip_btn = QPushButton("Play video clip")
        self.clip_btn.setEnabled(False)
        self.clip_btn.clicked.connect(self._play_clip)
        self.replay = TrajectoryReplay()

        mo_box = QGroupBox("Notable moments")
        mo_lay = QVBoxLayout(mo_box)
        mo_lay.addWidget(self.moments, 1)
        mo_lay.addWidget(self.clip_btn)
        right = QSplitter(Qt.Vertical)
        right.addWidget(mo_box)
        rep_box = QGroupBox("Trajectory replay")
        rep_lay = QVBoxLayout(rep_box)
        rep_lay.addWidget(self.replay)
        right.addWidget(rep_box)
        right.setSizes([240, 400])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_w)
        split.addWidget(right)
        split.setSizes([460, 460])

        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(split, 1)

    # ------------------------------------------------------------------
    def show_report(self, rep: RunReport, trace: MouseTrace | None = None) -> None:
        self.report = rep
        self.trace = trace
        if trace is None and rep.trace_file and Path(rep.trace_file).is_file():
            self.trace = MouseTrace.load(rep.trace_file)

        self.title.setText(f"{rep.scenario} — {rep.started_iso.replace('T', ' ')[:19]}")
        self.summary.setText(rep.summary_text)
        self._draw_bias(rep)
        self._draw_heat()
        self._fill_moments(rep)
        if self.trace is not None and len(self.trace) > 1:
            self.replay.load(self.trace, label="full run")

    def load_report_file(self, path: Path | str) -> None:
        self.show_report(RunReport.load(path))

    # ------------------------------------------------------------------
    def _draw_bias(self, rep: RunReport) -> None:
        self.bias_plot.clear()
        b = rep.bias or {}
        dirs = ["left", "vertical", "right"]
        vals = [
            (b.get(d) or {}).get("overshoot", 0.0)
            + 0.15 * (b.get(d) or {}).get("corrections", 0.0)
            for d in dirs
        ]
        colors = [BAD if v == max(vals) and v > 0 else ACCENT for v in vals]
        bars = pg.BarGraphItem(x=list(range(3)), height=vals, width=0.6,
                               brushes=colors, pens=[None] * 3)
        self.bias_plot.addItem(bars)
        ax = self.bias_plot.getAxis("bottom")
        ns = [(b.get(d) or {}).get("n", 0) for d in dirs]
        ax.setTicks([[(i, f"{d}\n({n} flicks)") for i, (d, n) in enumerate(zip(dirs, ns))]])

    def _draw_heat(self) -> None:
        if self.trace is None or len(self.trace) < 2:
            self.heat_img.clear()
            return
        heat, xe, ye = movement_heatmap(self.trace)
        img = np.log1p(heat.T)  # transpose: histogram2d x/y -> row-major image
        self.heat_img.setImage(img, autoLevels=True)
        self.heat_img.setRect(float(xe[0]), float(ye[0]),
                              float(xe[-1] - xe[0]), float(ye[-1] - ye[0]))

    def _fill_moments(self, rep: RunReport) -> None:
        self.moments.blockSignals(True)
        self.moments.clear()
        for i, m in enumerate(rep.notable):
            it = QListWidgetItem(m["text"])
            it.setForeground(pg.mkColor(_KIND_COLOR.get(m["kind"], ACCENT)))
            it.setData(Qt.UserRole, i)
            self.moments.addItem(it)
        self.moments.blockSignals(False)
        if rep.notable:
            self.moments.setCurrentRow(0)

    def _select_moment(self, row: int) -> None:
        if self.report is None or row < 0 or row >= len(self.report.notable):
            self.clip_btn.setEnabled(False)
            return
        m = self.report.notable[row]
        self.clip_btn.setEnabled(str(row) in (self.report.clip_files or {}))
        if self.trace is not None and len(self.trace) > 1:
            self.replay.load(self.trace, m["t_start"], m["t_end"],
                             label=m["kind"].replace("_", " "))

    def _play_clip(self) -> None:
        row = self.moments.currentRow()
        if self.report is None or row < 0:
            return
        p = (self.report.clip_files or {}).get(str(row))
        if p and Path(p).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p))))

    def _open_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open run report",
                                           str(Path.home() / ".kovadapt" / "reports"),
                                           "Run reports (*.json)")
        if p:
            self.load_report_file(p)
