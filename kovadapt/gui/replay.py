"""Trajectory replay: animated crosshair path from a MouseTrace window.

The full path is drawn dim; an animated bright segment sweeps through it in
real time (or scaled), with a marker at the current crosshair position and
click flashes. This is the free 'clip' every run gets even without video.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..telemetry.trace import MouseTrace
from .theme import ACCENT, BAD, FG_DIM


class TrajectoryReplay(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._t = np.empty(0)
        self._x = np.empty(0)
        self._y = np.empty(0)
        self._clicks = np.empty(0)
        self._pos = 0.0          # playhead (s from segment start)
        self._speed = 0.5        # default half speed: flicks are fast

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self._full = self.plot.plot([], [], pen=pg.mkPen(FG_DIM, width=1))
        self._live = self.plot.plot([], [], pen=pg.mkPen(ACCENT, width=2))
        self._head = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(ACCENT), pen=None)
        self._shots = pg.ScatterPlotItem(size=14, brush=None,
                                         pen=pg.mkPen(BAD, width=2), symbol="x")
        self.plot.addItem(self._head)
        self.plot.addItem(self._shots)

        self.btn = QPushButton("Replay")
        self.btn.clicked.connect(self.toggle)
        self.speed_btn = QPushButton("0.5x")
        self.speed_btn.clicked.connect(self._cycle_speed)
        self.info = QLabel("")
        self.info.setProperty("dim", True)

        bar = QHBoxLayout()
        bar.addWidget(self.btn)
        bar.addWidget(self.speed_btn)
        bar.addWidget(self.info)
        bar.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.plot, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    def load(self, trace: MouseTrace, t0: float | None = None,
             t1: float | None = None, label: str = "") -> None:
        """Show [t0, t1] of the trace (defaults: whole trace)."""
        self.stop()
        seg = trace if t0 is None else trace.window(t0, t1 if t1 is not None else trace.t[-1])
        t, x, y = seg.path()
        if t.size < 2:
            self._t = np.empty(0)
            self._full.setData([], [])
            self._live.setData([], [])
            self._head.setData([], [])
            self._shots.setData([], [])
            self.info.setText("no movement in this window")
            return
        self._t = t - t[0]
        self._x, self._y = x, y
        self._clicks = seg.clicks - t[0]
        self._full.setData(x, y)
        self._live.setData([], [])
        self._head.setData([x[0]], [y[0]])
        ci = np.searchsorted(t - t[0], self._clicks)
        ci = np.clip(ci, 0, t.size - 1)
        self._shots.setData(x[ci], y[ci])
        self.info.setText(label or f"{self._t[-1]:.2f}s, {seg.clicks.size} shots")
        self.plot.autoRange()

    # ------------------------------------------------------------------
    def toggle(self) -> None:
        if self._timer.isActive():
            self.stop()
        elif self._t.size:
            self._pos = 0.0
            self.btn.setText("Stop")
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.btn.setText("Replay")

    def _cycle_speed(self) -> None:
        order = [0.25, 0.5, 1.0]
        self._speed = order[(order.index(self._speed) + 1) % len(order)]
        self.speed_btn.setText(f"{self._speed:g}x")

    def _tick(self) -> None:
        self._pos += 0.016 * self._speed
        if self._pos >= self._t[-1]:
            self._pos = self._t[-1]
            self.stop()
        i = int(np.searchsorted(self._t, self._pos))
        i = max(min(i, self._t.size - 1), 1)
        self._live.setData(self._x[:i], self._y[:i])
        self._head.setData([self._x[i - 1]], [self._y[i - 1]])
