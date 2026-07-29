"""Trajectory replay: animated crosshair path with flick-quality overlays.

Visual language (all derived from the recorded MouseTrace — no video):

    faint grey line   full crosshair path of the window (deliberately subtle)
    green segments    clean flicks (low overshoot, <= 1 correction)
    red segments      flawed flicks (overshoot > 10% or >= 2 corrections)
    red ✕             shots (left clicks)
    bright dot+trail  playhead sweeping in (scaled) real time

The path/flicks/shots checkboxes in the control bar hide layers without
touching the item architecture — they only flip setVisible on the items.

Lightweight by construction: the overlays are exactly two PlotCurveItems
regardless of flick count (NaN-separated segments), the path is decimated
above ~50k points, and the QTimer only runs during playback. Playback time
comes from QElapsedTimer, so speed is wall-clock accurate under load.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..telemetry.trace import MouseTrace
from . import theme

_MAX_POINTS = 50_000          # decimation cap for the drawn path
_TRAIL_SECONDS = 1.2          # live comet-trail length (full path stays dim below)
_SLIDER_STEPS = 1000
# Flick quality thresholds (match analysis conventions: overshoot_rate uses
# 0.1, notable "clean" uses <= 1 correction).
_FLAWED_OVERSHOOT = 0.10
_FLAWED_CORRECTIONS = 2


class TrajectoryReplay(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._t = np.empty(0)
        self._x = np.empty(0)
        self._y = np.empty(0)
        self._pos = 0.0          # playhead (s from segment start)
        self._speed = 0.5        # default half speed: flicks are fast
        self._clock = QElapsedTimer()
        self._clock_base = 0.0   # _pos when the clock (re)started

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        # chrome-min: the canvas is pure trajectory — no context menu, no
        # autorange button (the surrounding ASCII viz has no chrome either)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self._full = self.plot.plot([], [])
        self._good = self.plot.plot([], [], connect="finite")
        self._bad = self.plot.plot([], [], connect="finite")
        self._live = self.plot.plot([], [])
        self._head = pg.ScatterPlotItem(size=10, pen=None)
        self._shots = pg.ScatterPlotItem(size=14, brush=None, symbol="x")
        self.plot.addItem(self._head)
        self.plot.addItem(self._shots)

        self.btn = QPushButton("Replay")
        self.btn.clicked.connect(self.toggle)
        self.speed_btn = QPushButton("0.5x")
        self.speed_btn.clicked.connect(self._cycle_speed)
        # layer toggles: hide/show existing items, never restructure them
        self.toggle_path = QCheckBox("path")
        self.toggle_path.setToolTip("Show the faint full crosshair path of the window")
        self.toggle_flicks = QCheckBox("flicks")
        self.toggle_flicks.setToolTip(
            "Show flick-quality overlays: green = clean, red = overshoot/correction")
        self.toggle_shots = QCheckBox("shots")
        self.toggle_shots.setToolTip("Show an ✕ where each shot was fired")
        for box in (self.toggle_path, self.toggle_flicks, self.toggle_shots):
            box.setChecked(True)
        self.toggle_path.toggled.connect(self._full.setVisible)
        self.toggle_flicks.toggled.connect(self._set_flicks_visible)
        self.toggle_shots.toggled.connect(self._shots.setVisible)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, _SLIDER_STEPS)
        self.scrub.sliderMoved.connect(self._scrubbed)
        self.info = QLabel("")
        self.info.setProperty("dim", True)
        self.legend = QLabel("")
        self.legend.setTextFormat(Qt.RichText)
        self.legend.setProperty("dim", True)

        bar = QHBoxLayout()
        bar.addWidget(self.btn)
        bar.addWidget(self.speed_btn)
        bar.addWidget(self.toggle_path)
        bar.addWidget(self.toggle_flicks)
        bar.addWidget(self.toggle_shots)
        bar.addWidget(self.scrub, 1)
        bar.addWidget(self.info)
        sub = QHBoxLayout()
        sub.addWidget(self.legend)
        sub.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addLayout(sub)
        lay.addWidget(self.plot, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)
        self.restyle()

    # ------------------------------------------------------------------
    def restyle(self, *_pal) -> None:
        """Re-pen every curve from the active palette (called on theme switch)."""
        pal = theme.current()
        self.plot.setBackground(pal.bg_alt)
        # ~40% alpha: the full path is context, not the story — the comet
        # trail and flick overlays must read on top of it
        full_pen = pg.mkColor(pal.fg_dim)
        full_pen.setAlphaF(0.4)
        self._full.setPen(pg.mkPen(full_pen, width=1))
        self._good.setPen(pg.mkPen(pal.good, width=2))
        self._bad.setPen(pg.mkPen(pal.bad, width=2))
        self._live.setPen(pg.mkPen(pal.accent, width=2))
        self._head.setBrush(pg.mkBrush(pal.accent))
        self._shots.setPen(pg.mkPen(pal.bad, width=2))
        self.legend.setText(
            f"<span style='color:{pal.good}'>—</span> clean flick&nbsp;&nbsp;"
            f"<span style='color:{pal.bad}'>—</span> overshoot/correction&nbsp;&nbsp;"
            f"<span style='color:{pal.bad}'>✕</span> shot")

    # ------------------------------------------------------------------
    def _set_flicks_visible(self, on: bool) -> None:
        """One toggle drives both flick overlays (they are a single layer)."""
        self._good.setVisible(on)
        self._bad.setVisible(on)

    # ------------------------------------------------------------------
    def load(self, trace: MouseTrace, t0: float | None = None,
             t1: float | None = None, label: str = "",
             flicks: list | None = None) -> None:
        """Show [t0, t1] of the trace (defaults: whole trace). `flicks` are
        analysis.movement.Flick objects for the SAME trace (absolute epoch
        times); the ones inside the window become quality overlays."""
        self.stop()
        seg = trace if t0 is None else trace.window(t0, t1 if t1 is not None else trace.t[-1])
        t, x, y = seg.path()
        if t.size < 2:
            self._t = np.empty(0)
            for item in (self._full, self._good, self._bad, self._live):
                item.setData([], [])
            self._head.setData([], [])
            self._shots.setData([], [])
            self.info.setText("no movement in this window")
            return
        # decimate for drawing: replay is an indicator, not a data export
        stride = max(1, t.size // _MAX_POINTS)
        t, x, y = t[::stride], x[::stride], y[::stride]
        base = t[0]
        self._t = t - base
        self._x, self._y = x, y
        self._full.setData(x, y)
        self._live.setData([], [])
        self._head.setData([x[0]], [y[0]])
        clicks = seg.clicks - base
        ci = np.clip(np.searchsorted(self._t, clicks), 0, t.size - 1)
        self._shots.setData(x[ci], y[ci])
        self._draw_flicks(base, flicks or [])
        self._pos = 0.0
        self.scrub.setValue(0)
        self.info.setText(label or f"{self._t[-1]:.2f}s · {seg.clicks.size} shots")
        self.plot.autoRange()

    def _draw_flicks(self, base: float, flicks: list) -> None:
        """Two NaN-separated polylines: clean (green) and flawed (red)."""
        good: list[list[float]] = [[], []]
        bad: list[list[float]] = [[], []]
        tmax = self._t[-1] if self._t.size else 0.0
        for f in flicks:
            o, c = f.t_onset - base, f.t_click - base
            if c <= 0 or o >= tmax or c <= o:
                continue
            i0, i1 = np.searchsorted(self._t, [o, c])
            i1 = min(int(i1) + 1, self._t.size)
            if i1 - i0 < 2:
                continue
            flawed = (f.overshoot > _FLAWED_OVERSHOOT
                      or f.corrections >= _FLAWED_CORRECTIONS)
            dest = bad if flawed else good
            dest[0].extend(self._x[i0:i1].tolist() + [np.nan])
            dest[1].extend(self._y[i0:i1].tolist() + [np.nan])
        self._good.setData(good[0], good[1])
        self._bad.setData(bad[0], bad[1])

    # ------------------------------------------------------------------
    def clear(self, message: str = "no trace for this run") -> None:
        """Empty the plot (used when a report arrives without telemetry, so
        the previous run's path can't masquerade as the current one)."""
        self.stop()
        self._t = np.empty(0)
        for item in (self._full, self._good, self._bad, self._live):
            item.setData([], [])
        self._head.setData([], [])
        self._shots.setData([], [])
        self.scrub.setValue(0)
        self.info.setText(message)

    # ------------------------------------------------------------------
    def toggle(self) -> None:
        if self._timer.isActive():
            self.stop()
        elif self._t.size:
            if self._pos >= self._t[-1]:
                self._pos = 0.0
            self._clock_base = self._pos
            self._clock.start()
            self.btn.setText("Stop")
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.btn.setText("Replay")

    def _cycle_speed(self) -> None:
        order = [0.25, 0.5, 1.0]
        self._speed = order[(order.index(self._speed) + 1) % len(order)]
        self.speed_btn.setText(f"{self._speed:g}x")
        if self._timer.isActive():           # rebase so speed changes mid-play
            self._clock_base = self._pos
            self._clock.restart()

    def _scrubbed(self, v: int) -> None:
        if not self._t.size:
            return
        self.stop()
        self._pos = self._t[-1] * v / _SLIDER_STEPS
        self._render()

    def _tick(self) -> None:
        if not self._t.size:
            self.stop()
            return
        self._pos = self._clock_base + self._clock.elapsed() / 1000.0 * self._speed
        if self._pos >= self._t[-1]:
            self._pos = self._t[-1]
            self.stop()
        self.scrub.blockSignals(True)
        self.scrub.setValue(int(self._pos / self._t[-1] * _SLIDER_STEPS))
        self.scrub.blockSignals(False)
        self._render()

    def _render(self) -> None:
        i = int(np.searchsorted(self._t, self._pos))
        i = max(min(i, self._t.size - 1), 1)
        # Comet trail, not the whole prefix: repainting an ever-growing
        # antialiased path each 16 ms tick stalls multi-minute replays
        # (the dim full path is already drawn once underneath).
        j = int(np.searchsorted(self._t, self._t[i - 1] - _TRAIL_SECONDS))
        self._live.setData(self._x[j:i], self._y[j:i])
        self._head.setData([self._x[i - 1]], [self._y[i - 1]])
