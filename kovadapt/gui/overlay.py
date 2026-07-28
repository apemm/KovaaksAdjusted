"""In-game performance overlay: frameless, translucent, always-on-top.

A compact card that floats over the game showing live session state: last
run vs your baseline, session run count, fatigue, current difficulty, input
health, and an accuracy sparkline. Click-through by default (pure Qt:
WindowTransparentForInput — no hooks, no injection, invisible to the game);
Unlock mode disables that so the card can be dragged, and the position
persists in settings.

KovaaK's must run Borderless or Windowed for any overlay to be visible —
true exclusive fullscreen bypasses the compositor. The Dashboard hint says
so next to the toggle.

Painting stays featherweight: one rounded-rect card + QLabels + a QPainter
sparkline (pyqtgraph deliberately not used here — the overlay redraws while
the game runs).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..config import Settings
from . import theme

_CARD_WIDTH = 250
_SPARK_RUNS = 40
_MARGIN = 24            # default distance from the screen's top-right corner

_BASE_FLAGS = (
    Qt.FramelessWindowHint
    | Qt.WindowStaysOnTopHint
    | Qt.Tool
    | Qt.WindowDoesNotAcceptFocus
)


class _Sparkline(QWidget):
    """Accuracy-per-run mini chart; stores plain floats, paints a polyline."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.values: list[float] = []
        self.setFixedHeight(34)

    def set_values(self, values: list[float]) -> None:
        self.values = list(values)[-_SPARK_RUNS:]
        self.update()

    def paintEvent(self, event) -> None:
        if len(self.values) < 2:
            return
        pal = theme.current()
        w, h = self.width(), self.height()
        lo, hi = min(self.values), max(self.values)
        span = (hi - lo) or 1.0
        pts = [
            QPoint(
                int(i * (w - 4) / (len(self.values) - 1)) + 2,
                int((h - 6) * (1.0 - (v - lo) / span)) + 3,
            )
            for i, v in enumerate(self.values)
        ]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(pal.accent), 2))
        p.drawPolyline(pts)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal.accent))
        p.drawEllipse(pts[-1], 3, 3)


class OverlayWindow(QWidget):
    """Top-level overlay card (create with no parent)."""

    def __init__(self, settings: Settings) -> None:
        flags = _BASE_FLAGS
        if settings.overlay_clickthrough:
            flags |= Qt.WindowTransparentForInput
        super().__init__(None, flags)
        self.s = settings
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("background: transparent;")
        self.setFixedWidth(_CARD_WIDTH)
        self.setWindowOpacity(settings.overlay_opacity)
        self._unlocked = False
        self._drag_from: QPoint | None = None
        self._session_runs = 0
        self._accs: list[float] = []

        self.title = QLabel("kovadapt")
        self.scenario = QLabel("")
        self.scenario.setWordWrap(True)
        self.status = QLabel("not watching")
        self.hint = QLabel("drag to move · lock from the Dashboard")
        self.hint.hide()

        self._stats: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setVerticalSpacing(2)
        for i, (key, cap) in enumerate([
            ("last", "Last run"), ("session", "Session"),
            ("diff", "Difficulty"), ("fatigue", "Fatigue"),
            ("input", "Input"),
        ]):
            capl = QLabel(cap)
            val = QLabel("—")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(capl, i, 0)
            grid.addWidget(val, i, 1)
            self._stats[key] = val
            setattr(self, f"_cap_{key}", capl)

        self.spark = _Sparkline()

        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(2)
        lay.addLayout(head)
        lay.addWidget(self.scenario)
        lay.addLayout(grid)
        lay.addWidget(self.spark)
        lay.addWidget(self.hint)
        self.restyle()

    # ----------------------------------------------------------------- theme
    def restyle(self, *_pal) -> None:
        pal = theme.current()
        f = "font-size: 12px; background: transparent;"
        self.title.setStyleSheet(f"{f} font-weight: 700; color: {pal.accent};")
        self.scenario.setStyleSheet(f"{f} color: {pal.fg};")
        self.status.setStyleSheet(f"{f} color: {pal.fg_dim};")
        self.hint.setStyleSheet(f"font-size: 11px; color: {pal.warn}; background: transparent;")
        for key, val in self._stats.items():
            val.setStyleSheet(f"{f} color: {pal.fg}; font-weight: 600;")
            getattr(self, f"_cap_{key}").setStyleSheet(f"{f} color: {pal.fg_dim};")
        self.update()

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QColor(pal.bg)
        bg.setAlpha(235 if pal.is_dark else 245)
        p.setBrush(bg)
        border = QColor(pal.warn if self._unlocked else pal.border)
        pen = QPen(border, 2 if self._unlocked else 1)
        if self._unlocked:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)

    # ------------------------------------------------------------ visibility
    def show_overlay(self) -> None:
        self._place()
        self.setWindowOpacity(self.s.overlay_opacity)
        self.show()

    def _place(self) -> None:
        # (-1, -1) is the only "never dragged" sentinel — single coordinates
        # are legitimately negative on monitors left of/above the primary.
        if (self.s.overlay_x, self.s.overlay_y) != (-1, -1):
            self.move(self.s.overlay_x, self.s.overlay_y)
            return
        screen = self.screen() or None
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.adjustSize()
        self.move(geo.right() - self.width() - _MARGIN, geo.top() + _MARGIN)

    def set_unlocked(self, unlocked: bool) -> None:
        """Unlocked = draggable (input NOT transparent); locked = click-through."""
        if unlocked == self._unlocked:
            return
        self._unlocked = unlocked
        visible = self.isVisible()
        flags = _BASE_FLAGS
        if not unlocked and self.s.overlay_clickthrough:
            flags |= Qt.WindowTransparentForInput
        pos = self.pos()
        self.setWindowFlags(flags)   # re-creates the native window
        self.move(pos)
        self.hint.setVisible(unlocked)
        if visible:
            self.show()
        self.update()

    def set_opacity(self, value: float) -> None:
        self.s.overlay_opacity = max(0.3, min(1.0, value))
        self.setWindowOpacity(self.s.overlay_opacity)

    # -------------------------------------------------------------- dragging
    def mousePressEvent(self, event) -> None:
        if self._unlocked and event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._unlocked and self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event) -> None:
        if self._unlocked and self._drag_from is not None:
            self._drag_from = None
            self.s.overlay_x, self.s.overlay_y = self.pos().x(), self.pos().y()
            try:
                self.s.save()
            except OSError:
                pass

    # ------------------------------------------------------------------ data
    def start_session(self, scenario: str) -> None:
        self._session_runs = 0
        self._accs = []
        self.scenario.setText(scenario)
        self.status.setText("watching")
        self.spark.set_values([])
        for val in self._stats.values():
            val.setText("—")

    def stop_session(self) -> None:
        self.status.setText("not watching")

    def on_report(self, rep, profile=None) -> None:
        """New RunReport from the watcher (and the freshly saved profile)."""
        self._session_runs += 1
        self._accs.append(float(rep.accuracy))
        self.spark.set_values(self._accs)
        pal = theme.current()

        base = getattr(profile, "ewma_accuracy", 0.0) if profile is not None else 0.0
        delta = rep.accuracy - base
        arrow = "▲" if delta >= 0.005 else ("▼" if delta <= -0.005 else "·")
        color = pal.good if delta >= 0.005 else (pal.bad if delta <= -0.005 else pal.fg)
        self._stats["last"].setText(
            f"<span style='color:{color}'>{rep.accuracy:.0%} {arrow}</span> "
            f"<span style='color:{pal.fg_dim}'>{rep.score:.0f}</span>")
        self._stats["session"].setText(f"{self._session_runs} runs")

        if profile is not None:
            self._stats["diff"].setText(
                f"{profile.target_scale:.2f}x · mov {profile.movement:.2f}")

        fat = rep.fatigue or {}
        level = fat.get("level", "")
        if level:
            fcol = {"fresh": pal.good, "declining": pal.warn}.get(level, pal.bad)
            self._stats["fatigue"].setText(
                f"<span style='color:{fcol}'>{level}</span>")

        ih = rep.input_health or {}
        if ih.get("polling_hz_est"):
            jit = ih.get("jitter_ms", 0.0)
            jcol = pal.good if jit <= 1.0 else pal.warn
            self._stats["input"].setText(
                f"{ih['polling_hz_est']:.0f}Hz "
                f"<span style='color:{jcol}'>±{jit:.1f}ms</span>")
