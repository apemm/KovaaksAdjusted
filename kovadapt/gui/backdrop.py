"""Parallax ASCII backdrop: the eye watches from behind the UI.

Three pre-rendered layers — a big faint ASCII eye off-center, and two depths
of drifting glyph dust — slide at different rates toward the cursor (plus a
slow ambient wander), so the whole app has depth without costing anything:
layers are QPixmaps rendered once per theme/resize, and each frame is three
blits. The cursor is polled at 30 Hz and offsets are eased, so motion stays
floaty, never twitchy.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetricsF, QPainter, QPixmap

from . import ascii_art, theme

_GLYPHS = ".:*+#@"


def _seed(i: int, salt: float) -> float:
    return (math.sin(i * 12.9898 + salt * 78.233) * 43758.5453) % 1.0


def _dust_pixmap(w: int, h: int, count: int, alpha: float, salt: float,
                 accent_share: float) -> QPixmap:
    pal = theme.current()
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    font = QFont("Cascadia Mono")
    if not QFontMetricsF(font).horizontalAdvance("@"):
        font = QFont("Consolas")
    for i in range(count):
        x = _seed(i, salt) * w
        y = _seed(i, salt + 1.0) * h
        s = _seed(i, salt + 2.0)
        font.setPixelSize(10 + int(s * 10))
        p.setFont(font)
        if s < accent_share:
            col = QColor(pal.accent)
        else:
            col = QColor(pal.fg_dim)
        col.setAlphaF(alpha * (0.4 + 0.6 * _seed(i, salt + 3.0)))
        p.setPen(col)
        p.drawText(QPointF(x, y), _GLYPHS[int(s * len(_GLYPHS)) % len(_GLYPHS)])
    p.end()
    return pm


class Backdrop:
    """Owns the layers + parallax state for one window. The window calls
    paint(painter) from its paintEvent and notify_theme()/notify_resize()
    when those change."""

    def __init__(self, window) -> None:
        self._win = window
        self._eye: QPixmap | None = None
        self._dust_far: QPixmap | None = None
        self._dust_near: QPixmap | None = None
        self._off = [QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)]
        self._phase = 0.0
        self._timer = QTimer(window)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._eye is not None:
            return
        w = max(self._win.width(), 640)
        h = max(self._win.height(), 480)
        self._eye = ascii_art.render_pixmap(int(w * 0.62))
        self._dust_far = _dust_pixmap(w + 80, h + 80, count=90, alpha=0.05,
                                      salt=3.7, accent_share=0.12)
        self._dust_near = _dust_pixmap(w + 120, h + 120, count=40, alpha=0.08,
                                       salt=9.1, accent_share=0.25)

    def notify_theme(self) -> None:
        self._eye = None            # re-render on next paint
        self._win.update()

    def notify_resize(self) -> None:
        self._eye = None

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if not self._win.isVisible():
            return
        self._phase += 0.033
        pos = self._win.mapFromGlobal(QCursor.pos())
        w, h = max(self._win.width(), 1), max(self._win.height(), 1)
        # -0.5..0.5 relative cursor position, clamped when outside the window
        rx = max(-0.5, min(0.5, pos.x() / w - 0.5))
        ry = max(-0.5, min(0.5, pos.y() / h - 0.5))
        drift_x = math.sin(self._phase * 0.11) * 6.0
        drift_y = math.cos(self._phase * 0.07) * 5.0
        targets = (
            QPointF(-rx * 18 + drift_x, -ry * 12 + drift_y),        # eye
            QPointF(rx * 10 + drift_x * 0.4, ry * 8 + drift_y * 0.4),   # far dust
            QPointF(-rx * 34 - drift_x, -ry * 26 - drift_y),        # near dust
        )
        moved = False
        for i, tgt in enumerate(targets):
            cur = self._off[i]
            nxt = QPointF(cur.x() + (tgt.x() - cur.x()) * 0.08,
                          cur.y() + (tgt.y() - cur.y()) * 0.08)
            if abs(nxt.x() - cur.x()) + abs(nxt.y() - cur.y()) > 0.05:
                moved = True
            self._off[i] = nxt
        if moved:
            self._win.update()

    # ------------------------------------------------------------------
    def paint(self, p: QPainter) -> None:
        self._ensure()
        pal = theme.current()
        w, h = self._win.width(), self._win.height()

        p.setOpacity(1.0)
        p.drawPixmap(QPointF(-40 + self._off[1].x(), -40 + self._off[1].y()),
                     self._dust_far)

        # the eye sits right-of-center, faint enough to read text over
        p.setOpacity(0.16 if pal.is_dark else 0.12)
        ex = w * 0.50 + self._off[0].x()
        ey = h * 0.22 + self._off[0].y()
        p.drawPixmap(QPointF(ex, ey), self._eye)
        p.setOpacity(1.0)

        p.drawPixmap(QPointF(-60 + self._off[2].x(), -60 + self._off[2].y()),
                     self._dust_near)
        # a soft vignette keeps content edges readable over the art
        p.setOpacity(0.0)
        _ = QRectF(0, 0, w, h)
