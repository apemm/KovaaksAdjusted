"""ASCII wavefront theme transition.

Switching themes sends a diagonal wave of monospace characters across the
window: ahead of the front the old theme still shows (a captured pixmap);
at the front a deep band of LED-bright glyphs churns — rainbow at the crest,
accent-tinted in the tail — and behind it the freshly restyled window shows
through. One overlay widget, deterministic per-cell seeds, self-destructs.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QRect, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from . import theme

_DURATION_MS = 720
_CELL_W = 11          # px; glyph cells of the wave grid
_CELL_H = 20
_BAND = 7.0           # wavefront depth, in cells
_GLYPHS = ".:*#@#*:."


def _seed(c: int, r: int) -> float:
    return (math.sin(c * 12.9898 + r * 78.233) * 43758.5453) % 1.0


class _Wave(QWidget):
    def __init__(self, parent: QWidget, old: QPixmap) -> None:
        super().__init__(parent)
        self._old = old
        self._t = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())
        self._cols = max(self.width() // _CELL_W + 2, 2)
        self._rows = max(self.height() // _CELL_H + 2, 2)
        # diagonal cell-distance of the far corner: front sweeps 0 -> this
        self._span = self._cols + self._rows * 0.65 + _BAND * 2
        self._font = QFont("Cascadia Mono")
        if not QFontMetricsF(self._font).horizontalAdvance("@"):
            self._font = QFont("Consolas")
        self._font.setPixelSize(_CELL_H - 3)
        self.raise_()
        self.show()

        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(_DURATION_MS)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.valueChanged.connect(self._step)
        anim.finished.connect(self.deleteLater)
        anim.start()

    def _step(self, t: float) -> None:
        self._t = float(t)
        self.update()

    def paintEvent(self, event) -> None:
        pal = theme.current()
        front = self._t * self._span - _BAND        # leading edge, in cells
        p = QPainter(self)
        p.setFont(self._font)

        # old theme still covers everything the front has not reached:
        # per row, the un-swept cells form one contiguous right-hand span
        for r in range(self._rows):
            first_unswept = front - r * 0.65        # d(c) = c + 0.65 r
            c0 = max(int(math.ceil(first_unswept)), 0)
            if c0 >= self._cols:
                continue
            x = c0 * _CELL_W
            y = r * _CELL_H
            src = QRect(x, y, self.width() - x, _CELL_H)
            p.drawPixmap(src, self._old, src)

        # the churning glyph band around the front
        sat, val = (0.65, 0.95) if not pal.is_dark else (0.6, 1.0)
        for r in range(self._rows):
            d_front = front - r * 0.65
            lo = max(int(d_front - _BAND), 0)
            hi = min(int(d_front + 1), self._cols - 1)
            for c in range(lo, hi + 1):
                depth = d_front - c                  # 0 at crest, grows in tail
                if depth < 0 or depth > _BAND:
                    continue
                s = _seed(c, r)
                k = 1.0 - depth / _BAND              # 1 at crest -> 0 at tail
                if k < s * 0.55:                     # ragged tail dissolve
                    continue
                if k > 0.72:                         # crest: rainbow LEDs
                    col = QColor.fromHsvF((s + depth * 0.13) % 1.0, sat, val)
                else:                                # tail: accent embers
                    col = QColor(pal.accent)
                    col.setAlphaF(0.25 + 0.75 * k)
                p.setPen(col)
                g = _GLYPHS[int(s * 7 + depth * 2.3) % len(_GLYPHS)]
                p.drawText(QRectF(c * _CELL_W, r * _CELL_H, _CELL_W * 2,
                                  _CELL_H * 1.2),
                           Qt.AlignLeft | Qt.AlignTop, g)


def ascii_wipe(window: QWidget) -> None:
    """Capture the window's current look and sweep the new one in behind a
    wave of glyphs. Call BEFORE restyling."""
    if not window.isVisible():
        return
    old = window.grab()
    old.setDevicePixelRatio(window.devicePixelRatioF())
    _Wave(window, old)
