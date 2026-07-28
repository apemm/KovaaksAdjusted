"""Iris-wipe theme transition: the old look is held as a pixmap over the
window and a growing circle — the eye dilating — reveals the new theme
beneath. One overlay widget, one animation, self-destructs on finish."""

from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QPoint, QRect, QVariantAnimation, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QWidget

_DURATION_MS = 340


class _Wipe(QWidget):
    def __init__(self, parent: QWidget, old: QPixmap, center: QPoint) -> None:
        super().__init__(parent)
        self._old = old
        self._center = center
        self._radius = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())
        w, h = parent.width(), parent.height()
        self._max_radius = max(
            math.hypot(center.x() - x, center.y() - y)
            for x in (0, w) for y in (0, h))
        self.raise_()
        self.show()

        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(_DURATION_MS)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.valueChanged.connect(self._step)
        anim.finished.connect(self.deleteLater)
        anim.start()

    def _step(self, t: float) -> None:
        self._radius = self._max_radius * float(t)
        self.update()

    def paintEvent(self, event) -> None:
        # everything EXCEPT the growing circle still shows the old theme
        p = QPainter(self)
        hole = QPainterPath()
        hole.addEllipse(self._center, self._radius, self._radius)
        full = QPainterPath()
        full.addRect(QRect(0, 0, self.width(), self.height()))
        p.setClipPath(full.subtracted(hole))
        p.drawPixmap(0, 0, self._old)


def iris_wipe(window: QWidget, center: QPoint) -> None:
    """Capture the window's current look and reveal the next one through a
    growing circle at `center` (window coordinates). Call BEFORE restyling."""
    if not window.isVisible():
        return
    ratio = window.devicePixelRatioF()
    old = window.grab()
    old.setDevicePixelRatio(ratio)
    _Wipe(window, old, center)
