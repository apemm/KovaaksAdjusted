"""The kovadapt mark: a geometric eye whose iris is a rainbow crosshair.

One parametric painter renders every form of the mark — the animated splash
("the eye wakes up": outline draws itself, the iris colors itself in with a
rainbow sweep, lashes pop in, the crosshair snaps, a glint settles), the
static window icon, and any in-app use. Pure QPainter: two bezier arcs for
the almond, a conical rainbow gradient pie for the iris, straight-tick
lashes along the upper lid's normals, a four-tick reticle. No images, no
GPU cost worth mentioning.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QConicalGradient,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from . import theme

_LASH_COUNT = 5
_LASH_SPAN = (0.22, 0.78)     # fraction of the upper arc that carries lashes


def _eye_path(rect: QRectF) -> tuple[QPainterPath, QPainterPath]:
    """(upper arc, lower arc) of the almond, both left corner -> right."""
    cx, cy = rect.center().x(), rect.center().y()
    w, h = rect.width(), rect.height()
    left = QPointF(cx - w / 2, cy)
    right = QPointF(cx + w / 2, cy)
    upper = QPainterPath(left)
    upper.quadTo(QPointF(cx, cy - h * 0.95), right)
    lower = QPainterPath(left)
    lower.quadTo(QPointF(cx, cy + h * 0.95), right)
    return upper, lower


def _partial_pen(pen: QPen, path: QPainterPath, t: float) -> QPen:
    """Dash-offset trick: stroke only the first fraction t of the path."""
    length = max(path.length() / max(pen.widthF(), 1e-6), 1e-6)
    pen = QPen(pen)
    pen.setDashPattern([length, length])
    pen.setDashOffset(length * (1.0 - max(0.0, min(1.0, t))))
    return pen


def rainbow(center: QPointF, is_dark: bool) -> QConicalGradient:
    grad = QConicalGradient(center, 90.0)
    sat, val = (0.72, 0.95) if not is_dark else (0.66, 1.0)
    for i in range(13):
        grad.setColorAt(i / 12.0, QColor.fromHsvF((i / 12.0) % 1.0, sat, val))
    return grad


def paint_eye(
    p: QPainter,
    rect: QRectF,
    *,
    outline: float = 1.0,   # 0..1: the almond drawing itself in
    iris: float = 1.0,      # 0..1: rainbow sweep filling the iris
    cross: float = 1.0,     # 0..1: reticle snap (scale + fade)
    lashes: float = 1.0,    # 0..1: lashes popping in, staggered
    glint: float = 1.0,     # 0..1: cornea glint
    ink: QColor | None = None,
    is_dark: bool | None = None,
) -> None:
    pal = theme.current()
    ink = ink or QColor(pal.fg)
    dark = pal.is_dark if is_dark is None else is_dark
    p.setRenderHint(QPainter.Antialiasing)
    stroke = max(rect.width() * 0.018, 1.6)

    eye_rect = QRectF(rect)
    eye_rect.adjust(rect.width() * 0.06, rect.height() * 0.22,
                    -rect.width() * 0.06, -rect.height() * 0.22)
    upper, lower = _eye_path(eye_rect)
    base_pen = QPen(ink, stroke, Qt.SolidLine, Qt.RoundCap)

    # --- almond outline (draws in symmetrically from the left corner) ------
    if outline > 0:
        p.setBrush(Qt.NoBrush)
        p.setPen(_partial_pen(base_pen, upper, outline))
        p.drawPath(upper)
        p.setPen(_partial_pen(base_pen, lower, outline))
        p.drawPath(lower)

    cx, cy = eye_rect.center().x(), eye_rect.center().y()
    ri = eye_rect.height() * 0.52          # iris radius

    # --- iris: rainbow pie sweeping clockwise from 12 o'clock --------------
    if iris > 0:
        iris_rect = QRectF(cx - ri, cy - ri, 2 * ri, 2 * ri)
        p.setPen(Qt.NoPen)
        p.setBrush(rainbow(QPointF(cx, cy), dark))
        p.drawPie(iris_rect, 90 * 16, -int(360 * 16 * min(iris, 1.0)))
        if iris >= 1.0:
            ring = QPen(ink, stroke * 0.66)
            p.setPen(ring)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(iris_rect)

    # --- reticle: four ticks + hub, snapping from 125% ----------------------
    if cross > 0:
        k = 1.25 - 0.25 * cross            # scale 1.25 -> 1.0
        col = QColor(ink)
        col.setAlphaF(cross)
        p.setPen(QPen(col, stroke, Qt.SolidLine, Qt.RoundCap))
        for ang in (0.0, 90.0, 180.0, 270.0):
            a = math.radians(ang)
            r0, r1 = ri * 0.55 * k, ri * 0.95 * k
            p.drawLine(
                QPointF(cx + r0 * math.cos(a), cy - r0 * math.sin(a)),
                QPointF(cx + r1 * math.cos(a), cy - r1 * math.sin(a)))
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        hub = ri * 0.16
        p.drawEllipse(QPointF(cx, cy), hub, hub)

    # --- lashes: straight ticks along the upper lid's normals ---------------
    if lashes > 0:
        p.setPen(QPen(ink, stroke * 0.9, Qt.SolidLine, Qt.RoundCap))
        lo, hi = _LASH_SPAN
        for i in range(_LASH_COUNT):
            # staggered pop: lash i lives in its own slice of the progress
            t0 = i / _LASH_COUNT
            li = max(0.0, min(1.0, (lashes - t0) * _LASH_COUNT))
            if li <= 0:
                continue
            frac = lo + (hi - lo) * i / (_LASH_COUNT - 1)
            pt = upper.pointAtPercent(frac)
            ang = math.radians(upper.angleAtPercent(frac) + 90.0)  # outward normal
            ln = eye_rect.height() * 0.22 * li
            p.drawLine(pt, QPointF(pt.x() + ln * math.cos(ang),
                                   pt.y() - ln * math.sin(ang)))

    # --- glint --------------------------------------------------------------
    if glint > 0 and iris > 0.5:
        g = QColor("#ffffff")
        g.setAlphaF(0.85 * glint)
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(cx - ri * 0.35, cy - ri * 0.38),
                      ri * 0.13, ri * 0.13)


def make_pixmap(size: int = 256, *, is_dark: bool | None = None) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    paint_eye(p, QRectF(0, 0, size, size), is_dark=is_dark)
    p.end()
    return pm


def make_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(make_pixmap(s))
    return icon


# ------------------------------------------------------------------- splash
def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _phase(t: float, start: float, end: float, ease=_ease_out_cubic) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return ease((t - start) / (end - start))


class SplashScreen(QWidget):
    """'The eye wakes up' — frameless splash that animates the mark while the
    main window constructs, then fades itself away. Call finish(callback)
    once the app is ready; the callback fires after the fade."""

    DURATION = 1.9      # seconds of animation before fade-out is allowed

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(440, 340)
        self._t = 0.0
        self._fade = 1.0
        self._done_cb = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        screen = self.screen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2,
                      geo.center().y() - self.height() // 2)

    def start(self) -> None:
        self.show()
        self._timer.start()

    def finish(self, callback) -> None:
        """Fade out (once the animation has played) and then call back."""
        self._done_cb = callback

    def _tick(self) -> None:
        self._t += 0.016
        if self._done_cb is not None and self._t >= self.DURATION:
            self._fade -= 0.08
            if self._fade <= 0.0:
                self._timer.stop()
                cb, self._done_cb = self._done_cb, None
                cb()          # show the main window BEFORE closing the last
                self.close()  # visible window, or the app would quit here
                return
        self.update()

    def paintEvent(self, event) -> None:
        pal = theme.current()
        t = self._t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setOpacity(max(self._fade, 0.0))

        # card
        p.setPen(QPen(QColor(pal.border), 1))
        p.setBrush(QColor(pal.bg))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 18, 18)

        # gentle settle: 2% exhale after the crosshair lands
        settle = _phase(t, 1.5, 1.9)
        scale = 1.02 - 0.02 * settle
        eye = QRectF(70, 30, 300, 220)
        p.save()
        p.translate(eye.center())
        p.scale(scale, scale)
        p.translate(-eye.center())
        paint_eye(
            p, eye,
            outline=_phase(t, 0.0, 0.55),
            iris=_phase(t, 0.45, 1.25),
            lashes=_phase(t, 0.85, 1.35),
            cross=_phase(t, 1.2, 1.5),
            glint=_phase(t, 1.4, 1.7),
        )
        p.restore()

        word = _phase(t, 1.0, 1.6)
        if word > 0:
            col = QColor(pal.fg)
            col.setAlphaF(word)
            p.setPen(col)
            f = QFont("Segoe UI", 21)
            f.setWeight(QFont.DemiBold)
            f.setLetterSpacing(QFont.PercentageSpacing, 108)
            p.setFont(f)
            p.drawText(QRectF(0, 252, self.width(), 44), Qt.AlignCenter, "kovadapt")
            sub = QColor(pal.fg_dim)
            sub.setAlphaF(word)
            p.setPen(sub)
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(0, 288, self.width(), 20), Qt.AlignCenter,
                       "adaptive KovaaK's")
