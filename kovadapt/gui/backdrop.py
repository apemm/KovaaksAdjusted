"""Parallax ASCII backdrop: the eye watches — live — from behind the UI.

Three layers — a big ASCII eye off-center and two depths of drifting glyph
dust — slide at different rates toward the cursor (plus a slow ambient
wander). The eye is no longer a static pixmap but a perfect 8 s loop:
everything except the iris and glints is rendered once into a base pixmap
(and the reticle into a small overlay pixmap); each animation frame
Source-blits the base into a reusable frame pixmap, redraws only the few
hundred live iris/glint glyphs with colors from ascii_art.loop_cell_color,
re-blits the reticle, and — once per loop — sweeps two eyelid wedges shut
for a blink (erase-composited parabolas matching the almond's curvature,
not a rectangle wipe). The loop clock rides the existing 30 Hz parallax
timer at half rate (~15 fps) and pauses while the window is hidden or
minimized. Every time-dependent term is periodic in ascii_art.LOOP_T, so
frame(LOOP_T) is byte-identical to frame(0).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QColor, QCursor, QFont, QFontMetricsF, QPainter,
                           QPen, QPixmap, QStaticText)

from . import ascii_art, theme

_GLYPHS = ".:*+#@"
_LIVE_ROLES = ("iris", "glint")


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
    """Owns the layers + parallax/loop state for one window. The window
    calls paint(painter) from its paintEvent and notify_theme()/
    notify_resize() when those change."""

    def __init__(self, window) -> None:
        self._win = window
        self._base: QPixmap | None = None      # eye minus iris/glint/reticle
        self._overlay: QPixmap | None = None   # reticle + hub, re-blit on top
        self._frame: QPixmap | None = None     # base + live cells, per frame
        self._live: list[tuple[QPointF, QStaticText, ascii_art.Cell]] = []
        self._font: QFont | None = None
        self._ink = QColor("#ffffff")
        self._iris_hue: float | None = None
        self._is_dark = True
        self._dust_far: QPixmap | None = None
        self._dust_near: QPixmap | None = None
        self._off = [QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)]
        self._phase = 0.0
        self._loop = 0.0        # live-eye loop clock, wraps at LOOP_T
        self._even = False      # tick parity: live frames render at ~15 fps
        self._timer = QTimer(window)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._frame is not None:
            return
        w = max(self._win.width(), 640)
        h = max(self._win.height(), 480)
        # the theme drives the backdrop's iris: accent-locked hue in normal
        # modes, the full rainbow in RGB mode
        pal = theme.current()
        self._iris_hue = None if pal.rgb else max(QColor(pal.accent).hueF(), 0.0)
        self._is_dark = pal.is_dark
        self._ink = QColor(pal.fg)
        eye_w = int(w * 0.62)
        self._base = ascii_art.render_pixmap(
            eye_w, iris_hue=self._iris_hue,
            exclude_roles=_LIVE_ROLES + ("reticle", "hub"))
        self._overlay = ascii_art.render_pixmap(
            eye_w, iris_hue=self._iris_hue,
            exclude_roles=_LIVE_ROLES + ("outline", "lash", "shade"))
        cw = self._base.width() / ascii_art.COLS
        ch = self._base.height() / ascii_art.ROWS
        self._font = ascii_art._mono()
        self._font.setPixelSize(max(int(ch * 1.05), 3))
        # QStaticText caches each glyph's layout once: ~9x faster per frame
        # than drawText(rect, flags) while anchoring the same top-left corner
        self._live = []
        for c in ascii_art.stencil():
            if c.role not in _LIVE_ROLES:
                continue
            st = QStaticText(c.ch)
            st.setTextFormat(Qt.PlainText)
            self._live.append((QPointF(c.col * cw, c.row * ch), st, c))
        self._frame = QPixmap(self._base.size())
        self._frame.fill(Qt.transparent)   # force the alpha channel
        self._render_frame()
        self._dust_far = _dust_pixmap(w + 80, h + 80, count=90, alpha=0.05,
                                      salt=3.7, accent_share=0.12)
        self._dust_near = _dust_pixmap(w + 120, h + 120, count=40, alpha=0.08,
                                       salt=9.1, accent_share=0.25)

    def notify_theme(self) -> None:
        self._frame = None          # re-render base + colors on next paint
        self._win.update()

    def notify_resize(self) -> None:
        self._frame = None

    # ------------------------------------------------------------------
    def _render_frame(self) -> None:
        """Compose one live frame at the current loop phase: base blit, the
        live iris/glint glyphs, the reticle overlay, the blink wedges."""
        pm = self._frame
        p = QPainter(pm)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawPixmap(0, 0, self._base)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setFont(self._font)
        phase = self._loop
        for pt, st, cell in self._live:
            p.setPen(ascii_art.loop_cell_color(
                cell, phase, is_dark=self._is_dark,
                iris_hue=self._iris_hue, ink=self._ink))
            p.drawStaticText(pt, st)
        p.drawPixmap(0, 0, self._overlay)
        k = ascii_art.blink_amount(phase)
        if k > 0.0:
            wedges, edges = ascii_art.blink_lid_paths(
                QRectF(0, 0, pm.width(), pm.height()), k)
            # the lids: erase the eye's interior between the resting lid
            # curve and the moving edge, then stroke the lid margin
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.fillPath(wedges, QColor(0, 0, 0))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            margin = QColor(self._ink)
            margin.setAlphaF(0.55 * k)
            p.setPen(QPen(margin, max(pm.height() / ascii_art.ROWS * 0.4, 1.0)))
            p.setBrush(Qt.NoBrush)
            p.drawPath(edges)
        p.end()

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if not self._win.isVisible() or self._win.isMinimized():
            return              # parallax AND the loop clock pause together
        self._phase += 0.033
        self._loop = (self._loop + 0.033) % ascii_art.LOOP_T
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
        self._even = not self._even
        animated = False
        if self._even and self._frame is not None:
            self._render_frame()        # the live eye at ~15 fps
            animated = True
        if moved or animated:
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
        p.drawPixmap(QPointF(ex, ey), self._frame)
        p.setOpacity(1.0)

        p.drawPixmap(QPointF(-60 + self._off[2].x(), -60 + self._off[2].y()),
                     self._dust_near)
        # a soft vignette keeps content edges readable over the art
        p.setOpacity(0.0)
        _ = QRectF(0, 0, w, h)
