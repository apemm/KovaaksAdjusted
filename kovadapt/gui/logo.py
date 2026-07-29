"""The kovadapt opening: the ASCII eye wakes up as an LED matrix.

SplashScreen drives gui/ascii_art.py's character choreography — cells warm
up as noise, organize into the almond, the iris rainbow-sweeps itself in,
the reticle types outward from the hub — while the boot worker does real
startup work (profile scan, cross-session skill fit) and narrates it in a
status line. The wordmark types itself with a terminal cursor. Deliberately
unhurried (~5 s): the LED show IS the loading experience, and finish() only
fades once both the choreography and the boot work are done.

make_icon() renders the same character art for the window icon.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import ascii_art, theme

_WORD = "kovadapt"


def make_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(ascii_art.render_pixmap(s))
    return icon


class SplashScreen(QWidget):
    """Frameless ASCII LED splash. start() begins the show; finish(callback)
    lets it fade once the animation has played out."""

    MIN_SECONDS = 5.0

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # portrait: the character grid is near-square (48 cols x 23 rows of
        # 2:1 cells) — squeezing it into a landscape card halves the eye
        self.setFixedSize(520, 620)
        self._t = 0.0
        self._fade = 1.0
        self._done_cb = None
        self._status = ""
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

    def set_status(self, text: str) -> None:
        """Boot-worker narration under the wordmark ('reading profiles…')."""
        self._status = text

    def finish(self, callback) -> None:
        self._done_cb = callback

    def _tick(self) -> None:
        self._t += 0.016
        if self._done_cb is not None and self._t >= self.MIN_SECONDS:
            self._fade -= 0.06
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
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setOpacity(max(self._fade, 0.0))

        p.setPen(QPen(QColor(pal.border), 1))
        p.setBrush(QColor(pal.bg))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)

        # near-square canvas so the 2:1 character cells keep the eye's shape
        ascii_art.paint_grid(
            p, QRectF(40, 26, 440, 422), t,
            QColor(pal.fg), QColor(pal.bg), pal.is_dark)

        # wordmark types itself, terminal cursor blinking while it does
        chars = _WORD[: max(0, int((t - 2.6) / 0.14))]
        if chars or t > 2.6:
            f = QFont(ascii_art._mono())
            f.setPixelSize(34)
            f.setWeight(QFont.DemiBold)
            p.setFont(f)
            cursor = "▌" if (len(chars) < len(_WORD) and int(t * 3) % 2 == 0) else ""
            p.setPen(QColor(pal.fg))
            p.drawText(QRectF(0, 468, self.width(), 46), Qt.AlignCenter,
                       chars + cursor)
        if t > 4.0:
            a = min((t - 4.0) / 0.5, 1.0)
            col = QColor(pal.fg_dim)
            col.setAlphaF(a)
            p.setPen(col)
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(0, 522, self.width(), 20), Qt.AlignCenter,
                       "adaptive KovaaK's")
        if self._status:
            col = QColor(pal.fg_dim)
            col.setAlphaF(0.9)
            p.setPen(col)
            f = QFont(ascii_art._mono())
            f.setPixelSize(12)
            p.setFont(f)
            p.drawText(QRectF(0, 576, self.width(), 20), Qt.AlignCenter,
                       self._status)
