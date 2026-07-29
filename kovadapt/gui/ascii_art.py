"""ASCII LED art: the kovadapt eye as character-grid choreography.

The brand language (Colicit-style): artwork built from monospace characters,
each cell behaving like an LED in a synchronized matrix — they warm up as
noise, organize into the almond outline, the iris colors itself in a
rainbow radial sweep, the reticle types outward from the hub, and then the
whole piece breathes. The stencil is procedural (almond from two parabolas,
iris circle, slope-aware outline glyphs, normal-following lashes), so it
renders crisp at any cell size — window icon to splash.

Per-cell state is a pure function of (cell, t): deterministic, loopable,
and cheap — a few hundred drawText calls per frame, only while visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from . import theme

COLS, ROWS = 49, 23          # odd x odd: true center column/row for the reticle
_W = 0.86                    # almond half-width   (world units, x in [-1, 1])
_H = 0.55                    # almond apex height
_RI = 0.335                  # iris radius
_CY = 0.02                   # iris center y (slightly low in the almond)
_LASHES = (0.20, 0.35, 0.50, 0.65, 0.80)   # positions along the upper arc


@dataclass(frozen=True)
class Cell:
    col: int
    row: int
    ch: str
    role: str        # outline | lash | iris | reticle | hub | glint | dust
    order: float     # 0..1 within its role's animation (sweep position)
    hue: float       # 0..1 rainbow hue (iris cells; 0 elsewhere)
    seed: float      # stable per-cell pseudo-random 0..1


def _seed(c: int, r: int) -> float:
    return (math.sin(c * 12.9898 + r * 78.233) * 43758.5453) % 1.0


def _world(c: int, r: int) -> tuple[float, float]:
    """Grid cell -> world coords with 2:1 cell aspect compensated."""
    half = COLS / 2.0
    return ((c - (COLS - 1) / 2.0) / half,
            (r - (ROWS - 1) / 2.0) * 2.0 / half)


def _upper(x: float) -> float:      # y is screen-down; the lid arcs upward
    return -_H * (1.0 - (x / _W) ** 2)


def _row_of(y: float) -> int:
    return round(y * COLS / 4.0 + (ROWS - 1) / 2.0)


def eye_stencil() -> list[Cell]:
    taken: dict[tuple[int, int], Cell] = {}
    half_cell = 1.2 / COLS * 2.0

    # ---- almond outline: rasterize per COLUMN — one cell per column per
    # arc, so the stroke can never break apart or double up
    for c in range(COLS):
        x, _ = _world(c, 0)
        if abs(x) > _W:
            continue
        slope = 2.0 * _H * x / (_W * _W)
        for curve_y, is_upper in ((_upper(x), True), (-_upper(x), False)):
            r = _row_of(curve_y)
            if not (0 <= r < ROWS) or (c, r) in taken:
                continue
            if abs(x) > _W * 0.86:
                ch = "(" if x < 0 else ")"
            elif abs(slope) < 0.45:
                ch = "-"
            elif (slope > 0) == is_upper:
                ch = "/"
            else:
                ch = "\\"
            order = (x + _W) / (2.0 * _W)
            taken[(c, r)] = Cell(c, r, ch, "outline", order, 0.0, _seed(c, r))

    # steep corner segments also rasterized per ROW so diagonal holes close
    for r in range(ROWS):
        _, y = _world(0, r)
        if abs(y) >= _H:
            continue
        # x on the upper arc at this height (both signs); lower arc mirrors
        x_abs = _W * math.sqrt(max(1.0 - abs(y) / _H, 0.0))
        for sign in (-1.0, 1.0):
            x = sign * x_abs
            slope = 2.0 * _H * x / (_W * _W)
            if abs(slope) < 1.0:          # shallow parts are column-covered
                continue
            c = round(x * (COLS / 2.0) + (COLS - 1) / 2.0)
            if not (0 <= c < COLS) or (c, r) in taken:
                continue
            is_upper = y < 0
            if abs(x) > _W * 0.86:
                ch = "(" if x < 0 else ")"
            elif (slope > 0) == is_upper:
                ch = "/"
            else:
                ch = "\\"
            order = (x + _W) / (2.0 * _W)
            taken[(c, r)] = Cell(c, r, ch, "outline", order, 0.0, _seed(c, r))

    # ---- iris / reticle / hub / glint / dust by cell scan ----------------
    for r in range(ROWS):
        for c in range(COLS):
            if (c, r) in taken:
                continue
            x, y = _world(c, r)
            dx, dy = x, y - _CY
            dist = math.hypot(dx, dy)
            if dist <= _RI * 1.02:
                ang = math.atan2(dx, -dy)            # 0 at 12 o'clock
                hue = (ang % (2 * math.pi)) / (2 * math.pi)
                gx, gy = dx + 0.35 * _RI, dy + 0.40 * _RI
                if math.hypot(gx, gy) < _RI * 0.16:
                    taken[(c, r)] = Cell(c, r, "o", "glint", 0.0, 0.0, _seed(c, r))
                    continue
                if dist < _RI * 0.15:
                    taken[(c, r)] = Cell(c, r, "@", "hub", 0.0, 0.0, _seed(c, r))
                    continue
                on_v = abs(dx) < half_cell * 0.5 and dist > _RI * 0.38
                on_h = abs(dy) < half_cell * 0.9 and dist > _RI * 0.38
                if on_v or on_h:
                    taken[(c, r)] = Cell(c, r, "|" if on_v else "-", "reticle",
                                         min(dist / _RI, 1.0), 0.0, _seed(c, r))
                    continue
                ch = "@" if dist < _RI * 0.5 else ("#" if dist < _RI * 0.8 else "*")
                taken[(c, r)] = Cell(c, r, ch, "iris", hue, hue, _seed(c, r))
            elif _seed(c, r) > 0.86 and abs(y) < 0.95:
                taken[(c, r)] = Cell(c, r, ".", "dust", _seed(c, r), 0.0,
                                     _seed(c, r))
    cells = list(taken.values())

    # ---- lashes: short ticks along the upper lid's outward normal --------
    for i, frac in enumerate(_LASHES):
        x = -_W + 2.0 * _W * frac
        y = _upper(x)
        slope = 2.0 * _H * x / (_W * _W)
        nx, ny = -(-slope), -1.0                      # outward (up) normal
        n = math.hypot(nx, ny)
        nx, ny = nx / n, ny / n
        ch = "|" if abs(nx) < 0.3 else ("\\" if nx < 0 else "/")
        for step in (1.6, 2.6, 3.6):
            wx = x + nx * step * (2.0 / COLS) * 2.2
            wy = y + ny * step * (2.0 / COLS) * 2.2 * 1.0
            c = round(wx * (COLS / 2.0) + (COLS - 1) / 2.0)
            r = round(wy * (COLS / 2.0) / 2.0 + (ROWS - 1) / 2.0)
            if 0 <= c < COLS and 0 <= r < ROWS:
                cells.append(Cell(c, r, ch, "lash",
                                  i / (len(_LASHES) - 1), 0.0, _seed(c, r)))
    return cells


_STENCIL: list[Cell] | None = None


def stencil() -> list[Cell]:
    global _STENCIL
    if _STENCIL is None:
        _STENCIL = eye_stencil()
    return _STENCIL


# ------------------------------------------------------------ choreography
def _pop(age: float) -> float:
    """LED turn-on: overshoot flash then settle (age in seconds since lit)."""
    if age <= 0.0:
        return 0.0
    if age < 0.22:
        return 1.0 + 0.5 * (1.0 - age / 0.22)
    return 1.0


def led_state(cell: Cell, t: float) -> float:
    """Brightness 0..~1.5 for one cell at time t (seconds). Phases:
    warmup noise -> outline sweep -> lashes -> iris rainbow sweep ->
    reticle from the hub -> glint -> breathing."""
    s = cell.seed
    warm = 0.0
    if t < 1.25 and cell.role != "dust":
        warm = 0.12 * max(0.0, math.sin(t * (5.0 + 4.0 * s) + s * 6.28)) \
            * min(t / 0.4, 1.0)

    if cell.role == "dust":
        base = 0.5 + 0.5 * math.sin(t * (0.7 + s) + s * 6.28)
        return min(t / 1.0, 1.0) * (0.25 + 0.35 * base)

    if cell.role == "outline":
        lit_at = 1.0 + cell.order * 0.9
    elif cell.role == "lash":
        lit_at = 2.0 + cell.order * 0.5
    elif cell.role == "iris":
        lit_at = 2.3 + cell.order * 1.05
    elif cell.role in ("reticle", "hub"):
        lit_at = 3.45 + (cell.order * 0.35 if cell.role == "reticle" else 0.0)
    else:  # glint
        lit_at = 3.95
    b = _pop(t - lit_at)
    if b == 0.0:
        return warm
    if t > 4.3:  # alive: gentle synchronized breathing + per-cell twinkle
        b *= 0.93 + 0.07 * math.sin(2.0 * math.pi * 0.35 * t + s * 1.2)
    return b


TOTAL = 4.5      # seconds until the art is fully lit (breathing continues)


# ------------------------------------------------------------------ widget
def _mono() -> QFont:
    f = QFont("Cascadia Mono")
    if not QFontMetricsF(f).horizontalAdvance("@"):
        f = QFont("Consolas")
    f.setStyleHint(QFont.Monospace)
    return f


def paint_grid(p: QPainter, rect: QRectF, t: float | None,
               ink: QColor, bg: QColor, is_dark: bool) -> None:
    """Draw the stencil into rect at time t (None = fully lit, no motion)."""
    cw = rect.width() / COLS
    ch = rect.height() / ROWS
    font = _mono()
    font.setPixelSize(max(int(ch * 0.98), 4))
    p.setFont(font)
    sat, val = (0.68, 0.92) if not is_dark else (0.62, 1.0)
    for cell in stencil():
        b = 1.0 if t is None else led_state(cell, t)
        if b <= 0.01:
            continue
        if cell.role == "iris":
            col = QColor.fromHsvF(cell.hue, sat, val)
        elif cell.role == "glint":
            # white reads on a dark ground; on cream the glint goes ink
            col = QColor("#ffffff") if is_dark else QColor(ink)
        elif cell.role == "dust":
            col = QColor(ink)
            col.setAlphaF(0.16)
        else:
            col = QColor(ink)
        if b < 1.0:
            r0, g0, b0 = bg.redF(), bg.greenF(), bg.blueF()
            col = QColor.fromRgbF(r0 + (col.redF() - r0) * b,
                                  g0 + (col.greenF() - g0) * b,
                                  b0 + (col.blueF() - b0) * b)
        elif b > 1.0:  # LED pop: push toward white
            k = min(b - 1.0, 0.5)
            col = QColor.fromRgbF(col.redF() + (1 - col.redF()) * k,
                                  col.greenF() + (1 - col.greenF()) * k,
                                  col.blueF() + (1 - col.blueF()) * k)
        p.setPen(col)
        p.drawText(QRectF(rect.x() + cell.col * cw, rect.y() + cell.row * ch,
                          cw * 1.6, ch * 1.2),
                   Qt.AlignLeft | Qt.AlignTop, cell.ch)


class AsciiEye(QWidget):
    """The mark as a live LED matrix; drive with set_time(t) or leave static."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._t: float | None = None

    def set_time(self, t: float | None) -> None:
        self._t = t
        self.update()

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        paint_grid(p, QRectF(self.rect()), self._t,
                   QColor(pal.fg), QColor(pal.bg), pal.is_dark)


def render_pixmap(width: int, *, is_dark: bool | None = None,
                  transparent: bool = True) -> QPixmap:
    """Static, fully-lit render (window icon, watermarks, docs)."""
    pal = theme.current()
    dark = pal.is_dark if is_dark is None else is_dark
    height = int(width * (ROWS * 2.0 / COLS))
    pm = QPixmap(width, height)
    bg = QColor(pal.bg)
    pm.fill(Qt.transparent if transparent else bg)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    paint_grid(p, QRectF(0, 0, width, height), None, QColor(pal.fg), bg, dark)
    p.end()
    return pm
