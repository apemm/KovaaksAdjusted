"""ASCII LED art: the kovadapt eye, rendered the way real ASCII art is made.

A shaded painting of the eye — almond stroke, lid crease, tapered curling
lashes, sclera form-shadow, iris fibers under a limbal ring, reticle, twin
glints — is rendered as a continuous ink field (numpy) and then converted
to characters through the classic Bourke density ramp, one character per
cell with intensity-driven alpha (antialiased strokes) and supersampled
edges. Every cell still behaves like an LED in the synchronized matrix:
noise warm-up, outline-and-lash sweep, rainbow iris fill, reticle typing
outward, glints, then a left-to-right shading wash and breathing.

The stencil is procedural and cached; ~2.5k lit cells render in a few ms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from . import theme

COLS, ROWS = 97, 47          # odd x odd: true center column/row for the reticle
_SS = 3                      # supersamples per cell axis
_ASPECT = 2.0                # cell height : width

# light -> dense (Paul Bourke's standard grayscale ramp, reversed)
_RAMP = (" `'.,:;\"^~-_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&%$#@")
_MIN_INK = 0.05

_W = 0.88                    # almond half-width (world x in [-1, 1])
_H = 0.52                    # almond apex height
_RI = 0.30                   # iris radius
_CY = 0.02                   # iris center y


@dataclass(frozen=True)
class Cell:
    col: int
    row: int
    ch: str
    role: str        # outline | lash | iris | reticle | hub | glint | shade
    order: float     # 0..1 within the role's animation
    hue: float       # rainbow hue for iris cells
    ink: float       # 0..1 field intensity (drives alpha / value)
    seed: float


def _seed_arr(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (np.sin(c * 12.9898 + r * 78.233) * 43758.5453) % 1.0


def _upper(x: np.ndarray | float) -> np.ndarray | float:
    return -_H * (1.0 - (x / _W) ** 2)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    half = COLS / 2.0
    xs = (np.arange(COLS * _SS) / _SS - (COLS - 1) / 2.0) / half
    ys = (np.arange(ROWS * _SS) / _SS - (ROWS - 1) / 2.0) * _ASPECT / half
    return np.meshgrid(xs, ys)


def _stroke(ink: np.ndarray, X: np.ndarray, Y: np.ndarray,
            pts: np.ndarray, width: float | np.ndarray, strength: float | np.ndarray) -> None:
    """Add a stroke (dense point cloud) to the ink field, gaussian falloff."""
    d2 = np.full_like(X, np.inf)
    w = np.broadcast_to(np.asarray(width, dtype=float), (len(pts),))
    s = np.broadcast_to(np.asarray(strength, dtype=float), (len(pts),))
    best_w = np.full_like(X, w[0])
    best_s = np.full_like(X, s[0])
    for (px, py), wi, si in zip(pts, w, s):
        di = (X - px) ** 2 + (Y - py) ** 2
        closer = di < d2
        d2 = np.where(closer, di, d2)
        best_w = np.where(closer, wi, best_w)
        best_s = np.where(closer, si, best_s)
    np.maximum(ink, best_s * np.exp(-d2 / np.maximum(best_w, 1e-9) ** 2), out=ink)


def _lash_points(frac: float, curl: float, length: float) -> np.ndarray:
    """Curved, tapering lash from a point on the upper lid."""
    x0 = -_W + 2.0 * _W * frac
    y0 = float(_upper(x0))
    slope = 2.0 * _H * x0 / (_W * _W)
    nx, ny = slope, -1.0                       # outward (up) normal
    n = math.hypot(nx, ny)
    nx, ny = nx / n, ny / n
    tx, ty = -ny, nx                           # tangent, for the curl
    s = np.linspace(0.05, 1.0, 10)[:, None]
    base = np.array([x0, y0])
    return base + s * length * np.array([nx, ny]) \
        + (s ** 2) * curl * np.array([tx, ty])


def _render() -> list[Cell]:
    X, Y = _grid()
    ink = np.zeros_like(X)

    # ---- almond outline (slightly heavier toward the corners) ------------
    xs = np.linspace(-_W, _W, 160)
    up = np.stack([xs, _upper(xs)], axis=1)
    lo = np.stack([xs, -_upper(xs)], axis=1)
    corner = np.abs(xs) / _W
    w_line = 0.012 + 0.010 * corner ** 4
    _stroke(ink, X, Y, up, w_line, 0.98)
    _stroke(ink, X, Y, lo, w_line, 0.92)

    # ---- lid crease above the upper lid ----------------------------------
    xs_c = np.linspace(-_W * 0.72, _W * 0.72, 90)
    crease = np.stack([xs_c * 1.06, _upper(xs_c) * 1.38 - 0.05], axis=1)
    _stroke(ink, X, Y, crease, 0.016, 0.30)

    # ---- lashes: nine up top (curling outward), four small below ---------
    rng = np.linspace(0.10, 0.90, 9)
    for i, frac in enumerate(rng):
        side = (frac - 0.5) * 2.0                   # -1 .. 1
        pts = _lash_points(float(frac), curl=0.10 * side,
                           length=0.16 + 0.05 * math.sin(i * 2.3))
        taper = np.linspace(1.0, 0.25, len(pts))
        _stroke(ink, X, Y, pts, 0.012 * taper, 0.95 * taper)
    for frac in (0.25, 0.45, 0.62, 0.80):
        x0 = -_W + 2.0 * _W * frac
        y0 = float(-_upper(x0))
        pts = np.stack([np.full(5, x0) + (frac - 0.5) * 0.05,
                        np.linspace(y0, y0 + 0.07, 5)], axis=1)
        _stroke(ink, X, Y, pts, 0.010, 0.5)

    # ---- sclera form shadow inside the almond ----------------------------
    inside = (np.abs(X) < _W) & (Y > _upper(np.clip(X, -_W, _W))) \
        & (Y < -_upper(np.clip(X, -_W, _W)))
    lid_dist = np.abs(Y - _upper(np.clip(X, -_W, _W)))
    sclera = 0.10 + 0.22 * np.exp(-lid_dist / 0.10)      # shadow under the lid
    sclera += 0.08 * (np.abs(X) / _W) ** 2               # corners darken
    ink = np.where(inside, np.maximum(ink, sclera * 0.55), ink)

    # ---- iris: fibers under a limbal ring, pupil hub, reticle ------------
    DX, DY = X, Y - _CY
    dist = np.hypot(DX, DY)
    ang = np.arctan2(DX, -DY)                            # 0 at 12 o'clock
    rr = dist / _RI
    in_iris = rr <= 1.0
    fibers = 0.42 + 0.20 * np.abs(np.sin(ang * 22.0 + rr * 4.0)) \
        + 0.12 * np.abs(np.sin(ang * 9.0 - rr * 2.0))
    fibers *= 0.75 + 0.25 * rr                           # lighter core, darker rim
    limbal = 0.38 * np.exp(-((rr - 0.97) / 0.09) ** 2)
    pupil = 0.98 * (rr < 0.16)
    iris_ink = np.clip(fibers + limbal, 0.0, 1.0)
    ink = np.where(in_iris, np.maximum(ink, iris_ink), ink)
    ink = np.where(in_iris & (rr < 0.16), np.maximum(ink, pupil), ink)

    on_v = (np.abs(DX) < 0.014) & (rr > 0.30) & (rr < 1.04)
    on_h = (np.abs(DY) < 0.014 * _ASPECT) & (rr > 0.30) & (rr < 1.04)
    ink = np.where(on_v | on_h, np.maximum(ink, 0.94), ink)

    # ---- twin glints subtract ink (they are light, not dark) -------------
    g1 = np.exp(-(((DX + 0.38 * _RI) / (0.16 * _RI)) ** 2
                  + ((DY + 0.42 * _RI) / (0.22 * _RI)) ** 2))
    g2 = np.exp(-(((DX - 0.52 * _RI) / (0.10 * _RI)) ** 2
                  + ((DY - 0.50 * _RI) / (0.14 * _RI)) ** 2))
    glint = np.clip(g1 + 0.7 * g2, 0.0, 1.0)
    ink = np.clip(ink - glint * 0.9, 0.0, 1.0)

    # ---- reduce to cells + classify --------------------------------------
    cell_ink = ink.reshape(ROWS, _SS, COLS, _SS).mean(axis=(1, 3))
    cc, rr_i = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    half = COLS / 2.0
    cx = (cc - (COLS - 1) / 2.0) / half
    cyv = (rr_i - (ROWS - 1) / 2.0) * _ASPECT / half
    cdx, cdy = cx, cyv - _CY
    cdist = np.hypot(cdx, cdy) / _RI
    cang = np.arctan2(cdx, -cdy) % (2 * math.pi)
    glint_cell = glint.reshape(ROWS, _SS, COLS, _SS).mean(axis=(1, 3))
    seeds = _seed_arr(cc.astype(float), rr_i.astype(float))

    up_c = _upper(np.clip(cx, -_W, _W))
    near_outline = (np.abs(cx) <= _W * 1.05) & (
        (np.abs(cyv - up_c) < 0.09) | (np.abs(cyv + up_c) < 0.09))
    above_lid = cyv < up_c - 0.02

    cells: list[Cell] = []
    for r in range(ROWS):
        for c in range(COLS):
            v = float(cell_ink[r, c])
            if v < _MIN_INK:
                continue
            ch = _RAMP[min(int(v * (len(_RAMP) - 1) + 0.5), len(_RAMP) - 1)]
            d = float(cdist[r, c])
            if d <= 1.04 and glint_cell[r, c] > 0.45:
                role, order, hue = "glint", 0.0, 0.0
            elif d < 0.16:
                role, order, hue = "hub", 0.0, 0.0
            elif d <= 1.04 and (abs(cdx[r, c]) < 0.014 or
                                abs(cdy[r, c]) < 0.014 * _ASPECT) and d > 0.30:
                role, order, hue = "reticle", min(d, 1.0), 0.0
            elif d <= 1.02:
                hue = float(cang[r, c]) / (2 * math.pi)
                role, order = "iris", hue
            elif near_outline[r, c] or above_lid[r, c]:
                role = "lash" if above_lid[r, c] else "outline"
                order, hue = (float(cx[r, c]) + _W) / (2 * _W), 0.0
                order = min(max(order, 0.0), 1.0)
            else:
                role, hue = "shade", 0.0
                order = (float(cx[r, c]) + 1.0) / 2.0
            cells.append(Cell(c, r, ch, role, order, hue, v,
                              float(seeds[r, c])))
    return cells


_STENCIL: list[Cell] | None = None


def stencil() -> list[Cell]:
    global _STENCIL
    if _STENCIL is None:
        _STENCIL = _render()
    return _STENCIL


# ------------------------------------------------------------ choreography
def _pop(age: float) -> float:
    if age <= 0.0:
        return 0.0
    if age < 0.22:
        return 1.0 + 0.5 * (1.0 - age / 0.22)
    return 1.0


def led_state(cell: Cell, t: float) -> float:
    """Brightness 0..~1.5 for one cell at time t. Phases: warmup noise ->
    outline+lash sweep -> rainbow iris fill -> reticle -> glints -> the
    shading wash sweeps in -> everything breathes."""
    s = cell.seed
    warm = 0.0
    if t < 1.3 and cell.role != "shade":
        warm = 0.12 * max(0.0, math.sin(t * (5.0 + 4.0 * s) + s * 6.28)) \
            * min(t / 0.4, 1.0)

    if cell.role == "outline":
        lit_at = 1.0 + cell.order * 0.95
    elif cell.role == "lash":
        lit_at = 1.35 + cell.order * 0.95
    elif cell.role == "iris":
        lit_at = 2.45 + cell.order * 1.05
    elif cell.role in ("reticle", "hub"):
        lit_at = 3.55 + (cell.order * 0.3 if cell.role == "reticle" else 0.0)
    elif cell.role == "glint":
        lit_at = 3.95
    else:  # shade: the painting's tone wash arrives last, left to right
        lit_at = 4.15 + cell.order * 0.75
    b = _pop(t - lit_at)
    if b == 0.0:
        return warm
    if t > 5.2:
        b *= 0.93 + 0.07 * math.sin(2.0 * math.pi * 0.35 * t + s * 1.2)
    return b


TOTAL = 5.2      # seconds until fully lit (breathing continues after)


# ------------------------------------------------------------------ widget
def _mono() -> QFont:
    f = QFont("Cascadia Mono")
    if not QFontMetricsF(f).horizontalAdvance("@"):
        f = QFont("Consolas")
    f.setStyleHint(QFont.Monospace)
    return f


def paint_grid(p: QPainter, rect: QRectF, t: float | None,
               ink: QColor, bg: QColor, is_dark: bool) -> None:
    """Draw the stencil into rect at time t (None = fully lit, static)."""
    cw = rect.width() / COLS
    ch = rect.height() / ROWS
    font = _mono()
    font.setPixelSize(max(int(ch * 1.05), 3))
    p.setFont(font)
    sat, val = (0.80, 0.95) if not is_dark else (0.72, 1.0)
    for cell in stencil():
        b = 1.0 if t is None else led_state(cell, t)
        if b <= 0.01:
            continue
        if cell.role == "iris":
            # bright constant value: the CHARACTER density carries the fiber
            # texture; darkening the color as well just muddies the rainbow
            col = QColor.fromHsvF(cell.hue, sat * (0.75 + 0.25 * cell.ink), val)
        elif cell.role == "glint":
            col = QColor("#ffffff") if is_dark else QColor(ink)
            col.setAlphaF(0.9 if is_dark else 0.35)
        elif cell.role == "shade":
            col = QColor(ink)
            col.setAlphaF(0.28 + 0.45 * cell.ink)
        else:
            col = QColor(ink)
            col.setAlphaF(0.35 + 0.65 * cell.ink)   # antialiased strokes
        if b < 1.0:
            a = col.alphaF()
            r0, g0, b0 = bg.redF(), bg.greenF(), bg.blueF()
            col = QColor.fromRgbF(r0 + (col.redF() - r0) * b,
                                  g0 + (col.greenF() - g0) * b,
                                  b0 + (col.blueF() - b0) * b)
            col.setAlphaF(a * (0.3 + 0.7 * b))
        elif b > 1.0:
            k = min(b - 1.0, 0.5)
            col = QColor.fromRgbF(col.redF() + (1 - col.redF()) * k,
                                  col.greenF() + (1 - col.greenF()) * k,
                                  col.blueF() + (1 - col.blueF()) * k,
                                  col.alphaF())
        p.setPen(col)
        p.drawText(QRectF(rect.x() + cell.col * cw, rect.y() + cell.row * ch,
                          cw * 1.8, ch * 1.25),
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
    height = int(width * (ROWS * _ASPECT / COLS))
    pm = QPixmap(width, max(height, 1))
    bg = QColor(pal.bg)
    pm.fill(Qt.transparent if transparent else bg)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    paint_grid(p, QRectF(0, 0, width, height), None, QColor(pal.fg), bg, dark)
    p.end()
    return pm


# ------------------------------------------------------- eye progress bar
class AsciiProgress(QWidget):
    """Progress as the iris motif: a strip of LED characters filling left to
    right through the rainbow, the tip cell popping — used for calibration."""

    CELLS = 26

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frac = 0.0
        self._label = ""
        self.setMinimumHeight(22)

    def set_progress(self, frac: float, label: str = "") -> None:
        self._frac = max(0.0, min(1.0, frac))
        self._label = label
        self.update()

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        font = _mono()
        font.setPixelSize(15)
        p.setFont(font)
        n = self.CELLS
        cw = 11.0
        x0 = 2.0
        filled = self._frac * n
        sat, val = (0.66, 0.9) if not pal.is_dark else (0.6, 1.0)
        for i in range(n):
            frac_i = i / (n - 1)
            if i < int(filled):
                col = QColor.fromHsvF(frac_i * 0.83, sat, val)   # red..violet
                chn = "@" if (i % 3) else "#"
            elif i == int(filled) and self._frac < 1.0:
                col = QColor(pal.accent)
                chn = "*"
            else:
                col = QColor(pal.border)
                chn = "."
            p.setPen(col)
            p.drawText(QRectF(x0 + i * cw, 1, cw * 2, 20),
                       Qt.AlignLeft | Qt.AlignTop, chn)
        if self._label:
            p.setPen(QColor(pal.fg_dim))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(x0 + n * cw + 10, 0, 260, 22),
                       Qt.AlignLeft | Qt.AlignVCenter, self._label)
