"""ASCII LED art: the kovadapt eye, rendered the way real ASCII art is made.

A shaded painting of the eye is built as a continuous ink field (numpy) and
converted to characters through the Bourke density ramp — 141x67 cells,
supersampled, intensity-driven alpha. The iris is macro-photography grade:
a deep soft-edged pupil, a wavy collarette ring, two layers of radial fiber
striations with crypts between them, a dark limbal ring, ambient shadow
from the upper lid, and layered highlights (a sharp specular with halo, a
small catchlight, a faint lower sheen). Rainbow hue runs around the iris
with a touch of radial iridescence.

Choreography: the EYE completes first — noise warm-up, outline and lashes,
the iris rainbow-sweeping in, glints, a tone wash — and only then the
crosshair scribes in OVER the finished eye as a separate accent-colored
overlay pass with a soft backing, so the reticle always reads against the
busy iris. Then everything breathes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QPainter,
                           QPainterPath, QPixmap)
from PySide6.QtWidgets import QWidget

from . import theme

COLS, ROWS = 141, 67         # odd x odd: true center column/row for the reticle
_SS = 3                      # supersamples per cell axis
_ASPECT = 2.0                # cell height : width

# light -> dense (Paul Bourke's standard grayscale ramp, reversed)
_RAMP = (" `'.,:;\"^~-_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&%$#@")
_MIN_INK = 0.05

_W = 0.88                    # almond half-width (world x in [-1, 1])
_H = 0.52                    # almond apex height
_RI = 0.315                  # iris radius
_CY = 0.02                   # iris center y
_PUPIL = 0.21                # pupil radius (fraction of iris)


@dataclass(frozen=True)
class Cell:
    col: int
    row: int
    ch: str
    role: str        # outline | lash | iris | reticle | hub | glint | shade
    order: float     # 0..1 within the role's animation
    hue: float       # rainbow hue for iris cells
    ink: float       # 0..1 field intensity (drives alpha / value)
    rad: float       # 0..1 radius inside the iris (iris cells only)
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
            pts: np.ndarray, width: float | np.ndarray,
            strength: float | np.ndarray) -> None:
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
    x0 = -_W + 2.0 * _W * frac
    y0 = float(_upper(x0))
    slope = 2.0 * _H * x0 / (_W * _W)
    nx, ny = slope, -1.0                       # outward (up) normal
    n = math.hypot(nx, ny)
    nx, ny = nx / n, ny / n
    tx, ty = -ny, nx
    s = np.linspace(0.05, 1.0, 12)[:, None]
    base = np.array([x0, y0])
    return base + s * length * np.array([nx, ny]) \
        + (s ** 2) * curl * np.array([tx, ty])


def _render() -> list[Cell]:
    X, Y = _grid()
    ink = np.zeros_like(X)

    # ---- almond outline (heavier toward the corners) ---------------------
    xs = np.linspace(-_W, _W, 240)
    up = np.stack([xs, _upper(xs)], axis=1)
    lo = np.stack([xs, -_upper(xs)], axis=1)
    corner = np.abs(xs) / _W
    w_line = 0.011 + 0.009 * corner ** 4
    _stroke(ink, X, Y, up, w_line, 0.98)
    _stroke(ink, X, Y, lo, w_line, 0.92)

    # ---- lid crease above the upper lid ----------------------------------
    xs_c = np.linspace(-_W * 0.74, _W * 0.74, 120)
    crease = np.stack([xs_c * 1.06, _upper(xs_c) * 1.38 - 0.05], axis=1)
    _stroke(ink, X, Y, crease, 0.015, 0.30)

    # ---- lashes: eleven curling on top, five small below -----------------
    for i, frac in enumerate(np.linspace(0.08, 0.92, 11)):
        side = (frac - 0.5) * 2.0
        pts = _lash_points(float(frac), curl=0.11 * side,
                           length=0.15 + 0.05 * math.sin(i * 2.1))
        taper = np.linspace(1.0, 0.22, len(pts))
        _stroke(ink, X, Y, pts, 0.010 * taper, 0.95 * taper)
    for frac in (0.22, 0.38, 0.55, 0.70, 0.84):
        x0 = -_W + 2.0 * _W * frac
        y0 = float(-_upper(x0))
        pts = np.stack([np.full(6, x0) + (frac - 0.5) * 0.05,
                        np.linspace(y0, y0 + 0.065, 6)], axis=1)
        _stroke(ink, X, Y, pts, 0.009, 0.5)

    # ---- sclera form shadow inside the almond ----------------------------
    inside = (np.abs(X) < _W) & (Y > _upper(np.clip(X, -_W, _W))) \
        & (Y < -_upper(np.clip(X, -_W, _W)))
    lid_dist = np.abs(Y - _upper(np.clip(X, -_W, _W)))
    sclera = 0.10 + 0.22 * np.exp(-lid_dist / 0.10)
    sclera += 0.08 * (np.abs(X) / _W) ** 2
    ink = np.where(inside, np.maximum(ink, sclera * 0.55), ink)

    # ---- the iris: macro detail ------------------------------------------
    DX, DY = X, Y - _CY
    dist = np.hypot(DX, DY)
    ang = np.arctan2(DX, -DY)                            # 0 at 12 o'clock
    rr = dist / _RI
    in_iris = rr <= 1.0

    # two layers of radial fibers, wavering slightly with radius
    fine = np.abs(np.sin(ang * 46.0 + np.sin(rr * 9.0) * 1.6))
    coarse = np.abs(np.sin(ang * 13.0 - rr * 3.0))
    fibers = 0.34 + 0.20 * fine + 0.14 * coarse
    fibers *= 0.72 + 0.28 * rr                           # denser toward the rim

    # collarette: wavy ring around the pupil (the iris's inner lace)
    col_r = 0.40 + 0.045 * np.sin(ang * 11.0)
    collarette = 0.30 * np.exp(-((rr - col_r) / 0.055) ** 2)

    # crypts: darker pockets between fiber bundles
    crypts = np.zeros_like(X)
    for k, (ca, cr) in enumerate(((0.7, 0.62), (1.9, 0.74), (3.1, 0.58),
                                  (4.3, 0.70), (5.5, 0.66))):
        d_ang = np.angle(np.exp(1j * (ang - ca)))
        crypts += 0.22 * np.exp(-((d_ang / 0.22) ** 2 + ((rr - cr) / 0.10) ** 2))

    # limbal ring + soft-edged deep pupil
    limbal = 0.42 * np.exp(-((rr - 0.965) / 0.075) ** 2)
    pupil = 0.98 / (1.0 + np.exp((rr - _PUPIL) / 0.025))     # soft edge
    # bright ring just outside the pupil (subtractive glow)
    glow = 0.30 * np.exp(-((rr - (_PUPIL + 0.09)) / 0.05) ** 2)

    # ambient shadow cast on the iris by the upper lid
    lid_ao = 0.28 * np.exp(-np.abs(Y - _upper(np.clip(X, -_W, _W))) / 0.13)

    iris_ink = np.clip(fibers + collarette + crypts + limbal + lid_ao - glow,
                       0.0, 1.0)
    ink = np.where(in_iris, np.maximum(ink, iris_ink), ink)
    ink = np.where(in_iris, np.maximum(ink, pupil), ink)

    # ---- highlights subtract ink (light against the detail) --------------
    g1 = np.exp(-(((DX + 0.36 * _RI) / (0.15 * _RI)) ** 2
                  + ((DY + 0.44 * _RI) / (0.20 * _RI)) ** 2))       # specular
    g1h = 0.45 * np.exp(-(((DX + 0.36 * _RI) / (0.30 * _RI)) ** 2
                          + ((DY + 0.44 * _RI) / (0.40 * _RI)) ** 2))  # halo
    g2 = 0.8 * np.exp(-(((DX - 0.50 * _RI) / (0.09 * _RI)) ** 2
                        + ((DY - 0.52 * _RI) / (0.12 * _RI)) ** 2))  # catchlight
    sheen = 0.22 * np.exp(-((rr - 0.72) / 0.16) ** 2) \
        * np.clip(-np.cos(ang), 0.0, 1.0)                            # lower sheen
    glint = np.clip(g1 + g1h + g2 + sheen, 0.0, 1.2)
    ink = np.clip(ink - glint * 0.85, 0.0, 1.0)

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
        (np.abs(cyv - up_c) < 0.07) | (np.abs(cyv + up_c) < 0.07))
    above_lid = cyv < up_c - 0.015

    cells: list[Cell] = []
    for r in range(ROWS):
        for c in range(COLS):
            v = float(cell_ink[r, c])
            if v < _MIN_INK:
                continue
            ch = _RAMP[min(int(v * (len(_RAMP) - 1) + 0.5), len(_RAMP) - 1)]
            d = float(cdist[r, c])
            if d <= 1.04 and glint_cell[r, c] > 0.5:
                role, order, hue = "glint", 0.0, 0.0
            elif d <= 1.02:
                hue = float(cang[r, c]) / (2 * math.pi)
                role, order = "iris", hue
            elif near_outline[r, c] or above_lid[r, c]:
                role = "lash" if above_lid[r, c] else "outline"
                order = min(max((float(cx[r, c]) + _W) / (2 * _W), 0.0), 1.0)
                hue = 0.0
            else:
                role, hue = "shade", 0.0
                order = (float(cx[r, c]) + 1.0) / 2.0
            # rad is the radial distance from the pupil for EVERY cell: the
            # whole eye fills outward from the spark
            cells.append(Cell(c, r, ch, role, order, hue, v, d,
                              float(seeds[r, c])))

    # ---- the crosshair: a separate overlay pass over the finished eye ----
    cc_mid = (COLS - 1) // 2
    rr_mid = (ROWS - 1) // 2 + round(_CY * COLS / 4.0)
    for r in range(ROWS):
        d = float(cdist[r, cc_mid])
        if 0.34 <= d <= 1.10:
            cells.append(Cell(cc_mid, r, "|", "reticle", min(d, 1.0),
                              0.0, 0.9, d, float(seeds[r, cc_mid])))
    for c in range(COLS):
        d = float(cdist[rr_mid, c])
        if 0.34 <= d <= 1.10:
            cells.append(Cell(c, rr_mid, "-", "reticle", min(d, 1.0),
                              0.0, 0.9, d, float(seeds[rr_mid, c])))
    cells.append(Cell(cc_mid, rr_mid, "+", "hub", 0.0, 0.0, 1.0, 0.0,
                      float(seeds[rr_mid, cc_mid])))
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


_SPARK = 0.85          # when the pupil ignites
_BOLTS = (0.35, 1.45, 2.30, 3.55, 4.40, 5.60)   # lightning angles (rad, cw)
_SWEEP_START = 4.65    # the reticle's rotating sweep
_SWEEP_LEN = 0.6
_RETICLE_RED = "#ff3b30"


def led_state(cell: Cell, t: float) -> float:
    """Brightness 0..~1.6 at time t. Out of total darkness, a spark at the
    pupil pulses rainbow outward along the iris veins; the fill escapes the
    rim into the lids, lashes and shading; the crosshair is imposed at the
    end by a rotating sweep. cell.rad is radial distance from the pupil, so
    every role's timing is 'when the pulse reaches me'."""
    s = cell.seed

    if cell.role == "iris":
        # LIGHTNING ignition: discrete bolts streak outward along seeded
        # fiber angles almost instantly (with strobe flicker), then the glow
        # spreads angularly outward from each bolt
        ang = cell.hue * 2 * math.pi
        d_bolt = min(abs(math.remainder(ang - b, 2 * math.pi)) for b in _BOLTS)
        near_bolt = d_bolt < 0.13
        if near_bolt:
            lit_at = _SPARK + cell.rad * 0.22 + (d_bolt / 0.13) * 0.05
            age = t - lit_at
            if 0.0 < age < 0.38:        # lightning strobe before settling
                return 1.55 if int(age * 26) % 3 != 0 else 0.25
        else:
            lit_at = _SPARK + 0.35 + cell.rad * 0.95 + d_bolt * 0.5
    elif cell.role == "glint":
        lit_at = 3.9
    elif cell.role in ("outline", "lash", "shade"):
        # the energy escapes the rim and keeps travelling outward
        reach = max(cell.rad - 1.0, 0.0)
        base = {"outline": 2.35, "lash": 2.5, "shade": 2.75}[cell.role]
        lit_at = base + reach * 0.85 + 0.1 * s
    elif cell.role == "reticle":
        # ignited by the rotating sweep as it passes this tick's angle
        ang = _reticle_angle(cell)
        lit_at = _SWEEP_START + _SWEEP_LEN * (ang / (2 * math.pi))
    else:  # hub locks after the sweep completes, double flash
        lit_at = _SWEEP_START + _SWEEP_LEN + 0.25
    age = t - lit_at

    if age <= 0.0:
        # darkness — except a faint heartbeat gathering at the pupil
        if cell.role == "iris" and cell.rad < 0.30 and t > 0.25:
            beat = max(0.0, math.sin((t - 0.25) * 5.2))
            return 0.10 * beat * min((t - 0.25) / 0.5, 1.0) * (1.0 - cell.rad / 0.30)
        return 0.0

    if cell.role == "hub" and (age < 0.18 or 0.32 < age < 0.46):
        return 1.6
    b = _pop(age)
    # an echo pulse ripples outward once the iris is lit
    if cell.role == "iris" and 2.6 < t < 3.6:
        echo = math.exp(-((t - 2.7 - cell.rad * 0.8) / 0.12) ** 2)
        b += 0.35 * echo
    if t > 6.2:
        b *= 0.93 + 0.07 * math.sin(2.0 * math.pi * 0.35 * t + s * 1.2)
    return b


def _reticle_angle(cell: Cell) -> float:
    """Clockwise angle from 12 o'clock of a reticle cell (for the sweep)."""
    half = COLS / 2.0
    x = (cell.col - (COLS - 1) / 2.0) / half
    y = (cell.row - (ROWS - 1) / 2.0) * _ASPECT / half - _CY
    return math.atan2(x, -y) % (2 * math.pi)


TOTAL = 6.1      # seconds until fully lit (breathing continues after)


# ------------------------------------------------------------------ widget
def _mono() -> QFont:
    f = QFont("Cascadia Mono")
    if not QFontMetricsF(f).horizontalAdvance("@"):
        f = QFont("Consolas")
    f.setStyleHint(QFont.Monospace)
    return f


def paint_grid(p: QPainter, rect: QRectF, t: float | None,
               ink: QColor, bg: QColor, is_dark: bool,
               iris_hue: float | None = None,
               exclude_roles: frozenset[str] | tuple[str, ...] = ()) -> None:
    """Draw the stencil into rect at time t (None = fully lit, static).
    Two passes: the eye, then the reticle overlay with a soft backing so
    the crosshair reads against the iris detail. `iris_hue` locks the iris
    to one hue (the theme accent drives the backdrop's iris); None keeps
    the full rainbow. `exclude_roles` skips whole roles — the live
    backdrop renders its base without the iris/glint cells and redraws
    only those each frame."""
    cw = rect.width() / COLS
    ch = rect.height() / ROWS
    font = _mono()
    font.setPixelSize(max(int(ch * 1.05), 3))
    p.setFont(font)
    sat, val = (0.80, 0.95) if not is_dark else (0.72, 1.0)

    excl = frozenset(exclude_roles)
    overlay: list[tuple[Cell, float]] = []
    for cell in stencil():
        if cell.role in excl:
            continue
        b = 1.0 if t is None else led_state(cell, t)
        if b <= 0.01:
            continue
        if cell.role in ("reticle", "hub"):
            overlay.append((cell, b))
            continue
        if cell.role == "iris":
            # bright constant value: character density carries the fiber
            # texture; a touch of radial iridescence in the hue
            base_hue = cell.hue if iris_hue is None else iris_hue
            hue = (base_hue + 0.05 * math.sin(cell.rad * 6.0)) % 1.0
            s_mod = sat * (0.72 + 0.28 * cell.ink)
            col = QColor.fromHsvF(hue, s_mod, val)
        elif cell.role == "glint":
            col = QColor("#ffffff") if is_dark else QColor(ink)
            col.setAlphaF(0.9 if is_dark else 0.35)
        elif cell.role == "shade":
            col = QColor(ink)
            col.setAlphaF(0.28 + 0.45 * cell.ink)
        else:
            col = QColor(ink)
            col.setAlphaF(0.35 + 0.65 * cell.ink)
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

    # ---- reticle overlay: accent color on a soft backing -----------------
    back = QColor(bg)
    for cell, b in overlay:
        x = rect.x() + cell.col * cw
        y = rect.y() + cell.row * ch
        back.setAlphaF(0.55 * min(b, 1.0))
        p.setPen(Qt.NoPen)
        p.setBrush(back)
        p.drawRoundedRect(QRectF(x - cw * 0.15, y, cw * 1.3, ch * 1.05), 2, 2)
        col = QColor(_RETICLE_RED)          # the crosshair is always red
        if b > 1.0:
            k = min(b - 1.0, 0.6)
            col = QColor.fromRgbF(col.redF() + (1 - col.redF()) * k,
                                  col.greenF() + (1 - col.greenF()) * k,
                                  col.blueF() + (1 - col.blueF()) * k)
        col.setAlphaF(min(b, 1.0))
        p.setPen(col)
        f2 = QFont(p.font())
        f2.setBold(True)
        p.setFont(f2)
        # double-strike with a sub-pixel offset: a touch thicker than bold
        p.drawText(QRectF(x, y, cw * 1.8, ch * 1.25),
                   Qt.AlignLeft | Qt.AlignTop, cell.ch)
        p.drawText(QRectF(x + 0.7, y, cw * 1.8, ch * 1.25),
                   Qt.AlignLeft | Qt.AlignTop, cell.ch)
        p.setFont(font)

    # ---- the imposed sweep: an accent hand rotating once over the iris,
    # igniting the reticle ticks as it passes
    if t is not None and _SWEEP_START <= t <= _SWEEP_START + _SWEEP_LEN + 0.12:
        from PySide6.QtGui import QPen

        prog = min((t - _SWEEP_START) / _SWEEP_LEN, 1.0)
        px_x = rect.width() / 2.0
        px_y = rect.height() * COLS / (2.0 * _ASPECT * ROWS)
        cx_px = rect.x() + rect.width() / 2.0
        cy_px = rect.y() + rect.height() / 2.0 + _CY * px_y
        radius = 1.18 * _RI
        for k in range(7):
            a = prog * 2 * math.pi - k * 0.11
            if a < 0:
                continue
            col = QColor(_RETICLE_RED)
            col.setAlphaF(max(0.65 - k * 0.09, 0.0))
            p.setPen(QPen(col, 2.0))
            p.drawLine(
                int(cx_px), int(cy_px),
                int(cx_px + radius * math.sin(a) * px_x),
                int(cy_px - radius * math.cos(a) * px_y))


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
                  transparent: bool = True,
                  iris_hue: float | None = None,
                  exclude_roles: frozenset[str] | tuple[str, ...] = ()) -> QPixmap:
    """Static, fully-lit render (window icon, watermarks, docs)."""
    pal = theme.current()
    dark = pal.is_dark if is_dark is None else is_dark
    height = int(width * (ROWS * _ASPECT / COLS))
    pm = QPixmap(width, max(height, 1))
    bg = QColor(pal.bg)
    pm.fill(Qt.transparent if transparent else bg)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    paint_grid(p, QRectF(0, 0, width, height), None, QColor(pal.fg), bg, dark,
               iris_hue=iris_hue, exclude_roles=exclude_roles)
    p.end()
    return pm


# ------------------------------------------------------ live backdrop loop
# The backdrop animates the iris/glints on a perfect loop. Every temporal
# term below is built from integer multiples of 2*pi*phase/LOOP_T, and each
# helper reduces phase mod LOOP_T first — so state at phase LOOP_T is
# bit-identical to state at phase 0 (T % T == 0.0 exactly in IEEE floats).
LOOP_T = 8.0            # loop period, seconds
BLINK_PHASE = 5.8       # loop phase (s) at which the blink begins
BLINK_LEN = 0.45        # lids close and reopen inside this window


def loop_cell_color(cell: Cell, phase: float, *, is_dark: bool,
                    iris_hue: float | None, ink: QColor) -> QColor:
    """Pure per-cell color for the live backdrop at loop `phase` seconds.

    Mirrors paint_grid's static colors, animated: a brightness ripple
    travelling around the iris, a radial saturation breath, the two glints
    swelling out of phase — and, when `iris_hue` is None (Gamer/rgb mode),
    the full rainbow rotated by exactly one hue cycle per loop. Roles other
    than iris/glint get their static ink color. Exactly periodic:
    loop_cell_color(c, 0) == loop_cell_color(c, LOOP_T)."""
    phase = phase % LOOP_T
    w = 2.0 * math.pi * phase / LOOP_T           # one turn per loop
    sat, val = (0.80, 0.95) if not is_dark else (0.72, 1.0)
    if cell.role == "iris":
        base_hue = cell.hue if iris_hue is None else iris_hue
        if iris_hue is None:                     # Gamer rainbow: one full
            base_hue = (base_hue + phase / LOOP_T) % 1.0   # cycle per loop
        hue = (base_hue + 0.05 * math.sin(cell.rad * 6.0)) % 1.0
        ang = cell.hue * 2.0 * math.pi           # angle around the iris
        # shimmer: a soft three-armed ripple orbiting once per loop, plus a
        # single-armed counter-ripple at double speed for organic motion
        sh = 0.7 * math.sin(3.0 * ang - w) + 0.3 * math.sin(ang + 2.0 * w)
        breathe = 1.0 + 0.08 * math.sin(2.0 * w - cell.rad * 3.5)
        s_mod = min(max(sat * (0.72 + 0.28 * cell.ink) * breathe, 0.0), 1.0)
        v_mod = min(max(val * (0.88 + 0.10 * sh), 0.0), 1.0)
        return QColor.fromHsvF(hue, s_mod, v_mod)
    if cell.role == "glint":
        # the two glints (specular left of center, catchlight right)
        # twinkle out of phase: two gentle swells per loop each
        left = cell.col < (COLS - 1) / 2.0
        tw = math.sin(2.0 * w + (0.0 if left else math.pi))
        col = QColor("#ffffff") if is_dark else QColor(ink)
        col.setAlphaF((0.9 if is_dark else 0.35) * (0.82 + 0.18 * tw))
        return col
    col = QColor(ink)                            # non-live roles: static
    col.setAlphaF(0.35 + 0.65 * cell.ink)
    return col


def blink_amount(phase: float) -> float:
    """0 (open) .. 1 (fully shut): one smooth close-and-reopen bump per
    loop starting at BLINK_PHASE, lasting BLINK_LEN. Periodic in LOOP_T."""
    u = ((phase % LOOP_T) - BLINK_PHASE) / BLINK_LEN
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return math.sin(math.pi * u) ** 2


def blink_lid_paths(rect: QRectF,
                    k: float) -> tuple[QPainterPath, QPainterPath] | None:
    """Eyelid geometry for the backdrop blink at amount k (0 open, 1 shut)
    in the pixel space paint_grid uses for `rect` (cell centers).

    Returns (wedges, edges). `wedges` holds two closed regions, one per
    lid, each spanning from just inside the resting lid parabola (inset so
    the almond outline and lid margin survive the blink) down/up to the
    moving lid edge — the same parabola eased toward the iris centerline
    y=_CY, so the sweep keeps the almond's curvature instead of reading as
    a rectangle wipe. `edges` holds the two moving lid-margin polylines
    for stroking. None when k <= 0."""
    if k <= 0.0:
        return None
    half = COLS / 2.0
    cw = rect.width() / COLS
    ch = rect.height() / ROWS

    def px(wx: float) -> float:
        return rect.x() + (wx * half + (COLS - 1) / 2.0 + 0.5) * cw

    def py(wy: float) -> float:
        return rect.y() + (wy * half / _ASPECT + (ROWS - 1) / 2.0 + 0.5) * ch

    inset = 0.03            # leave the outline / lid margin un-erased
    pad = 0.025 * k         # push the wedges past the meeting line when shut
    xs = np.linspace(-_W, _W, 49)
    rest_u = _upper(xs) + inset
    rest_l = -_upper(xs) - inset
    seam_u = (1.0 - k) * _upper(xs) + k * _CY      # the visible lid margins:
    seam_l = (1.0 - k) * -_upper(xs) + k * _CY     # meet at y=_CY when shut
    edge_u = np.maximum(seam_u + pad, rest_u)
    edge_l = np.minimum(seam_l - pad, rest_l)

    wedges = QPainterPath()
    # both wedges wind the same way (top boundary L->R, bottom R->L), and the
    # winding rule keeps the band where the shut lids overlap at y=_CY inside
    # (odd-even would punch it back out)
    wedges.setFillRule(Qt.WindingFill)
    edges = QPainterPath()
    for top, bottom, edge in ((rest_u, edge_u, seam_u),
                              (edge_l, rest_l, seam_l)):
        wedges.moveTo(px(float(xs[0])), py(float(top[0])))
        for x, y in zip(xs[1:], top[1:]):
            wedges.lineTo(px(float(x)), py(float(y)))
        for x, y in zip(xs[::-1], bottom[::-1]):
            wedges.lineTo(px(float(x)), py(float(y)))
        wedges.closeSubpath()
        edges.moveTo(px(float(xs[0])), py(float(edge[0])))
        for x, y in zip(xs[1:], edge[1:]):
            edges.lineTo(px(float(x)), py(float(y)))
    return wedges, edges


# ------------------------------------------------------- eye progress bar
_NYAN_BANDS = ("#ff3355", "#ff9f2e", "#ffe12e", "#4ddd55", "#3aa0ff", "#9a5cff")


class AsciiProgress(QWidget):
    """Progress as the iris motif: a strip of LED characters filling left to
    right through the rainbow, the tip cell popping — used for calibration.
    In RGB mode it becomes the cat: a pixel cat riding a scrolling rainbow
    trail (an homage in original pixels)."""

    CELLS = 26

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frac = 0.0
        self._label = ""
        self._phase = 0
        self.setMinimumHeight(24)
        from PySide6.QtCore import QTimer

        self._anim = QTimer(self)
        self._anim.setInterval(140)
        self._anim.timeout.connect(self._advance)

    def _advance(self) -> None:
        self._phase += 1
        self.update()

    def set_progress(self, frac: float, label: str = "") -> None:
        self._frac = max(0.0, min(1.0, frac))
        self._label = label
        self.update()

    def hideEvent(self, event) -> None:
        self._anim.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        pal = theme.current()
        if pal.rgb and not self._anim.isActive():
            self._anim.start()
        elif not pal.rgb and self._anim.isActive():
            self._anim.stop()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        if pal.rgb:
            self._paint_nyan(p, pal)
        else:
            self._paint_leds(p, pal)

    # ------------------------------------------------------------- classic
    def _paint_leds(self, p: QPainter, pal) -> None:
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
        self._draw_label(p, pal, x0 + n * cw + 10)

    # ----------------------------------------------------------------- cat
    def _paint_nyan(self, p: QPainter, pal) -> None:
        track = self.CELLS * 11.0
        x0 = 2.0
        filled = max(self._frac * track, 4.0)
        bh = 3
        y0 = 3
        # scrolling rainbow trail: dashed bands moving with the phase
        for i, hexcol in enumerate(_NYAN_BANDS):
            col = QColor(hexcol)
            y = y0 + i * bh
            x = x0 - (self._phase * 3) % 12
            while x < filled - 2:
                w = min(8.0, filled - 2 - x)
                if x + w > x0 and w > 0:
                    p.fillRect(QRectF(max(x, x0), y, w, bh - 0.4), col)
                x += 12
        # empty remainder of the track
        p.setPen(QColor(pal.border))
        p.drawLine(int(x0 + filled + 22), y0 + 3 * bh,
                   int(x0 + track), y0 + 3 * bh)

        # the cat, riding the tip (2px pixel blocks, gentle bob)
        cx = x0 + filled - 2
        cy = 1 + (self._phase % 2)
        body = QColor("#f2a7c3")
        body_d = QColor("#cf7ba2")
        grey = QColor("#8d8d99")
        dark = QColor("#101014")
        p.fillRect(QRectF(cx - 4, cy + 8, 6, 4), grey)          # tail
        p.fillRect(QRectF(cx, cy + 4, 16, 12), body_d)          # body border
        p.fillRect(QRectF(cx + 1, cy + 5, 14, 10), body)        # pastry body
        for sx, sy in ((4, 7), (8, 9), (5, 12), (10, 6), (11, 11)):
            p.fillRect(QRectF(cx + sx, cy + sy, 2, 2), body_d)  # sprinkles
        p.fillRect(QRectF(cx + 12, cy + 2, 10, 10), grey)       # head
        p.fillRect(QRectF(cx + 12, cy, 3, 3), grey)             # ears
        p.fillRect(QRectF(cx + 19, cy, 3, 3), grey)
        p.fillRect(QRectF(cx + 15, cy + 5, 2, 2), dark)         # eyes
        p.fillRect(QRectF(cx + 19, cy + 5, 2, 2), dark)
        p.fillRect(QRectF(cx + 17, cy + 9, 3, 1.6), dark)       # mouth
        for leg in range(4):
            p.fillRect(QRectF(cx + 2 + leg * 4, cy + 16, 3, 3), grey)
        self._draw_label(p, pal, x0 + track + 30)

    def _draw_label(self, p: QPainter, pal, x: float) -> None:
        if self._label:
            p.setPen(QColor(pal.fg_dim))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(x, 0, 280, self.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, self._label)
