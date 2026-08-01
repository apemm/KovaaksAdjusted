"""ASCII LED art: the kovadapt eye, rendered the way real ASCII art is made.

A shaded painting of the eye is built as a continuous ink field (numpy) and
converted to characters through the Bourke density ramp — 141x67 cells by
default, supersampled, intensity-driven alpha; stencil() also builds
higher-resolution grids for the splash. The iris is macro-photography
grade: a deep soft-edged pupil, a wavy collarette ring, two layers of
radial fiber striations with crypts between them, a dark limbal ring, and
ambient shadow from the upper lid. On top of the rainbow hue that runs
around the iris, every cell carries its own seeded color detail — per-
fiber-bundle hue jitter, radial color banding, heterochromatic flecks,
value jitter — so the iris never reads as one smooth wheel.

There are NO specular highlights, and that is a deliberate removal rather
than an omission. They existed as four overlapping gaussians (specular,
halo, catchlight, lower sheen) that behaved differently depending on the
caller: for static renders they SUBTRACTED ink, carving the highlight out
of the fiber detail, while the splash asked for an intact iris and got the
same highlights back as separate overlay cells so they could animate in.
One eye, two irises, and in both cases the brightest thing on screen sat
where the detail had been erased. The iris is now described the same way
everywhere.

Opening choreography: darkness, a heartbeat gathering at the pupil,
lightning ignition along seeded fiber angles, the glow escaping the rim
into lids/lashes/shading, one slow deliberate blink (lids as rows of
characters with lash silhouettes — see blink_cells), and then the red
crosshair FLICKERS on — see reticle_flicker, which is a hard-stepped strike
and not a fade, because the crosshair is meant to read as having been there
all along.
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
_SS = 3                      # supersamples per cell axis (the 141x67 default)
# Supersampling for the SPLASH's denser grids. It was 2, on the reasoning that
# small cells need it less; the opposite is true. A cell's glyph is chosen from
# its MEAN INK, so 2x supersampling decides each character from four samples,
# and measured at the shipped 255x121 tier that picks a different glyph for
# **37% of cells** than 4x does. Those are not cosmetic disagreements — they
# are cells landing on the wrong rung of the density ramp, which is exactly
# the fiber texture the iris is made of.
#
# 4 rather than 6: ss2->ss4 moves 37.0% of cells, ss2->ss6 only 39.3%, so the
# estimate has essentially converged by 4 and further samples buy noise.
#
# The cost is real and lands where it is most visible — the field is built in
# SplashScreen.__init__, before show(), so this is 0.24 s of black screen on
# launch (construction 0.133 s -> 0.376 s). It is worth it only because
# DENSITY cannot be raised instead: on a 1080p stage 381x181 gives 2.22 px
# glyphs, which is dither rather than finer art, and costs 32.2 ms against a
# 33 ms frame budget. Sharpening the tier we do use is the whole of the
# available gain. Per-frame cost is unchanged; this is paid once.
_SS_DENSE = 4
_ASPECT = 2.0                # cell height : width

# light -> dense (Paul Bourke's standard grayscale ramp, reversed)
_RAMP = (" `'.,:;\"^~-_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&%$#@")
_MIN_INK = 0.05

_W = 0.88                    # almond half-width (world x in [-1, 1])
_H = 0.52                    # almond apex height
_RI = 0.315                  # iris radius
_CY = 0.02                   # iris center y
_PUPIL = 0.30                # pupil radius (fraction of iris)
_PUPIL_EDGE = 0.020          # pupillary margin softness, in the same units

# How far value/saturation fall at the pupil's center (0 = black, 1 = no dim).
_PUPIL_V = 0.07
_PUPIL_S = 0.25

# The pupil RESOLVES rather than existing from the first frame: the opening's
# spark and its lub-dub heartbeat both originate at the pupil, so a core that
# is dark from t=0 would swallow them. It stays lit through the ignition and
# contracts out of the light once the iris has filled — the eye focusing.
PUPIL_T0, PUPIL_T1 = 2.60, 3.90


def pupil_dim(rad):
    """1 outside the pupil, ~0 inside it, soft across the pupillary margin.

    The pupil is the darkest part of an eye, but ink here drives character
    density, not brightness — iris cells all render near full value. The
    dense glyphs `_field` already lays over the pupil therefore lit at FULL
    brightness in rainbow, so the eye had a bright core and read as having
    no pupil at all. Every color path — paint_grid, loop_cell_color and
    logo._EyeField — scales iris value/saturation by this, which turns the
    iris into a ring around a dark core without touching the stencil, the
    roles, or the ignition choreography.

    Accepts a scalar or an array; returns the same shape.
    """
    x = np.clip((np.asarray(rad, dtype=float) - _PUPIL) / _PUPIL_EDGE,
                -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def pupil_mix(t: float | None) -> float:
    """How much of the pupil has resolved at opening time t: 0 during the
    ignition, 1 once focused. Static renders and the backdrop loop pass
    None and get a fully formed pupil."""
    if t is None or t >= PUPIL_T1:
        return 1.0
    if t <= PUPIL_T0:
        return 0.0
    s = (t - PUPIL_T0) / (PUPIL_T1 - PUPIL_T0)
    return s * s * (3.0 - 2.0 * s)


@dataclass(frozen=True)
class Cell:
    col: int
    row: int
    ch: str
    role: str        # outline | lash | iris | reticle | hub | shade
    order: float     # 0..1 within the role's animation
    hue: float       # rainbow hue for iris cells
    ink: float       # 0..1 field intensity (drives alpha / value)
    rad: float       # 0..1 radius inside the iris (iris cells only)
    seed: float
    # per-cell iris color detail (defaults keep non-iris construction terse)
    hue_j: float = 0.0    # seeded hue offset: fiber bundle + radial band
    fleck: float = -1.0   # heterochromatic fleck hue offset, or -1 (none)
    vj: float = 0.0       # seeded value jitter


def _seed_arr(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (np.sin(c * 12.9898 + r * 78.233) * 43758.5453) % 1.0


def _upper(x: np.ndarray | float) -> np.ndarray | float:
    return -_H * (1.0 - (x / _W) ** 2)


def _grid(cols: int, rows: int, ss: int) -> tuple[np.ndarray, np.ndarray]:
    half = cols / 2.0
    xs = ((np.arange(cols * ss) / ss - (cols - 1) / 2.0) / half)
    ys = ((np.arange(rows * ss) / ss - (rows - 1) / 2.0) * _ASPECT / half)
    return np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))


# exp(-CUTOFF**2) == 2e-16, i.e. below the float epsilon of the peak: a point
# farther than CUTOFF widths from a cell cannot measurably ink it.
_STROKE_CUTOFF = 6.0


def _stroke(ink: np.ndarray, X: np.ndarray, Y: np.ndarray,
            pts: np.ndarray, width: float | np.ndarray,
            strength: float | np.ndarray) -> None:
    """Add a stroke (dense point cloud) to the ink field, gaussian falloff.

    Nearest point wins. Each point is evaluated only over the sub-window it
    can measurably affect, which turns an O(points x whole grid) sweep into
    an O(points x footprint) one — the difference between a 1 s and a 4 s
    stencil build at the dense splash tiers. The window radius is CUTOFF x
    the stroke's WIDEST point (not each point's own width), so every cell
    that any point can reach still sees every competing point and the
    nearest-wins result is identical to the brute-force sweep.
    """
    xs, ys = X[0, :], Y[:, 0]          # _grid's axes: both monotonic
    d2 = np.full_like(X, np.inf)
    w = np.broadcast_to(np.asarray(width, dtype=float), (len(pts),))
    s = np.broadcast_to(np.asarray(strength, dtype=float), (len(pts),))
    best_w = np.full_like(X, w[0])
    best_s = np.full_like(X, s[0])
    reach = _STROKE_CUTOFF * float(np.max(w))
    for (px, py), wi, si in zip(pts, w, s):
        i0, i1 = np.searchsorted(xs, (px - reach, px + reach))
        j0, j1 = np.searchsorted(ys, (py - reach, py + reach))
        if i0 >= i1 or j0 >= j1:
            continue
        win = (slice(j0, j1), slice(i0, i1))
        di = (X[win] - px) ** 2 + (Y[win] - py) ** 2
        closer = di < d2[win]
        d2[win] = np.where(closer, di, d2[win])
        best_w[win] = np.where(closer, wi, best_w[win])
        best_s[win] = np.where(closer, si, best_s[win])
    with np.errstate(over="ignore"):   # untouched cells: exp(-inf) -> 0
        np.maximum(ink, best_s * np.exp(-d2 / np.maximum(best_w, 1e-9) ** 2),
                   out=ink)


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


def _render(cols: int, rows: int, ss: int) -> list[Cell]:
    X, Y = _grid(cols, rows, ss)
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

    # collarette: wavy ring around the pupil (the iris's inner lace) —
    # tighter and stronger, so the inner lace survives cell reduction
    col_r = 0.40 + 0.045 * np.sin(ang * 11.0)
    collarette = 0.38 * np.exp(-((rr - col_r) / 0.05) ** 2)

    # crypts: darker pockets between fiber bundles (seven, deeper)
    crypts = np.zeros_like(X)
    for k, (ca, cr) in enumerate(((0.7, 0.62), (1.9, 0.74), (3.1, 0.58),
                                  (4.3, 0.70), (5.5, 0.66), (2.5, 0.52),
                                  (5.0, 0.80))):
        d_ang = np.angle(np.exp(1j * (ang - ca)))
        crypts += 0.30 * np.exp(-((d_ang / 0.24) ** 2 + ((rr - cr) / 0.11) ** 2))

    # limbal ring + soft-edged deep pupil
    limbal = 0.42 * np.exp(-((rr - 0.965) / 0.075) ** 2)
    # soft edge; exponent clamped so float32 grids can't overflow exp()
    pupil = 0.98 / (1.0 + np.exp(np.minimum((rr - _PUPIL) / 0.025, 60.0)))
    # bright ring just outside the pupil (subtractive glow)
    glow = 0.30 * np.exp(-((rr - (_PUPIL + 0.09)) / 0.05) ** 2)

    # ambient shadow cast on the iris by the upper lid
    lid_ao = 0.28 * np.exp(-np.abs(Y - _upper(np.clip(X, -_W, _W))) / 0.13)

    iris_ink = np.clip(fibers + collarette + crypts + limbal + lid_ao - glow,
                       0.0, 1.0)
    ink = np.where(in_iris, np.maximum(ink, iris_ink), ink)
    ink = np.where(in_iris, np.maximum(ink, pupil), ink)

    # NO HIGHLIGHTS. There were four — a specular blob, its halo, a catchlight
    # and a lower sheen — and they carried a second, contradictory description
    # of the same iris: in static renders they SUBTRACTED ink (carving light
    # out of the fiber detail), while the splash kept the iris whole and
    # overlaid separate "glint" cells so the highlight could animate in. So
    # the eye had two different irises depending on where you met it, and the
    # highlight read as a bright smear over detail that was missing underneath
    # it. Removing the whole layer is what makes one iris: the fibers,
    # collarette, crypts and limbal ring now describe it everywhere, and the
    # opening's finale is the crosshair (see reticle_flicker) rather than a
    # highlight arriving late.

    # ---- reduce to cells + classify --------------------------------------
    cell_ink = ink.reshape(rows, ss, cols, ss).mean(axis=(1, 3))
    cc, rr_i = np.meshgrid(np.arange(cols), np.arange(rows))
    half = cols / 2.0
    cx = (cc - (cols - 1) / 2.0) / half
    cyv = (rr_i - (rows - 1) / 2.0) * _ASPECT / half
    cdx, cdy = cx, cyv - _CY
    cdist = np.hypot(cdx, cdy) / _RI
    cang = np.arctan2(cdx, -cdy) % (2 * math.pi)
    fc, fr = cc.astype(float), rr_i.astype(float)
    seeds = _seed_arr(fc, fr)

    # per-cell iris color detail, all seeded and resolution-local:
    # one hue offset per fiber bundle, one per radial band, a warm-shifted
    # core, sparse heterochromatic flecks, and per-cell value jitter
    bundle = np.floor(cang * (13.0 / (2.0 * math.pi)))
    hue_jit = (_seed_arr(bundle * 1.37 + 4.2, bundle * 0.61) - 0.5) * 0.11
    band = np.floor(cdist * 4.5)
    hue_jit = hue_jit + (_seed_arr(band + 9.0, band * 1.31) - 0.5) * 0.07
    hue_jit = hue_jit - 0.05 * np.clip(1.0 - cdist, 0.0, 1.0)
    fl = _seed_arr(fc + 31.0, fr + 17.0)
    flecks = np.where(fl < 0.075,
                      0.32 + 0.28 * _seed_arr(fc + 3.0, fr + 77.0), -1.0)
    vjit = (_seed_arr(fc + 57.0, fr + 91.0) - 0.5) * 0.22

    up_c = _upper(np.clip(cx, -_W, _W))
    near_outline = (np.abs(cx) <= _W * 1.05) & (
        (np.abs(cyv - up_c) < 0.07) | (np.abs(cyv + up_c) < 0.07))
    above_lid = cyv < up_c - 0.015

    cells: list[Cell] = []
    for r in range(rows):
        for c in range(cols):
            v = float(cell_ink[r, c])
            d = float(cdist[r, c])
            if v < _MIN_INK:
                continue
            ch = _RAMP[min(int(v * (len(_RAMP) - 1) + 0.5), len(_RAMP) - 1)]
            if d <= 1.02:
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
            if role == "iris":
                cells.append(Cell(c, r, ch, role, order, hue, v, d,
                                  float(seeds[r, c]), float(hue_jit[r, c]),
                                  float(flecks[r, c]), float(vjit[r, c])))
            else:
                cells.append(Cell(c, r, ch, role, order, hue, v, d,
                                  float(seeds[r, c])))

    # ---- the crosshair: overlay cells drawn in _RETICLE_RED everywhere the
    # eye appears — static renders, the live backdrop's reticle layer, AND
    # the opening, where it is now the closing beat (see reticle_flicker)
    cc_mid = (cols - 1) // 2
    rr_mid = (rows - 1) // 2 + round(_CY * cols / 4.0)
    for r in range(rows):
        d = float(cdist[r, cc_mid])
        if 0.34 <= d <= 1.10:
            cells.append(Cell(cc_mid, r, "|", "reticle", min(d, 1.0),
                              0.0, 0.9, d, float(seeds[r, cc_mid])))
    for c in range(cols):
        d = float(cdist[rr_mid, c])
        if 0.34 <= d <= 1.10:
            cells.append(Cell(c, rr_mid, "-", "reticle", min(d, 1.0),
                              0.0, 0.9, d, float(seeds[rr_mid, c])))
    cells.append(Cell(cc_mid, rr_mid, "+", "hub", 0.0, 0.0, 1.0, 0.0,
                      float(seeds[rr_mid, cc_mid])))
    return cells


_STENCILS: dict[tuple[int, int], list[Cell]] = {}


def stencil(cols: int = COLS, rows: int = ROWS) -> list[Cell]:
    """The character stencil at a given grid resolution (cached). The
    default 141x67 grid feeds every static render and the live backdrop;
    the splash asks for a denser grid (supersampling drops to 2x there —
    small cells need it less and the build must stay start-up friendly).

    There is ONE stencil per resolution now. It used to take a
    `subtract_glints` flag that produced two materially different irises
    from the same call — highlights carved out of the detail for static
    renders, an intact iris plus overlay cells for the splash — so the eye
    did not look like itself in two of the places it appeared."""
    key = (cols, rows)
    if key not in _STENCILS:
        _STENCILS[key] = _render(cols, rows, _SS if cols <= COLS else _SS_DENSE)
    return _STENCILS[key]


# ------------------------------------------------------------ choreography
def _pop(age: float) -> float:
    if age <= 0.0:
        return 0.0
    if age < 0.22:
        return 1.0 + 0.5 * (1.0 - age / 0.22)
    return 1.0


_SPARK = 1.15          # when the pupil ignites (two heartbeats first)
_BOLTS = (0.35, 1.45, 2.30, 3.55, 4.40, 5.60)   # lightning angles (rad, cw)
_RETICLE_RED = "#ff3b30"

# the opening blink: fast eased close, held full closure, slower cushioned
# reopen — real blinks are temporally asymmetric (Trutoiu et al., Modeling
# and Animating Eye Blinks)
BLINK_T = 4.55
BLINK_CLOSE, BLINK_HOLD, BLINK_OPEN = 0.38, 0.22, 0.70
RETICLE_T = 6.0        # the crosshair strikes only after the reopen
RETICLE_LEN = 0.80     # how long the strike takes to settle
BREATHE_T = 6.9

# The crosshair FLICKERS in — it does not draw in, sweep in, or fade in. The
# distinction is the whole point: a fade says "this is being created", a
# flicker says "this was always here and the power is only now holding". So
# the shape arrives whole on its first lit frame and the level cuts hard
# between steps; nothing here interpolates.
#
# The pattern is the one a gas tube makes striking: a weak catch, a dropout,
# a bright overshoot, two rapid stutters, then steady. It is a fixed table
# rather than noise because the opening must be identical every launch (the
# same reason the backdrop's blink track is seeded), and because a random
# flicker sampled at 30 fps is as likely to look like dropped frames as like
# a strike.
#
# Every segment is at least 0.07 s. The splash paints at 33 ms, so that is
# ~2 frames minimum: shorter steps get sampled at an arbitrary phase and
# some would simply never be drawn, which is the artifact the backdrop's
# _DUR_BAND comment describes for lids. The flicker also deliberately sits
# AFTER the last warp knot in logo.py, so it plays at 1:1 wall time — these
# durations are real seconds, not choreography seconds.
_FLICKER: tuple[tuple[float, float], ...] = (
    (0.00, 0.55),      # the tube catches, weakly
    (0.09, 0.00),      # and drops out
    (0.20, 1.00),      # a full strike
    (0.28, 0.00),
    (0.37, 0.72),
    (0.45, 0.10),      # nearly out, not quite
    (0.54, 1.25),      # overshoot: the moment it holds
    (0.63, 0.85),
    (0.72, 1.00),      # steady from here
)


def reticle_flicker(t: float) -> float:
    """Crosshair brightness 0..1.25 at choreography time t.

    0 before RETICLE_T (the opening has no crosshair until the eye is open),
    a hard-stepped strike through the table above, then a steady 1.0. Values
    over 1.0 are an overshoot the caller is expected to clip toward white,
    the same convention led_state uses for its own pop.
    """
    u = t - RETICLE_T
    if u < 0.0:
        return 0.0
    level = _FLICKER[0][1]
    for at, lv in _FLICKER:
        if u < at:
            break
        level = lv
    return level


def splash_blink(t: float) -> float:
    """0 (open) .. 1 (shut) for the opening's single deliberate blink."""
    u = t - BLINK_T
    if u <= 0.0:
        return 0.0
    if u < BLINK_CLOSE:
        s = u / BLINK_CLOSE
        return s * s * (3.0 - 2.0 * s)
    u -= BLINK_CLOSE
    if u < BLINK_HOLD:
        return 1.0
    u -= BLINK_HOLD
    if u < BLINK_OPEN:
        s = u / BLINK_OPEN
        return 1.0 - s * s * (3.0 - 2.0 * s)
    return 0.0


def led_state(cell: Cell, t: float) -> float:
    """Brightness 0..~1.6 at time t. Out of total darkness, a spark at the
    pupil pulses rainbow outward along the iris veins; the fill escapes the
    rim into the lids, lashes and shading; after the blink the crosshair
    flickers on. cell.rad is radial distance from the pupil, so every role's
    timing is 'when the pulse reaches me' — except the reticle, which has no
    such timing because it does not travel: it strikes whole (see
    reticle_flicker)."""
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
            lit_at = _SPARK + 0.40 + cell.rad * 0.95 + d_bolt * 0.5
    elif cell.role in ("outline", "lash", "shade"):
        # the energy escapes the rim and keeps travelling outward
        reach = max(cell.rad - 1.0, 0.0)
        base = {"outline": 2.85, "lash": 3.0, "shade": 3.25}[cell.role]
        lit_at = base + reach * 0.55 + 0.1 * s
    else:
        # reticle/hub: the crosshair does not propagate, it strikes. Every
        # cell shares one level so the shape appears whole, which is what
        # separates a flicker from something being drawn in.
        return reticle_flicker(t)
    age = t - lit_at

    if age <= 0.0:
        # darkness — except a lub-dub heartbeat gathering at the pupil
        if cell.role == "iris" and cell.rad < 0.30 and t > 0.25:
            u = t - 0.25
            beat = max(0.0, math.sin(u * 5.2)) \
                + 0.6 * max(0.0, math.sin(u * 5.2 - 1.1))
            return 0.14 * beat * min(u / 0.5, 1.0) * (1.0 - cell.rad / 0.30)
        return 0.0

    b = _pop(age)
    # an echo pulse ripples outward once the iris is lit
    if cell.role == "iris" and 3.0 < t < 3.9:
        echo = math.exp(-((t - 3.05 - cell.rad * 0.8) / 0.12) ** 2)
        b += 0.35 * echo
    if t > BREATHE_T:
        b *= 0.93 + 0.07 * math.sin(2.0 * math.pi * 0.35 * t + s * 1.2)
    return b


TOTAL = 7.1      # seconds until fully settled (breathing continues after)


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
    backdrop renders its base without the iris cells and redraws only
    those each frame."""
    cw = rect.width() / COLS
    ch = rect.height() / ROWS
    font = _mono()
    font.setPixelSize(max(int(ch * 1.05), 3))
    p.setFont(font)
    sat, val = (0.80, 0.95) if not is_dark else (0.72, 1.0)

    excl = frozenset(exclude_roles)
    p_mix = pupil_mix(t)                 # the core is lit until the eye focuses
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
            # character density carries the fiber texture; the hue carries
            # the cell's seeded detail — radial iridescence, fiber-bundle
            # jitter, radial banding, heterochromatic flecks, value jitter
            base_hue = cell.hue if iris_hue is None else iris_hue
            hue = (base_hue + 0.05 * math.sin(cell.rad * 6.0) + cell.hue_j) % 1.0
            s_mod = sat * (0.72 + 0.28 * cell.ink)
            v_mod = min(max(val * (1.0 + cell.vj * 0.5), 0.0), 1.0)
            if cell.fleck >= 0.0:
                hue = (hue + cell.fleck) % 1.0
                s_mod = min(s_mod * 1.15, 1.0)
            # the dark core: the rainbow is a ring, not a disc
            if p_mix > 0.0:
                dim = float(pupil_dim(cell.rad))
                s_mod *= 1.0 - p_mix * (1.0 - (_PUPIL_S + (1.0 - _PUPIL_S) * dim))
                v_mod *= 1.0 - p_mix * (1.0 - (_PUPIL_V + (1.0 - _PUPIL_V) * dim))
            col = QColor.fromHsvF(hue, s_mod, v_mod)
        elif cell.role == "shade":
            col = QColor(ink)
            col.setAlphaF(min(0.28 + 0.45 * cell.ink, 1.0))
        else:
            col = QColor(ink)
            col.setAlphaF(min(0.35 + 0.65 * cell.ink, 1.0))
        if b < 1.0:
            a = col.alphaF()
            r0, g0, b0 = bg.redF(), bg.greenF(), bg.blueF()
            col = QColor.fromRgbF(r0 + (col.redF() - r0) * b,
                                  g0 + (col.greenF() - g0) * b,
                                  b0 + (col.blueF() - b0) * b)
            col.setAlphaF(min(a * (0.3 + 0.7 * b), 1.0))
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
# The backdrop animates the iris on a perfect loop. Every temporal
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
    travelling around the iris, a radial saturation breath — and, when
    `iris_hue` is None (Gamer/rgb mode), the full rainbow rotated by exactly
    one hue cycle per loop. Roles other than iris get their static ink
    color. Exactly periodic:
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
        dim = float(pupil_dim(cell.rad))         # dark core, same as static
        s_mod *= _PUPIL_S + (1.0 - _PUPIL_S) * dim
        v_mod *= _PUPIL_V + (1.0 - _PUPIL_V) * dim
        return QColor.fromHsvF(hue, s_mod, v_mod)
    col = QColor(ink)                            # non-live roles: static
    col.setAlphaF(min(0.35 + 0.65 * cell.ink, 1.0))
    return col


def blink_amount(phase: float) -> float:
    """0 (open) .. 1 (fully shut): one smooth close-and-reopen bump per
    loop starting at BLINK_PHASE, lasting BLINK_LEN. Periodic in LOOP_T."""
    u = ((phase % LOOP_T) - BLINK_PHASE) / BLINK_LEN
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return math.sin(math.pi * u) ** 2


def lid_margins(x, k: float):
    """World-space y of the moving (upper, lower) lid margins at closure
    k for world x (scalar or array). The lids meet on a seam biased well
    below center — the upper lid does ~3/4 of the travel, the lower lid
    rises the rest, the way real lids close."""
    up = _upper(x)
    meet = _CY + 0.62 * (-up - _CY)
    return (1.0 - k) * up + k * meet, (1.0 - k) * (-up) + k * meet


# lid glyph alphabets: heavy chars build the margin silhouette, sparse
# directional strokes hang below it as lashes (solid-style ASCII craft:
# dense chars for silhouette, lighter ones to soften)
_MARGIN_UP = "wWmM"
_MARGIN_LO = "~-_~"
_FRINGE_LO = "'`,"


def blink_cells(k: float, cols: int = COLS,
                rows: int = ROWS) -> list[tuple[int, int, str, str, float]]:
    """The eyelids AS CHARACTERS at closure k: rows of lid-skin glyphs
    sweep down/up with the moving margins, the margin row itself is a
    heavy lash silhouette, and sparse directional lashes hang off it.
    Returns (col, row, ch, kind, shade) tuples, kind in {"skin", "lash"},
    shade 0..1 for the caller's alpha. Pure in (k, cols, rows) — the
    backdrop's perfect loop depends on that. Empty when k <= 0."""
    if k <= 0.0:
        return []
    half = cols / 2.0
    row_h = 2.0 * _ASPECT / cols                 # world height of one row
    c_all = np.arange(cols)
    x_all = (c_all - (cols - 1) / 2.0) / half
    keep = np.abs(x_all) < _W * 0.995
    ci = c_all[keep]
    xs = x_all[keep]
    um, lm = lid_margins(xs, k)
    rest_u = _upper(xs)
    rest_l = -rest_u
    r_all = np.arange(rows)
    y = ((r_all - (rows - 1) / 2.0) * _ASPECT / half)[:, None]
    UM, LM = um[None, :], lm[None, :]

    skin_u = (y >= rest_u[None, :] + 0.01) & (y < UM - 0.55 * row_h)
    lash_u = (np.abs(y - UM) <= 0.55 * row_h) & (y >= rest_u[None, :] - 0.2 * row_h)
    fringe_u = (y - UM > 0.55 * row_h) & (y - UM <= 1.7 * row_h)
    skin_l = (y > LM + 0.55 * row_h) & (y <= rest_l[None, :] - 0.01)
    lash_l = (np.abs(y - LM) <= 0.55 * row_h) \
        & (y <= rest_l[None, :] + 0.2 * row_h) & ~lash_u & ~skin_u
    fringe_l = (LM - y > 0.55 * row_h) & (LM - y <= 1.4 * row_h) \
        & ~skin_u & ~lash_u & ~fringe_u
    fringe_u &= ~skin_l & ~lash_l

    fc2 = np.broadcast_to(ci[None, :].astype(float), skin_u.shape)
    fr2 = np.broadcast_to(r_all[:, None].astype(float), skin_u.shape)
    seeds = _seed_arr(fc2 + 13.0, fr2 + 5.0)
    n_ramp = len(_RAMP) - 1

    out: list[tuple[int, int, str, str, float]] = []
    # ---- lid skin: banded rows of ramp chars riding the margin curve ----
    d_u = np.where(skin_u, UM - y, 0.0)
    d_l = np.where(skin_l, y - LM, 0.0)
    for mask, dist in ((skin_u, d_u), (skin_l, d_l)):
        shade = np.clip(0.30 + 0.45 * np.exp(-dist / 0.13)
                        + 0.10 * np.cos(dist * 48.0), 0.05, 1.0)
        for r, j in zip(*np.nonzero(mask)):
            sh = float(shade[r, j])
            out.append((int(ci[j]), int(r), _RAMP[int(sh * 0.72 * n_ramp)],
                        "skin", sh))

    # ---- the moving margins: heavy lash-silhouette rows -----------------
    slope_u = (2.0 * _H * xs / (_W * _W)) * ((1.0 - k) - 0.62 * k)
    for r, j in zip(*np.nonzero(lash_u)):
        s = float(slope_u[j])
        if s > 0.5:
            ch = "\\"
        elif s < -0.5:
            ch = "/"
        else:
            ch = _MARGIN_UP[int(seeds[r, j] * 3.99)]
        out.append((int(ci[j]), int(r), ch, "lash", 0.85))
    for r, j in zip(*np.nonzero(lash_l)):
        out.append((int(ci[j]), int(r), _MARGIN_LO[int(seeds[r, j] * 3.99)],
                    "lash", 0.55))

    # ---- lashes hanging off the margins (sparse, splayed outward) -------
    for r, j in zip(*np.nonzero(fringe_u)):
        if seeds[r, j] >= 0.42:
            continue
        x = float(xs[j])
        ch = "/" if x < -0.12 else ("\\" if x > 0.12 else "|")
        out.append((int(ci[j]), int(r), ch, "lash", 0.7))
    for r, j in zip(*np.nonzero(fringe_l)):
        if seeds[r, j] >= 0.30:
            continue
        out.append((int(ci[j]), int(r), _FRINGE_LO[int(seeds[r, j] * 9.99)],
                    "lash", 0.45))
    return out


def blink_lid_paths(rect: QRectF,
                    k: float) -> tuple[QPainterPath, QPainterPath] | None:
    """Eyelid ERASE geometry for the backdrop blink at amount k (0 open,
    1 shut) in the pixel space paint_grid uses for `rect` (cell centers).
    The character lids from blink_cells are drawn over the erased region.

    Returns (wedges, edges). `wedges` holds two closed regions, one per
    lid, each spanning from just inside the resting lid parabola (inset so
    the almond outline survives the blink) down/up to the moving lid
    margin from lid_margins — the same parabola eased toward the shut
    seam, so the sweep keeps the almond's curvature instead of reading as
    a rectangle wipe. `edges` holds the two moving lid-margin polylines
    (the curve the character lash rows ride). None when k <= 0."""
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
    pad = 0.025 * k         # push the wedges past the meeting seam when shut
    xs = np.linspace(-_W, _W, 49)
    rest_u = _upper(xs) + inset
    rest_l = -_upper(xs) - inset
    seam_u, seam_l = lid_margins(xs, k)
    edge_u = np.maximum(seam_u + pad, rest_u)
    edge_l = np.minimum(seam_l - pad, rest_l)

    wedges = QPainterPath()
    # both wedges wind the same way (top boundary L->R, bottom R->L), and the
    # winding rule keeps the band where the shut lids overlap at the seam
    # inside (odd-even would punch it back out)
    wedges.setFillRule(Qt.WindingFill)
    edges = QPainterPath()
    for top, bottom, edge in ((rest_u, edge_u, seam_u),
                              (edge_l, rest_l, seam_l)):
        wedges.moveTo(px(float(xs[0])), py(float(top[0])))
        for x, yv in zip(xs[1:], top[1:]):
            wedges.lineTo(px(float(x)), py(float(yv)))
        for x, yv in zip(xs[::-1], bottom[::-1]):
            wedges.lineTo(px(float(x)), py(float(yv)))
        wedges.closeSubpath()
        edges.moveTo(px(float(xs[0])), py(float(edge[0])))
        for x, yv in zip(xs[1:], edge[1:]):
            edges.lineTo(px(float(x)), py(float(yv)))
    return wedges, edges


# ------------------------------------------------------- eye progress bar
_NYAN_BANDS = ("#ff3355", "#ff9f2e", "#ffe12e", "#4ddd55", "#3aa0ff", "#9a5cff")

# The cat's coat follows the accent theme: indigo keeps the black cat.
_CAT_COATS = {
    "indigo": ("#17171c", "#33333d", "#7cfc9b"),   # black cat, green eyes
    "ocean": ("#2c3a52", "#48597a", "#ffd23e"),    # slate-blue, amber eyes
    "mint": ("#e8e6df", "#b9b6ac", "#3aa0ff"),     # cream-white, blue eyes
    "rose": ("#c96f3b", "#a04e21", "#4ddd55"),     # ginger, green eyes
}


def _cat_coat(pal) -> tuple[QColor, QColor, QColor]:
    """(body, edge, eyes) for the current accent; black cat by default.

    Reads the preset NAME off the palette — matching pal.accent against hex
    literals stopped working the moment accent colors became derived rather
    than hand-picked.

    The coat is then LIFTED OFF THE PAGE if it would otherwise disappear
    into it. The table is hand-picked hex that never consulted pal.is_dark,
    so the default black cat sat on the dark themes' near-black page at
    1.06:1 and mint's cream-white cat sat on cream at 1.17:1 — measured
    against theme.py's own 3.0 control floor. This cat is not decoration:
    CatSlider replaced QSlider as the overlay-opacity control, so the coat
    IS the handle, and on three of the four themes the only part of it a
    user could see was the 2px eyes.
    """
    from . import color

    key = getattr(pal, "accent_name", "indigo")
    body, edge, eye = _CAT_COATS.get(key, _CAT_COATS["indigo"])
    page = getattr(pal, "bg_alt", "#ffffff")
    if color.contrast_ratio(body, page) < 3.0:
        # Move away from the page, keeping the coat's hue: toward paper on a
        # dark theme, toward ink on a light one.
        target = "#ffffff" if getattr(pal, "is_dark", True) else "#000000"
        for t in (0.25, 0.4, 0.55, 0.7, 0.85):
            lifted = color.mix(body, target, t)
            if color.contrast_ratio(lifted, page) >= 3.0:
                body = lifted
                edge = color.mix(edge, target, t * 0.7)
                break
        else:
            body, edge = color.mix(body, target, 0.85), color.mix(edge, target, 0.6)
    return QColor(body), QColor(edge), QColor(eye)


class CatSlider(QWidget):
    """A slider that is a black pixel cat walking a scrolling RGB trail —
    the overlay opacity control. API-compatible enough with QSlider for the
    dashboard: value()/setValue(), valueChanged(int), setRange, tooltips."""

    from PySide6.QtCore import Signal as _Signal

    valueChanged = _Signal(int)

    def __init__(self, lo: int = 30, hi: int = 100, parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtCore import QTimer

        self._lo, self._hi = lo, hi
        self._value = hi
        self._phase = 0
        self.setMinimumSize(180, 26)
        self.setCursor(Qt.PointingHandCursor)
        self._anim = QTimer(self)
        self._anim.setInterval(140)
        self._anim.timeout.connect(self._advance)

    def _advance(self) -> None:
        self._phase += 1
        self.update()

    def showEvent(self, event) -> None:
        self._anim.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._anim.stop()
        super().hideEvent(event)

    # ----------------------------------------------------- QSlider-ish API
    def setRange(self, lo: int, hi: int) -> None:
        self._lo, self._hi = lo, hi

    def value(self) -> int:
        return self._value

    def setValue(self, v: int) -> None:
        v = int(max(self._lo, min(self._hi, v)))
        if v != self._value:
            self._value = v
            self.valueChanged.emit(v)
            self.update()

    # -------------------------------------------------------------- mouse
    def _set_from_x(self, x: float) -> None:
        span = max(self.width() - 30, 1)
        frac = max(0.0, min(1.0, (x - 4) / span))
        self.setValue(round(self._lo + frac * (self._hi - self._lo)))

    def mousePressEvent(self, event) -> None:
        self._set_from_x(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self._set_from_x(event.position().x())

    # -------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        span = max(self.width() - 30, 1)
        frac = (self._value - self._lo) / max(self._hi - self._lo, 1)
        knob_x = 4 + frac * span
        mid_y = self.height() / 2

        # empty remainder of the track
        p.setPen(QColor(pal.border))
        p.drawLine(int(knob_x + 24), int(mid_y), self.width() - 2, int(mid_y))

        # the RGB trail behind the cat: scrolling dashed bands
        bh = 2.6
        y0 = mid_y - 3 * bh
        for i, hexcol in enumerate(_NYAN_BANDS):
            col = QColor(hexcol)
            col.setAlphaF(0.45 + 0.55 * frac)   # trail glows with opacity
            y = y0 + i * bh
            x = 4 - (self._phase * 3) % 10
            while x < knob_x - 2:
                w = min(6.0, knob_x - 2 - x)
                if x + w > 4 and w > 0:
                    p.fillRect(QRectF(max(x, 4.0), y, w, bh - 0.4), col)
                x += 10

        # the cat (2px pixel blocks, walking bob) — coat follows the accent
        cx = knob_x - 2
        cy = mid_y - 10 + (self._phase % 2)
        black, edge, eye = _cat_coat(pal)
        p.fillRect(QRectF(cx - 5, cy + 3, 5, 3), edge)           # tail
        p.fillRect(QRectF(cx - 6, cy + 1, 3, 3), edge)           # tail curl
        p.fillRect(QRectF(cx, cy + 6, 16, 9), edge)              # body edge
        p.fillRect(QRectF(cx + 1, cy + 7, 14, 7), black)         # body
        p.fillRect(QRectF(cx + 10, cy + 1, 10, 9), edge)         # head edge
        p.fillRect(QRectF(cx + 11, cy + 2, 8, 7), black)         # head
        p.fillRect(QRectF(cx + 11, cy - 1, 3, 3), black)         # ears
        p.fillRect(QRectF(cx + 17, cy - 1, 3, 3), black)
        p.fillRect(QRectF(cx + 13, cy + 4, 2, 2), eye)           # eyes
        p.fillRect(QRectF(cx + 17, cy + 4, 2, 2), eye)
        step = self._phase % 2
        for i in range(3):
            p.fillRect(QRectF(cx + 2 + i * 5 + step, cy + 15, 3, 3), black)


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

        # the cat, riding the tip (2px pixel blocks, gentle bob); its coat
        # follows the accent theme — indigo keeps the black cat
        cx = x0 + filled - 2
        cy = 1 + (self._phase % 2)
        body, edge, eye = _cat_coat(pal)
        p.fillRect(QRectF(cx - 4, cy + 8, 6, 4), edge)          # tail
        p.fillRect(QRectF(cx - 5, cy + 6, 3, 3), edge)          # tail curl
        p.fillRect(QRectF(cx, cy + 4, 16, 12), edge)            # body edge
        p.fillRect(QRectF(cx + 1, cy + 5, 14, 10), body)        # body
        p.fillRect(QRectF(cx + 12, cy + 2, 10, 10), edge)       # head edge
        p.fillRect(QRectF(cx + 13, cy + 3, 8, 8), body)         # head
        p.fillRect(QRectF(cx + 12, cy, 3, 3), body)             # ears
        p.fillRect(QRectF(cx + 19, cy, 3, 3), body)
        p.fillRect(QRectF(cx + 15, cy + 5, 2, 2), eye)          # eyes
        p.fillRect(QRectF(cx + 19, cy + 5, 2, 2), eye)
        for leg in range(4):
            p.fillRect(QRectF(cx + 2 + leg * 4, cy + 16, 3, 3), edge)
        self._draw_label(p, pal, x0 + track + 30)

    def _draw_label(self, p: QPainter, pal, x: float) -> None:
        if self._label:
            p.setPen(QColor(pal.fg_dim))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(x, 0, 280, self.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, self._label)
