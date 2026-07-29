"""The kovadapt opening: the ASCII eye wakes, blinks, and catches the light.

SplashScreen is a near-fullscreen dark stage (~94% of the primary screen's
AVAILABLE geometry, so the taskbar is excluded)
running a high-resolution build of gui/ascii_art.py's character stencil.
Choreography: a heartbeat gathering at the pupil almost from the first
frame — lightning ignition along seeded iris fibers — the glow escapes
the rim into lids, lashes and shading — the completed eye blinks once,
slow and deliberate — on reopen a specular flash — a slanted reflection
streak — glances across the whole eye, and the twin glints shimmer in
behind it before settling into a live twinkle while the eye breathes.
There is no crosshair in the opening (the reticle cells stay in the
stencil for static renders, excluded here by role).

The whole show runs ~5 s of wall clock. ascii_art owns the beat times and
the live backdrop shares that module, so the tightening is done here, on a
warped clock (_choreo_time) whose knots ARE those beats — every beat still
fires, in order; only the air between them is squeezed. Any key or click
skips (SplashScreen.skip): the eye snaps to its settled pose and fades.

Two crafts inform the blink. Blink dynamics: real blinks are temporally
asymmetric — a fast close, held full closure, then a slower cushioned
reopen, with the upper lid doing most of the travel (Trutoiu et al.,
"Modeling and Animating Eye Blinks", Disney Research/ACM TAP). ASCII
motion: terminal ASCIImation conveys movement by substituting characters
frame to frame on a fixed grid — so the lids advance as whole rows of
lid-skin glyphs with a heavy lash-silhouette row riding the moving margin
(ascii_art.blink_cells), not as a sliding erase-wedge.

Per-frame math is vectorized (numpy) and glyphs go through a per-character
QStaticText cache — the same trick the live backdrop uses — which is what
holds ~30 fps.

Which tier _configure actually selects depends on AVAILABLE height, so the
taskbar decides it. Measured against the real ladder: 1080p and 1440p with a
48 px taskbar both land on 255x121 (3.2 / 4.3 px glyphs); 4K lands on
381x181 at 4.3 px; only a 1440p stage with no taskbar reaches 381x181, and
then at 2.96 px, which is under the ~3 px floor where the density ramp stops
discriminating. The gate is 5.8 px per row for exactly that reason — it is a
legibility limit, not a performance one.

At 381x181 (25.7k eye cells, offscreen raster) a dark heartbeat frame is
~2.5 ms, the fill averages ~12 ms, and an all-lit frame is ~27-35 ms against
a 33 ms budget; the floor is the drawStaticText count, not the pacing. The
choreography reads its time off QElapsedTimer, so a late frame costs
smoothness, never timing.

The boot worker narrates real startup work in a status line; finish() only
fades once both the choreography and the boot work are done.

make_icon() renders the same character art for the window icon.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import (QElapsedTimer, QEvent, QPointF, QRect, QRectF, Qt,
                            QTimer)
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QIcon, QPainter,
                           QPen, QStaticText)
from PySide6.QtWidgets import QWidget

from . import ascii_art

_WORD = "kovadapt"
_BG = QColor("#0a0a0e")
_INK = QColor("#d8dae2")
_BG_F = np.array([_BG.redF(), _BG.greenF(), _BG.blueF()])
_INK_F = np.array([_INK.redF(), _INK.greenF(), _INK.blueF()])

_BLINK_END = (ascii_art.BLINK_T + ascii_art.BLINK_CLOSE
              + ascii_art.BLINK_HOLD + ascii_art.BLINK_OPEN)      # 5.85
_GLEAM_END = ascii_art.GLEAM_T + ascii_art.GLEAM_LEN + 0.30       # 7.00

# ---- the opening clock: wall seconds -> choreography seconds --------------
# ascii_art owns every beat time (_SPARK, PUPIL_T0/T1, BLINK_T, GLEAM_T,
# BREATHE_T, TOTAL) and the live backdrop shares that module, so tightening
# the opening cannot mean moving those constants. The splash instead drives
# the field on a WARPED clock: a piecewise-linear map whose knots are the
# beats themselves. Every beat still fires, in order, keeping its own
# internal shape — only the air between them is squeezed, and each beat gets
# its own rate, so the blink and the reflection flash surrender less of their
# length than the fill does. The tail runs at 1:1 (the breathing is a loop,
# not a beat, and warping it would leave a visible speed seam at the knot).
_WARP: tuple[tuple[float, float], ...] = (
    (0.00, 0.00),
    (0.95, ascii_art._SPARK),          # 1.15  heartbeat gathers, pupil ignites
    (1.85, ascii_art.PUPIL_T0),        # 2.60  bolts have raced the fibers
    (2.90, ascii_art.BLINK_T),         # 4.55  fill escapes the rim, echo, pupil
    (3.77, _BLINK_END),                # 5.85  the deliberate blink
    (3.87, ascii_art.GLEAM_T),         # 6.00  reflection flash starts
    (4.67, _GLEAM_END),                # 7.00  flash + glints in, eye settles
)

# The heartbeat is the opening's FIRST beat and it plays alone on a
# near-fullscreen black stage, so it has to actually read. It now starts
# essentially with the window (first glow ~0.12 s of wall clock, the pulse
# topping out ~0.36 s); at the old 0.25 s start and 0.14 amplitude the pupil
# glow reached ~15% of the way from bg to ink and the opening read as a black
# slab right up to the spark a second later.
_HEART_T0 = 0.06        # choreography seconds
_HEART_A = 0.22         # scales a lub-dub whose own peak is ~1.38


def _choreo_time(wall: float) -> float:
    """Choreography time for a wall-clock instant. Monotonic and continuous:
    _WARP is sorted in both columns, and past the last knot the show simply
    runs in real time."""
    if wall <= 0.0:
        return 0.0
    pw, pc = _WARP[0]
    for w, c in _WARP[1:]:
        if wall < w:
            return pc + (c - pc) * (wall - pw) / (w - pw)
        pw, pc = w, c
    return pc + (wall - pw)


def _heartbeat(t: float) -> float:
    """Lub-dub amplitude at the pupil center at choreography time t; 0 outside
    the heartbeat window. Two systoles a beat (the second offset by 1.1 rad
    and two-thirds the height), ramped in so the eye does not simply switch
    on. The ramp is 0.35 s, not the half second it was: a 0.5 s ramp still had
    the lub at 60% height when it peaked, which handed the dub the accent and
    pushed the beat the viewer actually reads past 0.45 s of wall clock."""
    if not _HEART_T0 < t < ascii_art._SPARK + 0.6:
        return 0.0
    u = t - _HEART_T0
    beat = (max(0.0, math.sin(u * 5.2))
            + 0.6 * max(0.0, math.sin(u * 5.2 - 1.1))) * min(u / 0.35, 1.0)
    return _HEART_A * beat


def make_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(ascii_art.render_pixmap(s))
    return icon


def _hsv_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB, all components 0..1, shape (n, 3) out."""
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6)
    f = h6 - i
    pp = v * (1.0 - s)
    qq = v * (1.0 - s * f)
    tt = v * (1.0 - s * (1.0 - f))
    i = i.astype(int) % 6
    r = np.choose(i, [v, qq, pp, pp, tt, v])
    g = np.choose(i, [tt, v, v, qq, pp, pp])
    b = np.choose(i, [pp, pp, tt, v, v, qq])
    return np.stack([r, g, b], axis=1)


class _EyeField:
    """Vectorized splash renderer over one stencil resolution: numpy holds
    every cell's ignition time and full-brightness color; each frame is a
    handful of array ops plus one drawStaticText per visible glyph."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        cells = ascii_art.stencil(cols, rows, subtract_glints=False)
        eye = [c for c in cells
               if c.role in ("iris", "outline", "lash", "shade")]
        gls = [c for c in cells if c.role == "glint"]
        half = cols / 2.0
        code = {"iris": 0, "outline": 1, "lash": 2, "shade": 3}
        n = len(eye)

        def arr(get, dtype=float):
            return np.fromiter((get(c) for c in eye), dtype=dtype, count=n)

        col = arr(lambda c: c.col)
        row = arr(lambda c: c.row)
        role = arr(lambda c: code[c.role], dtype=int)
        rad = arr(lambda c: c.rad)
        seed = arr(lambda c: c.seed)
        ink = arr(lambda c: c.ink)
        hue = arr(lambda c: c.hue)
        hue_j = arr(lambda c: c.hue_j)
        fleck = arr(lambda c: c.fleck)
        vj = arr(lambda c: c.vj)
        self._col, self._row = col, row
        self._x = (col - (cols - 1) / 2.0) / half
        self._y = (row - (rows - 1) / 2.0) * ascii_art._ASPECT / half
        self._rad, self._seed = rad, seed
        self._iris = role == 0
        self._row_h = 2.0 * ascii_art._ASPECT / cols
        self._rest_u = ascii_art._upper(
            np.clip(self._x, -ascii_art._W, ascii_art._W))

        # ---- ignition schedule (mirrors ascii_art.led_state) -------------
        # Still a mirror for the IGNITION; the heartbeat deliberately is not
        # (see _heartbeat). led_state's copy is only reachable through
        # paint_grid(t)/AsciiEye.set_time, which nothing drives with a real
        # time — the splash is the only animated consumer of the stencil.
        ang = hue * 2.0 * math.pi
        d_bolt = np.min(np.stack(
            [np.abs(np.remainder(ang - b + math.pi, 2.0 * math.pi) - math.pi)
             for b in ascii_art._BOLTS]), axis=0)
        near = d_bolt < 0.13
        lit = np.where(
            near, ascii_art._SPARK + rad * 0.22 + (d_bolt / 0.13) * 0.05,
            ascii_art._SPARK + 0.40 + rad * 0.95 + d_bolt * 0.5)
        reach = np.maximum(rad - 1.0, 0.0)
        for rc, base in ((1, 2.85), (2, 3.0), (3, 3.25)):
            lit = np.where(role == rc, base + reach * 0.55 + 0.1 * seed, lit)
        self._lit = lit
        self._bolt = self._iris & near

        # ---- full-brightness colors (the detailed iris) ------------------
        h_full = (hue + 0.05 * np.sin(rad * 6.0) + hue_j) % 1.0
        s_full = 0.72 * (0.72 + 0.28 * ink)
        v_full = np.clip(1.0 + vj * 0.5, 0.0, 1.0)
        has_fl = fleck >= 0.0
        h_full = np.where(has_fl, (h_full + fleck) % 1.0, h_full)
        s_full = np.where(has_fl, np.minimum(s_full * 1.15, 1.0), s_full)
        rgb = _hsv_rgb(h_full, s_full, v_full)
        alpha = np.where(role == 3, 0.28 + 0.45 * ink, 0.35 + 0.65 * ink)
        mono = _BG_F[None, :] + (_INK_F - _BG_F)[None, :] \
            * np.clip(alpha, 0.0, 1.0)[:, None]
        self._rgb = np.where(self._iris[:, None], rgb, mono)
        self._chs = [c.ch for c in eye]
        # per-cell darkening once the pupil has fully resolved; folded into
        # each frame's brightness so the spark/heartbeat stay lit through
        # the ignition (see ascii_art.pupil_mix)
        pdim = ascii_art.pupil_dim(rad)
        self._pupil_v = np.where(
            self._iris,
            ascii_art._PUPIL_V + (1.0 - ascii_art._PUPIL_V) * pdim, 1.0)

        # ---- the twin glints: born from the gleam sweep ------------------
        m = len(gls)
        gcol = np.fromiter((c.col for c in gls), dtype=float, count=m)
        self._gg = np.fromiter((c.ink for c in gls), dtype=float, count=m)
        self._gseed = np.fromiter((c.seed for c in gls), dtype=float, count=m)
        gx = ((gcol - (cols - 1) / 2.0) / half) / ascii_art._RI
        self._g_left = gx < 0.0
        xn = np.clip(gx, -1.3, 1.3)
        self._g_ign = ascii_art.GLEAM_T \
            + ascii_art.GLEAM_LEN * (xn + 1.3) / 2.6
        pos = {(int(cc), int(rr)): i
               for i, (cc, rr) in enumerate(zip(col, row))}
        self._g_under = np.fromiter(
            (pos.get((c.col, c.row), -1) for c in gls), dtype=int, count=m)
        self._gcells = gls

        # geometry + glyph caches, filled by prepare()
        self._cw = self._ch = 1.0
        self._st: dict[str, QStaticText] = {}
        self._pts: list[QPointF] = []
        self._sts: list[QStaticText] = []
        self._gpts: list[QPointF] = []
        self._gsts: list[QStaticText] = []
        self._lid_qc: dict[tuple[str, int], QColor] = {}
        # Every cell's origin, flat (r * cols + c). The lids can cover most of
        # a 381x181 grid, and building ~21k QPointF per blink frame cost more
        # than the drawing did — the blink was the only phase over the 33 ms
        # frame budget (~38 ms; ~28 ms with this). It is built in prepare(),
        # not lazily on the first blink frame: 69k QPointF is ~50 ms and 4 MB,
        # which is free before show() and a dropped frame mid-blink after.
        self._all_pts: list[QPointF] = []

    def _st_of(self, ch: str) -> QStaticText:
        st = self._st.get(ch)
        if st is None:
            st = QStaticText(ch)
            st.setTextFormat(Qt.PlainText)
            self._st[ch] = st
        return st

    def prepare(self, cell_w: float, cell_h: float) -> None:
        self._cw, self._ch = cell_w, cell_h
        self._all_pts = [QPointF(c * cell_w, r * cell_h)
                         for r in range(self.rows) for c in range(self.cols)]
        self._pts = [QPointF(c * cell_w, r * cell_h)
                     for c, r in zip(self._col, self._row)]
        self._sts = [self._st_of(ch) for ch in self._chs]
        self._gpts = [QPointF(c.col * cell_w, c.row * cell_h)
                      for c in self._gcells]
        self._gsts = [self._st_of(c.ch) for c in self._gcells]

    # ------------------------------------------------------------- frame
    def paint(self, p: QPainter, t: float) -> None:
        """One choreography frame at time t; painter already translated to
        the eye origin with the stencil font set."""
        rad, seed = self._rad, self._seed
        k = ascii_art.splash_blink(t)

        age = t - self._lit
        b = np.where(age <= 0.0, 0.0,
                     np.where(age < 0.22, 1.0 + 0.5 * (1.0 - age / 0.22), 1.0))
        if ascii_art._SPARK - 0.01 < t < ascii_art._SPARK + 1.0:
            strobe = self._bolt & (age > 0.0) & (age < 0.38)
            if strobe.any():
                flick = np.floor(age * 26.0) % 3 != 0
                b = np.where(strobe, np.where(flick, 1.55, 0.25), b)
        hbv = _heartbeat(t)
        if hbv > 0.0:
            # the lub-dub heartbeat gathering at the pupil before the spark
            hb = self._iris & (b <= 0.0) & (rad < 0.30)
            b = np.where(hb, hbv * (1.0 - rad / 0.30), b)
        if 3.0 < t < 3.9:                       # echo pulse across the iris
            b = b + np.where(
                self._iris,
                0.35 * np.exp(-((t - 3.05 - rad * 0.8) / 0.12) ** 2), 0.0)
        if t > ascii_art.BREATHE_T:
            b = b * (0.93 + 0.07 * np.sin(
                2.0 * math.pi * 0.35 * t + seed * 1.2))
        if k > 0.0:                             # the lids shade the eye
            b = b * (1.0 - 0.35 * k)
        np.clip(b, 0.0, 1.6, out=b)

        # the pupil contracts out of the light as the eye focuses: darkening
        # toward bg is the same operation as dimming, so it folds into bb
        p_mix = ascii_art.pupil_mix(t)
        bb = np.minimum(b, 1.0)
        if p_mix > 0.0:
            bb = bb * (1.0 - p_mix * (1.0 - self._pupil_v))
        bb = bb[:, None]
        rgb = _BG_F[None, :] + (self._rgb - _BG_F[None, :]) * bb
        over = np.clip(b - 1.0, 0.0, 0.5)[:, None]
        rgb = rgb + (1.0 - rgb) * over

        # ---- the reflection flash: a slanted specular streak glancing
        #      across the whole eye, sharp and quick like light catching a
        #      curved lens; the glints shimmer in behind it ---------------
        g_end = ascii_art.GLEAM_T + ascii_art.GLEAM_LEN + 0.30
        if ascii_art.GLEAM_T <= t <= g_end:
            prog = (t - ascii_art.GLEAM_T) / (g_end - ascii_art.GLEAM_T)
            env = math.sin(math.pi * min(prog, 1.0)) ** 0.7  # fast rise, linger
            band = -1.55 + prog * 3.1
            # slant the streak (u = x tilted by y) so it reads as a reflection
            # sliding across a domed surface, not a flat vertical wipe
            u = (self._x - 0.5 * self._y) / ascii_art._RI
            streak = env * np.exp(-((u - band) / 0.22) ** 2)
            on_eye = self._iris | (rad <= 1.14)   # iris + onto the rim
            boost = np.where(on_eye, streak, 0.0)
            rgb = rgb + (1.0 - rgb) * (0.95 * boost[:, None])

        visible = b > 0.02
        if k > 0.0:
            um, lm = ascii_art.lid_margins(self._x, k)
            covered = ((self._y <= um + 0.6 * self._row_h)
                       & (self._y >= self._rest_u + 0.01)) \
                | ((self._y >= lm - 0.6 * self._row_h)
                   & (self._y <= -self._rest_u - 0.01))
            visible &= ~covered

        # glint intensities (born left-to-right as the band crosses them)
        inten = None
        if t >= ascii_art.GLEAM_T:
            gage = t - self._g_ign
            ga = np.clip(gage / 0.12, 0.0, 1.0)
            ga = ga * np.where((gage > 0.0) & (gage < 0.30),
                               1.35 - 0.35 * (gage / 0.30), 1.0)
            tw = 0.86 + 0.14 * np.sin(
                2.0 * math.pi * 0.5 * t
                + np.where(self._g_left, 0.0, math.pi)) \
                + 0.05 * np.sin(2.0 * math.pi * 1.1 * t + self._gseed * 6.0)
            ga = ga * np.where(gage > 0.35, tw, 1.0)
            inten = np.clip(ga * (0.55 + 0.45 * self._gg), 0.0, 1.0)
            hide = self._g_under[(ga > 0.5) & (self._g_under >= 0)]
            visible[hide] = False

        # ---- draw: one cached QStaticText per glyph ----------------------
        q = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint32)
        packed = ((q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]).tolist()
        pts, sts = self._pts, self._sts
        for i in np.nonzero(visible)[0].tolist():
            p.setPen(QColor.fromRgb(packed[i]))
            p.drawStaticText(pts[i], sts[i])

        if k > 0.0:      # the character lids over the covered eye
            # locals: this loop runs up to ~21k times per blink frame
            pts_all, qcs, sts_all = self._all_pts, self._lid_qc, self._st
            cols = self.cols
            for cc, rr, ch, kind, shade in ascii_art.blink_cells(
                    k, cols, self.rows):
                key = (kind, int(shade * 31.9))
                qc = qcs.get(key)
                if qc is None:
                    a = 0.26 + 0.45 * shade if kind == "skin" \
                        else 0.40 + 0.50 * shade
                    qc = QColor.fromRgbF(
                        *(_BG_F + (_INK_F - _BG_F) * min(a, 1.0)))
                    qcs[key] = qc
                st = sts_all.get(ch)
                if st is None:
                    st = self._st_of(ch)
                p.setPen(qc)
                p.drawStaticText(pts_all[rr * cols + cc], st)

        if inten is not None:                   # the glints, on top
            for j in np.nonzero(inten > 0.05)[0].tolist():
                v = float(inten[j])
                p.setPen(QColor.fromRgbF(*(_BG_F + (1.0 - _BG_F) * v)))
                p.drawStaticText(self._gpts[j], self._gsts[j])


class SplashScreen(QWidget):
    """Frameless near-fullscreen ASCII stage. start() begins the show;
    finish(callback) lets it fade once the animation has played out."""

    # Wall clock, not choreography time: _WARP fits the whole show (heartbeat,
    # spark, fill, blink, gleam) into 4.67 s and the eye then breathes settled
    # for the remainder before the fade.
    MIN_SECONDS = 5.0

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)     # so a key press can skip
        self._t = 0.0               # wall seconds since start()
        self._ct = 0.0              # choreography seconds (warped)
        self._fade = 1.0
        self._done_cb = None
        self._status = ""
        self._skipped = False
        self._filtering = False
        self._frame_ms = 0.0        # rolling paint cost, for diagnostics
        self._timer = QTimer(self)
        self._timer.setInterval(33)             # the field targets ~30 fps
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None \
            else QRect(0, 0, 1280, 800)
        # near-fullscreen: the stage is the art, and a bigger stage is also
        # what buys the denser stencil tier its physical glyph size
        self._configure(int(geo.width() * 0.94), int(geo.height() * 0.94))
        self.move(geo.center().x() - self.width() // 2,
                  geo.center().y() - self.height() // 2)

    def _configure(self, w: int, h: int) -> None:
        """Size the dark stage and build the eye field at the densest
        stencil the stage height supports (small screens fall back to the
        shared 141x67 stencil, which is already cached).

        The gate is physical glyph size, not cell count: _ASPECT is 2, so a
        row pitch of R px makes each character R/2 px wide, and below ~3 px
        the density ramp stops discriminating — the art turns to dither
        rather than getting finer (measured across tiers; 2.2 px glyphs
        also lose ~25% luminance). 5.8 px/row keeps the densest tier just
        above that floor on a 1440p stage and comfortably above it on 4K.
        """
        self.setFixedSize(w, h)
        band = max(int(h * 0.16), 130)          # wordmark / status strip
        self._band_top = h - band
        eye_h = h - band - int(h * 0.05)
        rows = next((r for r in (181, 121, 95, 67) if eye_h / r >= 5.8), 67)
        cols = {181: 381, 121: 255, 95: 199, 67: 141}[rows]
        eye_w = int(eye_h * cols / (rows * ascii_art._ASPECT))
        if eye_w > int(w * 0.94):
            eye_w = int(w * 0.94)
            eye_h = int(eye_w * rows * ascii_art._ASPECT / cols)
        self._eye_rect = QRectF((w - eye_w) / 2.0, int(h * 0.04),
                                eye_w, eye_h)
        self._field = _EyeField(cols, rows)
        ch = eye_h / rows
        self._eye_font = ascii_art._mono()
        self._eye_font.setPixelSize(max(int(ch * 1.05), 3))
        self._field.prepare(eye_w / cols, ch)

    def start(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()       # a SplashScreen window is not activated
        self.setFocus(Qt.OtherFocusReason)
        app = QGuiApplication.instance()
        if app is not None:
            # Belt and braces: activation is a request the window manager may
            # refuse for a splash-flagged window, and a key press delivered
            # anywhere in the app must still skip.
            app.installEventFilter(self)
            self._filtering = True
        self._clock.start()
        self._timer.start()

    def set_status(self, text: str) -> None:
        """Boot-worker narration under the wordmark ('reading profiles…')."""
        self._status = text

    def finish(self, callback) -> None:
        self._done_cb = callback

    # ---------------------------------------------------------------- skip
    def skip(self) -> None:
        """End the opening now — the escape hatch for someone who has seen it.

        (Settings.skip_splash is the other lever: that one never shows the
        splash at all.) The clock jumps to the settled pose rather than
        cutting to black mid-blink, and the fade runs ~3x faster. If the boot
        worker has not called finish() yet the app cannot be revealed, so the
        settled eye simply holds until it does.
        """
        if self._skipped:
            return
        self._skipped = True
        self._t = max(self._t, self.MIN_SECONDS)
        self._ct = _choreo_time(self._t)
        self.update()

    def keyPressEvent(self, event) -> None:
        self.skip()

    def mousePressEvent(self, event) -> None:
        self.skip()

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.KeyPress, QEvent.MouseButtonPress):
            self.skip()
        return False            # never swallow: the splash owns no input

    def closeEvent(self, event) -> None:
        app = QGuiApplication.instance()
        if self._filtering and app is not None:
            app.removeEventFilter(self)
        self._filtering = False
        # The normal fade-out stops the timer, but closing or hiding the
        # splash by any other route left a 33 ms timer repainting an
        # invisible near-fullscreen widget — the one thing the motion budget
        # forbids outright.
        self._timer.stop()
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._done_cb is not None or self._t < self.MIN_SECONDS:
            self._timer.start()      # re-shown mid-show: resume, never restart

    def _tick(self) -> None:
        wall = self._clock.elapsed() / 1000.0 if self._clock.isValid() \
            else self._t + 0.033
        # once skipped the clock never walks back to an unfinished pose
        self._t = max(wall, self.MIN_SECONDS) if self._skipped else wall
        self._ct = _choreo_time(self._t)
        if self._done_cb is not None and self._t >= self.MIN_SECONDS:
            self._fade -= 0.34 if self._skipped else 0.12
            if self._fade <= 0.0:
                self._timer.stop()
                cb, self._done_cb = self._done_cb, None
                cb()          # show the main window BEFORE closing the last
                self.close()  # visible window, or the app would quit here
                return
        self.update()

    def paintEvent(self, event) -> None:
        # The splash is ALWAYS darkness — the eye is born out of black,
        # whatever theme the app itself uses.
        # Everything below is on the CHOREOGRAPHY clock, so the wordmark and
        # the tagline stay pinned to the beats they were cued against (the
        # word types over the fill, the tagline lands after the reopen)
        # instead of drifting when the pacing changes.
        t = self._ct
        t0 = time.perf_counter()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setOpacity(max(self._fade, 0.0))

        p.setPen(QPen(QColor("#23252e"), 1))
        p.setBrush(_BG)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 24, 24)

        p.save()
        p.translate(self._eye_rect.x(), self._eye_rect.y())
        p.setFont(self._eye_font)
        self._field.paint(p, t)
        p.restore()

        # wordmark types itself, terminal cursor blinking while it does
        w, h = self.width(), self.height()
        chars = _WORD[: max(0, int((t - 3.2) / 0.14))]
        word_px = max(int(h * 0.030), 24)
        if chars or t > 3.2:
            f = QFont(ascii_art._mono())
            f.setPixelSize(word_px)
            f.setWeight(QFont.DemiBold)
            p.setFont(f)
            cursor = "▌" if (len(chars) < len(_WORD)
                             and int(t * 3) % 2 == 0) else ""
            p.setPen(QColor("#e8e9ee"))
            p.drawText(QRectF(0, self._band_top + h * 0.008, w, word_px + 14),
                       Qt.AlignCenter, chars + cursor)
        if t > 5.9:
            a = min((t - 5.9) / 0.5, 1.0)
            col = QColor("#8e94a3")
            col.setAlphaF(a)
            p.setPen(col)
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(QRectF(0, self._band_top + h * 0.008 + word_px + 16,
                              w, 24), Qt.AlignCenter, "adaptive KovaaK's")
        if self._status:
            col = QColor("#8e94a3")
            col.setAlphaF(0.9)
            p.setPen(col)
            f = QFont(ascii_art._mono())
            f.setPixelSize(12)
            p.setFont(f)
            p.drawText(QRectF(0, h - 40, w, 20), Qt.AlignCenter, self._status)
        ms = (time.perf_counter() - t0) * 1000.0
        self._frame_ms = ms if self._frame_ms == 0.0 \
            else 0.9 * self._frame_ms + 0.1 * ms
