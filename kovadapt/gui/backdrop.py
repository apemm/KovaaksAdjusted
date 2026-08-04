"""Parallax ASCII backdrop: the eye, alive, behind the UI.

Three layers — a big ASCII eye off-center and two depths of drifting glyph
dust — slide at different rates toward the cursor (plus a slow ambient
wander). Everything except the iris is rendered once into a base
pixmap (and the reticle into a small overlay pixmap); each animation frame
Source-blits the base into a reusable frame pixmap, redraws only the few
hundred live iris glyphs, re-blits the reticle, and blinks: the eye's
interior is erase-composited between the resting lid parabolas and the
moving margins (ascii_art.blink_lid_paths), then the lids are drawn AS
CHARACTERS from ascii_art.blink_cells — the same visual style as the
opening's blink.

`set_session()` binds the eye to the live watch session as AMBIENT
CHARACTER — not as a readout, and the difference was measured rather than
guessed. Composite the frame the way paint() does (0.16 opacity over
pal.bg, 0.12 in light) and sample whole regions of the result:

    iris ring vs the page behind it      1.04:1 dark   1.02:1 light
    iris ring vs the pupil core          1.04:1 dark   1.07:1 light
    brightest 5% of iris pixels vs page  1.21:1 dark   1.00:1 light
    a 20% pupil vs a 99% pupil           1.00:1 dark   1.01:1 light
      ...its most favourable annulus     1.03:1 dark   1.03:1 light

WCAG's floor for a UI element is 3.0:1. Those numbers are not a defect to
be tuned out — they are what "faint enough to read text over" costs, and at
that contrast a pupil DIAMETER cannot carry a value, so this file does not
claim it does and no other surface may cite it. Raising the opacity until it
could would turn a backdrop into foreground noise, and binding accuracy to
the iris hue instead would break the accent lock the theme tests pin: there
is no version of this that is both perceptible and still a backdrop, so it
stays subtle and stays described as subtle. What the bindings do buy is a
room that responds to the session instead of looping obliviously:

* PUPIL — session accuracy moves the pupil's diameter through the same
  ascii_art.pupil_dim curve every static render uses, and fatigue makes it
  oscillate (hippus). Both directions are borrowed from real oculomotor
  behaviour rather than invented — effort dilates a pupil (the task-evoked
  pupillary response), sleepiness destabilizes it (the pupillographic
  sleepiness test) — and both land under the threshold at which a user could
  say which way they went. This file used to advertise the pupil as the
  app's live indicator for the adaptation loop. It never was one.
* BLINK — with live state the loop's once-per-8s blink is replaced by a
  stochastic event track: a refractory point process over the observed
  4-19 s interval band, fast-close/cushioned-open, its rate raised by
  fatigue and suppressed while watching (a visual task suppresses blinking).
  This is the largest thing the layer does: it swaps the whole iris for lid
  glyphs inside ~240 ms, which modulates the composited footprint's mean
  luminance by 1.10% against the pupil binding's 0.22% (dark; 0.18 vs 0.04%
  light) and does it as a moving edge rather than a static difference. Five
  times the pupil is still 1% on a layer nobody is looking at, so this is
  not a signal either — but it is the one thing here whose TIMING was worth
  getting right, which is what the section below is about.
* Nothing set: the eye behaves EXACTLY as it did before any of the above —
  a resting pupil and the deterministic loop blink.

Two guarantees, and it matters which one is in force:

* WITH NO SESSION STATE every rendered frame is a pure function of the
  quantized loop phase: every time-dependent term is periodic in
  ascii_art.LOOP_T, the colour table holds exactly one step per glyph frame
  of it, and the lid plan is quantized to _BLINK_STEPS closures. So
  frame(LOOP_T) is byte-identical to frame(0) and the eye keeps no state
  across a loop; tests/test_backdrop.py pins that and still passes
  unmodified. It is not a metronome, though — ticks are timed rather than
  counted (see _DT_MAX), so which phases a given run lands on varies.
* WITH SESSION STATE the blink track and the pupil ride a session clock, so
  the loop is no longer periodic — deliberately, since ambient life that
  repeats every 8 s is not responding to anything. What replaces the
  guarantee is reproducibility: the track is drawn from a fixed seed and
  consumed only at scheduling points, so the same state history always
  yields the same blinks.

Timing comes from gui/motion.py and nowhere else; ambient life runs only
while `motion.ambient(settings)`, and the timer only while the window it
paints is actually on screen.
"""

from __future__ import annotations

import math
import random
import weakref

import numpy as np
from PySide6.QtCore import (QElapsedTimer, QEvent, QObject, QPointF, QRectF,
                            Qt, QTimer)
from PySide6.QtGui import (QColor, QCursor, QFont, QFontMetricsF, QPainter,
                           QPixmap, QStaticText)

from . import ascii_art, motion, theme

_GLYPHS = ".:*+#@"
_LIVE_ROLES = ("iris",)
# height / width of the rendered stencil, from its grid and cell aspect
_EYE_ASPECT = ascii_art.ROWS * ascii_art._ASPECT / ascii_art.COLS

# --- clocks ----------------------------------------------------------------
# Parallax is continuous transform motion, so it ticks at twice the glyph
# rate and recomposes the character frame only when a glyph frame is due.
# Both come off motion.GLYPH_HZ; neither is a number chosen here.
_TICK_MS = motion.GLYPH_MS // 2
_DT = _TICK_MS / 1000.0
_GLYPH_DT = 2.0 * _DT

# A tick is worth the wall time that actually passed, not the interval it
# asked for — and it asks precisely. BOTH halves were needed, and neither
# was visible in code: Qt's default CoarseTimer delivered this 33 ms request
# every 46.9 ms (measured median on this machine: 21.3 Hz, not 30.3), and
# crediting each tick with a nominal 33 ms then ran the session clock at
# 0.71x real time. Every duration below was therefore a lie on screen — a
# 240 ms blink drew for 340 ms and a 300 ms one for 425 ms, both outside the
# 120-300 ms band the blink section cites as its authority, and the 4 s
# interval floor was really 5.7 s, capping the rate at 10.6/min rather than
# the 15/min it claims. PreciseTimer fixes the request; measuring the gap
# keeps the clock true anyway when the host is busy.
#
# One cap and no floor: a stall costs the eye some animation rather than
# fast-forwarding it through a blink, and everything shorter is taken at face
# value. There WAS a floor at one nominal interval, to keep programmatic
# _tick() calls advancing; it ran the clock 4% fast, because Qt delivers a
# compressed timeout on the first pass through the event loop and that
# 0.9 ms tick then got credited with 33 ms. _on_timeout draws the line where
# it actually is instead.
_DT_MAX = 4.0 * _DT

# The cursor follow was a flat 0.08 lerp per tick, which made the follow
# SPEED a function of the tick rate. As a time constant it is the same
# motion at _DT and stays that motion when a tick runs long.
_FOLLOW_TAU = -_DT / math.log(1.0 - 0.08)

# The loop's colours are quantized to exactly one step per glyph frame: at
# GLYPH_HZ no finer phase is distinguishable, and it turns the per-cell
# colour work — 2.9 of an uncached 4.0 ms frame — into a table lookup, so a
# composed frame settles at ~1.2 ms. Be honest about when that arrives: the
# table is built one step per frame ON DEMAND, so the first loop after a
# palette change still runs at ~4.3 ms/frame and only the loops after it are
# cheap. That window is not a regression to hunt — measured with the cache
# ripped out, a frame cost 3.95 ms, so the warm-up IS the old cost — but it
# is 8 s long, it lands on launch, and _ensure must not re-enter it for
# anything the table does not depend on. 3.1 MB per palette.
_STEPS = int(round(ascii_art.LOOP_T * motion.GLYPH_HZ))

# --- pupil -----------------------------------------------------------------
# Pupil radius as a fraction of the iris radius. ascii_art._PUPIL (0.30) is
# the resting size every static render uses; the two extremes bound the
# STENCIL, where the same curve is also seen at full strength (static
# renders, the splash): under ~0.22 the dark core stops reading as a pupil,
# over ~0.44 it eats the collarette and the iris reads as noise around a
# hole. At the backdrop's own opacity neither extreme is distinguishable
# from the other — see the module docstring; these bounds exist so the eye
# survives being rendered somewhere it IS visible.
_PUPIL_REST = ascii_art._PUPIL
_PUPIL_FOCUS = 0.225
_PUPIL_WIDE = 0.415
_PUPIL_STEPS = 500      # quantization of the radius; a step is 1/10 of the
#                         pupillary margin's own softness, so invisible
_REST_KEY = int(round(_PUPIL_REST * _PUPIL_STEPS))

# The accuracy band the dilation spans; outside it the mapping saturates
# rather than extrapolating, so a 5% run and a 40% one land in the same
# place instead of implying a resolution the eye does not have.
_ACC_WIDE, _ACC_FOCUS = 0.45, 0.85

# Pupillary unrest ("hippus"): a real 0.05-0.3 Hz oscillation whose
# amplitude grows with sleepiness — the basis of the pupillographic
# sleepiness test — so fatigue shows as an UNSTEADY pupil rather than a
# differently sized one. 5 s per cycle sits in that band, and far above
# motion.AMBIENT_MIN. Amplitude 0 when fatigue is unknown: no claim.
_HIPPUS_T = 5.0
_HIPPUS_REST, _HIPPUS_TIRED = 0.010, 0.038

# A new dilation settles over twice the ambient floor: inside the 1-3 s a
# real pupil takes to redilate, and slow enough that a run landing reads as
# the eye adjusting rather than as a UI state flip.
_PUPIL_SETTLE = 2.0 * motion.AMBIENT_MIN / 1000.0

# --- blink -----------------------------------------------------------------
# Spontaneous blinking is a point process with a refractory period, so the
# gap is MIN + Exponential(mean - MIN) clamped to the observed 4-19 s band.
# That 4 s floor caps the rate at 15/min, the bottom of the 15-20/min quoted
# for conversational rest — and someone staring at a monitor blinks slower
# than that anyway (screen-task studies measure ~5-10/min), which is why the
# resting mean here is 6.5 s (~9/min) rather than 3.5 s.
_GAP_MIN, _GAP_MAX = 4.0, 19.0
_GAP_REST = 6.5             # mean interval, idle
_GAP_WATCH = 9.0            # a demanding visual task suppresses blinking
_GAP_TIRED = 4.6            # fatigue raises the rate

# Blinks are temporally asymmetric — fast close, brief full closure,
# cushioned reopen (Trutoiu et al., Modeling and Animating Eye Blinks) —
# unlike the symmetric sin^2 bump of the fallback loop blink. At 240 ms that
# is an 82 ms close and a 134 ms reopen, which is what those measurements
# report; _lids_moving is what makes those numbers survive to the screen.
_CLOSE, _HOLD = 0.34, 0.10          # fractions of the blink; rest reopens
_BLINK_JITTER = (0.95, 1.10)        # per-blink duration spread

# Measured blink durations run 120-300 ms; every drawn blink stays inside
# that band. The RESTED end is motion.SLOW and the tired end is the band's
# own ceiling (an ocular datum, not a ladder rung — motion.ms only applies
# the user's intensity scale to it).
#
# The floor cannot be raised to buy legibility, so the SAMPLE RATE was. An
# 82 ms close is barely one glyph frame, and rendering the strip at every
# sub-tick alignment proved what that looks like: worst case a 240 ms blink
# got THREE lit frames, k = 0.90, 0.90, 0.23 — the eye jumped from wide open
# to nine tenths shut in one step and read as a dropped frame. That is the
# exact artifact the old comment claimed a 240 ms floor had fixed, and its
# "open, mid-close, shut, mid-open" never happened at any alignment. So
# _lids_moving samples the LID SWEEP at the tick rate instead of the glyph
# rate: a lid is a moving silhouette, not a character ramp crossing
# thresholds, so motion.GLYPH_HZ's reason for existing does not apply to it.
# The same worst case now draws seven — 0.36, 0.90, 1.00, 0.90, 0.59, 0.23,
# 0.01 — with two steps on the way down, a real full closure, and four
# cushioned steps back up.
_DUR_BAND = (0.12, 0.30)
_BLINK_SEED = 0x5EED1D              # fixed: the track is reproducible

# Watching used to lift the catchlights a touch. The glint role is gone (see
# ascii_art: the eye had two different irises depending on which caller asked
# for the stencil), so there is nothing left to lift and the lift is not
# reinstated on the iris. That is a real, if tiny, loss of one redundant cue
# — by the module docstring's own measurements it moved the composited
# footprint by a fraction of the blink's 1.10%, i.e. below anything a viewer
# could report — and the state is still carried by the pupil and the blink
# cadence, which is what the comment already said was doing the work.
_WATCH_WANDER = 0.5     # ambient wander shrinks while fixating on a task

_LID_LEVELS = 24        # lid alphas bucketed so one QPen serves hundreds
# Closure quantization for the lid memo. 32 steps put the lid margin within
# a single character row of its true position at every closure, and the whole
# table can be prefetched inside two seconds of open-eye frames.
_BLINK_STEPS = 32
# Activation belongs here with the rest. This app's whole purpose is to sit
# behind a game you are playing, and a window you alt-tabbed away from is
# still `isVisible()` to Qt — so the ambient layer went on animating a
# full-window parallax at 21Hz while the user was in Valorant. Measured on
# this machine: 23% of a core at motion=full, 275 QLabel repaints a second
# cascading off one backdrop tick.
_VISIBILITY = frozenset((QEvent.Type.Show, QEvent.Type.Hide,
                         QEvent.Type.WindowStateChange,
                         QEvent.Type.WindowActivate,
                         QEvent.Type.WindowDeactivate))


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


class _WindowEvents(QObject):
    """Show/hide/minimize relay for the window the backdrop paints into.

    A timer must never tick while the widget it animates is off screen, and
    a plain (non-QObject) helper cannot receive the window's own showEvent /
    hideEvent — so the backdrop watches for them instead of polling.
    """

    def __init__(self, backdrop: Backdrop, parent) -> None:
        super().__init__(parent)
        # WEAK, deliberately: this filter is parented to the window and the
        # window is reachable from the backdrop, so a strong ref here closes
        # a reference cycle — and the cyclic collector clears cycle members
        # in arbitrary order, so Qt delivers the dying window's Hide into a
        # half-cleared Backdrop. That crashed.
        self._bd = weakref.ref(backdrop)

    def eventFilter(self, obj, event) -> bool:
        if event.type() in _VISIBILITY:
            bd = self._bd()
            if bd is not None:
                try:
                    if event.type() == QEvent.Type.WindowDeactivate:
                        bd._active = False
                    elif event.type() in (QEvent.Type.WindowActivate,
                                          QEvent.Type.Show):
                        # Show counts as coming to the front. Hiding a window
                        # also emits Deactivate, so without this a hide/show
                        # cycle left the flag stuck false and the backdrop
                        # never came back.
                        bd._active = True
                    bd._sync_timer()
                except RuntimeError:
                    pass    # teardown order: the window's C++ side can
                    #         already be gone while Qt still delivers here
        return False


class Backdrop:
    """Owns the layers, the parallax/loop state and the live session state
    for one window. The window calls paint(painter) from its paintEvent and
    notify_theme()/notify_resize() when those change; whoever owns the watch
    session calls set_session() to let the eye respond to it."""

    def __init__(self, window, settings=None) -> None:
        self._win = window
        # Motion intensity is read off this at USE time, never cached. None
        # is a valid value: motion.level(None) is FULL, which is exactly the
        # behaviour the backdrop had before it took a Settings at all.
        self._s = settings
        self._base: QPixmap | None = None      # eye minus iris/reticle
        self._overlay: QPixmap | None = None   # reticle + hub, re-blit on top
        self._frame: QPixmap | None = None     # base + live cells, per frame
        self._live: list[tuple[QPointF, QStaticText, ascii_art.Cell]] = []
        self._cell_pt: list[QPointF] = []             # one per grid cell
        self._blink_st: dict[str, QStaticText] = {}    # lid glyphs, by char
        self._blink_cache: dict[int, tuple] = {}       # lid draw plan, by k
        self._warm = 1          # next lid plan to prefetch
        self._rgba: dict[int, np.ndarray] = {}         # loop colours, by step
        self._rgba_key: tuple | None = None            # palette the table is of
        self._pens: list[QColor] = []                  # this frame's pens
        self._pen_key: tuple | None = None
        self._font: QFont | None = None
        self._ink = QColor("#ffffff")
        self._iris_hue: float | None = None
        self._is_dark = True
        self._dust_far: QPixmap | None = None
        self._dust_near: QPixmap | None = None
        self._off = [QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)]
        self._phase = 0.0
        self._loop = 0.0        # live-eye loop clock, wraps at LOOP_T
        self._clock = 0.0       # session clock: only advances while animating
        self._since = 0.0       # wall seconds since the last composed frame
        self._el = QElapsedTimer()      # monotonic: what a tick is worth
        self._timed = False     # inside _on_timeout, i.e. the timer's tick
        self._ambient = True    # last known motion.ambient(), for the settle

        # --- live session state; all None/False = nothing bound -----------
        self._acc: float | None = None
        self._fatigue: float | None = None
        self._watching = False
        self._session = False
        self._pupil = _PUPIL_REST
        self._pupil_key = -1
        self._rads: np.ndarray | None = None      # per live cell
        self._iris_mask: np.ndarray | None = None
        self._fs: np.ndarray | None = None        # saturation factor per cell
        self._fv: np.ndarray | None = None        # value factor per cell
        self._rng: random.Random | None = None
        self._blink_at = math.inf
        self._blink_len = 0.0

        self._timer = QTimer(window)
        # PreciseTimer, not the default: see _DT_MAX. A coarse 33 ms request
        # came back every 46.9 ms, and the whole eye then ran at 0.71x speed.
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_timeout)
        # Assumed active until Qt says otherwise, rather than polled through
        # isActiveWindow(): a window that has never been activated at all —
        # every headless and offscreen context — reports False there, which
        # would freeze the backdrop in exactly the places it is measured.
        # The DEACTIVATION event is the reliable signal, and it is the one
        # that actually matters: alt-tabbing to the game.
        self._active = True
        self._events = _WindowEvents(self, window)
        window.installEventFilter(self._events)
        self._sync_timer()

    # ------------------------------------------------------------------
    def _cfg(self):
        """The Settings that drives motion: the one handed in, else the
        window's own. The backdrop is constructed BY the window, so that is
        its owner's setting rather than a global reached for."""
        return self._s if self._s is not None else getattr(self._win, "s", None)

    def _sync_timer(self) -> None:
        # ON SCREEN, IN FRONT, and asked for. `motion=off` used to leave this
        # timer running at 30Hz with a no-op tick: cheaper than full, but
        # "off" should mean the frames stop, not that they cost less.
        live = (self._win.isVisible()
                and not self._win.isMinimized()
                and self._active
                and motion.ambient(self._cfg()))
        if live and not self._timer.isActive():
            # Start the measurement WITH the timer, so even the first tick
            # measures a real gap. Deferring it to that first tick credited it
            # with a nominal interval instead, which is up to 33 ms of invented
            # time — invisible over a session, 2.5% over the first second.
            self._el.start()
            self._timer.start()
        elif not live and self._timer.isActive():
            self._timer.stop()
            self._el.invalidate()   # the eye is not credited with the time it
            #                         spends off screen

    def _on_timeout(self) -> None:
        """The timer's own slot, and the only place a tick counts as timed.

        _tick is also called directly — that is how the tests drive the eye
        without an event loop — and from inside _tick the two are
        indistinguishable, which is exactly the trap: a real tick delivered
        0.9 ms after its predecessor looks like a programmatic one. Marking
        the timer's path is the honest discriminator; guessing from the size
        of the gap was what ran the clock 4% fast.
        """
        self._timed = True
        try:
            self._tick()
        finally:
            self._timed = False

    def _tick_dt(self) -> float:
        """Wall seconds this tick stands for, capped at _DT_MAX.

        Timed, not assumed — that is the whole point (see _DT_MAX). A tick
        that did not come from the timer, or the first one after it starts,
        has nothing to measure and is worth one interval.
        """
        if not self._timed or not self._el.isValid():
            self._el.start()
            return _DT
        dt = self._el.nsecsElapsed() / 1e9
        self._el.start()
        return min(dt, _DT_MAX)

    # ------------------------------------------------------ public: session
    def set_session(self, accuracy: float | None = None,
                    fatigue: float | None = None,
                    watching: bool = False) -> None:
        """Bind the eye to the live session. GUI thread only.

        Each call is a complete SNAPSHOT, not a patch: an omitted value means
        "not known", which is the honest state before run 2 exists (every
        EWMA is seeded to the first run's own value, so there is no accuracy
        to speak of yet). Passing nothing at all — or nothing knowable — puts
        the eye back on its deterministic loop.

        None of these is legible as a quantity at the opacity paint() uses
        (the module docstring has the measurements); they change the eye's
        behaviour, not its readings.

        accuracy  0..1 hit rate; low dilates the pupil, high focuses it.
        fatigue   0..1; raises the blink rate, lengthens each blink, and
                  grows the pupil's hippus. Never sizes the pupil: fatigue
                  does not tonically dilate one.
        watching  a watch session is running: blinking is suppressed the way
                  a visual task suppresses it, and the catchlights lift.
        """
        acc = None if accuracy is None else float(min(max(accuracy, 0.0), 1.0))
        fat = None if fatigue is None else float(min(max(fatigue, 0.0), 1.0))
        watch = bool(watching)
        live = acc is not None or fat is not None or watch
        if (acc, fat, watch) == (self._acc, self._fatigue, self._watching):
            return
        self._acc, self._fatigue, self._watching = acc, fat, watch
        if live and self._rng is None:
            # a fixed seed: the "random" blink track is reproducible, which
            # is what replaces the loop's byte-identity guarantee
            self._rng = random.Random(_BLINK_SEED)
            self._schedule_blink()
        elif not live:
            self._rng = None
            self._blink_at = math.inf
        self._session = live
        if not motion.ambient(self._cfg()):
            self._pupil = self._pupil_goal()     # no ambient: end state now
        if self._refresh_pupil(force=True) and self._frame is not None:
            self._render_frame()
        self._win.update()

    # ------------------------------------------------------------- pupil
    def _pupil_goal(self) -> float:
        """Target radius from accuracy alone. Unknown accuracy holds the
        resting pupil — the eye must not imply a number it does not have."""
        if self._acc is None:
            return _PUPIL_REST
        u = (self._acc - _ACC_WIDE) / (_ACC_FOCUS - _ACC_WIDE)
        u = motion.ease_in_out(u)
        return _PUPIL_WIDE + (_PUPIL_FOCUS - _PUPIL_WIDE) * u

    def _pupil_now(self) -> float:
        r = self._pupil
        if self._fatigue is not None and motion.ambient(self._cfg()):
            amp = _HIPPUS_REST \
                + (_HIPPUS_TIRED - _HIPPUS_REST) * self._fatigue
            r += amp * math.sin(2.0 * math.pi * self._clock / _HIPPUS_T)
        return min(max(r, 0.08), 0.85)

    def _refresh_pupil(self, force: bool = False) -> bool:
        """Rebuild the per-cell pupil factors when the radius has moved a
        visible step. True when the composed frame is now stale.

        The factors turn ascii_art's fixed-radius pupil into a variable one
        without touching the stencil, the roles or ascii_art itself: the
        colours loop_cell_color returns are already scaled by pupil_dim at
        the RESTING radius, so each cell needs the ratio between the resting
        and wanted dim terms. pupil_dim bakes its radius in, so the wanted
        term comes from shifting its INPUT rather than reimplementing its
        sigmoid — pupil_dim(rad + (_PUPIL - r)) is the same curve centred on
        r. Saturation and value carry their own floors (_PUPIL_S/_PUPIL_V),
        hence two factors.
        """
        key = int(round(self._pupil_now() * _PUPIL_STEPS))
        if key == self._pupil_key and not force:
            return False
        self._pupil_key = key
        if self._rads is None:
            return False                # before _ensure(): nothing to build
        r = key / _PUPIL_STEPS
        # Identity is compared on the quantized KEY, not on the float: a
        # resting pupil must take the untouched code path exactly, and
        # key == _REST_KEY stays true whatever ascii_art._PUPIL becomes.
        if not self._session or key == _REST_KEY:
            self._fs = self._fv = None          # identity: the untouched path
        else:
            dim0 = ascii_art.pupil_dim(self._rads)
            want = ascii_art.pupil_dim(self._rads + (_PUPIL_REST - r))
            s0, v0 = ascii_art._PUPIL_S, ascii_art._PUPIL_V
            fs = (s0 + (1.0 - s0) * want) / (s0 + (1.0 - s0) * dim0)
            fv = (v0 + (1.0 - v0) * want) / (v0 + (1.0 - v0) * dim0)
            # only the iris is a ring around a core (see _iris_mask)
            self._fs = np.where(self._iris_mask, fs, 1.0)[:, None]
            self._fv = np.where(self._iris_mask, fv, 1.0)[:, None]
        self._pen_key = None                    # pens are stale
        return True

    def _advance_pupil(self, dt: float = _GLYPH_DT) -> None:
        """Ease the pupil `dt` wall seconds toward its goal. The caller passes
        the interval it actually measured, so the settle takes _PUPIL_SETTLE
        of real time whatever rate the frames arrive at; the default is one
        glyph frame's worth."""
        goal = self._pupil_goal()
        a = 1.0 - math.exp(-dt / (_PUPIL_SETTLE / 3.0))
        self._pupil += (goal - self._pupil) * a
        self._refresh_pupil()

    # ------------------------------------------------------------- blink
    def _schedule_blink(self) -> None:
        """Draw the next blink's start and duration off the seeded track."""
        rng = self._rng
        fat = self._fatigue or 0.0
        mean = _GAP_WATCH if self._watching else _GAP_REST
        mean += (_GAP_TIRED - mean) * fat
        gap = _GAP_MIN + rng.expovariate(1.0 / max(mean - _GAP_MIN, 0.2))
        self._blink_at = self._clock + min(gap, _GAP_MAX)
        lo, hi = _BLINK_JITTER
        rung = motion.SLOW + (_DUR_BAND[1] * 1000.0 - motion.SLOW) * fat
        # motion.ms applies the user's intensity scale; the band then keeps
        # the result inside measured human blink durations whatever that scale
        # did, which is also what _lids_moving needs to draw it legibly. Scale
        # 0 is motion off, which means no ambient life at all — leave the
        # length at zero rather than clamping a non-blink up to 120 ms, and
        # let _tick redraw it if ambient ever resumes.
        dur = motion.ms(self._cfg(), rung) / 1000.0 \
            * (lo + (hi - lo) * rng.random())
        self._blink_len = (min(max(dur, _DUR_BAND[0]), _DUR_BAND[1])
                           if dur > 0.0 else 0.0)

    def _advance_blink(self) -> None:
        if self._rng is not None \
                and self._clock >= self._blink_at + self._blink_len:
            self._schedule_blink()

    def _lids_moving(self) -> bool:
        """Whether a lid is in flight this tick — the frames that get sampled
        at the tick rate rather than the glyph rate (see _DUR_BAND).

        One tick of lead time, so the first step of the close is a real
        intermediate position instead of the jump the strip showed.
        """
        if not motion.ambient(self._cfg()):
            return False
        if self._rng is None:
            return (ascii_art.blink_amount(self._loop) > 0.0
                    or ascii_art.blink_amount(self._loop + _DT) > 0.0)
        if self._blink_len <= 0.0:
            return False
        return -_DT <= self._clock - self._blink_at < self._blink_len

    def _blink_k(self) -> float:
        """Closure 0 (open) .. 1 (shut) for this frame."""
        if not motion.ambient(self._cfg()):
            return 0.0                          # end state: an open eye
        if self._rng is None:
            return ascii_art.blink_amount(self._loop)   # the pinned loop
        if self._blink_len <= 0.0:
            return 0.0
        u = (self._clock - self._blink_at) / self._blink_len
        if u <= 0.0 or u >= 1.0:
            return 0.0
        if u < _CLOSE:
            return motion.ease_in_out(u / _CLOSE)
        if u < _CLOSE + _HOLD:
            return 1.0
        return 1.0 - motion.ease_in_out(
            (u - _CLOSE - _HOLD) / (1.0 - _CLOSE - _HOLD))

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
        # Size the eye to fit BOTH axes. A flat 0.62*w was drawn from a
        # top-left at 0.50*w, so 12% of it always hung off the right edge —
        # and at 1600x1000 its 0.95 aspect also ran 160px past the bottom.
        # The art was clipped through the outer lash on two sides at every
        # window size, which reads as a rendering fault rather than a crop.
        eye_w = int(min(w * 0.66, h * 0.80 / _EYE_ASPECT))
        self._eye_w = eye_w
        self._eye_h = eye_w * _EYE_ASPECT
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
        # One QPointF per grid cell, shared by the live glyphs and the lid
        # glyphs: constructing them per frame measured 0.2 us each, which at
        # ~3000 lid cells was most of a 240 Hz frame on its own.
        self._cell_pt = [QPointF(c * cw, r * ch)
                         for r in range(ascii_art.ROWS)
                         for c in range(ascii_art.COLS)]
        # QStaticText caches each glyph's layout once: ~9x faster per frame
        # than drawText(rect, flags) while anchoring the same top-left corner
        self._live = []
        rads, iris = [], []
        for c in ascii_art.stencil():
            if c.role not in _LIVE_ROLES:
                continue
            st = QStaticText(c.ch)
            st.setTextFormat(Qt.PlainText)
            self._live.append(
                (self._cell_pt[c.row * ascii_art.COLS + c.col], st, c))
            rads.append(c.rad)
            iris.append(c.role == "iris")
        self._rads = np.asarray(rads, dtype=float)
        # Every live cell is an iris cell now that the glint role is gone, so
        # this mask is all-True. It is kept rather than dropped because it is
        # what _refresh_pupil uses to say "the pupil factors apply to the ring
        # around the core, not to everything live" — a distinction that comes
        # back the moment any second live role does.
        self._iris_mask = np.asarray(iris, dtype=bool)
        # The lid plans hold pixel geometry and cell origins, so they die with
        # the size. The COLOUR table does not: loop_cell_color reads the
        # stencil, which is a fixed COLS x ROWS grid, so the table is a pure
        # function of the palette. Clearing it here regardless made every
        # delivered resize event re-enter the whole warm-up window — 516 ms of
        # work across the following loop against 197 ms now, and a window drag
        # delivers dozens — for a table that had not changed. Verified by
        # rendering: reusing it is byte-identical to a backdrop built fresh at
        # the new size, which is the thing that would break if the live-cell
        # order the table is indexed by ever stopped being stable.
        key = (self._is_dark, self._iris_hue, self._ink.rgba(), len(self._live))
        if key != self._rgba_key:
            self._rgba.clear()
            self._rgba_key = key
        self._blink_cache.clear()
        self._blink_st.clear()
        self._warm = 1
        self._pen_key = None
        self._refresh_pupil(force=True)
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
    def _rgba_at(self, step: int) -> np.ndarray:
        """The loop's per-cell colours at a quantized phase step, as an
        (n, 4) float64 RGBA table. Built once per step per palette — the
        first loop pays what every frame used to, and after that the whole
        colour pass is a lookup. float64 round-trips QColor's 16-bit
        channels exactly, so the recomposed pens are bit-identical to what
        loop_cell_color returned."""
        arr = self._rgba.get(step)
        if arr is None:
            phase = step * (ascii_art.LOOP_T / _STEPS)
            # getRgbF() in one call and one asarray at the end, rather than
            # four accessors and a per-row numpy store: same float64 values
            # bit for bit, 2.99 -> 2.77 ms per step. The remaining 2.5 ms is
            # loop_cell_color itself and belongs to ascii_art.
            arr = np.asarray([
                ascii_art.loop_cell_color(
                    cell, phase, is_dark=self._is_dark,
                    iris_hue=self._iris_hue, ink=self._ink).getRgbF()
                for _pt, _st, cell in self._live])
            self._rgba[step] = arr
        return arr

    def _pen_list(self, step: int) -> list[QColor]:
        """This frame's pen per live cell: the loop colour, then the pupil
        factors. Memoized on (step, pupil) — a steady pupil makes a whole
        loop of frames free.

        `watching` is no longer part of the key: it only ever entered the
        pens through the glint catchlight lift, and with the glint role gone
        it changes nothing here. Leaving it in would throw the whole pen
        cache away every time a watch session started or stopped, for an
        identical result."""
        key = (step, self._pupil_key if self._session else -1)
        if key == self._pen_key:
            return self._pens
        rgba = self._rgba_at(step)
        if not self._session:
            pens = [QColor.fromRgbF(r, g, b, a) for r, g, b, a in rgba.tolist()]
        else:
            rgb = rgba[:, :3]           # a VIEW on the cached table
            if self._fs is not None:
                # scaling HSV value is a uniform RGB scale; scaling
                # saturation is a lerp toward V, since each channel is
                # V - V*S*k. Both without an HSV round trip per cell. The
                # result must be a NEW array — writing through the view
                # would poison the cached colours for every later frame.
                v = rgb.max(axis=1, keepdims=True)
                rgb = np.clip((v - (v - rgb) * self._fs) * self._fv, 0.0, 1.0)
            alpha = rgba[:, 3]
            pens = [QColor.fromRgbF(r, g, b, a)
                    for (r, g, b), a in zip(rgb.tolist(), alpha.tolist())]
        self._pen_key, self._pens = key, pens
        return pens

    def _blink_plan(self, k: float):
        """(erase wedges, [(alpha, points, glyphs), ...]) at closure k,
        memoized on a quantized k.

        blink_cells + blink_lid_paths rebuild the whole lid geometry from
        scratch (1.2 ms), and drawing the result cell by cell — a QPointF
        and a QPen each — measured another 2.1 ms at mid-closure. Together
        that was a dropped 240 Hz frame every blink, forever. Quantizing the
        closure caps the table at _BLINK_STEPS entries, and bucketing the
        lid alphas to _LID_LEVELS turns thousands of setPen calls into a
        couple of dozen. A live blink is only 3-5 glyph frames long, so the
        table is prefetched on open frames (_warm_lids) rather than paid for
        while the lids are moving.
        """
        # never quantize DOWN to zero: the caller only asks when the lids are
        # moving, and blink_lid_paths returns None at k == 0
        key = max(int(min(k, 1.0) * _BLINK_STEPS + 0.5), 1)
        plan = self._blink_cache.get(key)
        if plan is None:
            kq = key / _BLINK_STEPS
            pm = self._frame
            wedges, _edges = ascii_art.blink_lid_paths(
                QRectF(0, 0, pm.width(), pm.height()), kq)
            buckets: dict[int, tuple[list, list]] = {}
            for col, row, chr_, kind, shade in ascii_art.blink_cells(kq):
                a = min(0.22 + 0.40 * shade if kind == "skin"
                        else 0.30 + 0.55 * shade, 1.0)
                pts, sts = buckets.setdefault(
                    int(a * (_LID_LEVELS - 1) + 0.5), ([], []))
                pts.append(self._cell_pt[row * ascii_art.COLS + col])
                st = self._blink_st.get(chr_)
                if st is None:
                    st = QStaticText(chr_)
                    st.setTextFormat(Qt.PlainText)
                    self._blink_st[chr_] = st
                sts.append(st)
            plan = (wedges, [(lvl / (_LID_LEVELS - 1), pts, sts)
                             for lvl, (pts, sts) in sorted(buckets.items())])
            self._blink_cache[key] = plan
        return plan

    def _warm_lids(self) -> None:
        """Build ONE missing lid plan. Driven from _tick, and only while the
        eye is open: an open frame costs ~1.2 ms, so a 1.4 ms plan build still
        fits the tick budget many times over, whereas the same build during a
        blink lands on a frame that is already the most expensive one there
        is. The first blink is at least four seconds out either way, so by the
        time the lids move every closure is in the table.

        Deliberately NOT called from _render_frame: that would put a build on
        the first paint, which happens inside MainWindow's construction, and
        on every set_session() and theme switch. Prefetch rides the
        animation's own frames or it does not happen.
        """
        while self._warm <= _BLINK_STEPS and self._warm in self._blink_cache:
            self._warm += 1
        if self._warm <= _BLINK_STEPS:
            self._blink_plan(self._warm / _BLINK_STEPS)
            self._warm += 1

    def _render_frame(self) -> None:
        """Compose one live frame at the current loop phase: base blit, the
        live iris glyphs, the reticle overlay, the blink wedges."""
        pm = self._frame
        step = int(round(self._loop / ascii_art.LOOP_T * _STEPS)) % _STEPS
        pens = self._pen_list(step)
        p = QPainter(pm)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawPixmap(0, 0, self._base)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setFont(self._font)
        for (pt, st, _cell), col in zip(self._live, pens):
            p.setPen(col)
            p.drawStaticText(pt, st)
        p.drawPixmap(0, 0, self._overlay)
        k = self._blink_k()
        if k > 0.0:
            wedges, groups = self._blink_plan(k)
            # the lids: erase the eye's interior between the resting lid
            # curve and the moving margin, then draw the lids AS CHARACTERS
            # — the same lid-skin rows + lash silhouettes as the opening
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.fillPath(wedges, QColor(0, 0, 0))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            lid = QColor(self._ink)
            for alpha, pts, sts in groups:
                lid.setAlphaF(alpha)
                p.setPen(lid)
                for pt, st in zip(pts, sts):
                    p.drawStaticText(pt, st)
        p.end()

    # ------------------------------------------------------------------
    def _settle(self) -> None:
        """Jump to the END STATE, for when ambient motion is switched off
        mid-flight: the resting open eye, no parallax offset, the pupil
        already where the session says it belongs (a dilation is
        information, not decoration, so it survives motion being off)."""
        self._loop = 0.0
        self._off = [QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)]
        self._pupil = self._pupil_goal()
        self._since = 0.0
        self._el.invalidate()           # a frozen spell is not elapsed time
        if self._rng is not None:
            self._schedule_blink()      # resume clean, not mid-blink
        self._refresh_pupil(force=True)
        if self._frame is not None:
            self._render_frame()
        self._win.update()

    def _tick(self) -> None:
        if not self._win.isVisible() or self._win.isMinimized():
            self._sync_timer()  # belt and braces: the filter should have
            return              # stopped us before the window went away
        if not motion.ambient(self._cfg()):
            if self._ambient:               # just switched off
                self._ambient = False
                self._settle()
            return
        if not self._ambient:
            self._ambient = True
            self._el.invalidate()
            if self._rng is not None:
                self._schedule_blink()  # redraw it under the live intensity
        dt = self._tick_dt()
        self._phase += dt
        self._clock += dt
        self._loop = (self._loop + dt) % ascii_art.LOOP_T
        pos = self._win.mapFromGlobal(QCursor.pos())
        w, h = max(self._win.width(), 1), max(self._win.height(), 1)
        # -0.5..0.5 relative cursor position, clamped when outside the window
        rx = max(-0.5, min(0.5, pos.x() / w - 0.5))
        ry = max(-0.5, min(0.5, pos.y() / h - 0.5))
        # a fixating eye wanders less: the ambient drift shrinks on task
        wander = _WATCH_WANDER if self._watching else 1.0
        drift_x = math.sin(self._phase * 0.11) * 6.0 * wander
        drift_y = math.cos(self._phase * 0.07) * 5.0 * wander
        targets = (
            QPointF(-rx * 18 + drift_x, -ry * 12 + drift_y),        # eye
            QPointF(rx * 10 + drift_x * 0.4, ry * 8 + drift_y * 0.4),   # far dust
            QPointF(-rx * 34 - drift_x, -ry * 26 - drift_y),        # near dust
        )
        moved = False
        follow = 1.0 - math.exp(-dt / _FOLLOW_TAU)
        for i, tgt in enumerate(targets):
            cur = self._off[i]
            nxt = QPointF(cur.x() + (tgt.x() - cur.x()) * follow,
                          cur.y() + (tgt.y() - cur.y()) * follow)
            if abs(nxt.x() - cur.x()) + abs(nxt.y() - cur.y()) > 0.05:
                moved = True
            self._off[i] = nxt
        # Glyph work is TIMED at motion.GLYPH_HZ rather than counted every
        # other tick, so a late tick cannot slow the character animation down;
        # a moving lid overrides it and composes every tick (see _DUR_BAND).
        self._since += dt
        lids = self._lids_moving()
        animated = False
        if lids or self._since >= _GLYPH_DT - 1e-9:
            if self._frame is not None:
                self._advance_pupil(self._since)
                self._advance_blink()
                self._render_frame()
                animated = True
                if not lids:
                    self._warm_lids()   # prefetch only on open frames
            self._since = 0.0
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

        # The eye sits right-of-center, faint enough to read text over —
        # anchored off its RIGHT edge with a margin wide enough to swallow
        # the parallax drift, so the whole eye stays on screen.
        p.setOpacity(0.16 if pal.is_dark else 0.12)
        ex = w - self._eye_w - w * 0.04 + self._off[0].x()
        ey = (h - self._eye_h) * 0.42 + self._off[0].y()
        p.drawPixmap(QPointF(ex, ey), self._frame)
        p.setOpacity(1.0)

        p.drawPixmap(QPointF(-60 + self._off[2].x(), -60 + self._off[2].y()),
                     self._dust_near)
