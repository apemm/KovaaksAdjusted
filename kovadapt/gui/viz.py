"""Character-art data viz: the Analysis charts drawn as glyph matrices in
the ascii_art visual language (monospace glyphs, density ramps, LED color)
instead of pyqtgraph. pyqtgraph survives only inside TrajectoryReplay.

These three are LEGIBLE first and decorative second — the standing complaint
was that no value could be read off any of them:

    AsciiBars     one constant-pitch track per bar: filled cells, a HARD tip
                  rule at the exact value, the unfilled remainder left as a
                  faint dotted rule so the scale stays visible, the numerals
                  in their own right-hand column (never over the track), and a
                  0..max ruler under the rows. The old glyph run faded
                  '@' -> '.' along its length, so 0.22 and 0.24 differed by
                  one dissolving character and the chart could not be compared
                  at all. The footer ratio is OPT-IN (see ratio_footer).
    AsciiHeatmap  a zone lattice (Settings.region_cols x region_rows) with
                  position words on both axes in the app's one region
                  vocabulary (analysis.insights._region_words, the wording the
                  Coach uses on the same screen), each measured zone's value
                  printed on it, unmeasured zones drawn hollow and COUNTED in
                  the footer, and a colour key only when the data's range
                  really does map to distinguishable colour. The r{row}c{col}
                  keys are the engine contract, not user-facing text: they
                  live on hover.
    AsciiTrend    a CONNECTED glyph line (slope glyphs plus vertical risers)
                  over a dim mean rule, with first/min/max/newest marked and
                  a run axis underneath. Unconnected marks at 1400px read as
                  a dot field with no baseline, which left the title carrying
                  all of the meaning and the graph none. Two runs draw as a
                  SEGMENT between two marked points, and a spread too small to
                  be a shape draws FLAT — on one row, with the mean rule ON it.
                  The axis names an absolute run number only when the caller
                  states the offset: it is handed a WINDOW of the history
                  (history[-60:]) and cannot know the index it starts at.

Nothing here may draw a claim its data cannot carry, and no two parts of one
panel may disagree. Three rules earn their own machinery:

    * a microstructure sentence (the bars' ratio) needs the caller's
      input-health verdict, so it is opt-in and gated on sample size;
    * a y axis printing the same number at both ends, or a hairline spread
      stretched into an oscillation, is a lie about movement — hence
      AsciiTrend._y_range's span floor;
    * a colour key is a promise that colour maps to number, so it appears
      only when its own endpoints differ visibly — and a lattice whose whole
      spread is narrower than one ramp step is drawn uniformly FAINT rather
      than stretched, however far from zero it sits.

All three are pure QPainter and keep only their data: they read
theme.current() at paint time, so restyle() is nothing but update() — never
cache a palette. Grid row 0 is the BOTTOM row everywhere (aim convention,
+y up), matching the r{row}c{col} region-key contract.

Motion comes from gui/motion.py and nowhere else. New data IGNITES: cells
light in a wave staggered by DISTANCE from a meaningful origin (the weakest
zone, the bars' zero anchor, the newest run), and the heatmap answers a hover
with a ring rippling outward. Both are quantized to a character ramp, so both
run at motion.GLYPH_HZ; both are skipped entirely when motion is off. The one
timer per widget ticks only while the widget is VISIBLE and something is
actually in flight — an idle chart schedules nothing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

# The lattice axes speak the app's ONE region vocabulary. analysis/ is the leaf
# (it never imports gui/), so gui/ imports the wording from it rather than
# restating it — analysis_view.py imports the same function for its titles, and
# the Coach prints it two panels below these charts.
from ..analysis.insights import _region_words
from . import motion, theme

_BAR_FILL = "#"                  # one filled cell of a bar's track
_BAR_EMPTY = "."                 # the unfilled remainder: a faint scale rule
_HEAT_RAMP = " .:-=+*#%@"        # light -> dense zone texture
# Where inside its cell a flat segment sits: bottom, CENTRE, top. The middle
# bucket has to be the glyph that reads flat, because dead centre is exactly
# where a trend that did not move lands. With four glyphs ('_-~"') the buckets
# were [0,.25) [.25,.5) [.5,.75) [.75,1) — none of them centred on 0.5 — so a
# value sitting on a row's centre drew '~', and a flat panel whose whole
# finding is "this did not move" was a run of ripples. Three buckets put "-"
# on [1/3, 2/3), i.e. astride 0.5, at the cost of one sub-cell level.
_TREND_SUB = "_-\""
_IGNITE_RAMP = ".:-+#"           # a cell part-way through the ignite wave

# Band widths, in px, for the furniture that makes a value readable.
_RULER_H = 20.0                  # the bars' 0..max ruler
_FOOTER_H = 16.0                 # the bars' ratio line / the heatmap's key
_AXIS_H = 16.0                   # the trend's run axis

# The bar track and the trend's run axis are normalized to this many stagger
# units, so the ignite wave takes the same time on a 400px panel as on a
# 1400px one instead of scaling with the pixel count.
_BAR_SPAN = 22.0
_TREND_SPAN = 20.0

# A 5x5 zone lattice is only ~5.7 units across: at the dense-grid default
# (motion.STAGGER_PER_CELL) that whole wave resolves inside a single 66 ms
# glyph frame, i.e. invisibly. motion.STAGGER_CAP still governs the spread.
_ZONE_PER_UNIT = 38
# The ring's WIDTH is its flash duration over this rate, so 34 ms/unit puts a
# motion.INSTANT flash across ~2 zones: any faster and the "ring" is the whole
# lattice lit at once, which is a flash, not a ripple.
_RIPPLE_PER_UNIT = 34


def _seed(r: int, c: int) -> float:
    """Stable per-cell noise in [0, 1) (same hash family as ascii_art)."""
    return (math.sin(c * 12.9898 + r * 78.233) * 43758.5453) % 1.0


# A deviation smaller than this many z is treated as noise, so a flat map
# renders flat instead of being stretched to fill the ramp.
NOISE_FLOOR = 0.60

# The bars' footer sentence is a MICROSTRUCTURE claim (these costs are
# overshoot plus corrective submovements), so it answers to both gates any such
# claim answers to: this run's input health, and a sample big enough to mean
# something. viz.py cannot check input health itself — analysis.report's
# input_degraded() is the app's only definition of it and it needs a RunReport,
# which a chart widget has none of — so that half is the CALLER's to assert, by
# handing over the per-bar counts that justify the line. No counts, no footer.
# Otherwise a panel titled "input timing too noisy to compare directions this
# run" spelled out "top two: left 0.42 / right 0.31 = 1.35x" underneath.
_RATIO_MIN_N = 3          # per-bar floor; analysis.movement.directional_bias's

# A spread this much smaller than the metric's own magnitude is float noise,
# not movement: np.mean over identical values lands one ULP off them and the
# mean is folded into the trend's range, so a PERFECTLY flat history was
# stretched over a ~1e-16 span — the line drawn on the bottom row with its own
# mean rule nine rows above it, both axis ends printing the same number.
_FLAT_SPAN_REL = 0.005
_FLAT_SPAN_ABS = 1e-9

# A colour key promises that colour maps to number. Ten swatches spanning
# #f6571c..#ff5414 are one colour with two labels. 40 of 255 on the widest
# channel is kovadapt's calibration for "visibly different" at swatch size —
# the all-positive occupancy map that shipped this managed 9.
_KEY_MIN_DELTA = 40

# One step of _HEAT_RAMP, in normalized units. A zone map whose whole measured
# spread is narrower than this cannot separate two zones by even one glyph, so
# the only thing normalization decides is WHICH single step they all land on.
# Zero-anchoring a one-signed range sends them all to the LOUDEST: an occupancy
# map reading 3.60..3.79 normalizes to 0.95..1.00, i.e. 25 zones of '@' at full
# saturation reading "everything is maximally bad". Below one step the map is
# drawn at the faintest step the ramp has instead (_FLAT_MAP_LEVEL) and the
# numbers printed on the zones carry the values — the same lie min-max
# normalization used to tell the deficit branch, amplifying noise to fill the
# ramp, kept alive by the offset instead of by the scale.
_RAMP_STEP = 1.0 / (len(_HEAT_RAMP) - 1)
# Inside _HEAT_RAMP's first non-blank bucket ([0.056, 0.167) rounds to '.'),
# and low enough that a flat map can never read as data.
_FLAT_MAP_LEVEL = 0.75 * _RAMP_STEP

# --- live Settings ---------------------------------------------------------
# The charts need the motion intensity, which lives on Settings. They hold the
# OBJECT and re-read `motion` off it at use time, never a cached value: the
# setting is user-editable while the app runs, exactly like the palette.
_SETTINGS = None


def use_settings(settings) -> None:
    """Register the app's live Settings object for every chart.

    Optional: with none registered the charts behave as `full`, the shipped
    default. A widget's own `.settings` attribute wins over this.
    """
    global _SETTINGS
    _SETTINGS = settings


def _heat_color(v: float, pal) -> QColor:
    """Signed value -1..+1 -> DIVERGING ramp color.

    Zero is the player's own average and gets the neutral midpoint; positive
    (weaker than your average) runs warm, negative (stronger) runs cool. A
    sequential ramp over a signed quantity hid the sign entirely — "worst of
    a good set" and "genuinely bad" looked identical.
    """
    v = min(max(v, -1.0), 1.0)
    mag = abs(v)
    if pal.rgb:
        hue = 0.62 - 0.62 * (v * 0.5 + 0.5)      # cyan -> red across the range
        return QColor.fromHsvF(max(hue, 0.0) % 1.0, 0.85, 0.35 + 0.65 * mag)
    warm, cool = (0.045, 0.55)                   # ember / ice
    hue = warm if v >= 0 else cool
    if pal.is_dark:
        return QColor.fromHsvF(hue, 0.30 + 0.62 * mag, 0.30 + 0.70 * mag)
    return QColor.fromHsvF(hue, 0.18 + 0.68 * mag, 0.86 - 0.28 * mag)


def pool(field: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean-pool a 2D field down to (rows, cols) zone means. Row order is
    preserved: feed row 0 = bottom and it stays the bottom row."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((rows, cols))
    return np.array([[chunk.mean() for chunk in np.array_split(band, cols, axis=1)]
                     for band in np.array_split(arr, rows, axis=0)])


def region_grid(deficits: dict[str, float], cols: int,
                rows: int) -> tuple[np.ndarray, list[list[str]]]:
    """r{row}c{col} dict (aim convention: higher row = higher on the wall)
    -> (grid with row 0 at the bottom, matching labels).

    Regions with no observation are NaN, not 0.0. The values are z-scores,
    so 0.0 IS the player's mean — filling absent regions with it rendered
    "never measured" as a confident "exactly average" tile, and a run that
    touched 8 of 25 regions still painted a full, authoritative 5x5 grid.
    NaN lets the widget draw them as unmeasured instead of inventing them.
    """
    labels = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
    grid = np.array([[float(deficits.get(labels[r][c], np.nan))
                      for c in range(cols)] for r in range(rows)])
    return grid, labels


def _axis_words(rows: int, cols: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(row words, bottom row first; column words, left first) for a lattice.

    The wording is analysis.insights._region_words' — the app's ONE region
    vocabulary, the same words the Coach prints on this screen. It is not
    restated here but ASKED for, per band, so a change to the shared wording
    arrives here for free: the row words come from column 0 ("lower left" ->
    "lower") and the column words from row 0 ("lower center" -> "center").
    Only (middle, center) collapses to a single word in that function, and
    neither probe can hit it, so both probes always return two tokens.

    Row 0 is the BOTTOM row, matching the r{row}c{col} contract.
    """
    dims = SimpleNamespace(region_rows=max(int(rows), 1),
                           region_cols=max(int(cols), 1))
    row_words = tuple(_region_words(f"r{r}c0", dims).rsplit(" ", 1)[0]
                      for r in range(max(rows, 0)))
    col_words = tuple(_region_words(f"r0c{c}", dims).split(" ", 1)[-1]
                      for c in range(max(cols, 0)))
    return row_words, col_words


def _label_groups(words: Sequence[str]) -> list[tuple[str, int, int]]:
    """Runs of the same word collapsed to (word, first band, last band).

    The shared vocabulary has three bands per axis, so a 5-zone lattice really
    does have one "center" spanning three zones. Printing the word once per
    zone would claim three distinct positions that the vocabulary — and the
    Coach reading it — does not distinguish.
    """
    out: list[tuple[str, int, int]] = []
    for i, word in enumerate(words):
        if out and out[-1][0] == word:
            out[-1] = (word, out[-1][1], i)
        else:
            out.append((word, i, i))
    return out


def _paint_title(p: QPainter, pal, title: str, width: int) -> float:
    """Dim uppercase mono header line; returns the content top y.

    Elided rather than clipped: a title cut mid-glyph reads as a broken panel,
    and these titles are the chart's finding — the caller cannot know how many
    characters this width holds."""
    if not title:
        return 8.0
    font = theme.mono(12)
    p.setFont(font)
    p.setPen(QColor(pal.fg_dim))
    text = QFontMetricsF(font).elidedText(title.upper(), Qt.ElideRight, width - 22)
    p.drawText(QRectF(10, 4, width - 20, 16), Qt.AlignLeft | Qt.AlignVCenter, text)
    return 26.0


def _paint_empty(p: QPainter, pal, rect: QRectF, text: str) -> None:
    p.setFont(theme.mono(12))
    col = QColor(pal.fg_dim)
    col.setAlphaF(0.75)
    p.setPen(col)
    p.drawText(rect, Qt.AlignCenter, f"· {text} ·")


def _dim(hex_or_col, alpha: float) -> QColor:
    col = QColor(hex_or_col)
    col.setAlphaF(min(max(alpha, 0.0), 1.0))
    return col


def _fmt_value(fmt: str, v: float) -> str:
    """Format a value, never as a signed zero. "{:+.2f}" turns -0.001 into
    "-0.00", which reads as a rendering bug rather than as the zero it is.

    The test is "the rendered text carries no nonzero digit", not "it is
    nothing but zeros and a point": a signed zero can carry a UNIT, and
    stripping only "0." left "%" behind, so "{:.0%}" sailed through and put
    "-0%" on the trend's y axis. It reads the SIGN off the text too, because
    -0.0 is not less than zero in Python but still formats with a minus.
    """
    txt = fmt.format(v)
    if txt.lstrip().startswith("-") and not any(c in "123456789" for c in txt):
        return fmt.format(0.0)
    return txt


def _prints_zero(v: float) -> bool:
    """Whether a bar's value RENDERS as zero in the value column, which is the
    only zero a reader can see. A footer sentence about zeros has to agree with
    the numerals printed beside it."""
    return f"{v:.2f}" in ("0.00", "-0.00")


def _ignite_glyph(ch: str, q: float) -> str:
    """The character a cell shows part-way through the ignite wave. The
    output IS quantized to a ramp, which is why these run at GLYPH_HZ."""
    if q >= 1.0:
        return ch
    return _IGNITE_RAMP[min(int(q * len(_IGNITE_RAMP)), len(_IGNITE_RAMP) - 1)]


class _Ignite:
    """Staggered reveal + hover ripple for the charts, on ONE timer each.

    Every duration comes from gui/motion.py. Two rules are absolute:

    * the timer runs only while the widget is visible AND something is in
      flight — it is started by _ignite/_ripple/showEvent and stopped by
      hideEvent and by the first tick that finds nothing left to animate;
    * with motion off nothing animates at all: _ignite jumps to the end
      state and never schedules a frame.

    A reveal asked for while the widget is hidden is held PENDING and fires
    on showEvent (igniting a chart nobody can see would burn frames behind a
    hidden page), and a hidden widget paints its END state — offscreen grabs
    have to show the finished chart.
    """

    def _init_motion(self) -> None:
        self.settings = None            # per-widget override of use_settings()
        self._ign_t0 = 0.0
        self._ign_span = 0.0            # ms until the last cell has arrived
        self._ign_grow = 1.0            # ms for one cell to come up
        self._ign_per = motion.STAGGER_PER_CELL
        self._ign_origin: tuple[float, float] = (0.0, 0.0)
        self._ign_pending = False
        self._rip_t0 = 0.0
        self._rip_span = 0.0
        self._rip_flash = 1.0
        self._rip_far = 1.0
        self._rip_per = _RIPPLE_PER_UNIT
        self._rip_origin: tuple[float, float] = (0.0, 0.0)
        self._timer = QTimer(self)
        self._timer.setInterval(motion.GLYPH_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    def _motion_settings(self):
        """The live Settings object (may be None -> motion defaults to full).
        Read at use time; never cached into a value."""
        return self.settings if self.settings is not None else _SETTINGS

    def _ignite(self, origin: tuple[float, float], far: float, *,
                per_unit: int = motion.STAGGER_PER_CELL) -> None:
        """Start a staggered reveal from `origin`; `far` is the greatest
        distance any cell sits from it, which sets the total spread."""
        s = self._motion_settings()
        self._ign_origin = origin
        self._ign_per = per_unit
        if not motion.animates(s):
            self._ign_span = 0.0        # end state, no timer, no frames
            self._ign_pending = False
            self.update()
            return
        self._ign_grow = max(motion.ms(s, motion.BASE), 1)
        self._ign_span = motion.stagger(s, far, per_unit=per_unit) + self._ign_grow
        if self.isVisible():
            self._ign_pending = False
            self._ign_t0 = time.monotonic()
            self._timer.start()
        else:
            self._ign_pending = True
        self.update()

    def _ignite_skip(self) -> None:
        """No reveal at all — there is nothing to reveal (data cleared, or too
        little of it to draw). An empty chart must schedule no frames either."""
        self._ign_span = 0.0
        self._ign_pending = False
        self.update()

    def _ignite_lit(self):
        """distance -> 0..1 reveal factor for THIS frame, or None when the
        chart is fully lit (the common case — callers then skip the work)."""
        if self._ign_span <= 0.0 or self._ign_pending:
            return None
        el = (time.monotonic() - self._ign_t0) * 1000.0
        if el >= self._ign_span:
            return None
        s = self._motion_settings()
        per, grow = self._ign_per, self._ign_grow

        def lit(distance: float) -> float:
            # motion.stagger owns the delay rule; this only asks it per cell
            return motion.ease_out((el - motion.stagger(s, distance, per_unit=per)) / grow)
        return lit

    def _ripple(self, origin: tuple[float, float], far: float, *,
                per_unit: int = _RIPPLE_PER_UNIT) -> None:
        """A ring travelling outward from `origin` — the hover acknowledgement.
        Ambient-gated: `reduced` and `off` get no idle-ish flourishes."""
        s = self._motion_settings()
        if not motion.ambient(s):
            return
        self._rip_origin = origin
        self._rip_per = per_unit
        self._rip_far = max(far, 1.0)
        self._rip_flash = max(motion.ms(s, motion.INSTANT), 1)
        self._rip_span = motion.stagger(s, far, per_unit=per_unit) + self._rip_flash
        self._rip_t0 = time.monotonic()
        if self.isVisible():
            self._timer.start()
        self.update()

    def _ripple_glow(self):
        """distance -> 0..1 ring brightness for this frame, or None."""
        if self._rip_span <= 0.0:
            return None
        el = (time.monotonic() - self._rip_t0) * 1000.0
        if el >= self._rip_span:
            return None
        s = self._motion_settings()
        flash, per, far = self._rip_flash, self._rip_per, self._rip_far

        def glow(distance: float) -> float:
            u = (el - motion.stagger(s, distance, per_unit=per)) / flash
            if u <= 0.0 or u >= 1.0:
                return 0.0
            # rises and falls as the ring passes, and DECAYS outward — a ring of
            # even brightness reads as the whole lattice being outlined
            return math.sin(math.pi * u) * (1.0 - 0.55 * min(distance / far, 1.0))
        return glow

    # ------------------------------------------------------------------
    def _in_flight(self) -> bool:
        """True while any animation still has frames left; retires the ones
        that have finished so the next tick can stop the timer."""
        now = time.monotonic()
        live = False
        if self._ign_span > 0.0 and not self._ign_pending:
            if (now - self._ign_t0) * 1000.0 >= self._ign_span:
                self._ign_span = 0.0
            else:
                live = True
        if self._rip_span > 0.0:
            if (now - self._rip_t0) * 1000.0 >= self._rip_span:
                self._rip_span = 0.0
            else:
                live = True
        return live

    def _tick(self) -> None:
        live = self._in_flight()
        self.update()
        if not live:
            self._timer.stop()           # idle charts schedule nothing

    def showEvent(self, event) -> None:
        if self._ign_pending:
            self._ign_pending = False
            self._ign_t0 = time.monotonic()
        if self._in_flight():
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()               # never tick behind a hidden widget
        super().hideEvent(event)


class AsciiBars(_Ignite, QWidget):
    """Horizontal bars as constant-pitch glyph tracks, zero-anchored.

    A bar is a bar: filled cells at one density, a hard tip rule at the exact
    value (so 0.22 and 0.24 differ by a visible distance rather than by one
    faded character), the unfilled remainder as a faint dotted rule carrying
    the scale, and the numerals in a right-hand column of their own. Every
    bar sharing the max (when positive) paints pal.bad — but ONLY when the
    caller passed `ratio_counts`, which is its permission to make a
    side-vs-side claim at all. Highlighting a worst bar is that claim in
    colour, so it cannot outlive the sentence version of it.
    """

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._init_motion()
        self._title = title
        self._labels: list[str] = []
        self._sublabels: list[str] = []
        self._values: list[float] = []
        self._counts: list[int] | None = None
        self._floor = 0.0
        self.setMinimumHeight(220)

    # ------------------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, labels: Sequence[str], values: Sequence[float],
                 sublabels: Sequence[str] | None = None, *,
                 ratio_counts: Sequence[int] | None = None,
                 floor: float = 0.0) -> None:
        """`ratio_counts` are the per-bar sample sizes, and passing them is the
        caller's explicit permission to print the footer ratio at all (see
        ratio_footer). Omit them — the default — and no footer sentence is
        drawn: only the caller holds this run's input-health verdict, and a
        ratio under a title reading "too noisy to compare directions" is a
        panel contradicting itself.

        `floor` is the smallest value allowed to fill the track — the size a
        bar has to reach before its length means anything. It is the caller's
        because only the caller knows the units: this widget cannot tell a
        cost of 0.004 from a temperature of 0.004. Leave it at zero and the
        axis is pure max-normalization, which draws noise at full scale.
        """
        self._labels = [str(x) for x in labels]
        self._values = [float(v) for v in values]
        self._sublabels = [str(s) for s in (sublabels or [])]
        self._counts = (None if ratio_counts is None
                        else [int(c) for c in ratio_counts])
        self._floor = max(float(floor), 0.0)
        # The wave starts at the zero anchor, mid-stack, so every bar GROWS
        # out of the axis instead of the rows appearing one after another.
        n = len(self._values)
        if not n:
            self._ignite_skip()
            return
        origin = ((n - 1) / 2.0, 0.0)
        far = math.hypot((n - 1) / 2.0, _BAR_SPAN)
        self._ignite(origin, far)
        self.update()

    def clear(self) -> None:
        self.set_data([], [])

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------------
    def _label_at(self, i: int, fallback: str) -> str:
        """A bar's label, indexed defensively — a short label list must not
        crash a chart or make the footer name a bar that is not there."""
        return self._labels[i] if 0 <= i < len(self._labels) else fallback

    def axis_max(self) -> float:
        """Where the track ends — the largest bar or the caller's floor,
        whichever is bigger. ONE definition, because the bar lengths and the
        ruler label under them are two renderings of this same number and
        they were computed separately: the bars scaled to the floor while the
        label kept printing the largest value, so the axis ran to 0.05 under
        a caption reading "0.00"."""
        if not self._values:
            return 1.0
        vmax = max(self._values)
        return max(vmax, self._floor) if vmax > 0 else 1.0

    def ratio_note(self) -> str:
        """The candidate evidence line: the top two bars and the ratio between
        them. "These two are even" is itself a finding, and a 1.09x difference
        is exactly what no bar chart can be eyeballed for.

        A pure formatter over the plotted values — every sentence it can return
        has to agree with the numerals in the value column. ratio_footer() is
        what decides whether any of it may be SHOWN.
        """
        if not self._values:
            return ""
        order = sorted(range(len(self._values)), key=lambda i: -self._values[i])
        hi = self._values[order[0]]
        lab = self._label_at(order[0], "top")
        if len(self._values) < 2:
            return f"{lab} {_fmt_value('{:.2f}', hi)} — one direction measured"
        nxt = self._values[order[1]]
        lab2 = self._label_at(order[1], "next")
        if all(_prints_zero(v) for v in self._values):
            return "every bar is 0.00 — no cost recorded to compare"
        if hi <= 0:
            # Every bar NEGATIVE is not every bar zero: the value column prints
            # -0.20 / -0.10 / -0.30 beside this line, and "every bar is 0.00"
            # contradicted every numeral on the panel.
            return (f"no bar is above zero — {lab} "
                    f"{_fmt_value('{:.2f}', hi)} is the highest")
        if nxt <= 0 or _prints_zero(nxt):
            return f"{lab} {hi:.2f} is the only bar above zero"
        return f"top two: {lab} {hi:.2f} / {lab2} {nxt:.2f} = {hi / nxt:.2f}x"

    def _cited_counts(self) -> list[int]:
        """The sample counts behind the bars the footer sentence NAMES: the top
        two by value, or the two largest samples when every bar reads 0.00
        (that sentence is a claim about the whole set, so the biggest samples
        are the ones that have to carry it)."""
        counts = self._counts or []

        def cnt(i: int) -> int:
            return int(counts[i]) if 0 <= i < len(counts) else 0

        idx = range(len(self._values))
        if all(_prints_zero(v) for v in self._values):
            return sorted((cnt(i) for i in idx), reverse=True)[:2]
        order = sorted(idx, key=lambda i: -self._values[i])
        return [cnt(i) for i in order[:2]]

    def ratio_footer(self) -> str:
        """The sentence actually painted under the bars — "" unless BOTH gates
        pass, which is the default.

        * the caller supplied per-bar sample counts (set_data's
          ratio_counts): a microstructure comparison needs this run's
          input-health verdict and only the caller holds it;
        * every bar the sentence names carries at least _RATIO_MIN_N samples,
          so "4.00x" can never be printed off one flick per side.
        """
        if self._counts is None or not self._values:
            return ""
        note = self.ratio_note()
        cited = self._cited_counts()
        if not note or not cited or min(cited) < _RATIO_MIN_N:
            return ""
        return note

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        top = _paint_title(p, pal, self._title, w)
        if not self._values:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "waiting for flick data")
            return

        n = len(self._values)
        body_h = max(h - top - _RULER_H - _FOOTER_H - 4, 20.0)
        row_h = body_h / n
        roomy = row_h >= 34
        gf = theme.mono(14 if roomy else 12)
        fm = QFontMetricsF(gf)
        cw = max(fm.horizontalAdvance(_BAR_FILL), 4.0)
        chh = fm.height()
        lf = theme.mono(12)
        lfm = QFontMetricsF(lf)

        texts = [f"{v:.2f}" for v in self._values]
        lab_w = max([lfm.horizontalAdvance(s) for s in self._labels]
                    + [lfm.horizontalAdvance(s) for s in self._sublabels]
                    + [40.0]) + 10
        val_w = max(fm.horizontalAdvance(t) for t in texts) + 14
        x_axis = 10 + lab_w                       # the zero anchor rule
        bar_x0 = x_axis + cw * 0.7
        bar_x1 = w - val_w - 10
        ncells = max(int((bar_x1 - bar_x0) // cw), 4)
        track_w = ncells * cw
        vmax = max(self._values)
        # The axis tops out at the largest bar OR the caller's floor,
        # whichever is bigger. Pure max-normalization made the biggest value
        # fill the track no matter how small it was: [0.004, 0.002, 0.003] —
        # three flicks that overshot by well under one percent of their own
        # amplitude — drew at 496/272/408px, pixel-for-pixel the same picture
        # as a real [0.60, 0.33, 0.49] spread. Same failure the zone heatmap
        # had: normalizing noise to full scale renders it as a verdict.
        scale = self.axis_max()
        lit = self._ignite_lit()

        # the zero axis every bar grows out of
        p.setPen(_dim(pal.fg_dim, 0.70))
        p.drawLine(QPointF(x_axis, top + 2), QPointF(x_axis, top + body_h))

        # Marking a worst bar IS a side-vs-side claim — the same claim the
        # footer sentence makes, in colour instead of words — so it answers to
        # the same permission. Without this the title could read "input timing
        # too noisy to compare directions this run" while a red bar underneath
        # named the worst direction anyway: the gate held for the words and
        # leaked through the paint.
        #
        # It also answers to the same zero test the sentence does: when every
        # bar prints 0.00 the footer says "no cost recorded to compare", and a
        # red bar beside that sentence is the panel arguing with itself.
        may_compare = (self._counts is not None
                       and not all(_prints_zero(x) for x in self._values))
        for i, v in enumerate(self._values):
            cy = top + i * row_h + row_h / 2
            is_max = may_compare and v == vmax and vmax > 0
            color = QColor(pal.bad if is_max else pal.accent)

            # ---- direction label, with the flick count dim underneath
            # (indexed defensively: a short label list must not crash a chart)
            label = self._labels[i] if i < len(self._labels) else ""
            p.setFont(lf)
            p.setPen(QColor(pal.fg))
            if i < len(self._sublabels) and roomy:
                p.drawText(QRectF(6, cy - 17, lab_w - 8, 16),
                           Qt.AlignRight | Qt.AlignBottom, label)
                p.setPen(QColor(pal.fg_dim))
                p.drawText(QRectF(6, cy + 1, lab_w - 8, 14),
                           Qt.AlignRight | Qt.AlignTop, self._sublabels[i])
            else:
                p.drawText(QRectF(6, cy - row_h / 2, lab_w - 8, row_h),
                           Qt.AlignRight | Qt.AlignVCenter, label)

            # ---- the track: constant pitch, one density, hard tip
            p.setFont(gf)
            exact = min(v / scale, 1.0) * ncells
            full = int(exact)
            front = -1                            # rightmost cell lit this frame
            gy = cy - chh / 2
            # the unfilled remainder is the SCALE, not data: it never animates,
            # so it is one drawText (integer advance = same cells as a loop)
            if ncells > full:
                p.setPen(_dim(pal.border, 1.0))
                p.drawText(QRectF(bar_x0 + full * cw, gy,
                                  (ncells - full) * cw + cw, chh),
                           Qt.AlignLeft | Qt.AlignVCenter, _BAR_EMPTY * (ncells - full))
            if lit is None:
                if full:
                    front = full - 1
                    p.setPen(color)
                    p.drawText(QRectF(bar_x0, gy, full * cw + cw, chh),
                               Qt.AlignLeft | Qt.AlignVCenter, _BAR_FILL * full)
            else:
                for j in range(full):
                    u = j / max(ncells - 1, 1) * _BAR_SPAN
                    q = min(max(lit(motion.grid_distance(i, u, self._ign_origin)), 0.0), 1.0)
                    if q <= 0.02:
                        continue
                    front = j
                    p.setPen(_dim(color, 0.45 + 0.55 * q))
                    p.drawText(QRectF(bar_x0 + j * cw, gy, cw * 2, chh),
                               Qt.AlignLeft | Qt.AlignVCenter,
                               _ignite_glyph(_BAR_FILL, q))

            # ---- the tip: unambiguous, sub-cell exact, riding the ignite
            tip = exact if lit is None else min(exact, front + 1.0)
            if tip > 0.0:
                tx = bar_x0 + tip * cw
                # the part-cell remainder as a solid nub, so the rule sits
                # flush against the run instead of floating past its end
                if tip - int(tip) > 0.05:
                    p.setPen(Qt.NoPen)
                    p.setBrush(color)
                    p.drawRect(QRectF(bar_x0 + int(tip) * cw, cy - 3.0,
                                      (tip - int(tip)) * cw, 6.0))
                over = 4.0 if is_max else 2.0
                p.setPen(QPen(color, 3.0 if is_max else 2.0))
                p.drawLine(QPointF(tx, cy - chh / 2 - over), QPointF(tx, cy + chh / 2 + over))

            # ---- the number, in its own column: never over the track
            p.setFont(gf)
            p.setPen(QColor(pal.bad) if is_max else QColor(pal.fg))
            p.drawText(QRectF(bar_x1 + 4, cy - row_h / 2, val_w - 8, row_h),
                       Qt.AlignRight | Qt.AlignVCenter, texts[i])

        # ---- the ruler: what a length is worth
        ry = top + body_h + 3
        p.setPen(_dim(pal.fg_dim, 0.7))
        # quarter ticks only when there IS a scale: with every bar at zero the
        # track means nothing, and ticks would imply a range that has no data
        for k in range(5 if vmax > 0 else 1):
            x = bar_x0 + track_w * k / 4.0
            p.drawLine(QPointF(x, ry), QPointF(x, ry + (5.0 if k % 4 == 0 else 3.0)))
        p.setFont(lf)
        p.setPen(_dim(pal.fg_dim, 0.95))
        p.drawText(QRectF(x_axis - 2, ry + 4, 40, 14), Qt.AlignLeft | Qt.AlignTop, "0")
        if vmax > 0:
            # `scale`, not vmax: the label names where the TRACK ends, and
            # once a floor is in play those stop being the same number. It
            # read "0.00" under a track running to 0.05 — an axis captioned
            # with a value that is not on it, which is the same lie the bars
            # were just stopped from telling.
            p.drawText(QRectF(bar_x0 + track_w - 70, ry + 4, 70, 14),
                       Qt.AlignRight | Qt.AlignTop, f"{scale:.2f}")

        # ---- the ratio, because a 1.09x gap is not eyeballable — but only
        # when the caller has vouched for the comparison and the sample carries
        # it. The band stays reserved either way, so the bars keep their pitch.
        footer = self.ratio_footer()
        if footer:
            p.setPen(QColor(pal.fg_dim))
            p.drawText(QRectF(10, h - _FOOTER_H, w - 20, _FOOTER_H - 1),
                       Qt.AlignLeft | Qt.AlignVCenter, footer)


class AsciiHeatmap(_Ignite, QWidget):
    """Zone-grid heatmap: each zone a block of glyphs whose density and ramp
    color carry the value, its number printed on it, position words on both
    axes, unmeasured zones hollow and counted. Grid row 0 = bottom (aim
    convention). Hovering a zone tooltips its exact region key and value and
    sends a ripple out from it."""

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._init_motion()
        self._title = title
        self._grid: np.ndarray | None = None
        self._norm: np.ndarray | None = None
        self._labels: list[list[str]] | None = None
        self._flat_map = False
        self._fmt = "{:.2f}"
        self._hover: tuple[int, int] | None = None
        self.setMinimumHeight(220)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, grid: np.ndarray | None,
                 labels: Sequence[Sequence[str]] | None = None,
                 fmt: str = "{:.2f}") -> None:
        """grid: 2D array, row 0 = bottom. labels: same shape, shown in the
        hover tooltip (defaults to the r{row}c{col} region keys)."""
        if grid is None:
            self._grid = self._norm = self._labels = None
            self._flat_map = False
            self._ignite_skip()
        else:
            g = np.asarray(grid, dtype=float)
            self._grid = g
            # Zero-anchored and symmetric, NOT min-max. Min-max always sent
            # the smallest cell to 0 and the largest to 1 whatever their
            # magnitudes, so 25 regions of pure noise rendered as a
            # full-range map with one cell screaming "your weakest zone".
            # These are z-scores: 0 is the player's own mean, and a deviation
            # only earns colour once it is a real fraction of a standard
            # deviation. NOISE_FLOOR keeps a flat map looking flat.
            finite = g[np.isfinite(g)]
            span = float(np.max(np.abs(finite))) if finite.size else 0.0
            span = max(span, NOISE_FLOOR)
            norm = np.clip(g / span, -1.0, 1.0)
            # Zero-anchoring is right for a signed z-score, but it does nothing
            # for a ONE-SIGNED range sitting far from zero: 3.60..3.79 becomes
            # 0.95..1.00 and every zone paints at the ramp's last step. When the
            # whole measured spread is under one ramp step there is no colour
            # difference to draw, so it is drawn uniformly faint (sign kept, so
            # the hue still says which way the metric runs) and the numbers on
            # the zones carry it. Under two measured zones there is no spread to
            # judge at all — a lone zone keeps its own magnitude.
            self._flat_map = bool(
                finite.size >= 2
                and (float(finite.max()) - float(finite.min())) / span < _RAMP_STEP)
            self._norm = np.sign(norm) * _FLAT_MAP_LEVEL if self._flat_map else norm
            self._labels = ([[str(s) for s in row] for row in labels]
                            if labels is not None else None)
            self._fmt = fmt
            self._ignite(*self._ignite_from(g), per_unit=_ZONE_PER_UNIT)
        self.update()

    @staticmethod
    def _ignite_from(g: np.ndarray) -> tuple[tuple[float, float], float]:
        """(origin, farthest distance) for the reveal — the WEAKEST measured
        zone, in display coordinates (mean > 0 = weaker there, per the
        RegionPosterior contract). All-NaN maps light from the middle."""
        if g.ndim != 2 or g.size == 0:
            return (0.0, 0.0), 0.0
        rows, cols = g.shape
        fin = np.isfinite(g)
        if fin.any():
            r, c = np.unravel_index(int(np.argmax(np.where(fin, g, -np.inf))), g.shape)
            origin = (float(rows - 1 - r), float(c))
        else:
            origin = ((rows - 1) / 2.0, (cols - 1) / 2.0)
        far = max(motion.grid_distance(rr, cc, origin)
                  for rr in (0, rows - 1) for cc in (0, cols - 1))
        return origin, far

    def clear(self) -> None:
        self.set_data(None)

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------------
    def coverage_note(self) -> str:
        """How much of the lattice this run actually measured. A 5x5 grid with
        two live zones is not a broken chart, but it has to SAY so — 23 hollow
        outlines and no explanation read as a rendering failure."""
        if self._grid is None or self._grid.size == 0:
            return ""
        total = int(self._grid.size)
        seen = int(np.isfinite(self._grid).sum())
        if seen == total:
            return ""
        if seen == 0:
            return f"no zone measured this run — all {total} drawn hollow"
        return f"{seen} of {total} zones measured — the rest are drawn hollow"

    def spread_note(self) -> str:
        """Why every zone is the same shade, when the spread is too narrow to
        colour. 25 identically faint zones and no explanation read as a broken
        chart for exactly the reason 25 hollow ones do."""
        ends = self._key_ends()
        if not self._flat_map or ends is None:
            return ""
        lo_v, hi_v, _span = ends
        # Kept short on purpose: it shares one footer line with coverage_note,
        # and both together have to fit a chart column before elision starts.
        return (f"measured zones {self._fmt.format(lo_v)}"
                f"..{self._fmt.format(hi_v)} — too narrow to shade; "
                f"the numbers carry it")

    def footer_note(self) -> str:
        """The whole footer sentence: what was measured, and why it is flat if
        it is. Both halves are about the same lattice, so they share one line
        rather than one of them going unsaid."""
        return " · ".join(p for p in (self.coverage_note(), self.spread_note()) if p)

    # ------------------------------------------------------------------
    def _layout(self):
        """Full geometry: (x0, y0, zw, zh, gap, rows, cols, gutter, words_h,
        foot_h). The lattice is inset by a word gutter on the left and a word
        row plus a key row underneath, all dropped when the panel is too
        small to spend the pixels."""
        if self._grid is None or self._grid.ndim != 2 or self._grid.size == 0:
            return None
        rows, cols = self._grid.shape
        top = 26.0 if self._title else 8.0
        gap = 3.0
        w, h = float(self.width()), float(self.height())
        gutter = 62.0 if w >= 320 else 0.0
        words_h = 15.0 if h >= 150 else 0.0
        foot_h = 15.0 if h >= 175 else 0.0
        x0, y0 = 10.0 + gutter, top
        zw = (w - x0 - 10.0 - gap * (cols - 1)) / cols
        zh = (h - y0 - 8.0 - words_h - foot_h - gap * (rows - 1)) / rows
        if zw <= 4 or zh <= 4:
            return None
        return x0, y0, zw, zh, gap, rows, cols, gutter, words_h, foot_h

    def _geom(self) -> tuple[float, float, float, float, float, int, int] | None:
        """(x0, y0, zone_w, zone_h, gap, rows, cols) of the zone lattice —
        screen row 0 at y0 is the TOP row (data row rows-1)."""
        lay = self._layout()
        return None if lay is None else lay[:7]

    def _zone_at(self, x: float, y: float) -> tuple[int, int] | None:
        """(display_row, col) under widget coords, or None (gutter/outside)."""
        geom = self._geom()
        if geom is None:
            return None
        x0, y0, zw, zh, gap, rows, cols = geom
        c = int((x - x0) // (zw + gap))
        disp_r = int((y - y0) // (zh + gap))
        if not (0 <= c < cols and 0 <= disp_r < rows) or x < x0 or y < y0:
            return None
        if (x - x0) - c * (zw + gap) > zw or (y - y0) - disp_r * (zh + gap) > zh:
            return None                                   # in the gutter
        return disp_r, c

    def zone_info(self, x: float, y: float) -> str | None:
        """'label · value' for the zone under widget coords (x, y), else None.

        The label is the caller's — the r{row}c{col} region key — because the
        key is the cross-module contract and hover is where it belongs now
        that the lattice itself is labelled in words."""
        cell = self._zone_at(x, y)
        if cell is None or self._grid is None:
            return None
        disp_r, c = cell
        rows = self._grid.shape[0]
        r = rows - 1 - disp_r                             # data row (bottom = 0)
        label = self._labels[r][c] if self._labels else f"r{r}c{c}"
        value = float(self._grid[r, c])
        if not math.isfinite(value):
            return f"{label} · not measured this run"
        return f"{label} · {self._fmt.format(value)}"

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        info = self.zone_info(x, y)
        self.setToolTip(info or "")
        if info:
            QToolTip.showText(event.globalPosition().toPoint(), info, self)
        cell = self._zone_at(x, y)
        if cell != self._hover:
            self._hover = cell
            if cell is not None and self._grid is not None:
                rows, cols = self._grid.shape
                far = max(motion.grid_distance(rr, cc, cell)
                          for rr in (0, rows - 1) for cc in (0, cols - 1))
                self._ripple((float(cell[0]), float(cell[1])), far,
                             per_unit=_RIPPLE_PER_UNIT)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        top = _paint_title(p, pal, self._title, w)
        lay = self._layout()
        if lay is None:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "no movement data")
            return
        x0, y0, zw, zh, gap, rows, cols, gutter, words_h, foot_h = lay

        gf = theme.mono(12)
        fm = QFontMetricsF(gf)
        cw = max(fm.horizontalAdvance("@"), 4.0)
        chh = max(fm.height() * 0.92, 6.0)
        lit = self._ignite_lit()
        glow = self._ripple_glow()

        # ---- the value that will be printed on every measured zone
        finite = self._grid[np.isfinite(self._grid)]
        show_values = False
        if finite.size:
            widest = max(fm.horizontalAdvance(self._fmt.format(float(v)))
                         for v in (float(finite.min()), float(finite.max())))
            show_values = zw >= widest + 14 and zh >= fm.height() + 12

        for disp_r in range(rows):
            r = rows - 1 - disp_r
            for c in range(cols):
                v = float(self._norm[r, c])
                zx = x0 + c * (zw + gap)
                zy = y0 + disp_r * (zh + gap)
                zone = QRectF(zx, zy, zw, zh)
                q = 1.0 if lit is None else min(
                    max(lit(motion.grid_distance(disp_r, c, self._ign_origin)), 0.0), 1.0)
                if q <= 0.02:
                    continue
                ring = 0.0 if glow is None else glow(
                    motion.grid_distance(disp_r, c, self._rip_origin))

                if not math.isfinite(v):
                    # Never measured: say so rather than paint a confident
                    # "average" tile. A DOTTED hairline, no fill, no glyphs —
                    # dotted so a mostly-hollow lattice reads as deliberate.
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(_dim(pal.border, q), 1.0, Qt.DotLine))
                    p.drawRoundedRect(zone, 4, 4)
                    p.setFont(gf)
                    p.setPen(_dim(pal.fg_dim, 0.30 * q + 0.35 * ring))
                    p.drawText(zone, Qt.AlignCenter, "·")
                    if ring > 0.02:
                        p.setBrush(Qt.NoBrush)
                        p.setPen(QPen(_dim(pal.accent, 0.75 * ring), 1.4))
                        p.drawRoundedRect(zone, 4, 4)
                    continue

                mag = abs(v)
                color = _heat_color(v, pal)

                # soft backing tint so a zone reads even between glyphs
                p.setPen(Qt.NoPen)
                p.setBrush(_dim(color, (0.07 + 0.20 * mag) * q + 0.10 * ring))
                p.drawRoundedRect(zone, 4, 4)

                # the glyph block: density carries the value, seeded per-row
                # jitter keeps it from reading as a flat stamp
                p.setFont(gf)
                gcols = max(int(zw // cw), 1)
                grows = max(int(zh // chh), 1)
                mx = zx + (zw - gcols * cw) / 2
                my = zy + (zh - grows * chh) / 2
                # Jitter alpha, never the VALUE: perturbing the value moves
                # cells across ramp steps, so a zone's reported density
                # stopped matching its number.
                d = min(max(mag, 0.0), 1.0)
                ch = _HEAT_RAMP[int(round(d * (len(_HEAT_RAMP) - 1)))]
                if ch != " ":
                    # One drawText per ROW of the zone, not per cell. Per-cell
                    # alpha meant ~4600 drawText calls and 50 ms a frame on a
                    # 1400x500 panel — survivable while the map only repainted
                    # on new data, fatal now that the reveal repaints it at
                    # GLYPH_HZ. theme.mono()'s integer advance is what puts the
                    # run on exactly the cells the per-glyph loop used.
                    row_txt = _ignite_glyph(ch, q) * gcols
                    row_w = gcols * cw + cw
                    for gr in range(grows):
                        jit = (_seed(disp_r * 31 + gr, c * 17) - 0.5) * 0.22
                        p.setPen(_dim(color, (0.55 + 0.45 * d + jit) * q + 0.35 * ring))
                        p.drawText(QRectF(mx, my + gr * chh, row_w, chh * 1.4),
                                   Qt.AlignLeft | Qt.AlignVCenter, row_txt)

                # the number, on a chip so it survives the texture under it
                if show_values:
                    txt = _fmt_value(self._fmt, float(self._grid[r, c]))
                    tw = fm.horizontalAdvance(txt)
                    chip = QRectF(zx + (zw - tw - 10) / 2,
                                  zy + (zh - fm.height() - 4) / 2,
                                  tw + 10, fm.height() + 4)
                    p.setPen(Qt.NoPen)
                    p.setBrush(_dim(pal.bg, 0.80 * q))
                    p.drawRoundedRect(chip, 3, 3)
                    p.setFont(gf)
                    p.setPen(_dim(pal.fg, 0.95 * q))
                    p.drawText(chip, Qt.AlignCenter, txt)

                if ring > 0.02:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(_dim(pal.accent, 0.8 * ring), 1.4))
                    p.drawRoundedRect(zone, 4, 4)

        # ---- position words instead of region keys, in the Coach's vocabulary
        # (_axis_words): two spellings of one grid read as two findings, and
        # the Coach names these zones a few hundred pixels below this lattice.
        # A word spanning several zones is drawn ONCE over the whole band, so
        # the axis distinguishes exactly what the vocabulary distinguishes.
        p.setFont(gf)
        row_words, col_words = _axis_words(rows, cols)
        if gutter and row_words:
            p.setPen(_dim(pal.fg_dim, 0.95))
            for word, lo_band, hi_band in _label_groups(row_words):
                ya = y0 + (rows - 1 - hi_band) * (zh + gap)      # bands run up
                yb = y0 + (rows - 1 - lo_band) * (zh + gap) + zh
                p.drawText(QRectF(6, ya, gutter - 8, yb - ya),
                           Qt.AlignRight | Qt.AlignVCenter, word)
        if words_h and col_words:
            groups = _label_groups(col_words)
            widths = [fm.horizontalAdvance(word) for word, _a, _b in groups]
            centres = [x0 + (a + b) / 2.0 * (zw + gap) + zw / 2
                       for _word, a, b in groups]
            # A word can be wider than the band it sits over, and a rect that
            # clips "center" to "cent" is worse than no label: let a word
            # overflow and thin the axis until no two labels collide.
            step = 1
            while step < len(groups) and any(
                    centres[i] + widths[i] / 2 + 6 > centres[i + step] - widths[i + step] / 2
                    for i in range(0, len(groups) - step, step)):
                step += 1
            wy = y0 + rows * zh + (rows - 1) * gap + 1
            p.setPen(_dim(pal.fg_dim, 0.95))
            for i in range(0, len(groups), step):
                p.drawText(QRectF(centres[i] - widths[i] / 2 - 4, wy,
                                  widths[i] + 8, words_h),
                           Qt.AlignHCenter | Qt.AlignVCenter, groups[i][0])

        # ---- footer: how much was measured, and the key to the colours
        if foot_h:
            fy = h - foot_h - 2
            p.setPen(QColor(pal.fg_dim))
            # Elided, not clipped: this line explains the lattice, and a
            # sentence cut mid-glyph reads as a broken panel (same rule as
            # _paint_title). The key measures the ELIDED width, so it can only
            # take room the sentence really left.
            note = fm.elidedText(self.footer_note(), Qt.ElideRight, w - 24)
            p.drawText(QRectF(10, fy, w - 20, foot_h), Qt.AlignLeft | Qt.AlignVCenter,
                       note)
            self._paint_key(p, pal, fm, w, fy, foot_h,
                            left_used=fm.horizontalAdvance(note) + 24)

    def _key_ends(self) -> tuple[float, float, float] | None:
        """(lowest measured value, highest, the ramp's own span) or None when
        there is no range to key at all — fewer than two measured zones."""
        if self._grid is None or self._grid.size == 0:
            return None
        finite = self._grid[np.isfinite(self._grid)]
        if finite.size < 2:
            return None
        lo_v, hi_v = float(finite.min()), float(finite.max())
        return lo_v, hi_v, max(abs(lo_v), abs(hi_v), NOISE_FLOOR)

    def key_is_readable(self, pal=None) -> bool:
        """Whether the key's two ends actually map to distinguishable colour.

        The key's whole claim is "this colour means this number". The ramp is
        zero-anchored, so a narrow one-signed range (an occupancy map reading
        3.60 .. 3.79) sends every zone to nearly the same colour: ten swatches
        of #f6571c..#ff5414 captioned with two numbers claim a scale that is
        not on screen. Below _KEY_MIN_DELTA the key is dropped and the values
        printed on the zones carry the numbers instead."""
        ends = self._key_ends()
        if ends is None or self._flat_map:
            # A flat map paints every zone the SAME shade on purpose; a ramp
            # beside it would key colours the lattice does not contain.
            return False
        if pal is None:
            pal = theme.current()          # palette read at use time, never held
        lo_v, hi_v, span = ends
        a, b = _heat_color(lo_v / span, pal), _heat_color(hi_v / span, pal)
        return max(abs(a.red() - b.red()), abs(a.green() - b.green()),
                   abs(a.blue() - b.blue())) >= _KEY_MIN_DELTA

    def _paint_key(self, p: QPainter, pal, fm: QFontMetricsF, w: float,
                   fy: float, foot_h: float, left_used: float) -> None:
        """The colour key, labelled with the data's OWN range.

        The ramp used to be captioned "stronger .. weaker", which is a claim
        about z-scored deficits and simply false over the occupancy map the
        same widget draws. The endpoints are the numbers being coloured, so
        the key cannot say anything the data does not — and when those
        endpoints are the same colour there is no scale to key."""
        ends = self._key_ends()
        if ends is None or not self.key_is_readable(pal):
            return
        lo_v, hi_v, span = ends
        lo_t, hi_t = self._fmt.format(lo_v), self._fmt.format(hi_v)
        steps = 10
        cell = fm.horizontalAdvance("#")
        need = fm.horizontalAdvance(lo_t) + fm.horizontalAdvance(hi_t) + steps * cell + 16
        if 10 + left_used + need > w:
            return                              # no room: the numbers already show
        x = w - 10 - need
        base = fy + foot_h / 2 + fm.ascent() / 2 - 1
        p.setPen(QColor(pal.fg_dim))
        p.drawText(QPointF(x, base), lo_t)
        x += fm.horizontalAdvance(lo_t) + 8
        for k in range(steps):
            frac = k / (steps - 1)
            p.setPen(_heat_color((lo_v + (hi_v - lo_v) * frac) / span, pal))
            p.drawText(QPointF(x, base), "#")
            x += cell
        p.setPen(QColor(pal.fg_dim))
        p.drawText(QPointF(x + 8, base), hi_t)


class AsciiTrend(_Ignite, QWidget):
    """Metric-over-runs as a CONNECTED glyph line: '/' and '\\' for slope,
    '|' risers across a jump, and '_-~"' resolving where inside its cell a
    flat segment sits. A dim rule marks the mean of the runs shown, the
    first/min/max points are labelled, the newest carries a value tag, and a
    run axis sits underneath. In RGB mode the columns run the rainbow.

    Two shapes are NOT drawn as a trend, because they are not one:

    * n == 2 draws a single rule between two marked points. A pair of runs
      interpolated over ~170 glyph columns is ~170 marks the eye counts as
      runs, which is the exact thing the run ticks exist to prevent.
    * a spread under the _y_range span floor draws FLAT, on one row, with no
      min/max labels. Anything else is a picture of float noise.

    The run axis names an absolute run number ONLY when a caller states the
    offset (set_data's first_run) — see run_axis_text.
    """

    def __init__(self, title: str = "", fmt: str = "{:.0%}", parent=None) -> None:
        super().__init__(parent)
        self._init_motion()
        self._title = title
        self._fmt = fmt
        self._values: list[float] = []
        self._tag: str | None = None
        self._baseline: float | None = None
        self._first_run: int | None = None
        self.setMinimumHeight(200)

    # ------------------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, values: Sequence[float], tag: str | None = None,
                 baseline: float | None = None, *,
                 first_run: int | None = None) -> None:
        """`baseline` overrides the reference rule (default: the mean of the
        runs shown, which is the only baseline this widget can compute from
        what it was given).

        `first_run` is the true 1-based run number of ``values[0]``. Supply it
        when `values` is a WINDOW of a longer history and the axis may name
        absolute runs; omit it and the axis says "oldest shown" instead of
        naming a run it cannot identify (see run_axis_text).
        """
        self._values = [float(v) for v in values]
        self._tag = tag
        self._baseline = None if baseline is None else float(baseline)
        self._first_run = None if first_run is None else int(first_run)
        if len(self._values) < 2:
            self._ignite_skip()               # one run is not a trend to reveal
            return
        # Origin: the NEWEST run. The history unrolls right to left out of the
        # run that just landed, which is the one the reveal is about.
        self._ignite((0.0, 0.0), _TREND_SPAN)
        self.update()

    def clear(self) -> None:
        self.set_data([])

    def restyle(self, *_pal) -> None:
        self.update()

    def baseline(self) -> float | None:
        """The reference the dim rule marks: the mean of the runs SHOWN, unless
        a caller supplied one. None when there is nothing to average — the rule
        is the widget's only claim of its own, so it never invents a value."""
        if self._baseline is not None:
            return self._baseline
        return float(np.mean(self._values)) if self._values else None

    def run_axis_text(self, ncols: int) -> tuple[str, str]:
        """(left label, right label) for the run axis under the line.

        A label that names a specific run has to BE that run. This widget knows
        only the LIST it was handed, and both callers hand over
        ``profile.history[-60:]`` — so on a 137-run profile the axis printed
        "run 1 · 62%" under what is really run 78 and "run 60 · newest" under
        run 137. Two labels, both naming the wrong run, from an index the
        widget cannot know the offset of.

        So the absolute numbers appear only when the caller states the offset
        (set_data's `first_run`). Without it the axis says what the widget can
        actually support: which end is oldest, how many runs are drawn, and
        whether they all got a column of their own.
        """
        n = len(self._values)
        if n < 2:
            return "", ""            # no axis is drawn: there is no trend yet
        first = self._first_run
        left = (f"run {first} · " if first is not None else "oldest shown · ")
        left += _fmt_value(self._fmt, self._values[0])
        if n == 2:
            # Two runs are a segment, not a trend anyone should read a
            # direction off — the shape says so and the axis has to agree.
            right = "2 runs · segment"
        elif n <= ncols:
            right = (f"run {first + n - 1} · newest" if first is not None
                     else f"{n} runs · newest")
        else:
            right = (f"runs {first}..{first + n - 1} · {ncols} columns"
                     if first is not None else f"{n} runs · {ncols} columns")
        return left, right

    def _y_range(self) -> tuple[float, float, bool]:
        """(lo, hi, flat) for the value axis.

        Two honesty rules live here:

        * a SPAN FLOOR. np.mean over identical values lands one ULP off them
          and the mean is folded into the range, so a perfectly flat history
          used to be spread over ~1e-16: the line on the bottom row, its own
          mean rule nine rows above it. A spread under _FLAT_SPAN_REL of the
          metric's magnitude is float noise, so the range is centred on the
          data and `flat` is returned — the caller draws one row, not a
          full-panel oscillation over a hairline ([0.6666, 0.6667] drew a
          dramatic sawtooth while every label on the panel printed 67%).
        * the two axis labels must FORMAT differently. An axis printing "67%"
          at both ends is not an axis, so the range widens until the strings
          differ (bounded, so a format with no placeholder cannot spin). The
          comparison runs through _fmt_value, i.e. the strings that get
          PAINTED: raw format() calls a hairline negative "-0%", which differs
          from "0%" as a string while reading as the same number on the axis,
          so a range centred on zero stopped widening at once and put a signed
          zero in the gutter.
        """
        src = self._values
        base = float(self.baseline())
        lo, hi = min(min(src), base), max(max(src), base)
        floor = max(_FLAT_SPAN_REL * max(abs(lo), abs(hi)), _FLAT_SPAN_ABS)
        flat = (hi - lo) < floor
        if flat:
            mid = (hi + lo) / 2.0
            lo, hi = mid - floor / 2.0, mid + floor / 2.0
        else:
            pad = (hi - lo) * 0.10
            lo, hi = lo - pad, hi + pad
        # 64, not 24: each pass only DOUBLES the span, and from _FLAT_SPAN_ABS
        # (1e-9) a percentage format needs ~23 of them just to reach 1%. The
        # magnitude test below is the real bound — this count is the backstop.
        for _ in range(64):
            if _fmt_value(self._fmt, lo) != _fmt_value(self._fmt, hi):
                break
            mid = (hi + lo) / 2.0
            if hi - lo > max(1.0, 8.0 * abs(mid)):
                break
            half = max(hi - mid, _FLAT_SPAN_ABS) * 2.0
            lo, hi = mid - half, mid + half
        return lo, hi, flat

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        top = _paint_title(p, pal, self._title, w)
        if len(self._values) < 2:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "not enough runs yet")
            return

        gf = theme.mono(14)
        fm = QFontMetricsF(gf)
        cw = fm.horizontalAdvance("#")
        chh = fm.height()
        lf = theme.mono(12)
        lfm = QFontMetricsF(lf)

        src = self._values
        n = len(src)
        base = float(self.baseline())
        lo, hi, flat = self._y_range()
        # Two runs is a SEGMENT, not a trend: one rule between two marked
        # points. Interpolating them across ~170 glyph columns drew a long
        # steady climb of glyph marks that reads as a long history — exactly
        # what the run ticks below exist to stop a viewer counting.
        segment = n == 2

        tag_text = self._tag if self._tag is not None else _fmt_value(self._fmt, src[-1])
        tag_w = fm.horizontalAdvance(tag_text) + 14
        # every number this widget paints goes through _fmt_value, including the
        # two it measures the gutter against
        hi_txt, lo_txt = _fmt_value(self._fmt, hi), _fmt_value(self._fmt, lo)
        gutter = max(lfm.horizontalAdvance(t) for t in (hi_txt, lo_txt)) + 12
        x_left, x_right = 10 + gutter, w - tag_w - 10
        y_top, y_bot = top + 8, h - _AXIS_H - 6
        if x_right - x_left < 40 or y_bot - y_top < 3 * chh:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "panel too small")
            return
        ncols = max(int((x_right - x_left) // cw), 4)
        grid_rows = max(int((y_bot - y_top) // chh), 3)

        # One column per run when they fit, resampled when there are more runs
        # than columns; either way the line spans the panel and stays joined.
        run_x = np.linspace(0.0, ncols - 1.0, n)
        cvals = np.interp(np.arange(ncols, dtype=float), run_x, np.asarray(src))
        if flat:
            # The runs do not differ enough to draw a shape. Interpolating that
            # difference across the panel drew the noise as an oscillation.
            cvals = np.full(ncols, float(np.mean(src)))
        levels = (cvals - lo) / (hi - lo) * (grid_rows - 1)
        rows_i = np.clip(np.rint(levels).astype(int), 0, grid_rows - 1)
        if flat:
            # SNAP the flat panel to its row. A flat range centres the level
            # mid-cell as often as not, and left fractional that put the mean
            # rule up to half a glyph row off the line it is the mean OF, while
            # the sub-cell glyph leaned out of its cell to reach it — a run of
            # '"' or '_' under a finding that reads "this did not move". On the
            # row centre the line, the rule, the head and every run mark share
            # one y, and the sub-cell glyph is dead centre, i.e. "-".
            levels = rows_i.astype(float)
        flat_level = float(rows_i[0])

        def level_of(v: float) -> float:
            """Row level of one run's value — the flat panel puts every run on
            the same row, so a mark can never sit off the line it belongs to."""
            return flat_level if flat else (v - lo) / (hi - lo) * (grid_rows - 1)

        def ypix(level: float) -> float:
            """Pixel y of a fractional row level (row centres, 0 = bottom)."""
            return y_bot - (level + 0.5) * chh

        # ---- the reference rule: where this metric has been sitting
        p.setPen(QPen(_dim(pal.fg_dim, 0.55), 1, Qt.DotLine))
        by = ypix(level_of(base))
        p.drawLine(QPointF(x_left, by), QPointF(x_right, by))

        # ---- the line itself
        lit = self._ignite_lit()

        def q_at(j: int) -> float:
            """Reveal factor for column j — the wave runs right to left out of
            the newest run, so every mark fades in with its own column."""
            if lit is None:
                return 1.0
            u = (ncols - 1 - j) / max(ncols - 1, 1) * _TREND_SPAN
            return min(max(lit(motion.grid_distance(0, u, self._ign_origin)), 0.0), 1.0)

        p.setFont(gf)
        head_y = ypix(float(levels[-1]))

        # where the actual runs sit; the marks are drawn below, but the segment
        # rule needs the endpoints before anything else is painted
        run_px = x_left + run_x * cw + cw * 0.5
        run_py = np.array([ypix(level_of(v)) for v in src])
        run_q = [q_at(int(round(x))) for x in run_x]

        # one glyph per column, computed before any painting: a slope character
        # where the line is climbing or falling, a sub-cell character where it
        # is flat (that is what recovers resolution a 10-row grid cannot hold)
        chs: list[str] = []
        for j in range(ncols):
            prev = levels[max(j - 1, 0)]
            nxt = levels[min(j + 1, ncols - 1)]
            slope = (nxt - prev) / (1.0 if j in (0, ncols - 1) else 2.0)
            if slope > 0.75:
                chs.append("/")
            elif slope < -0.75:
                chs.append("\\")
            else:
                sub = min(max(levels[j] - rows_i[j] + 0.5, 0.0), 0.999)
                chs.append(_TREND_SUB[int(sub * len(_TREND_SUB))])

        def col_color(j: int) -> QColor:
            return (QColor.fromHsvF((j / max(ncols - 1, 1)) * 0.83, 0.8, 1.0)
                    if pal.rgb else QColor(pal.accent))

        if segment:
            # ONE plain rule, and the two runs marked as points. No glyph
            # columns: 170 of them between two measurements read as 170
            # measurements, and the eye counts marks, not the axis label.
            q = q_at(ncols // 2)
            p.setPen(QPen(_dim(pal.accent, 0.60 * q), 1.6))
            p.drawLine(QPointF(float(run_px[0]), float(run_py[0])),
                       QPointF(float(run_px[-1]), float(run_py[-1])))
            p.setPen(Qt.NoPen)
            p.setBrush(_dim(pal.accent, q))
            p.drawEllipse(QPointF(float(run_px[0]), float(run_py[0])), 3.0, 3.0)
        elif lit is None and not pal.rgb:
            # Idle and one colour: batch neighbours that share a row AND a
            # glyph into a single drawText. Per-column drawing costs ~2.5 ms a
            # frame at 1400px, and this panel repaints on every theme switch,
            # resize and hover elsewhere — not only while animating.
            p.setPen(QColor(pal.accent))
            j = 0
            while j < ncols:
                k = j + 1
                while k < ncols and rows_i[k] == rows_i[j] and chs[k] == chs[j]:
                    k += 1
                p.drawText(QRectF(x_left + j * cw, ypix(float(rows_i[j])) - chh / 2,
                                  (k - j) * cw + cw, chh),
                           Qt.AlignLeft | Qt.AlignVCenter, chs[j] * (k - j))
                j = k
        else:
            for j in range(ncols):
                q = q_at(j)
                if q <= 0.02:
                    continue
                p.setPen(_dim(col_color(j), q))
                p.drawText(QRectF(x_left + j * cw, ypix(float(rows_i[j])) - chh / 2,
                                  cw * 2, chh),
                           Qt.AlignLeft | Qt.AlignVCenter, _ignite_glyph(chs[j], q))

        # When more runs than columns land in one cell, that cell's single
        # glyph cannot show their spread — resampling alone would quietly
        # flatten an oscillating history into a mid-range line. Each column
        # carries the min..max ENVELOPE of the runs that fell in it.
        band_lo, band_hi = rows_i.copy(), rows_i.copy()
        if n > ncols and not flat:
            # `flat` means that spread is float noise: drawing an envelope of it
            # would put a riser under a line the panel says did not move.
            src_rows = np.clip(
                np.rint((np.asarray(src) - lo) / (hi - lo) * (grid_rows - 1)),
                0, grid_rows - 1).astype(int)
            for i, x in enumerate(run_x):
                j = int(round(x))
                band_lo[j] = min(band_lo[j], src_rows[i])
                band_hi[j] = max(band_hi[j], src_rows[i])

        # risers: without them a steep step (or a decimated column) reads as
        # two unrelated marks instead of one line. The segment draws its own
        # rule, so it has nothing to join up.
        for j in range(0 if segment else ncols):
            fill = set(range(int(band_lo[j]), int(band_hi[j]) + 1))
            if j + 1 < ncols:
                a, b = int(rows_i[j]), int(rows_i[j + 1])
                if abs(b - a) >= 2:
                    fill.update(range(min(a, b) + 1, max(a, b)))
            fill.discard(int(rows_i[j]))          # the line's own glyph sits there
            if not fill:
                continue
            q = q_at(j)
            if q <= 0.02:
                continue
            p.setPen(_dim(col_color(j), 0.55 * q))
            for k in sorted(fill):
                p.drawText(QRectF(x_left + j * cw, ypix(float(k)) - chh / 2,
                                  cw * 2, chh),
                           Qt.AlignLeft | Qt.AlignVCenter, "|")

        # ---- where the actual runs are: the line between them is drawn
        # interpolated, so without these a viewer counting glyphs would count
        # data points that do not exist
        if n <= ncols:
            for x, q in zip(run_px, run_q):
                p.setPen(_dim(pal.fg_dim, 0.6 * q))
                p.drawLine(QPointF(float(x), y_bot + 2), QPointF(float(x), y_bot + 5))
        if n <= 30:
            p.setPen(Qt.NoPen)
            for x, y, q in zip(run_px[1:-1], run_py[1:-1], run_q[1:-1]):
                p.setBrush(_dim(pal.accent, 0.5 * q))
                p.drawEllipse(QPointF(float(x), float(y)), 1.7, 1.7)

        # ---- marks: first, min, max, newest
        first_x = float(run_px[0])
        last_x = float(run_px[-1])
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(_dim(pal.fg_dim, 0.9 * run_q[0]), 1.4))
        p.drawEllipse(QPointF(first_x, float(run_py[0])), 3.0, 3.0)
        i_max, i_min = int(np.argmax(src)), int(np.argmin(src))
        p.setFont(lf)
        for i, kind in ((i_max, "max"), (i_min, "min")):
            if flat or i in (0, n - 1) or src[i_max] == src[i_min]:
                # `flat`: the spread between them is float noise, and "max 67%"
                # over "min 67%" on one flat line is two labels claiming a
                # difference the picture (rightly) refuses to draw.
                continue                       # the tag or the first mark has it
            jx = float(run_px[i])
            py = float(run_py[i])
            if min(abs(jx - first_x), abs(jx - last_x)) < 4 * cw:
                continue                       # would collide with those marks
            q = run_q[i]
            p.setPen(QPen(_dim(pal.fg_dim, 0.9 * q), 1.2))
            p.drawEllipse(QPointF(jx, py), 2.5, 2.5)
            txt = f"{kind} {_fmt_value(self._fmt, src[i])}"
            tw = lfm.horizontalAdvance(txt)
            tx = min(max(jx - tw / 2, x_left), x_right - tw)
            ty = py - chh - 4 if kind == "max" else py + 4
            ty = min(max(ty, y_top - 2), y_bot - 12)
            p.setPen(_dim(pal.fg_dim, q))
            p.drawText(QRectF(tx, ty, tw + 4, 14), Qt.AlignLeft | Qt.AlignVCenter, txt)
        head_col = QColor.fromHsvF(0.83, 0.8, 1.0) if pal.rgb else QColor(pal.accent)
        p.setPen(Qt.NoPen)
        p.setBrush(head_col)
        p.drawEllipse(QPointF(last_x, head_y), 3.5, 3.5)

        # ---- the newest value, riding the line's end
        p.setFont(gf)
        p.setPen(QColor(pal.accent))
        ty = min(max(head_y, y_top + chh / 2), y_bot - chh / 2)
        p.drawText(QRectF(x_right + 6, ty - chh / 2, tag_w, chh),
                   Qt.AlignLeft | Qt.AlignVCenter, tag_text)

        # ---- the mean's label LAST: the line would otherwise draw over it.
        # OPAQUE backing, sized to the label's own line box. On a flat panel the
        # line sits exactly on the mean rule (it IS the mean), so this chip
        # always lands on the line's glyph run — and at 0.85 the glyphs read
        # straight through the text in both themes. A label you cannot read is
        # not a label, and the line loses ~9 characters of an unchanging rule.
        btxt = f"mean {_fmt_value(self._fmt, base)}"
        bth = lfm.height() + 4
        btw = lfm.horizontalAdvance(btxt) + 10
        chip = QRectF(x_right - btw, by - bth / 2, btw, bth)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal.bg))
        p.drawRoundedRect(chip, 3, 3)
        p.setFont(lf)
        p.setPen(QColor(pal.fg_dim))
        p.drawText(chip.adjusted(5, 0, -5, 0), Qt.AlignLeft | Qt.AlignVCenter, btxt)

        # ---- y scale in the gutter, run axis underneath
        p.setFont(lf)
        p.setPen(QColor(pal.fg_dim))
        # hi_txt/lo_txt, not self._fmt.format: the widening loop above can
        # centre a hairline range on zero, and raw format() prints its low end
        # "-0%" — a signed zero on the axis reads as a rendering bug.
        p.drawText(QRectF(6, ypix(float(grid_rows - 1)) - 7, gutter - 10, 14),
                   Qt.AlignRight | Qt.AlignVCenter, hi_txt)
        p.drawText(QRectF(6, ypix(0.0) - 7, gutter - 10, 14),
                   Qt.AlignRight | Qt.AlignVCenter, lo_txt)
        p.setPen(QPen(_dim(pal.border, 1.0), 1, Qt.DotLine))
        p.drawLine(QPointF(x_left, y_bot + 2), QPointF(x_right, y_bot + 2))
        p.setPen(QColor(pal.fg_dim))
        # The axis says what the picture is, and never names a run it cannot
        # identify — run_axis_text owns both rules.
        left_txt, right_txt = self.run_axis_text(ncols)
        p.drawText(QRectF(x_left, h - _AXIS_H, (x_right - x_left) / 2, _AXIS_H - 1),
                   Qt.AlignLeft | Qt.AlignVCenter, left_txt)
        p.drawText(QRectF(x_left + (x_right - x_left) / 2, h - _AXIS_H,
                          (x_right - x_left) / 2, _AXIS_H - 1),
                   Qt.AlignRight | Qt.AlignVCenter, right_txt)
