"""In-game performance overlay: frameless, translucent, always-on-top.

A compact card that floats over the game showing live session state: this
run against your baseline, score, current difficulty, fatigue, input health,
and a baseline-anchored accuracy spark. Click-through by default (pure Qt:
WindowTransparentForInput — no hooks, no injection, invisible to the game);
Unlock mode disables that so the card can be dragged, and the position
persists in settings.

KovaaK's must run Borderless or Windowed for any overlay to be visible —
true exclusive fullscreen bypasses the compositor. The Dashboard hint says
so next to the toggle.

Two painted surfaces carry the data, in gui/viz.py's character-art language
(its ramps are restated here rather than imported — viz.py is the Analysis
tab's module and this card must stay standalone):

    _StatDeck        caption/value rows on a monospace lattice, so a digit
                     stays in its column as the value updates mid-session;
                     a glyph meter is drawn only on rows whose value has a
                     real bounded range (size, movement, fatigue)
    _BaselineSpark   one glyph column per run against a FIXED reference
                     line — see its docstring for why that matters

Translucency is BACKGROUND alpha, not window opacity: setWindowOpacity
fades the glyphs along with the panel, which is exactly what made the card
read muddy at low settings. The panel and its border scale with
Settings.overlay_opacity; the text stays fully opaque.

Painting stays featherweight — this redraws while the game runs. The only
animation is the newest spark column breathing at ~15 Hz; it repaints that
column's rect alone, and its timer never runs while the overlay is hidden
or no session is live.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..config import Settings
from . import theme

_CARD_WIDTH = 272
_SPARK_RUNS = 40
_MARGIN = 24            # default distance from the screen's top-right corner

# Character ramps, same conventions as gui/viz.py.
_BAR_RAMP = "@#*+=-:."           # dense at the anchor -> light at the tip
_TREND_RAMP = ".:-=+*#"          # light -> dense: fraction of a cell filled
_METER_CELLS = 7

# The spark's scale. The half-height NEVER shrinks below _MIN_HALF_SPAN, so a
# session that drifts a point renders as a drift and not as a climb; it grows
# only when a run actually leaves that band.
_MIN_HALF_SPAN = 0.10            # +/- 10 accuracy points around the reference
_FLAT_EPS = 0.005                # under half a point: neither up nor down
_SPARK_ROWS = 5                  # odd: a true center row for the reference
_TICK_MS = 66                    # ~15 Hz; the head marker only

_BASE_FLAGS = (
    Qt.FramelessWindowHint
    | Qt.WindowStaysOnTopHint
    | Qt.Tool
    | Qt.WindowDoesNotAcceptFocus
)


def _mono_css(px: int = 12) -> str:
    """Mono face for a QLabel, as STYLESHEET text.

    A stylesheet outranks setFont, so setting the face on the widget looked
    right and rendered in the app-wide "Segoe UI" rule anyway — which is how
    the header's run counter ended up on a proportional face. `px` must be a
    theme.CELL_SIZES value or the character cell lands off the pixel grid.
    """
    return f'font-family: "{theme.mono_family()}"; font-size: {px}px;'


def _role_color(pal, role: str) -> QColor:
    """Palette role name -> color, resolved at PAINT time (never cached)."""
    return QColor({
        "fg": pal.fg, "dim": pal.fg_dim, "good": pal.good,
        "warn": pal.warn, "bad": pal.bad, "accent": pal.accent,
    }.get(role, pal.fg))


def _paint_meter(p: QPainter, pal, x: float, y: float, cw: float, ch: float,
                 frac: float, color: QColor) -> None:
    """viz.AsciiBars' glyph run, shrunk to _METER_CELLS: dense at the anchor,
    fading toward the tip, unfilled cells as dim dots."""
    filled = int(round(min(max(frac, 0.0), 1.0) * _METER_CELLS))
    for j in range(_METER_CELLS):
        if j < filled:
            f = j / max(filled - 1, 1)
            glyph = _BAR_RAMP[int(f * (len(_BAR_RAMP) - 1))]
            col = QColor(color)
            col.setAlphaF(1.0 - 0.35 * f)
        else:
            glyph = "."
            col = QColor(pal.border)
        p.setPen(col)
        p.drawText(QRectF(x + j * cw, y, cw * 2, ch),
                   Qt.AlignLeft | Qt.AlignVCenter, glyph)


class _StatDeck(QWidget):
    """The card's numbers on a monospace lattice.

    Values are stored as (text, palette-role) segments and the palette is
    resolved in paintEvent, so restyle() is nothing but update(). Everything
    is drawn in theme.mono: with a proportional face every digit change
    reflowed the whole row, and on a surface that updates between runs while
    you are aiming, dancing digits are the thing you notice.
    """

    ROWS = (("acc", "ACC"), ("score", "SCORE"), ("size", "SIZE"),
            ("move", "MOVE"), ("fatigue", "FATIGUE"), ("input", "INPUT"))

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vals: dict[str, list[tuple[str, str]]] = {}
        self._meters: dict[str, tuple[float, str]] = {}
        fm = QFontMetricsF(self.value_font())
        self._row_h = max(fm.height(), 15.0) + 1.0
        self.setFixedHeight(int(self._row_h * len(self.ROWS)) + 7)
        self.clear()

    # ------------------------------------------------------------------
    def value_font(self):
        return theme.mono(14)

    def caption_font(self):
        return theme.mono(12)

    def set_row(self, key: str, segments: list[tuple[str, str]],
                meter: float | None = None, meter_role: str = "accent") -> None:
        """segments: (text, role) run left to right, right-aligned as a block.
        `meter` is a 0..1 fraction — pass it ONLY where the value has a real
        bounded range, otherwise the bar is decoration pretending to be data."""
        self._vals[key] = list(segments)
        if meter is None:
            self._meters.pop(key, None)
        else:
            self._meters[key] = (float(meter), meter_role)
        self.update()

    def row_text(self, key: str) -> str:
        return "".join(text for text, _ in self._vals.get(key, []))

    def clear(self) -> None:
        self._vals = {key: [("—", "dim")] for key, _ in self.ROWS}
        self._meters = {}
        self.update()

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w = self.width()

        cap_f, val_f = self.caption_font(), self.value_font()
        cap_fm, val_fm = QFontMetricsF(cap_f), QFontMetricsF(val_f)
        cap_cw = max(cap_fm.horizontalAdvance("M"), 4.0)
        val_cw = max(val_fm.horizontalAdvance("M"), 4.0)
        cap_w = cap_cw * 8.0                       # "FATIGUE" + one space
        meter_w = cap_cw * (_METER_CELLS + 1)

        # hairline rule separating the header from the numbers
        p.setPen(QColor(pal.border))
        p.drawLine(0, 1, w, 1)

        y = 5.0
        for key, caption in self.ROWS:
            p.setFont(cap_f)
            p.setPen(QColor(pal.fg_dim))
            p.drawText(QRectF(0, y, cap_w, self._row_h),
                       Qt.AlignLeft | Qt.AlignVCenter, caption)

            meter = self._meters.get(key)
            if meter is not None:
                frac, role = meter
                _paint_meter(p, pal, cap_w, y, cap_cw, self._row_h, frac,
                             _role_color(pal, role))

            # right-aligned value block: monospace, so the block width is
            # exactly its character count and columns line up run to run
            p.setFont(val_f)
            segs = self._vals.get(key, [])
            total = sum(len(text) for text, _ in segs) * val_cw
            x = max(w - total, cap_w + meter_w)
            for text, role in segs:
                p.setPen(_role_color(pal, role))
                p.drawText(QRectF(x, y, len(text) * val_cw + val_cw, self._row_h),
                           Qt.AlignLeft | Qt.AlignVCenter, text)
                x += len(text) * val_cw
            y += self._row_h


class _BaselineSpark(QWidget):
    """Session accuracy as glyph columns against a FIXED reference line.

    The old sparkline normalised to its own min/max, so a session drifting
    61% -> 62% painted a dramatic climb: the scale was invented out of the
    very data it was measuring, and every session looked eventful. Same bug
    class as the heatmap's min-max ramp.

    Now the anchor is the player's baseline accuracy (the profile EWMA,
    captured ONCE at the session's first report and held — a reference that
    folds in each new run chases the session, and a reference that chases
    can never show the session moving), and the half-span is
    max(largest deviation, _MIN_HALF_SPAN). A flat session therefore sits on
    the line; the scale only opens up when a run genuinely leaves the band.
    The header states which reference is in force and how wide the band is:
    a column height means nothing without them.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._values: list[float] = []
        self._ref: float | None = None
        self._ref_kind = "base"
        self._live = False
        fm = QFontMetricsF(theme.mono(12))
        self._cw = max(fm.horizontalAdvance("M"), 4.0)
        self._ch = max(fm.height(), 10.0)
        self._pad = 3.0
        self.setFixedHeight(int(self._ch * (_SPARK_ROWS + 1) + self._pad * 2))
        self._anim = QTimer(self)
        self._anim.setInterval(_TICK_MS)
        self._anim.timeout.connect(self._beat)

    # ------------------------------------------------------------------ data
    def set_reference(self, value: float | None, kind: str = "base") -> None:
        self._ref = None if value is None else float(value)
        self._ref_kind = kind
        self.update()

    def reference(self) -> tuple[float | None, str]:
        """(value, "base" | "avg") — what the line the columns hang off IS."""
        return self._ref, self._ref_kind

    def set_values(self, values: list[float]) -> None:
        self._values = [float(v) for v in values][-_SPARK_RUNS:]
        self._sync_timer()
        self.update()

    def set_live(self, live: bool) -> None:
        """A live session is what earns the head marker its frames."""
        self._live = bool(live)
        self._sync_timer()
        self.update()

    def clear(self) -> None:
        self._values = []
        self._ref = None
        self._ref_kind = "base"
        self._sync_timer()
        self.update()

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------- geometry
    def _lay(self) -> tuple[float, float, float, int]:
        """(x0, y_top, x_end, ncols) of the glyph lattice. y_top is the top
        row; the reference row sits _SPARK_ROWS // 2 rows below it."""
        cw = self._cw
        x0 = 2.0
        tag_w = 4.0 * cw                       # room for the "100%" tag
        ncols = max(int((self.width() - 4.0 - tag_w - cw - x0) // cw), 4)
        return x0, self._pad + self._ch, x0 + ncols * cw, ncols

    def visible(self) -> list[float]:
        """The runs that actually fit on the lattice, oldest first."""
        return self._values[-self._lay()[3]:] if self._values else []

    def span(self) -> float:
        """Half-height of the chart in accuracy points. Never below the floor
        — that floor is the whole reason a flat session reads flat."""
        vals, ref = self.visible(), self._ref
        if ref is None or not vals:
            return _MIN_HALF_SPAN
        return max(max(abs(v - ref) for v in vals), _MIN_HALF_SPAN)

    def levels(self) -> list[float]:
        """Each visible run's deviation from the reference in [-1, +1], where
        1 is the top of the chart. Flat session -> everything near zero."""
        vals, ref = self.visible(), self._ref
        if ref is None or not vals:
            return []
        span = self.span()
        return [max(min((v - ref) / span, 1.0), -1.0) for v in vals]

    def _head_rect(self) -> QRect | None:
        """Just the newest column — the only thing the animation touches."""
        if not self._values or self._ref is None:
            return None
        _, y_top, x_end, _ = self._lay()
        return QRect(int(x_end - self._cw - 4), int(y_top - 2),
                     int(self._cw + 10), int(self._ch * _SPARK_ROWS + 4))

    # ------------------------------------------------------------ animation
    def _sync_timer(self) -> None:
        """Never a busy timer on a hidden widget, or on a dead session."""
        want = self._live and self.isVisible() and bool(self._values) \
            and self._ref is not None
        if want and not self._anim.isActive():
            self._anim.start()
        elif not want and self._anim.isActive():
            self._anim.stop()

    def _beat(self) -> None:
        rect = self._head_rect()
        if rect is not None:
            self.update(rect)               # dirty-rect: ~15x80 px, not the card

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event) -> None:
        self._anim.stop()
        super().hideEvent(event)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(theme.mono(12))
        cw, ch, w = self._cw, self._ch, self.width()

        vals, ref = self.visible(), self._ref
        if ref is None or not vals:
            col = QColor(pal.fg_dim)
            col.setAlphaF(0.75)
            p.setPen(col)
            p.drawText(QRectF(0, 0, w, self.height()), Qt.AlignCenter,
                       "· waiting for runs ·")
            return

        x0, y_top, x_end, ncols = self._lay()
        span = self.span()
        half = (_SPARK_ROWS - 1) // 2                  # glyph rows per side
        base_y = y_top + half * ch

        # ---- header: the scale, stated. A column height is meaningless
        # without the reference it is measured from and the band it spans.
        p.setPen(QColor(pal.fg_dim))
        head_r = QRectF(x0, self._pad - 1, w - x0 - 4, ch)
        p.drawText(head_r, Qt.AlignLeft | Qt.AlignVCenter,
                   "ACC vs BASELINE" if self._ref_kind == "base"
                   else "ACC vs SESSION AVG")
        p.drawText(head_r, Qt.AlignRight | Qt.AlignVCenter,
                   f"±{span * 100:.0f}pp")

        # only the columns the update region actually covers get drawn
        dirty = QRectF(event.rect()).adjusted(-cw, -ch, cw, ch)
        j0 = max(int((dirty.left() - x0) // cw), 0)
        j1 = min(int((dirty.right() - x0) // cw) + 1, ncols)

        # ---- the reference line itself, as a run of glyphs
        line = QColor(pal.fg_dim)
        line.setAlphaF(0.85)
        p.setPen(line)
        for j in range(j0, j1):
            p.drawText(QRectF(x0 + j * cw, base_y, cw * 2, ch),
                       Qt.AlignLeft | Qt.AlignVCenter, "·")
        p.setPen(QColor(pal.fg))
        p.drawText(QRectF(x_end + cw * 0.5, base_y, w - x_end, ch),
                   Qt.AlignLeft | Qt.AlignVCenter, f"{ref:.0%}")

        # ---- one column per run, newest at the right edge
        offset = ncols - len(vals)
        for j in range(max(j0, offset), j1):
            v = vals[j - offset]
            dev = v - ref
            # cells = position inside (half + 1) rows, so the extreme run
            # lands densely on the outermost row instead of faintly past it
            cells = min(abs(dev) / span, 0.9999) * (half + 1)
            k = int(cells)
            sub = cells - k
            up = dev > 0.0
            if abs(dev) < _FLAT_EPS:
                base_col = QColor(pal.fg_dim)
            else:
                base_col = QColor(pal.good if up else pal.bad)

            x = x0 + j * cw
            for m in range(k):               # dim stem back to the line
                fill = QColor(base_col)
                fill.setAlphaF(0.20 + 0.22 * (m / max(k, 1)))
                p.setPen(fill)
                p.drawText(QRectF(x, base_y + (-m if up else m) * ch, cw * 2, ch),
                           Qt.AlignLeft | Qt.AlignVCenter, ":")
            gy = base_y + (-k if up else k) * ch
            p.setPen(base_col)
            p.drawText(QRectF(x, gy, cw * 2, ch), Qt.AlignLeft | Qt.AlignVCenter,
                       _TREND_RAMP[int(sub * (len(_TREND_RAMP) - 1))])

            if j == ncols - 1:               # the newest run, breathing
                alpha = 1.0
                if self._live:
                    alpha = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(
                        2.0 * math.pi * 0.8 * time.monotonic()))
                mark = QColor(base_col)
                mark.setAlphaF(alpha)
                p.setPen(Qt.NoPen)
                p.setBrush(mark)
                p.drawEllipse(QPointF(x + cw * 0.5, gy + ch * 0.5), 2.6, 2.6)


class OverlayWindow(QWidget):
    """Top-level overlay card (create with no parent)."""

    def __init__(self, settings: Settings) -> None:
        flags = _BASE_FLAGS
        if settings.overlay_clickthrough:
            flags |= Qt.WindowTransparentForInput
        super().__init__(None, flags)
        self.s = settings
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("background: transparent;")
        self.setFixedWidth(_CARD_WIDTH)
        # Opacity is applied to the PANEL in paintEvent, never here:
        # setWindowOpacity composites the whole window, glyphs included, and
        # half-faded text over a moving game is what made this read muddy.
        self.setWindowOpacity(1.0)
        self._unlocked = False
        self._live = False
        self._drag_from: QPoint | None = None
        self._session_runs = 0
        self._accs: list[float] = []
        # Session references, frozen at the first report (see _BaselineSpark).
        self._base_acc: float | None = None
        self._base_score: float | None = None

        self.title = QLabel("kovadapt")
        self.scenario = QLabel("")
        self.scenario.setWordWrap(True)
        self.status = QLabel("not watching")
        self.hint = QLabel("drag to move · lock from the Dashboard")
        self.hint.hide()

        self.deck = _StatDeck()
        self.spark = _BaselineSpark()

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(3)
        lay.addLayout(head)
        lay.addWidget(self.scenario)
        lay.addWidget(self.deck)
        lay.addWidget(self.spark)
        lay.addWidget(self.hint)
        self.restyle()

    # ----------------------------------------------------------------- theme
    def restyle(self, *_pal) -> None:
        pal = theme.current()
        f = "font-size: 12px; background: transparent;"
        mono = f"{_mono_css(12)} background: transparent;"
        self.title.setStyleSheet(f"{mono} font-weight: 700; color: {pal.accent};")
        self.scenario.setStyleSheet(f"{f} color: {pal.fg};")
        self.status.setStyleSheet(f"{mono} color: {pal.fg_dim};")
        self.hint.setStyleSheet(f"font-size: 11px; color: {pal.warn}; background: transparent;")
        self.deck.restyle()
        self.spark.restyle()
        self.update()

    def panel_alpha(self) -> int:
        """Card-background alpha for the configured opacity. This is what the
        opacity slider drives — the text keeps its own full alpha."""
        pal = theme.current()
        base = 235 if pal.is_dark else 245
        return int(base * max(0.0, min(1.0, self.s.overlay_opacity)))

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        alpha = self.panel_alpha()
        bg = QColor(pal.bg)
        bg.setAlpha(alpha)
        p.setBrush(bg)
        border = QColor(pal.warn if self._unlocked else pal.border)
        border.setAlpha(max(alpha, 140))     # the edge stays findable when faint
        pen = QPen(border, 2 if self._unlocked else 1)
        if self._unlocked:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)

    # ------------------------------------------------------------ visibility
    def show_overlay(self) -> None:
        self._place()
        self.show()

    def _place(self) -> None:
        from PySide6.QtGui import QGuiApplication

        # (-1, -1) is the only "never dragged" sentinel — single coordinates
        # are legitimately negative on monitors left of/above the primary.
        if (self.s.overlay_x, self.s.overlay_y) != (-1, -1):
            self.move(self.s.overlay_x, self.s.overlay_y)
            if any(scr.geometry().intersects(self.frameGeometry())
                   for scr in QGuiApplication.screens()):
                return
            # Saved position is on a monitor that no longer exists — fall
            # through to the default corner instead of showing off-screen.
        screen = self.screen() or None
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.adjustSize()
        self.move(geo.right() - self.width() - _MARGIN, geo.top() + _MARGIN)

    def set_unlocked(self, unlocked: bool) -> None:
        """Unlocked = draggable (input NOT transparent); locked = click-through."""
        if unlocked == self._unlocked:
            return
        self._unlocked = unlocked
        visible = self.isVisible()
        flags = _BASE_FLAGS
        if not unlocked and self.s.overlay_clickthrough:
            flags |= Qt.WindowTransparentForInput
        pos = self.pos()
        self.setWindowFlags(flags)   # re-creates the native window
        self.move(pos)
        self.hint.setVisible(unlocked)
        if visible:
            self.show()
        self.update()

    def set_opacity(self, value: float) -> None:
        self.s.overlay_opacity = max(0.3, min(1.0, value))
        self.update()                # panel alpha only; glyphs stay crisp

    # -------------------------------------------------------------- dragging
    def mousePressEvent(self, event) -> None:
        if self._unlocked and event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._unlocked and self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event) -> None:
        if self._unlocked and self._drag_from is not None:
            self._drag_from = None
            self.s.overlay_x, self.s.overlay_y = self.pos().x(), self.pos().y()
            try:
                self.s.save()
            except OSError:
                pass

    # ------------------------------------------------------------------ data
    def start_session(self, scenario: str) -> None:
        self._session_runs = 0
        self._accs = []
        self._base_acc = self._base_score = None
        self._live = True
        self.scenario.setText(scenario)
        self.deck.clear()
        self.spark.clear()
        self.spark.set_live(True)
        self._render_status()

    def stop_session(self) -> None:
        self._live = False
        self.spark.set_live(False)
        self._render_status()

    def _render_status(self) -> None:
        if not self._live:
            self.status.setText("not watching")
        elif self._session_runs:
            self.status.setText(f"watching · {self._session_runs} run"
                                f"{'' if self._session_runs == 1 else 's'}")
        else:
            self.status.setText("watching")

    def _reference(self) -> tuple[float | None, str]:
        """(accuracy reference, what it is). The profile EWMA when we have
        one; otherwise the session's own mean, LABELLED as such — the card
        never presents a made-up anchor as your baseline."""
        if self._base_acc is not None:
            return self._base_acc, "base"
        if self._accs:
            return sum(self._accs) / len(self._accs), "avg"
        return None, "avg"

    def on_report(self, rep, profile=None) -> None:
        """New RunReport from the watcher (and the freshly saved profile)."""
        self._session_runs += 1
        self._accs = (self._accs + [float(rep.accuracy)])[-_SPARK_RUNS:]

        # Freeze the references on the first report we see. The profile handed
        # over here has ALREADY folded this run in, so re-reading the EWMA
        # every run gives a baseline that walks with the session — against
        # which even a real climb looks flat.
        if profile is not None and self._base_acc is None:
            # And require MORE than one run behind it: observe_run seeds
            # ewma_accuracy = accuracy exactly when run_count is 0, so on a
            # fresh scenario the "baseline" would be this very run — the card
            # would draw your own run as the line it measures you against.
            # Until then the labelled session-average reference is the honest
            # one.
            ewma = float(getattr(profile, "ewma_accuracy", 0.0) or 0.0)
            if ewma > 0.0 and int(getattr(profile, "run_count", 0) or 0) > 1:
                self._base_acc = ewma
                self._base_score = float(getattr(profile, "ewma_score", 0.0) or 0.0)

        ref, kind = self._reference()
        self.spark.set_reference(ref, kind)
        self.spark.set_values(self._accs)
        self._render_status()

        acc = float(rep.accuracy)
        segs: list[tuple[str, str]] = []
        if ref is None:
            segs.append((f"{acc:.1%}", "fg"))
        else:
            dev = acc - ref
            role = "good" if dev >= _FLAT_EPS else ("bad" if dev <= -_FLAT_EPS else "fg")
            segs.append((f"{acc:.1%}", role))
            segs.append((f" {dev * 100:+.1f}pp", "dim"))
        self.deck.set_row("acc", segs)

        score: list[tuple[str, str]] = [(f"{rep.score:.0f}", "fg")]
        if self._base_score:
            score.append((f" {rep.score - self._base_score:+.0f}", "dim"))
        self.deck.set_row("score", score)

        if profile is not None:
            lo, hi = self.s.min_target_scale, self.s.max_target_scale
            scale = float(profile.target_scale)
            # The meter reads the target SIZE inside its allowed band (full =
            # biggest = easiest); inverting it to "difficulty" would put a
            # full bar next to a small number.
            self.deck.set_row("size", [(f"{scale:.2f}x", "fg")],
                              meter=(scale - lo) / max(hi - lo, 1e-9))
            move = float(profile.movement)
            self.deck.set_row("move", [(f"{move:.2f}", "fg")], meter=move)

        fat = rep.fatigue or {}
        need = max(int(self.s.fatigue_min_runs), 2)
        runs = int(fat.get("runs", 0))
        if runs < need:
            # Say what is missing rather than report the default "fresh" as a
            # verdict the tracker has not actually reached yet.
            self.deck.set_row("fatigue", [(f"{runs}/{need} runs", "dim")])
        else:
            level = str(fat.get("level", "fresh"))
            role = {"fresh": "good", "declining": "warn"}.get(level, "bad")
            self.deck.set_row("fatigue", [(level, role)],
                              meter=float(fat.get("score", 0.0)), meter_role=role)

        ih = rep.input_health or {}
        if ih.get("polling_hz_est"):
            jit = float(ih.get("jitter_ms", 0.0))
            self.deck.set_row("input", [
                (f"{ih['polling_hz_est']:.0f}Hz", "fg"),
                (f" ±{jit:.1f}ms", "good" if jit <= 1.0 else "warn"),
            ])
        else:
            self.deck.set_row("input", [("no telemetry", "dim")])
