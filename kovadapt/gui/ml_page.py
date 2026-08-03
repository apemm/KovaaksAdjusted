"""How it learns — the frozen model explained in prose, with live ASCII diagrams.

The final page-space section: a detailed explainer of the adaptation model as
it actually ships (the 85-95 accuracy governor, the Fitts throughput
sub-controller, pace-plateau progression, OU movement, and the Thompson
bandit over the 5x5 amplitude-aware zone grid). Written per the standing
cite-everything rule: every part carries a dim sources line naming the
analysis/kb.py ids behind it, with the full citations in the tooltip, and
kovadapt's own constructs are labeled as such in the text.

Diagrams follow ascii_art.py's craft: QPainter-drawn character cells from the
Bourke density ramp, monospace, colors read from theme.current() at paint
time (never cached across theme switches). Each diagram runs a ~12 fps timer
only while visible (show/hideEvent), and every random sequence is seeded at
construction so a given frame is deterministic.

API contract (wired into the shell by gui/app.py): ``MLPage(settings)`` with
``.restyle(pal)``; root objectName "tabPage" so the backdrop shows through;
plain vertical layout — the shell owns page-space scrolling, so no scroll
area is nested here.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..adapt.stochastic import OrnsteinUhlenbeck
from ..analysis import kb
from ..config import Settings
from . import theme
from .ascii_art import _RAMP, _mono
from .onboarding import HintBar

_RLEN = len(_RAMP) - 1


def _ramp_char(i: float) -> str:
    """Character for intensity i in [0, 1] from the shared density ramp."""
    return _RAMP[min(int(max(i, 0.0) * _RLEN + 0.5), _RLEN)]


def _hash01(a: int, b: int) -> float:
    """Deterministic per-(cell, frame) pseudo-noise (ascii_art's sin hash)."""
    return (math.sin(a * 12.9898 + b * 78.233) * 43758.5453) % 1.0


# ------------------------------------------------------------------ diagrams
class _Diagram(QWidget):
    """Base for the live ASCII diagrams: a ~12 fps phase timer that runs only
    while the widget is visible, theme colors read at paint time."""

    INTERVAL_MS = 80

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

    def _advance(self) -> None:
        self._phase += 1
        self.update()

    def showEvent(self, event) -> None:
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def restyle(self, *_pal) -> None:
        self.update()   # colors are read from theme.current() in paintEvent


class ZoneGridDiagram(_Diagram):
    """A Thompson round on the 5x5 zone grid: cells flicker as their
    posteriors are sampled, the worst draw locks in as the focus — and every
    so often a wide, barely-visited posterior wins instead (exploration)."""

    GRID = 5
    ROUND = 24            # frames per round (~2 s)
    X0, Y0, CW, CH = 8.0, 6.0, 22.0, 17.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(276, 100)
        means = np.zeros((self.GRID, self.GRID))
        for r in range(self.GRID):
            for c in range(self.GRID):
                ring = max(abs(r - 2), abs(c - 2)) / 2.0
                means[r, c] = 0.08 + 0.30 * ring     # edges weaker (doctrine)
        stds = np.full((self.GRID, self.GRID), 0.10)
        means[1, 4] = 0.68                           # the mapped weakness
        means[3, 0] = 0.46
        stds[0, 0] = 0.30                            # barely-visited arms:
        stds[4, 4] = 0.30                            # wide posteriors
        stds[2, 2] = 0.26
        span = means.max() - means.min()
        self._meannorm = (means - means.min()) / max(span, 1e-9)
        best = np.unravel_index(int(np.argmax(means)), means.shape)
        rng = np.random.default_rng(20260728)
        self._draws: list[np.ndarray] = []
        self._winners: list[tuple[int, int]] = []
        self._explore: list[bool] = []
        for _ in range(48):
            d = means + stds * rng.standard_normal(means.shape)
            self._draws.append(d)
            w = np.unravel_index(int(np.argmax(d)), d.shape)
            self._winners.append((int(w[0]), int(w[1])))
            self._explore.append(w != best)

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        font = _mono()
        font.setPixelSize(15)
        p.setFont(font)
        k = (self._phase // self.ROUND) % len(self._winners)
        u = (self._phase % self.ROUND) / self.ROUND
        draws = self._draws[k]
        norm = (draws - draws.min()) / max(float(draws.max() - draws.min()), 1e-9)
        wr, wc = self._winners[k]
        explore = self._explore[k]
        locked = u >= 0.45
        for r in range(self.GRID):
            for c in range(self.GRID):
                x = self.X0 + c * self.CW
                y = self.Y0 + r * self.CH
                if locked and (r, c) == (wr, wc):
                    col = QColor(pal.warn if explore else pal.accent)
                    if u < 0.55:
                        # The lock-in pop has to move AWAY from the page.
                        # lighter() raises HSV value, which is emphasis on a
                        # dark page and de-emphasis on cream — and every accent
                        # is fitted to land at exactly 4.5:1, so on light the
                        # "winner" flashed DIMMER than the ordinary ramp cells
                        # around it while the sidebar word beside it held 4.5.
                        # Both halves of the same beat, disagreeing.
                        col = col.lighter(130) if pal.is_dark else col.darker(125)
                    ch = "@"
                else:
                    if locked:
                        i = 0.15 + 0.55 * float(self._meannorm[r, c])
                    else:
                        flick = 0.55 + 0.45 * _hash01(r * self.GRID + c, self._phase)
                        i = (0.2 + 0.8 * float(norm[r, c])) * flick
                    ch = _ramp_char(0.18 + 0.72 * i)
                    col = QColor(pal.fg)
                    col.setAlphaF(0.16 + 0.60 * i)
                p.setPen(col)
                p.drawText(QRectF(x, y, self.CW, self.CH), Qt.AlignCenter, ch)
        sx = self.X0 + self.GRID * self.CW + 16
        font.setPixelSize(12)
        p.setFont(font)
        if locked:
            p.setPen(QColor(pal.fg))
            p.drawText(QRectF(sx, self.Y0 + 18, 130, 16), Qt.AlignLeft,
                       f"focus r{wr}c{wc}")
            p.setPen(QColor(pal.warn if explore else pal.accent))
            p.drawText(QRectF(sx, self.Y0 + 38, 130, 16), Qt.AlignLeft,
                       "explore" if explore else "exploit")
        else:
            p.setPen(QColor(pal.fg_dim))
            p.drawText(QRectF(sx, self.Y0 + 18, 130, 16), Qt.AlignLeft, "sampling…")


class DeadbandDiagram(_Diagram):
    """The accuracy governor as a thermostat: the accuracy dot drifts against
    the bracketed 85-95 band and target size reacts only outside it."""

    LOW, HIGH = 0.85, 0.95
    W, H = 440, 58

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.W, self.H)

    def _acc(self) -> float:
        t = self._phase * 0.08
        return 0.90 + 0.052 * math.sin(0.23 * t) + 0.028 * math.sin(0.61 * t + 1.7)

    def _x(self, acc: float) -> float:
        return 66.0 + (acc - 0.70) / 0.30 * (self.W - 78.0)

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        font = _mono()
        font.setPixelSize(15)
        p.setFont(font)
        acc = self._acc()
        in_band = self.LOW <= acc <= self.HIGH
        y = 8.0

        dim = QColor(pal.fg_dim)
        dim.setAlphaF(0.5)
        p.setPen(dim)
        for i in range(16):                          # axis ticks, 0.70 .. 1.00
            a = 0.70 + i * 0.02
            if abs(a - self.LOW) < 0.005 or abs(a - self.HIGH) < 0.005:
                continue
            p.drawText(QRectF(self._x(a) - 7, y, 14, 16), Qt.AlignCenter, "·")
        bold = _mono()
        bold.setPixelSize(16)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QColor(pal.accent))
        p.drawText(QRectF(self._x(self.LOW) - 7, y, 14, 16), Qt.AlignCenter, "[")
        p.drawText(QRectF(self._x(self.HIGH) - 7, y, 14, 16), Qt.AlignCenter, "]")
        p.setFont(font)
        dot = QColor(pal.good if in_band else pal.warn)
        p.setPen(dot)
        p.drawText(QRectF(self._x(acc) - 7, y, 14, 16), Qt.AlignCenter, "@")
        font12 = _mono()
        font12.setPixelSize(12)
        p.setFont(font12)
        p.drawText(QRectF(4, y + 1, 58, 16), Qt.AlignLeft, f"{acc * 100.0:.1f}%")

        ry = 34.0
        cx = self.W / 2.0
        if in_band:
            p.setPen(QColor(pal.fg_dim))
            p.drawText(QRectF(cx - 130, ry, 260, 16), Qt.AlignCenter,
                       "hold — inside the band")
        else:
            shrink = acc > self.HIGH
            label = "targets shrink" if shrink else "targets grow"
            arrow = "v" if shrink else "^"
            col = QColor(pal.accent if shrink else pal.warn)
            p.setPen(QColor(pal.fg))
            p.drawText(QRectF(cx - 90, ry, 180, 16), Qt.AlignCenter, label)
            for i, ax in enumerate((cx - 108, cx - 92, cx + 86, cx + 102)):
                c = QColor(col)
                c.setAlphaF(0.35 + 0.65 * float((self._phase + i) % 3 == 0))
                p.setPen(c)
                p.drawText(QRectF(ax, ry, 14, 16), Qt.AlignCenter, arrow)


class OUTraceDiagram(_Diagram):
    """The Ornstein-Uhlenbeck movement drift drawing itself: one step per
    run, always pulled back toward the mean, never the same twice."""

    N, ROWS = 56, 9
    LX, TY, CWD, CHT = 48.0, 8.0, 7.2, 10.0
    SEG = 70              # frames per segment: N reveal + hold
    CAP_H = 15.0          # line box an 8pt caption actually needs

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Height must clear the axis CAPTION, not just the trace. At 112 the
        # caption's own rect ran past the widget and the widget clipped it;
        # deriving the height from the same terms that place the caption keeps
        # the two from drifting apart again.
        self.setFixedSize(
            470, int(self.TY + self.ROWS * self.CHT + 4 + self.CAP_H + 4))
        path = OrnsteinUhlenbeck(theta=0.35, sigma=0.25).path(0.0, 400, seed=11)
        self._vals = 1.0 / (1.0 + np.exp(-2.0 * path))   # squash into (0, 1)

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        font = _mono()
        font.setPixelSize(12)
        p.setFont(font)
        seg = (self._phase // self.SEG) % 8
        off = seg * 43
        tip = min(self._phase % self.SEG, self.N - 1)

        mid = QColor(pal.fg_dim)
        mid.setAlphaF(0.25)
        p.setPen(mid)
        ymid = self.TY + (self.ROWS - 1) / 2.0 * self.CHT
        for j in range(0, self.N, 2):                # the OU mean (squash(0))
            p.drawText(QRectF(self.LX + j * self.CWD, ymid, 8, 13),
                       Qt.AlignCenter, "-")

        for j in range(tip + 1):
            v = float(self._vals[off + j])
            r = (self.ROWS - 1) - v * (self.ROWS - 1)
            yy = self.TY + r * self.CHT
            d = tip - j
            if d == 0:
                col, ch = QColor(pal.accent), "@"
            elif d <= 4:
                col, ch = QColor(pal.fg), "o"
                col.setAlphaF(0.85)
            else:
                col, ch = QColor(pal.fg_dim), "·"
                col.setAlphaF(0.8)
            p.setPen(col)
            p.drawText(QRectF(self.LX + j * self.CWD, yy, 8, 13),
                       Qt.AlignCenter, ch)

        p.setPen(QColor(pal.fg_dim))
        p.setFont(QFont("Segoe UI", 8))
        # CAP_H, not 12: an 8pt face needs ~14px of line box, so a 12px rect
        # clipped its own descenders — "twitchy" lost its y and the axis
        # caption was cut through the middle.
        p.drawText(QRectF(2, self.TY, 44, self.CAP_H), Qt.AlignLeft, "twitchy")
        p.drawText(QRectF(2, self.TY + (self.ROWS - 1) * self.CHT, 44, self.CAP_H),
                   Qt.AlignLeft, "calm")
        p.drawText(QRectF(self.LX, self.TY + self.ROWS * self.CHT + 4, 220,
                          self.CAP_H),
                   Qt.AlignLeft, "one step per run →")


class FittsDiagram(_Diagram):
    """Per-session Fitts scatter: flick times against difficulty in bits,
    the fitted line locking in, and the ms-per-bit slope falling session
    after session while old fits ghost behind it."""

    SLOPES = (118, 106, 96, 87, 80, 74)   # ms per bit, falling across sessions
    A = 90.0                              # intercept (ms)
    P, F = 13, 44                         # points per session, frames per session
    PL, PR, PT, PB = 44.0, 356.0, 10.0, 124.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(470, 150)
        rng = np.random.default_rng(3)
        self._xs = [rng.uniform(1.15, 4.85, self.P) for _ in self.SLOPES]
        self._ns = [rng.normal(0.0, 26.0, self.P) for _ in self.SLOPES]

    def _xp(self, bits: float) -> float:
        return self.PL + (bits - 1.0) / 4.0 * (self.PR - self.PL)

    def _yp(self, ms: float) -> float:
        return self.PB - (ms - 80.0) / 640.0 * (self.PB - self.PT)

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        s = (self._phase // self.F) % len(self.SLOPES)
        fp = self._phase % self.F

        p.setPen(QColor(pal.border_control))
        p.drawLine(int(self.PL), int(self.PT), int(self.PL), int(self.PB))
        p.drawLine(int(self.PL), int(self.PB), int(self.PR), int(self.PB))

        font = _mono()
        font.setPixelSize(12)
        p.setFont(font)
        ghost = QColor(pal.fg_dim)
        ghost.setAlphaF(0.28)
        for q in range(s):                            # earlier sessions' fits
            p.setPen(ghost)
            for xcol in range(int(self.PL) + 4, int(self.PR), 7):
                bits = 1.0 + (xcol - self.PL) / (self.PR - self.PL) * 4.0
                yy = self._yp(self.A + self.SLOPES[q] * bits)
                p.drawText(QRectF(xcol - 4, yy - 7, 8, 13), Qt.AlignCenter, "·")

        shown = min(self.P, fp // 2 + 1)              # scatter fills in first
        pt = QColor(pal.fg)
        pt.setAlphaF(0.85)
        p.setPen(pt)
        for i in range(shown):
            x = float(self._xs[s][i])
            ms = self.A + self.SLOPES[s] * x + float(self._ns[s][i])
            p.drawText(QRectF(self._xp(x) - 4, self._yp(ms) - 7, 9, 13),
                       Qt.AlignCenter, "×")

        lp = 0.0 if fp < 26 else min((fp - 26) / 10.0, 1.0)
        if lp > 0:                                    # then the fit sweeps in
            p.setPen(QColor(pal.accent))
            xmax = self.PL + (self.PR - self.PL) * lp
            for xcol in range(int(self.PL) + 4, int(xmax), 7):
                bits = 1.0 + (xcol - self.PL) / (self.PR - self.PL) * 4.0
                yy = self._yp(self.A + self.SLOPES[s] * bits)
                p.drawText(QRectF(xcol - 4, yy - 7, 8, 13), Qt.AlignCenter, "*")

        ix = 366.0
        p.setPen(QColor(pal.fg_dim))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(ix, 22, 100, 14), Qt.AlignLeft,
                   f"session {s + 1}/{len(self.SLOPES)}")
        p.setFont(font)
        p.setPen(QColor(pal.accent))
        p.drawText(QRectF(ix, 40, 104, 16), Qt.AlignLeft,
                   f"b = {self.SLOPES[s]} ms/bit")
        if s > 0:
            p.setPen(QColor(pal.good))
            p.drawText(QRectF(ix, 60, 100, 16), Qt.AlignLeft, "v falling")
        p.setPen(QColor(pal.fg_dim))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(self.PL, self.PB + 6, self.PR - self.PL, 14),
                   Qt.AlignCenter, "ID (bits)")
        p.drawText(QRectF(4, self.PT, 36, 14), Qt.AlignLeft, "MT")


# ------------------------------------------------------------------- content
_LEDE = (
    "kovadapt is built on one result from motor-learning research: practice improves "
    "fastest at an intermediate difficulty relative to the learner — the <i>challenge "
    "point</i> — because a too-easy task carries no new information to learn, and a "
    "too-hard one exceeds the capacity to use the information available. The consequence "
    "is inconvenient for any fixed drill: the right difficulty is a property of the "
    "player and the session, not of the scenario file. KovaaK's has no modding API, so "
    "kovadapt cannot chase that target while you play; this came at a cost: adaptation "
    "is only possible between runs — and the cost turned out to be the method. Each "
    "finished run becomes one controlled observation, the model updates its beliefs, and "
    "the next variant is written before the next load. Playing and adapting are not "
    "competing activities but subsequent steps, repeated until the scenario has bent "
    "itself around whatever you are currently worst at."
)

_GOVERNOR = (
    "The controller at the center of the model is a band, not a target: hold accuracy "
    "between 85 and 95 percent, and train speed against it. The doctrine is "
    "primary-sourced — below the floor you are spraying flicks you cannot land, building "
    "sloppy habits that are expensive to unlearn; above the ceiling you are "
    "<i>obligated</i> to push pace, because parked accuracy means the speed axis is idle "
    "— and the research agrees: in the largest longitudinal aim dataset, hit rate "
    "improved only modestly with practice while hits per second improved considerably. "
    "Slowing down and speeding up are therefore not rival schools but subsequent steps: "
    "accuracy first, then speed rebuilt at the new pace, then accuracy again. kovadapt "
    "implements the rule as a deadband controller on target size — its own construction "
    "on top of the sourced band. Inside the band nothing happens: the deadband is what "
    "keeps difficulty from oscillating around a point no player can hold exactly. "
    "Outside it, size moves multiplicatively against only the excess beyond the nearest "
    "edge, so a run at 96 percent is nudged and a run at 99 percent is shoved. (The "
    "clicking band is doctrine; the tracking and switching bands are kovadapt's own "
    "extrapolations of the same law, and are labeled as such.)"
)

_FLICK = (
    "Outcomes conflate distinct failures, which is why the model refuses to reason from "
    "the stats file alone: a miss may be a badly aimed flick or a well-aimed flick that "
    "arrived too slowly, and accuracy cannot tell the two apart. Telemetry can, because "
    "a flick has anatomy. The standard motor-control decomposition — validated on "
    "professional FPS players — splits every aimed movement into a primary ballistic "
    "phase that lands long (overshoot) or short, followed by corrective submovements "
    "that settle onto the target; kovadapt's per-flick overshoot and correction counts "
    "are that decomposition, measured from Raw Input deltas exactly as the game receives "
    "them. The distinction matters because it reverses verdicts: overshoot with the shot "
    "fired mid-movement and at most one correction is a <i>swipe</i> — a timing strategy "
    "skilled players deliberately adopt on speed tasks — while the same overshoot "
    "followed by a chain of corrections is a control failure. One number, two opposite "
    "coaching decisions: only the microstructure separates them, which is precisely why "
    "the model demands it."
)

_BANDIT = (
    "Weakness has a geography, and the model maps it. Every flick is projected onto a "
    "5×5 grid over the wall — its direction sets the heading, its amplitude sets the "
    "ring: short flicks credit the inner cells, where micro-corrections live, and long "
    "flicks the edges, where control decays with distance from the hand's rest position "
    "on the pad. (The amplitude-ring mapping is kovadapt's own construction; the "
    "doctrine it encodes — skill degrades away from rest position — is sourced.) Each "
    "cell carries a Gaussian belief about how much weaker you are there than your own "
    "average, and focus is chosen by Thompson sampling: draw once from every cell's "
    "belief, commit to the worst draw. Most draws exploit what the model already knows, "
    "but roughly a fifth land somewhere else, and this waste is deliberate: a weak side "
    "moves as you improve, so a purely greedy policy would keep drilling a weakness that "
    "no longer exists. Forgetting serves the same end from the opposite direction — "
    "every run decays each cell's evidence three percent back toward the prior, so no "
    "verdict is permanent and every region has to keep re-earning its reputation, in "
    "both directions."
)

_FITTS = (
    "Scores plateau while skill improves, and high scores are luck-noisy — so the model "
    "keeps one number the scoreboard cannot fake. Fitts's law says movement time grows "
    "with the logarithm of task difficulty, MT = a + b·ID with ID = "
    "log<sub>2</sub>(D/W&nbsp;+&nbsp;1): smaller and farther targets cost more time "
    "because the motor system has a limited information capacity. The slope b — "
    "milliseconds per bit of difficulty — is the fair price you pay to aim, and kovadapt "
    "fits it inside every run as a regression of flick duration on log<sub>2</sub> "
    "amplitude. Across sessions the slope's trend is the honest progress report: a "
    "falling ms-per-bit under a flat scoreboard is genuine motor improvement, and the "
    "model says so instead of letting the plateau read as stagnation. Within a session "
    "the same number also acts. Two EWMAs of the slope run at different speeds, and when "
    "the fast one stops undercutting the slow one while accuracy sits comfortably in its "
    "band — throughput stalled, comfort intact — the engine adds one extra gentle shrink "
    "per run, about 1.75 percent at the default gain, pushing the task back toward the "
    "challenge point. Diagnosis and control are not separate systems here but subsequent "
    "uses of the same measurement."
)

_MOVEMENT = (
    "A fixed drill trains the drill; a varying one trains the player. Target "
    "micro-movement is therefore driven by an Ornstein-Uhlenbeck process — a "
    "mean-reverting random walk stepped once per run — so intensity drifts smoothly (no "
    "jarring jump between consecutive variants) but unpredictably (no rhythm to "
    "memorize): the anti-autopilot property, and the same reason community routines "
    "alternate scenarios rather than grinding one. Pace couples into the drift twice. "
    "Run hotter than your own kills-per-second norm and movement rises immediately — "
    "comfort read as a symptom rather than a reward. And when accuracy sits parked "
    "in-band while kill pace has been flat across the last ten runs, a bounded "
    "progression push raises the movement target, because long-term growth shows up as "
    "speed, not accuracy — the plateau is broken for you before it settles in. Strafe "
    "timing, last, is skewed toward whichever direction your flicks measurably favor "
    "less, so the weak side sees more traffic without a single spawn being scripted."
)

_REFUSALS = (
    "Some of the model's most deliberate behavior is refusal. It never forces a "
    "prescription: every insight carries its evidence, its reasoning, and its "
    "confidence, and anything the knowledge base marks contested or extrapolated is "
    "surfaced that way instead of being laundered into settled fact. Sensitivity is the "
    "standing example — the research is two-sided (high gain inflates overshoot, worst "
    "on small and distant targets; very low gain costs time through clutching and "
    "limb-speed limits; performance is U-shaped with a broad usable middle) while the "
    "community's sens-stability rule is contested, with documented professionals at both "
    "poles — so kovadapt reasons from both sides, suggests with the caveat attached, and "
    "never commands a change. And when input health is bad — timing jitter high, the "
    "effective polling rate unstable or far below the mouse's spec — the model "
    "suppresses every skill diagnosis built on flick microstructure, because device "
    "noise degrades each inference downstream, and blaming the player for the hardware "
    "would be worse than saying nothing: the honest move is a hardware fix, not a "
    "training plan. (The jitter cutoffs themselves are kovadapt's editorial calibration, "
    "and the knowledge base says so.)"
)

_CLOSING = (
    "All of it runs in the gap the game left open — the pause between runs that a "
    "modding API would have erased. Like tuning an instrument between movements rather "
    "than mid-phrase, the model does its work where you cannot feel it happening: the "
    "run you just played is measured, the belief is updated, the next variant is written "
    "— and by the time the scenario loads again, the task has already turned to face "
    "whatever you are currently worst at."
)


# --------------------------------------------------------------------- page
class MLPage(QWidget):
    """The "How it learns" section: prose in the project voice, sources lines
    per part (analysis/kb.py ids, citations in the tooltip), and four live
    ASCII diagrams of the controllers at work."""

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.setObjectName("tabPage")
        self.section_titles: list[str] = []
        self.prose: list[QLabel] = []
        self.diagrams: list[_Diagram] = []
        self.source_lines: list[QLabel] = []
        self.cited_ids: set[str] = set()

        self._lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(12)
        self._lay.addWidget(HintBar(settings, (
            "This page is the model documenting itself. Every part ends in a dim "
            "<b>sources</b> line naming the knowledge-base entries behind it — hover "
            "one for the full citations. The diagrams are live; ids like "
            "<code>p-…</code>/<code>dx-…</code> live in <code>analysis/kb.py</code>.")))

        self._prose(_LEDE)
        self._sources("p-challenge-point")

        self._section("The governor: accuracy is the constraint, speed is the variable")
        self._prose(_GOVERNOR)
        self._figure(DeadbandDiagram(), (
            "Live: the accuracy dot drifts against the [ 85 – 95 ] band; target size "
            "reacts only outside the brackets — shrink above, grow below, hold inside."))
        self._sources("p-speed-accuracy-governor", "p-speed-is-growth-axis",
                      "dx-acc-above-band", "dx-acc-below-band")

        self._section("What the mouse actually says")
        self._prose(_FLICK)
        self._sources("p-two-phase-flick", "p-swipiness", "dx-overshoot-strategic")

        self._section("The zone bandit: weakness has a geography")
        self._prose(_BANDIT)
        self._figure(ZoneGridDiagram(), (
            "Live: a Thompson round on the 5×5 zone grid — every cell's belief is "
            "sampled and the worst draw takes the focus; amber wins are exploration."))
        self._sources("p-weakness-isolation", "p-rest-position", "dx-region-deficit")

        self._section("The honest number: milliseconds per bit")
        self._prose(_FITTS)
        self._figure(FittsDiagram(), (
            "Live: one session's flicks (×) against difficulty in bits, with the fitted "
            "line — the slope, in ms per bit, falls across sessions as the motor system "
            "learns; earlier fits ghost behind it."))
        self._sources("p-fitts-throughput", "dx-fitts-progress")

        self._section("Movement that refuses to be memorized")
        self._prose(_MOVEMENT)
        self._figure(OUTraceDiagram(), (
            "Live: the Ornstein-Uhlenbeck drift that sets movement intensity — always "
            "wandering, always pulled back toward the middle, never the same twice."))
        self._sources("p-contextual-interference", "p-speed-is-growth-axis", "dx-bias")

        self._section("What kovadapt refuses to do")
        self._prose(_REFUSALS)
        self._sources("p-sensitivity-doctrine", "dx-input-health")

        self._lay.addSpacing(6)
        self._prose(_CLOSING)
        self._lay.addStretch(1)

    # ---------------------------------------------------------- construction
    def _section(self, title: str) -> None:
        head = QLabel(title)
        head.setProperty("headline", True)
        self._lay.addSpacing(8)
        self._lay.addWidget(head)
        self.section_titles.append(title)

    def _prose(self, html: str) -> None:
        lab = QLabel(html)
        lab.setTextFormat(Qt.RichText)
        lab.setWordWrap(True)
        self._lay.addWidget(lab)
        self.prose.append(lab)

    def _figure(self, diagram: _Diagram, caption: str) -> None:
        box = QWidget()
        box.setObjectName("tabPage")
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 4, 0, 2)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(diagram)
        row.addStretch(1)
        v.addLayout(row)
        cap = QLabel(caption)
        cap.setProperty("dim", True)
        cap.setWordWrap(True)
        cap.setAlignment(Qt.AlignHCenter)
        v.addWidget(cap)
        self._lay.addWidget(box)
        self.diagrams.append(diagram)

    def _sources(self, *ids: str) -> None:
        """Dim per-part sources line (the cite-everything rule made visible):
        text names the kb ids, the tooltip carries topic, confidence, and the
        full citation list, straight from analysis/kb.py."""
        tips: list[str] = []
        for kid in ids:
            entry = kb.PRINCIPLES.get(kid) or kb.DIAGNOSTICS.get(kid)
            if entry is None:      # unknown ids never render as fake citations
                continue
            head = entry.get("topic") or entry.get("signal") or kid
            srcs = "\n    ".join(entry.get("sources", ()))
            tips.append(f"{kid} — {head} [{entry.get('confidence', '')}]\n    {srcs}")
            self.cited_ids.add(kid)
        lab = QLabel("sources: " + "  ·  ".join(
            k for k in ids if k in self.cited_ids) + "    — analysis/kb.py")
        lab.setProperty("dim", True)
        lab.setStyleSheet("font-size: 11px;")
        lab.setWordWrap(True)
        lab.setToolTip("\n\n".join(tips))
        self._lay.addWidget(lab)
        self.source_lines.append(lab)

    # ------------------------------------------------------------------ API
    def restyle(self, *_pal) -> None:
        for d in self.diagrams:
            d.restyle()
