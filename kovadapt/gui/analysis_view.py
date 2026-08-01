"""Post-run analysis tab: the run's four headline numbers, then the charts,
then the Coach.

Reading order is the design. A KPI strip readable in a second leads; the
character-art charts (gui/viz.py) follow side by side, each carrying a
TAKEAWAY title that states what its data shows rather than what the chart
is; the Coach lands last, folded to its two most severe cards with the rest
one click away — nothing is dropped, because the citations are the product.

Every takeaway is derived from the values actually on screen and falls back
to the neutral descriptor the moment the data stops supporting a claim: a
title that overstates is worse than no title at all.

pyqtgraph remains only inside TrajectoryReplay's canvas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# The KPI reads reuse the Coach's own cutoffs and its region wording rather
# than restating them: a tile reading "clean" above a card that says
# "overshooting, then repairing" would be one body of evidence contradicting
# itself, and two spellings of the same region would read as two findings.
from ..analysis.insights import (
    _CORRECTIONS_CHAIN,
    _CORRECTIONS_CLEAN,
    _MIN_FLICKS,
    _OVERSHOOT_HIGH,
    Insight,
    _region_words,
    generate_insights,
)
from ..analysis.movement import movement_heatmap, segment_flicks
from ..analysis.report import input_degraded
from ..analysis.report import RunReport
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from ..telemetry.trace import MouseTrace
from . import theme, viz
from .onboarding import HintBar
from .replay import TrajectoryReplay

# Neutral chart descriptors — what the chart IS. Used whenever the run's own
# numbers do not clear the floors below.
_BIAS_TITLE = "flick quality by direction · lower is better"
_TRAVEL_TITLE = "aim travel around engagements"
_DEFICIT_TITLE = "weakness by wall region · brighter = weaker"
_TREND_TITLE = "accuracy over runs"

# Captions say how to READ the chart; the title carries what it says. Both
# stay short — the page used to open with ~700 characters of caption prose.
_BIAS_CAPTION = ("Cost per flick = overshoot + 0.15 x corrective submovements, "
                 "one bar per direction; the red bar is this run's worst.")
_TRAVEL_CAPTION = ("Where the crosshair spent its time around each engagement — "
                   "denser glyphs = more time. Hover a zone for its value.")
_DEFICIT_CAPTION = ("Weakness per wall region as z-scores against this run's own "
                    "average, on the r{row}c{col} grid the engine targets. Hover "
                    "a zone; outlined zones were never measured.")
_TREND_CAPTION = ("Accuracy per run for this scenario — the loop should bend it "
                  "upward without ever pinning it at 100%.")

# Claim floors. A takeaway has to clear one of these or the chart keeps its
# neutral title; each is kovadapt's own editorial calibration except the
# per-side flick count, which is analysis.directional_bias's own gate.
_EVEN_RATIO = 1.25       # side-vs-side cost ratio that stops reading as "even"
_MIN_SIDE_FLICKS = 3     # per-side floor (matches analysis.directional_bias)
_TREND_MIN_RUNS = 6      # runs before a direction is called on the sparkline
_TREND_STEP = 0.02       # accuracy points that count as a move, not noise
_LOPSIDED_SHARE = 0.55   # occupancy share that stops reading as balanced
_PACE_STEP = 0.05        # pace change vs the EWMA that counts as faster/slower

# Coach folding: severity order, then how many cards stay unfolded.
_SEVERITY_RANK = {"warning": 0, "attention": 1, "info": 2}
_COACH_FOLD = 2


class _InsightCard(QFrame):
    """One coach insight: severity dot, title, sourced body + prescription,
    and the reasoning/citations chain (the cite-everything rule made visible)."""

    def __init__(self, ins: Insight, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.insight = ins       # the card's evidence, still queryable after build
        pal = theme.current()
        color = {"warning": pal.warn, "attention": pal.bad}.get(ins.severity, pal.good)
        head = QLabel(f"<span style='color:{color}'>●</span>  <b>{ins.title}</b>"
                      f"  <span style='color:{pal.fg_dim}'>{ins.confidence}</span>")
        head.setTextFormat(Qt.RichText)
        body = QLabel(f"{ins.body}<br><b>Suggestion:</b> {ins.prescription}")
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        why = QLabel(f"why: {ins.reasoning}")
        why.setWordWrap(True)
        why.setProperty("dim", True)
        cites = QLabel(f"{len(ins.sources)} source{'s' if len(ins.sources) != 1 else ''}")
        cites.setProperty("dim", True)
        cites.setToolTip("\n".join(ins.sources))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        lay.addWidget(head)
        lay.addWidget(body)
        lay.addWidget(why)
        lay.addWidget(cites)


def _mono_css(px: int) -> str:
    """Family + size for mono text, as a widget-level sheet.

    theme.py's app-wide `* { font-family: "Segoe UI"; font-size: 13px }`
    outranks setFont(), so a numeral styled only with setFont renders at body
    size in the running app — and looks right in tests, where no app QSS is
    installed. `px` must stay on theme.CELL_SIZES."""
    return f'font-family: "{theme.mono_family()}"; font-size: {px}px;'


class _KpiTile(QFrame):
    """One headline number: caption, mono value + unit, and a one-word read.

    The read is always a comparison against a stated baseline, and the tile's
    tooltip carries that baseline with the live numbers — the strip is the
    first thing on the page, so it has to be as citable as a Coach card."""

    def __init__(self, caption: str, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self._tone = "dim"
        self.cap = QLabel(caption.upper())
        self.value = QLabel("—")
        self.unit = QLabel("")
        self.read = QLabel("")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.value, 0, Qt.AlignBottom)
        row.addWidget(self.unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(2)
        lay.addWidget(self.cap)
        lay.addLayout(row)
        lay.addWidget(self.read)
        self.restyle()

    def set_value(self, value: str, unit: str, read: str, tone: str,
                  why: str) -> None:
        """tone is a palette role name ('good' | 'warn' | 'bad' | 'dim');
        `why` is the tooltip that must justify the read."""
        self.value.setText(value)
        self.unit.setText(unit)
        self.read.setText(read)
        self._tone = tone
        self.setToolTip(why)
        self.restyle()

    def restyle(self, *_pal) -> None:
        pal = theme.current()
        tone = {"good": pal.good, "warn": pal.warn,
                "bad": pal.bad}.get(self._tone, pal.fg_dim)
        self.cap.setStyleSheet(
            f"color: {pal.fg_dim}; letter-spacing: 0.8px; {_mono_css(12)}")
        self.value.setStyleSheet(
            f"color: {pal.fg}; font-weight: 700; {_mono_css(24)}")
        self.unit.setStyleSheet(f"color: {pal.fg_dim}; {_mono_css(12)}")
        self.read.setStyleSheet(f"color: {tone}; {_mono_css(12)}")
        # setFont as well as the sheet: the sheet is what survives theme.py's
        # app-wide font rule, setFont is what gives the labels honest metrics
        # (and theme.mono snaps to the sizes Cascadia Mono renders crisply at).
        self.cap.setFont(theme.mono(12))
        self.value.setFont(theme.mono(24, bold=True))
        self.unit.setFont(theme.mono(12))
        self.read.setFont(theme.mono(12))


# ---------------------------------------------------------------- takeaways
def _bias_title(vals: list[float], ns: list[int],
                degraded: bool = False) -> str:
    """Headline for the direction bars: which side actually costs more.

    `vals`/`ns` are the plotted [left, vertical, right] costs and flick
    counts — the claim comes from the same numbers the bars draw, and only
    once both sides carry the flicks analysis.directional_bias needs before
    it will score them at all.

    `degraded` is the shared input-health gate: these costs are overshoot
    plus corrections, i.e. pure flick microstructure, so a run too noisy to
    diagnose must not get a directional verdict here either — that was a
    "your left flicks cost 3.2x more than your right" headline sitting
    directly above a tile reading "noisy-input"."""
    left, vert, right = vals
    n_left, n_vert, n_right = ns
    if sum(ns) == 0:
        return _BIAS_TITLE
    if degraded:
        return "input timing too noisy to compare directions this run"
    if n_left < _MIN_SIDE_FLICKS or n_right < _MIN_SIDE_FLICKS:
        return f"only {n_left} left / {n_right} right flicks — too few to call a side"
    hi, lo = max(left, right), min(left, right)
    weak = "left" if left >= right else "right"
    other = "right" if weak == "left" else "left"
    # vert > 0 guard: a run with no cost anywhere satisfies vert >= 1.25 * hi
    # arithmetically, and would headline "vertical flicks cost most — 0.00".
    if vert > 0 and n_vert >= _MIN_SIDE_FLICKS and vert >= _EVEN_RATIO * hi:
        return f"vertical flicks cost most — {vert:.2f} vs {hi:.2f} horizontal"
    if lo <= 0:
        if hi <= 0:
            return "no overshoot or correction cost in either direction"
        return f"only your {weak} flicks carry any cost — {hi:.2f} vs 0.00"
    ratio = hi / lo
    if ratio >= _EVEN_RATIO:
        return (f"your {weak} flicks cost {ratio:.1f}x more than your {other} "
                f"— {hi:.2f} vs {lo:.2f}")
    return f"left and right flicks are even — {left:.2f} vs {right:.2f}"


def _deficit_title(deficits: dict[str, float], settings: Settings | None) -> str:
    """Headline for the region map: the weakest zone, when one really is.

    Below viz.NOISE_FLOOR the map itself renders flat (that constant is what
    stops noise being stretched across the ramp), so naming a "weakest" zone
    there would claim a finding the picture deliberately refuses to draw."""
    if not deficits:
        return _DEFICIT_TITLE
    key, z = max(deficits.items(), key=lambda kv: kv[1])
    # "All within N SD" has to be tested on the largest ABSOLUTE deviation,
    # the same quantity the map colours. Testing only the maximum let a
    # strongly negative zone (a genuine strength, drawn cool and saturated)
    # sit under a title claiming nothing deviated at all.
    peak = max(abs(v) for v in deficits.values())
    if peak < viz.NOISE_FLOOR:
        return (f"no zone stands out — all within "
                f"{viz.NOISE_FLOOR:.1f} SD of average")
    if z < viz.NOISE_FLOOR:
        # Nothing is weak; the spread that colours the map is on the strong
        # side, so report THAT rather than a weakness the run does not show.
        skey, sz = min(deficits.items(), key=lambda kv: kv[1])
        swhere = _region_words(skey, settings) if settings is not None else skey
        return (f"no zone is weaker than your average — {swhere} is your "
                f"strongest, {sz:+.2f} SD")
    where = _region_words(key, settings) if settings is not None else key
    return f"weakest zone this run: {where}, {z:+.2f} SD above average"


def _travel_title(heat: np.ndarray) -> str:
    """Headline for the occupancy map: which side the crosshair lives on.

    movement_heatmap recenters at every click, so axis 0 is displacement
    left/right of the last shot — the share is of resampled samples, not of
    distance travelled."""
    arr = np.asarray(heat, dtype=float)
    half = arr.shape[0] // 2
    if half == 0:
        return _TRAVEL_TITLE
    low, high = float(arr[:half].sum()), float(arr[-half:].sum())
    total = low + high
    if total <= 0:
        return _TRAVEL_TITLE
    share = max(low, high) / total
    if share < _LOPSIDED_SHARE:
        return "aim travel is balanced left/right of your shots"
    side = "left" if low > high else "right"
    return f"aim travel leans {side} — {share:.0%} of the time on that side"


def _trend_title(accs: list[float]) -> str:
    """Headline for the sparkline: where accuracy is actually going.

    Halves, not endpoints: a single hot or cold run at either end must not
    become a direction. Under _TREND_MIN_RUNS runs no direction is claimed
    at all."""
    n = len(accs)
    if n < 2:
        return _TREND_TITLE
    half = max(n // 2, 1)
    first, second = float(np.mean(accs[:half])), float(np.mean(accs[half:]))
    if n < _TREND_MIN_RUNS:
        return f"accuracy over {n} runs — too few to call a direction"
    delta = second - first
    if abs(delta) < _TREND_STEP:
        return f"accuracy flat near {second:.0%} across {n} runs"
    verb = "up" if delta > 0 else "down"
    return (f"accuracy {verb} {abs(delta) * 100:.0f} points over {n} runs "
            f"— {first:.0%} → {second:.0%}")


def _kind_color(kind: str) -> str:
    pal = theme.current()
    return {"overshoot": pal.bad, "hesitation": pal.bad,
            "slow_flick": pal.warn, "clean_flick": pal.good}.get(kind, pal.accent)


def _clips_available() -> bool:
    """Whether the [clips] extra (dxcam/opencv) is importable. Lazy on purpose:
    kovadapt.capture must never be pulled in at module import."""
    try:
        from ..capture.clips import CLIPS_AVAILABLE
    except ImportError:
        return False
    return CLIPS_AVAILABLE


def _caption(text: str) -> QLabel:
    """Dim, word-wrapped how-to-read-this caption shown under a plot."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("dim", True)
    return lab


class AnalysisView(QWidget):
    def __init__(self, settings: Settings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.report: RunReport | None = None
        self.trace: MouseTrace | None = None
        self.flicks: list = []
        self._settings = settings
        self._last_insights: tuple[RunReport, PlayerProfile] | None = None
        self._coach_cards: list[_InsightCard] = []
        self.coach_more: QPushButton | None = None
        self._coach_open = False

        # header
        self.title = QLabel("No run analyzed yet")
        self.title.setProperty("headline", True)
        self.summary = QLabel("Finish a run while watching (or open a saved report).")
        self.summary.setWordWrap(True)
        # Secondary now: the strip below carries the numbers this line opens
        # with, so it reads as the caption to the headline rather than a
        # second copy of the run.
        self.summary.setProperty("dim", True)
        open_btn = QPushButton("Open report…")
        open_btn.clicked.connect(self._open_dialog)

        head = QHBoxLayout()
        head_col = QVBoxLayout()
        head_col.addWidget(self.title)
        head_col.addWidget(self.summary)
        head.addLayout(head_col, 1)
        head.addWidget(open_btn, 0, Qt.AlignTop)

        # ---- KPI strip: the four numbers that answer "how did that go?"
        self.kpi_strip = QWidget()
        self.kpi_strip.setObjectName("tabPage")     # transparent; backdrop shows
        kpi_lay = QHBoxLayout(self.kpi_strip)
        kpi_lay.setContentsMargins(0, 0, 0, 0)
        kpi_lay.setSpacing(12)
        self.kpis: dict[str, _KpiTile] = {}
        for key, cap in (("accuracy", "accuracy"), ("kills", "kills"),
                         ("pace", "pace"), ("flick", "mean flick")):
            tile = _KpiTile(cap)
            self.kpis[key] = tile
            kpi_lay.addWidget(tile, 1)

        # ---- charts: side by side, each with a takeaway title + short caption
        self.bias_bars = viz.AsciiBars(title=_BIAS_TITLE)
        self.bias_caption = _caption(_BIAS_CAPTION)
        self.heat_map = viz.AsciiHeatmap(title=_TRAVEL_TITLE)
        self.heat_caption = _caption(_TRAVEL_CAPTION)
        self.trend_spark = viz.AsciiTrend(title=_TREND_TITLE, fmt="{:.0%}")
        self.trend_caption = _caption(_TREND_CAPTION)

        bias_w = QWidget()
        bv = QVBoxLayout(bias_w)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(self.bias_bars, 1)
        bv.addWidget(self.bias_caption)
        heat_w = QWidget()
        hv = QVBoxLayout(heat_w)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(self.heat_map, 1)
        hv.addWidget(self.heat_caption)
        self.trend_w = QWidget()
        tv = QVBoxLayout(self.trend_w)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(self.trend_spark, 1)
        tv.addWidget(self.trend_caption)
        self.trend_w.hide()                    # appears once the profile has history

        # The section is 1400px wide (shell.COLUMN_WIDTHS "wide"): the two
        # per-run charts sit next to each other instead of stacking a narrow
        # column and pushing everything below the fold.
        self.charts = QSplitter(Qt.Horizontal)
        self.charts.addWidget(bias_w)
        self.charts.addWidget(heat_w)
        self.charts.setSizes([700, 700])
        self.charts.setMinimumHeight(300)

        # ---- notable moments + replay, also side by side
        self.moments = QListWidget()
        # Moment text is a full sentence, and the panel is ~330px wide: elided
        # to one line it read "Overshot a right flick by 36% of its distance,
        # then corrected 1x before" and pushed a horizontal scrollbar under the
        # list. Wrapping is what makes the sentence readable; ElideNone stops
        # Qt truncating instead of wrapping, and the off switch on the
        # horizontal bar keeps the wrap authoritative.
        self.moments.setWordWrap(True)
        self.moments.setTextElideMode(Qt.ElideNone)
        self.moments.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.moments.setResizeMode(QListWidget.Adjust)
        self.moments.currentRowChanged.connect(self._select_moment)
        self.full_btn = QPushButton("Whole run")
        self.full_btn.setEnabled(False)
        self.full_btn.setToolTip("Replay the whole run instead of one moment")
        self.full_btn.clicked.connect(self._show_full_run)
        self.clip_btn = QPushButton("Play video clip")
        self.clip_btn.setEnabled(False)
        self.clip_btn.clicked.connect(self._play_clip)
        self.clip_hint = QLabel("")           # why clips are off, when they are
        self.clip_hint.setWordWrap(True)
        self.clip_hint.setProperty("dim", True)
        self.replay = TrajectoryReplay()

        mo_box = QGroupBox("Notable moments")
        mo_lay = QVBoxLayout(mo_box)
        mo_lay.addWidget(self.moments, 1)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.full_btn)
        btn_row.addWidget(self.clip_btn, 1)
        mo_lay.addLayout(btn_row)
        mo_lay.addWidget(self.clip_hint)
        self._update_clip_state(-1)
        rep_box = QGroupBox("Trajectory replay")
        rep_lay = QVBoxLayout(rep_box)
        rep_lay.addWidget(self.replay)
        self.detail = QSplitter(Qt.Horizontal)
        self.detail.addWidget(mo_box)
        self.detail.addWidget(rep_box)
        self.detail.setSizes([440, 900])
        self.detail.setMinimumHeight(380)

        # ---- coach last: folded to its two most severe cards
        self.coach_box = QGroupBox("Coach — every insight shows its evidence and sources")
        self.coach_lay = QVBoxLayout(self.coach_box)
        self.coach_lay.setSpacing(10)
        self.coach_box.hide()

        # room to breathe: generous spacing between every panel
        self.charts.setHandleWidth(14)
        self.detail.setHandleWidth(14)
        lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        if settings is not None:
            lay.addWidget(HintBar(settings, (
                "The four numbers up top are this run; each chart's title is what "
                "its data says. Click a notable moment to replay just that flick "
                "(green is clean, red overshot) or <b>Whole run</b> for all of it. "
                "Coach cards carry their sources — hover a source count.")))
        lay.addLayout(head)
        lay.addWidget(self.kpi_strip)
        lay.addWidget(self.charts)
        lay.addWidget(self.trend_w)
        lay.addWidget(self.detail, 1)
        lay.addWidget(self.coach_box)

    # ------------------------------------------------------------------
    def restyle(self, *_pal) -> None:
        # the viz widgets read theme.current() at paint time — update() is all
        for chart in (self.bias_bars, self.heat_map, self.trend_spark):
            chart.restyle()
        for tile in self.kpis.values():
            tile.restyle()
        self.replay.restyle()
        if self._last_insights is not None:
            self._fill_insights(*self._last_insights)   # cards bake colors
        if self.report is not None:
            for r in range(self.moments.count()):
                it = self.moments.item(r)
                # By ROW this painted every moment in the NEXT moment's kind
                # colour and dropped the caption's dim grey. The list is
                # severity-sorted and the clean reference flick scores 1.0,
                # so a theme switch on a degraded run rendered "this is your
                # benchmark" in the BAD colour. On this page the colour IS
                # the evidence.
                i = self._moment_index(r)
                if i < 0:
                    # the caption row: re-dim it to the NEW palette, since
                    # skipping would leave the old theme's grey behind
                    it.setForeground(QColor(theme.current().fg_dim))
                    continue
                it.setForeground(QColor(_kind_color(self.report.notable[i]["kind"])))

    # ------------------------------------------------------------------
    def set_trends(self, trends) -> None:
        """Cross-session SkillTrends from the boot worker."""
        self._trends = trends

    def show_report(self, rep: RunReport, trace: MouseTrace | None = None,
                    profile: PlayerProfile | None = None) -> None:
        self.report = rep
        self.trace = trace
        profile = self._resolve_profile(rep, profile)
        self._coach_open = False           # a new run opens folded
        self._fill_kpis(rep, profile)
        self._fill_trend(profile)
        self._fill_insights(rep, profile)
        if trace is None and rep.trace_file and Path(rep.trace_file).is_file():
            self.trace = MouseTrace.load(rep.trace_file)
        # flicks aren't serialized in the report — recompute from the trace
        self.flicks = (segment_flicks(self.trace)
                       if self.trace is not None and len(self.trace) > 10 else [])

        self.title.setText(f"{rep.scenario} — {rep.started_iso.replace('T', ' ')[:19]}")
        self.summary.setText(rep.summary_text)
        self._draw_bias(rep)
        self._draw_heat(rep)
        has_trace = self.trace is not None and len(self.trace) > 1
        self.full_btn.setEnabled(has_trace)
        # Moments FIRST: filling the list selects the top moment, which loads
        # that window into the replay. Loading the full run first as well cost
        # a whole extra path() + decimate + setData pass on every report — up
        # to ~6.8k points thrown away — and, before the ordering was fixed,
        # left the highlighted row and the replay describing different
        # segments. The full run is loaded only when nothing selected one.
        self._fill_moments(rep)
        if not has_trace:
            self.replay.clear()
        elif self.moments.currentRow() < 0:
            self.replay.load(self.trace, label="full run", flicks=self.flicks)
        self._update_clip_state(self._moment_index(self.moments.currentRow()))

    def load_report_file(self, path: Path | str) -> None:
        self.show_report(RunReport.load(path))

    # ------------------------------------------------------------------
    def _resolve_profile(self, rep: RunReport,
                         profile: PlayerProfile | None) -> PlayerProfile | None:
        """The profile this run belongs to — reloaded from disk on the
        saved-report path, where the caller has none to hand us."""
        if profile is not None or self._settings is None:
            return profile
        name = rep.scenario
        if not name.endswith(ADAPTIVE_SUFFIX):
            name += ADAPTIVE_SUFFIX
        return PlayerProfile.load(name, self._settings.profile_path)

    def _fill_kpis(self, rep: RunReport, profile: PlayerProfile | None) -> None:
        """The four headline numbers. Every read names the baseline it is a
        comparison against, and the tooltip carries that baseline with the
        live numbers — no bare adjectives."""
        arche = (profile.archetype if profile is not None else "") or "clicking"
        eff = (self._settings.for_archetype(arche)
               if self._settings is not None else None)

        # accuracy vs the archetype's band (the size controller's setpoint)
        if eff is None:
            self.kpis["accuracy"].set_value(
                f"{rep.accuracy:.0%}", "hit rate", "—", "dim",
                "No settings loaded, so the accuracy band this run would be "
                "judged against is unknown.")
        else:
            lo, hi = eff.target_accuracy_low, eff.target_accuracy_high
            if rep.accuracy > hi:
                read, tone = "above-band", "warn"
            elif rep.accuracy < lo:
                read, tone = "below-band", "warn"
            else:
                read, tone = "in-band", "good"
            band_note = ("the 85-95% band is primary-sourced for clicking"
                         if arche == "clicking" else
                         f"kovadapt's {arche} band is an extrapolation of the "
                         "same control law")
            self.kpis["accuracy"].set_value(
                f"{rep.accuracy:.0%}", "hit rate", read, tone,
                f"{rep.accuracy:.1%} of shots hit, against the {lo:.0%}-{hi:.0%} "
                f"{arche} band the size controller holds you in ({band_note}).")

        # kills, plus how much of the run telemetry actually saw
        n = rep.n_flicks
        self.kpis["kills"].set_value(
            str(rep.kills), "kills",
            f"{n} flicks" if n else "no-telemetry", "dim" if n else "warn",
            f"{rep.kills} kills in the stats file. Mouse telemetry segmented "
            f"{n} flicks out of this run — every chart below is built from "
            "those, so a run without telemetry shows stats only.")

        # pace against the profile's own EWMA
        base = profile.ewma_kps if profile is not None else 0.0
        # run_count > 1, not > 0: observe_run seeds every EWMA to the first
        # run's own value, so at run_count == 1 this compares the run against
        # itself and reports a confident "steady · +0%".
        if profile is not None and profile.run_count > 1 and base > 0:
            delta = rep.kps / base - 1.0
            if delta >= _PACE_STEP:
                read, tone = "faster", "good"
            elif delta <= -_PACE_STEP:
                read, tone = "slower", "warn"
            else:
                read, tone = "steady", "dim"
            why = (f"{rep.kps:.2f} kills/s against your {base:.2f} EWMA over "
                   f"{profile.run_count} runs ({delta:+.0%}). The "
                   f"+/-{_PACE_STEP:.0%} cutoff for calling that a change is "
                   "kovadapt's editorial calibration.")
        else:
            read, tone = "no-baseline", "dim"
            runs = profile.run_count if profile is not None else 0
            why = (f"{rep.kps:.2f} kills/s. This scenario has {runs} run"
                   f"{'' if runs == 1 else 's'} of history — the pace EWMA is "
                   "seeded from the first run, so it needs a second before it "
                   "is a baseline rather than a copy of this run.")
        self.kpis["pace"].set_value(f"{rep.kps:.2f}", "kills/s", read, tone, why)

        # mean flick time, read through the Coach's own microstructure gate —
        # the SAME gate, not a copy of its cutoffs. Reading overshoot and
        # corrections without the input-health check let this tile call a run
        # "repaired" on the same screen where the Coach reported that
        # microstructure diagnoses were suppressed for it.
        degraded = input_degraded(rep)
        if degraded:
            read, tone = "noisy-input", "warn"
        elif n < _MIN_FLICKS:
            read, tone = "thin-data", "dim"
        elif (rep.overshoot_rate > _OVERSHOOT_HIGH
              and rep.mean_corrections >= _CORRECTIONS_CHAIN):
            read, tone = "repaired", "bad"
        elif (rep.overshoot_rate <= _OVERSHOOT_HIGH
              and rep.mean_corrections <= _CORRECTIONS_CLEAN):
            read, tone = "clean", "good"
        else:
            read, tone = "mixed", "warn"
        why = (f"Mean flick {rep.mean_flick_ms:.0f} ms over {n} flicks: "
               f"{rep.overshoot_rate:.0%} overshot with "
               f"{rep.mean_corrections:.1f} corrective submovements each. The "
               f"{_OVERSHOOT_HIGH:.0%} / {_CORRECTIONS_CHAIN:.0f} cutoffs and "
               f"the {_MIN_FLICKS}-flick floor are the same ones the Coach "
               "reads microstructure through (kovadapt editorial calibration).")
        if degraded:
            why = (f"Mean flick {rep.mean_flick_ms:.0f} ms over {n} flicks, but "
                   "this run's input timing is too noisy to read flick "
                   "microstructure from, so no overshoot or directional "
                   "verdict is offered for it anywhere on this page. The "
                   "flick time itself is still measured.")
        self.kpis["flick"].set_value(
            f"{rep.mean_flick_ms:.0f}" if rep.mean_flick_ms > 0 else "—", "ms",
            read, tone, why)

    # ------------------------------------------------------------------ coach
    def _fill_insights(self, rep: RunReport, profile: PlayerProfile | None) -> None:
        self._clear_coach()
        if profile is None or self._settings is None or profile.run_count == 0:
            # Drop the cached pair too: restyle() refills from it, so leaving
            # the previous run's (rep, profile) here meant any theme or accent
            # change resurrected that run's Coach cards and trend underneath
            # the CURRENT run's header — stale advice presented as live.
            self._last_insights = None
            self.coach_box.hide()
            return
        insights = generate_insights(
            rep, profile, self._settings,
            trends=getattr(self, "_trends", None))
        self._last_insights = (rep, profile)
        # Severity first, generate_insights' own actionability order within a
        # severity (sorted is stable). The two cards left showing when the
        # section is folded have to be the two that matter most.
        for ins in sorted(insights, key=lambda i: _SEVERITY_RANK.get(i.severity, 9)):
            card = _InsightCard(ins)
            self._coach_cards.append(card)
            self.coach_lay.addWidget(card)
        if len(self._coach_cards) > _COACH_FOLD:
            self.coach_more = QPushButton("")
            self.coach_more.setProperty("flat", True)
            self.coach_more.setCursor(Qt.PointingHandCursor)
            self.coach_more.clicked.connect(self._toggle_coach)
            self.coach_lay.addWidget(self.coach_more, 0, Qt.AlignLeft)
        self._apply_fold()
        self.coach_box.setVisible(bool(insights))

    def _clear_coach(self) -> None:
        while self.coach_lay.count():
            item = self.coach_lay.takeAt(0)
            if item.widget():
                # Unparent before deleting: takeAt only drops the layout item,
                # so a card left parented stays painted until the event loop
                # gets round to the deletion.
                item.widget().setParent(None)
                item.widget().deleteLater()
        self._coach_cards = []
        self.coach_more = None

    def _apply_fold(self) -> None:
        """Fold the Coach to its most severe cards. The rest are built and
        kept — their citations are the product, not decoration — and only
        hidden behind the disclosure."""
        for i, card in enumerate(self._coach_cards):
            card.setVisible(self._coach_open or i < _COACH_FOLD)
        if self.coach_more is not None:
            self.coach_more.setText(
                "show fewer" if self._coach_open
                else f"show all ({len(self._coach_cards)})")

    def _toggle_coach(self) -> None:
        self._coach_open = not self._coach_open
        self._apply_fold()

    # ------------------------------------------------------------------
    def _fill_trend(self, profile: PlayerProfile | None) -> None:
        """Accuracy-over-runs sparkline from the profile history (hidden
        until at least two runs exist)."""
        hist = profile.history if profile is not None else []
        accs = [float(h.get("accuracy", 0.0)) for h in hist[-60:]]
        if len(accs) >= 2:
            self.trend_spark.set_title(_trend_title(accs))
            # first_run is the true 1-based run number of accs[0]. Without it
            # the widget can only say "oldest shown", because it cannot know
            # this list was sliced — on a 137-run profile it was labelling
            # run 78 as "run 1".
            self.trend_spark.set_data(accs, tag=f"{accs[-1]:.0%}",
                                      first_run=len(hist) - len(accs) + 1)
            self.trend_w.show()
        else:
            self.trend_spark.set_title(_TREND_TITLE)
            self.trend_spark.clear()
            self.trend_w.hide()

    def _draw_bias(self, rep: RunReport) -> None:
        b = rep.bias or {}
        dirs = ["left", "vertical", "right"]
        vals = [
            (b.get(d) or {}).get("overshoot", 0.0)
            + 0.15 * (b.get(d) or {}).get("corrections", 0.0)
            for d in dirs
        ]
        ns = [(b.get(d) or {}).get("n", 0) for d in dirs]
        degraded = input_degraded(rep)
        self.bias_bars.set_title(_bias_title(vals, ns, degraded))
        # ratio_counts is the caller's explicit permission for the chart to
        # spell out a side-vs-side ratio, and only this layer can grant it:
        # viz.py cannot reach input_degraded (it would have to import analysis
        # and take a RunReport). Withheld on a degraded run, so the footer can
        # never contradict a title that says the comparison cannot be made;
        # granted otherwise, since a 1.09x gap genuinely cannot be eyeballed
        # and the sentence is the only way to read it.
        self.bias_bars.set_data(dirs, vals, [f"{n} flicks" for n in ns],
                                ratio_counts=None if degraded else ns)

    def _draw_heat(self, rep: RunReport | None = None) -> None:
        """Zone heatmap on the Settings.region_cols x region_rows grid:
        the run's region deficits when the report has them, else the
        movement heatmap pooled down to the same zones."""
        cols = self._settings.region_cols if self._settings is not None else 5
        rows = self._settings.region_rows if self._settings is not None else 5
        if rep is not None and rep.region_deficits:
            grid, labels = viz.region_grid(rep.region_deficits, cols, rows)
            self.heat_map.set_title(_deficit_title(rep.region_deficits, self._settings))
            self.heat_map.set_data(grid, labels, fmt="{:+.2f}")
            self.heat_caption.setText(_DEFICIT_CAPTION)
        elif self.trace is not None and len(self.trace) >= 2:
            heat, _xe, _ye = movement_heatmap(self.trace)
            pooled = viz.pool(np.log1p(heat.T), rows, cols)  # heat.T row 0 = bottom
            labels = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
            self.heat_map.set_title(_travel_title(heat))
            self.heat_map.set_data(pooled, labels, fmt="{:.2f}")
            self.heat_caption.setText(_TRAVEL_CAPTION)
        else:
            self.heat_map.set_title(_TRAVEL_TITLE)
            self.heat_map.set_data(None)
            self.heat_caption.setText(_TRAVEL_CAPTION)

    def _fill_moments(self, rep: RunReport) -> None:
        self.moments.blockSignals(True)
        self.moments.clear()
        # Each moment narrates per-flick overshoot and correction counts, so
        # the same input-health gate applies. The clips stay listed and
        # replayable — the events are real — but the page must not quantify
        # them as if the timing behind them were trustworthy.
        if rep.notable and input_degraded(rep):
            note = QListWidgetItem(
                "· input timing was noisy — treat these overshoot figures as "
                "indicative only ·")
            note.setForeground(QColor(theme.current().fg_dim))
            note.setFlags(Qt.NoItemFlags)          # a caption, not a choice
            self.moments.addItem(note)
        for i, m in enumerate(rep.notable):
            it = QListWidgetItem(m["text"])
            it.setForeground(QColor(_kind_color(m["kind"])))
            it.setData(Qt.UserRole, i)
            self.moments.addItem(it)
        self.moments.blockSignals(False)
        if rep.notable:
            # First SELECTABLE row — row 0 may be the input-health caption,
            # which carries NoItemFlags and no moment index.
            first = next((r for r in range(self.moments.count())
                          if self._moment_index(r) >= 0), -1)
            self.moments.setCurrentRow(first)  # drives the replay via _select_moment

    def _moment_index(self, row: int) -> int:
        """Moment index behind a list row, or -1.

        Rows are NOT moment indices: the list can carry a non-selectable
        caption row, so every row stamps its own index in Qt.UserRole and
        that is the only mapping to trust.
        """
        item = self.moments.item(row) if row >= 0 else None
        if item is None:
            return -1
        idx = item.data(Qt.UserRole)
        return int(idx) if idx is not None else -1

    def _select_moment(self, row: int) -> None:
        idx = self._moment_index(row)
        if self.report is None or idx < 0 or idx >= len(self.report.notable):
            self._update_clip_state(-1)
            return
        m = self.report.notable[idx]
        self._update_clip_state(idx)
        if self.trace is not None and len(self.trace) > 1:
            self.replay.load(self.trace, m["t_start"], m["t_end"],
                             label=m["kind"].replace("_", " "),
                             flicks=self.flicks)

    def _show_full_run(self) -> None:
        """Back out to the whole run. The selection is cleared first so the
        list highlight and the replay never describe different segments."""
        if self.trace is None or len(self.trace) <= 1:
            return
        self.moments.setCurrentRow(-1)        # _select_moment(-1) clears clip state
        self.replay.load(self.trace, label="full run", flicks=self.flicks)

    # ------------------------------------------------------------------
    def _clips_off_reason(self) -> str | None:
        """Why the clips feature can't produce clips at all right now, or None
        when it can (then a missing clip is just a moment without one)."""
        if self._settings is not None and not self._settings.clips_enabled:
            return ("Enable 'Capture video clips' in Adaptability, then new "
                    "notable moments get clips")
        if not _clips_available():
            return "pip install kovadapt[clips] — dxcam/opencv are not installed"
        return None

    def _update_clip_state(self, moment_idx: int) -> None:
        """Enable the clip button when the selected moment has a clip; when it
        doesn't, the disabled button's tooltip says exactly why.

        The parameter is a MOMENT index, never a list row. It was called
        `row`, and that name was the whole bug: `clip_files` is keyed by
        moment (watcher.py enumerates rep.notable), one caller correctly
        passed an index into a parameter called row, the other passed an
        actual row, and on clean runs the two are equal — so nothing ever
        caught it.
        """
        has_clip = (self.report is not None and moment_idx >= 0
                    and str(moment_idx) in (self.report.clip_files or {}))
        self.clip_btn.setEnabled(has_clip)
        off = self._clips_off_reason()
        self.clip_btn.setToolTip(
            "" if has_clip else off or "No clip was captured for this moment")
        self.clip_hint.setText(off or "")
        self.clip_hint.setVisible(off is not None)

    def _play_clip(self) -> None:
        idx = self._moment_index(self.moments.currentRow())
        if self.report is None or idx < 0:
            return
        p = (self.report.clip_files or {}).get(str(idx))
        if p and Path(p).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p))))

    def _open_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open run report",
                                           str(Path.home() / ".kovadapt" / "reports"),
                                           "Run reports (*.json)")
        if p:
            self.load_report_file(p)
