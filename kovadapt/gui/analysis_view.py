"""Post-run analysis tab: summary, bias, heatmap, notable moments with
trajectory replays and (when captured) video clips — side by side.

The charts are character art (gui/viz.py) in the ascii_art visual
language; pyqtgraph remains only inside TrajectoryReplay's canvas."""

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

from ..analysis.insights import Insight, generate_insights
from ..analysis.movement import movement_heatmap, segment_flicks
from ..analysis.report import RunReport
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from ..telemetry.trace import MouseTrace
from . import ascii_art, theme, viz
from .onboarding import HintBar
from .replay import TrajectoryReplay

_TRAVEL_CAPTION = (
    "Where your crosshair spent its time around engagements — brighter, denser "
    "glyphs = more travel. A lopsided grid means your engagements cluster on "
    "one side. Hover a zone for its value.")
_DEFICIT_CAPTION = (
    "Estimated weakness per wall region from this run's flicks — brighter, "
    "denser glyphs mark zones your flicks reach slowly or overshoot. Same "
    "r{row}c{col} grid the adaptive engine targets; hover a zone for its score.")


class _InsightCard(QFrame):
    """One coach insight: severity dot, title, sourced body + prescription,
    and the reasoning/citations chain (the cite-everything rule made visible)."""

    def __init__(self, ins: Insight, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
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

        # header (with the ASCII mark keeping the empty state company)
        self.title = QLabel("No run analyzed yet")
        self.title.setProperty("headline", True)
        self.summary = QLabel("Finish a run while watching (or open a saved report).")
        self.summary.setWordWrap(True)
        open_btn = QPushButton("Open report…")
        open_btn.clicked.connect(self._open_dialog)
        self.empty_mark = ascii_art.AsciiEye()
        self.empty_mark.setFixedSize(230, 110)

        head = QHBoxLayout()
        head_col = QVBoxLayout()
        head_col.addWidget(self.title)
        head_col.addWidget(self.summary)
        head.addLayout(head_col, 1)
        head.addWidget(self.empty_mark)
        head.addWidget(open_btn, 0, Qt.AlignTop)

        # left column: bias bars + zone heatmap + accuracy trend, each with a
        # plain-language caption — all character art (gui/viz.py)
        self.bias_bars = viz.AsciiBars(title="flick quality by direction · lower is better")
        self.bias_caption = _caption(
            "Each bar scores flicks toward that side — longer is worse. Built from "
            "overshoot plus corrective submovements; the red bar is your weakest "
            "direction this run.")
        self.heat_map = viz.AsciiHeatmap(title="aim travel around engagements")
        self.heat_caption = _caption(_TRAVEL_CAPTION)
        self.trend_spark = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
        self.trend_caption = _caption(
            "Session-to-session accuracy for this scenario. The adaptive loop "
            "should bend this upward without ever letting it pin at 100%.")

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
        # vertical splitter: the flick-quality bars get real room instead of
        # being crushed under the heatmap
        left_w = QSplitter(Qt.Vertical)
        left_w.addWidget(bias_w)
        left_w.addWidget(heat_w)
        left_w.addWidget(self.trend_w)
        left_w.setSizes([300, 340, 240])

        # right column: notable moments + replay
        self.moments = QListWidget()
        self.moments.currentRowChanged.connect(self._select_moment)
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
        mo_lay.addWidget(self.clip_btn)
        mo_lay.addWidget(self.clip_hint)
        self._update_clip_state(-1)
        right = QSplitter(Qt.Vertical)
        right.addWidget(mo_box)
        rep_box = QGroupBox("Trajectory replay")
        rep_lay = QVBoxLayout(rep_box)
        rep_lay.addWidget(self.replay)
        right.addWidget(rep_box)
        right.setSizes([240, 400])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_w)
        split.addWidget(right)
        split.setSizes([460, 460])

        # coach insights (populated per report; hidden when there are none)
        self.coach_box = QGroupBox("Coach — every insight shows its evidence and sources")
        self.coach_lay = QVBoxLayout(self.coach_box)
        self.coach_lay.setSpacing(6)
        self.coach_box.hide()

        # room to breathe: generous spacing between every panel
        split.setHandleWidth(14)
        right.setHandleWidth(14)
        left_w.setHandleWidth(14)
        self.coach_lay.setSpacing(10)
        lay = QVBoxLayout(self)
        lay.setSpacing(16)
        if settings is not None:
            lay.addWidget(HintBar(settings, (
                "Every watched run lands here automatically — or use "
                "<b>Open report…</b> for a saved one. Click a notable moment "
                "to replay just that flick; green is clean, red overshot. "
                "Hover a card's source count for its citations.")))
        lay.addLayout(head)
        lay.addWidget(self.coach_box)
        lay.addWidget(split, 1)

    # ------------------------------------------------------------------
    def restyle(self, *_pal) -> None:
        # the viz widgets read theme.current() at paint time — update() is all
        for chart in (self.bias_bars, self.heat_map, self.trend_spark):
            chart.restyle()
        self.replay.restyle()
        if getattr(self, "_last_insights", None) is not None:
            self._fill_insights(*self._last_insights)   # cards bake colors
        if self.report is not None:
            for i in range(self.moments.count()):
                it = self.moments.item(i)
                kind = self.report.notable[i]["kind"] if i < len(self.report.notable) else ""
                it.setForeground(QColor(_kind_color(kind)))

    # ------------------------------------------------------------------
    def set_trends(self, trends) -> None:
        """Cross-session SkillTrends from the boot worker."""
        self._trends = trends

    def show_report(self, rep: RunReport, trace: MouseTrace | None = None,
                    profile: PlayerProfile | None = None) -> None:
        self.report = rep
        self.trace = trace
        self.empty_mark.hide()
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
        self._fill_moments(rep)
        self._update_clip_state(self.moments.currentRow())
        if self.trace is not None and len(self.trace) > 1:
            self.replay.load(self.trace, label="full run", flicks=self.flicks)
        else:
            self.replay.clear()

    def load_report_file(self, path: Path | str) -> None:
        self.show_report(RunReport.load(path))

    # ------------------------------------------------------------------
    def _fill_insights(self, rep: RunReport, profile: PlayerProfile | None) -> None:
        while self.coach_lay.count():
            item = self.coach_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if profile is None and self._settings is not None:
            # Saved-report path: reload the profile the run belongs to.
            name = rep.scenario
            if not name.endswith(ADAPTIVE_SUFFIX):
                name += ADAPTIVE_SUFFIX
            profile = PlayerProfile.load(name, self._settings.profile_path)
        self._fill_trend(profile)
        if profile is None or self._settings is None or profile.run_count == 0:
            self.coach_box.hide()
            return
        insights = generate_insights(
            rep, profile, self._settings,
            trends=getattr(self, "_trends", None))
        self._last_insights = (rep, profile)
        for ins in insights:
            self.coach_lay.addWidget(_InsightCard(ins))
        self.coach_box.setVisible(bool(insights))

    def _fill_trend(self, profile: PlayerProfile | None) -> None:
        """Accuracy-over-runs sparkline from the profile history (hidden
        until at least two runs exist)."""
        hist = profile.history if profile is not None else []
        accs = [float(h.get("accuracy", 0.0)) for h in hist[-60:]]
        if len(accs) >= 2:
            self.trend_spark.set_data(accs, tag=f"{accs[-1]:.0%}")
            self.trend_w.show()
        else:
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
        self.bias_bars.set_data(dirs, vals, [f"{n} flicks" for n in ns])

    def _draw_heat(self, rep: RunReport | None = None) -> None:
        """Zone heatmap on the Settings.region_cols x region_rows grid:
        the run's region deficits when the report has them, else the
        movement heatmap pooled down to the same zones."""
        cols = self._settings.region_cols if self._settings is not None else 5
        rows = self._settings.region_rows if self._settings is not None else 5
        if rep is not None and rep.region_deficits:
            grid, labels = viz.region_grid(rep.region_deficits, cols, rows)
            self.heat_map.set_title("weakness by wall region · brighter = weaker")
            self.heat_map.set_data(grid, labels, fmt="{:+.2f}")
            self.heat_caption.setText(_DEFICIT_CAPTION)
        elif self.trace is not None and len(self.trace) >= 2:
            heat, _xe, _ye = movement_heatmap(self.trace)
            pooled = viz.pool(np.log1p(heat.T), rows, cols)  # heat.T row 0 = bottom
            labels = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
            self.heat_map.set_title("aim travel around engagements")
            self.heat_map.set_data(pooled, labels, fmt="{:.2f}")
            self.heat_caption.setText(_TRAVEL_CAPTION)
        else:
            self.heat_map.set_title("aim travel around engagements")
            self.heat_map.set_data(None)
            self.heat_caption.setText(_TRAVEL_CAPTION)

    def _fill_moments(self, rep: RunReport) -> None:
        self.moments.blockSignals(True)
        self.moments.clear()
        for i, m in enumerate(rep.notable):
            it = QListWidgetItem(m["text"])
            it.setForeground(QColor(_kind_color(m["kind"])))
            it.setData(Qt.UserRole, i)
            self.moments.addItem(it)
        self.moments.blockSignals(False)
        if rep.notable:
            self.moments.setCurrentRow(0)

    def _select_moment(self, row: int) -> None:
        if self.report is None or row < 0 or row >= len(self.report.notable):
            self._update_clip_state(-1)
            return
        m = self.report.notable[row]
        self._update_clip_state(row)
        if self.trace is not None and len(self.trace) > 1:
            self.replay.load(self.trace, m["t_start"], m["t_end"],
                             label=m["kind"].replace("_", " "),
                             flicks=self.flicks)

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

    def _update_clip_state(self, row: int) -> None:
        """Enable the clip button when the selected moment has a clip; when it
        doesn't, the disabled button's tooltip says exactly why."""
        has_clip = (self.report is not None and row >= 0
                    and str(row) in (self.report.clip_files or {}))
        self.clip_btn.setEnabled(has_clip)
        off = self._clips_off_reason()
        self.clip_btn.setToolTip(
            "" if has_clip else off or "No clip was captured for this moment")
        self.clip_hint.setText(off or "")
        self.clip_hint.setVisible(off is not None)

    def _play_clip(self) -> None:
        row = self.moments.currentRow()
        if self.report is None or row < 0:
            return
        p = (self.report.clip_files or {}).get(str(row))
        if p and Path(p).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p))))

    def _open_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open run report",
                                           str(Path.home() / ".kovadapt" / "reports"),
                                           "Run reports (*.json)")
        if p:
            self.load_report_file(p)
