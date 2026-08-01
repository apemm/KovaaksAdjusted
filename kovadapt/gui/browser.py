"""Scenario browser tab: every local scenario, its training state, one click
to play — deep links can't open local scenarios, so this replaces the game's
own browser as the place you pick what to train.

Rows come from three cheap sources joined by name: the Scenarios directory
(*.sce, adaptive variants folded into their base), name-based archetype
detection, and the profile store (loaded only for scenarios that have one —
a handful of small JSONs). Refresh is synchronous and fast at hundreds of
scenarios; nothing here touches the network or the game.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..adapt.archetype import detect_archetype
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from . import theme
from .onboarding import HintBar

_COLS = ("Scenario", "Archetype", "Runs", "Accuracy", "Calibration", "Last played")


@dataclass
class _Row:
    name: str
    archetype: str
    has_adaptive: bool
    runs: int = 0
    accuracy: float = 0.0
    calibration: float = 0.0
    last_played: str = ""      # ISO ts ("" = never)


class ScenarioBrowser(QWidget):
    play_requested = Signal(str)    # base scenario name
    watch_requested = Signal(str)

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self._rows: list[_Row] = []

        hint = HintBar(settings, (
            "Every scenario installed in KovaaK's, with what the adaptive "
            "model knows about it. <b>Play</b> queues the adaptive task and "
            "launches the game; <b>●</b> marks scenarios that already have an "
            "adaptive variant."))

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search scenarios…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        self.arch_filter = QComboBox()
        self.arch_filter.addItems(["All types", "clicking", "tracking", "switching"])
        self.arch_filter.currentIndexChanged.connect(self._apply_filter)
        self.sort_by = QComboBox()
        self.sort_by.addItems(["Name", "Recently played", "Most runs"])
        self.sort_by.currentIndexChanged.connect(self._rebuild)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(self.search, 1)
        top.addWidget(self.arch_filter)
        top.addWidget(self.sort_by)
        top.addWidget(refresh)

        # full-width, tall table: the section's centerpiece
        self.table = QTableWidget(0, len(_COLS))
        self.table.setMinimumHeight(150)   # height follows row count
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(_COLS)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        # A header centred over a left-aligned column reads as a misprint;
        # each label sits over its own data's alignment.
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for i in (2, 3, 4):
            self.table.horizontalHeaderItem(i).setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter)
        header.setHighlightSections(False)   # no bold-on-select native tic
        self.table.setAlternatingRowColors(False)
        self.table.setCornerButtonEnabled(False)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _it: self._emit_play())

        self.detail = QLabel("select a scenario")
        self.detail.setProperty("dim", True)
        self.detail.setWordWrap(True)
        self.play_btn = QPushButton("▶  Play adaptive task")
        self.play_btn.setProperty("accent", True)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._emit_play)
        self.watch_btn = QPushButton("Start adapting")
        self.watch_btn.setEnabled(False)
        self.watch_btn.clicked.connect(self._emit_watch)
        self.gen_btn = QPushButton("Generate variant")
        self.gen_btn.setToolTip(
            "Write/refresh the [Adaptive] .sce from the learned profile "
            "without starting a session")
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._generate)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 6, 0, 0)
        actions.setSpacing(12)
        actions.addWidget(self.detail, 1)
        actions.addWidget(self.gen_btn)
        actions.addWidget(self.watch_btn)
        actions.addWidget(self.play_btn)

        lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        lay.addWidget(hint)
        lay.addLayout(top)
        # No stretch on the table — its height follows the row count now. The
        # slack has to be given an explicit home at the bottom, or Qt hands it
        # to whichever widget will take it (the hint bar grew to fill the
        # whole section).
        lay.addWidget(self.table)
        lay.addLayout(actions)
        lay.addStretch(1)

        self.refresh()

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        """Rescan scenarios + profiles. Cheap: name heuristics and a handful
        of small profile JSONs (only scenarios that have been trained)."""
        self._rows = []
        if not self.s.scenarios_dir.is_dir():
            self._rebuild()
            return
        stems = {p.stem for p in self.s.scenarios_dir.glob("*.sce")}
        bases = sorted(s for s in stems if not s.endswith(ADAPTIVE_SUFFIX))
        region_count = self.s.region_cols * self.s.region_rows
        for name in bases:
            row = _Row(
                name=name,
                archetype=detect_archetype(name),
                has_adaptive=(name + ADAPTIVE_SUFFIX) in stems,
            )
            adaptive = name + ADAPTIVE_SUFFIX
            if PlayerProfile.path_for(adaptive, self.s.profile_path).is_file():
                prof = PlayerProfile.load(adaptive, self.s.profile_path)
                row.runs = prof.run_count
                row.accuracy = prof.ewma_accuracy
                row.calibration = prof.readiness(region_count)["score"]
                row.last_played = prof.last_run_ts
            self._rows.append(row)
        self._rebuild()

    def _sorted_rows(self) -> list[_Row]:
        mode = self.sort_by.currentText()
        if mode == "Recently played":
            return sorted(self._rows, key=lambda r: r.last_played, reverse=True)
        if mode == "Most runs":
            return sorted(self._rows, key=lambda r: r.runs, reverse=True)
        return sorted(self._rows, key=lambda r: r.name.lower())

    def _rebuild(self) -> None:
        # A rebuild is cosmetic — sort change, Refresh, theme/accent switch —
        # but it clears the table, and clearing drops the selection. That
        # disabled Play / Start adapting / Generate variant every time the
        # theme changed, so the picked scenario is carried across.
        keep = self.selected()
        pal = theme.current()
        self.table.setRowCount(0)
        for row in self._sorted_rows():
            i = self.table.rowCount()
            self.table.insertRow(i)
            name = QTableWidgetItem(
                (f"● {row.name}") if row.has_adaptive else f"   {row.name}")
            name.setData(Qt.UserRole, row.name)
            if row.has_adaptive:
                name.setForeground(theme_color(pal.accent))
            arch = QTableWidgetItem(row.archetype)
            runs = QTableWidgetItem(str(row.runs) if row.runs else "—")
            acc = QTableWidgetItem(f"{row.accuracy:.0%}" if row.runs else "—")
            cal = QTableWidgetItem(f"{row.calibration:.0%}" if row.runs else "—")
            last = QTableWidgetItem(
                row.last_played.replace("T", " ")[:16] if row.last_played else "never")
            # Numeric columns are data: they get the mono face so digits sit
            # on one grid and the column scans vertically. In a proportional
            # face "27" and "115" are different widths and a long list stops
            # being comparable at a glance.
            # 14, not 13: mono() SNAPS to CELL_SIZES (12, 14, 20, 24) and 13
            # is not on that grid, so it silently became 12 — a size smaller
            # than the 13px row it sits in, with its baseline 3.0px high.
            # Measured against Segoe UI 13: mono12 is cap -0.8px / baseline
            # -3.0px, mono14 is cap +0.6px / baseline -1.0px. The 13 here
            # meant "match the row", and 14 is the neighbour that does.
            num_font = theme.mono(14)
            for col, item in enumerate((name, arch, runs, acc, cal, last)):
                if col:
                    item.setForeground(theme_color(pal.fg_dim if col in (1, 5)
                                                   else pal.fg))
                if col in (2, 3, 4):
                    item.setFont(num_font)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, item)
        self._apply_filter()
        if keep:
            self._reselect(keep)
        self._selection_changed()

    def _reselect(self, name: str) -> None:
        """Re-select the row for `name` after a rebuild. No-op when the
        scenario is gone or the current filter hides it — selecting a hidden
        row would arm the action buttons for something invisible."""
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if (item is not None and item.data(Qt.UserRole) == name
                    and not self.table.isRowHidden(i)):
                self.table.selectRow(i)
                return

    def _fit_table_height(self) -> None:
        """Height the table to its visible rows, within sane bounds.

        The table used to take all the section's stretch, so four scenarios
        left ~500px of empty ruled box below them; a full library still needs
        room to scroll, hence the ceiling rather than a pure fit.

        BOTH bounds, not just the maximum. Setting only a maximum made this
        fit work downward and never upward: a QVBoxLayout gives a
        non-stretched child its sizeHint, and QTableWidget.sizeHint() is
        Qt's content-independent QSize(256, 192), so raising a MAXIMUM above
        192 changes nothing. Measured at 80 scenarios: height 192, viewport
        156, FIVE rows painted, with 531px of empty page below it — a
        porthole over the whole library, while the docstring above described
        a fit that never happened.
        """
        visible = sum(1 for r in range(self.table.rowCount())
                      if not self.table.isRowHidden(r))
        row_h = self.table.verticalHeader().defaultSectionSize()
        header_h = self.table.horizontalHeader().height()
        wanted = max(150, min(header_h + row_h * max(visible, 1) + 8, 620))
        self.table.setMinimumHeight(wanted)
        self.table.setMaximumHeight(wanted)

    def _apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        arch = self.arch_filter.currentText()
        for i in range(self.table.rowCount()):
            name = (self.table.item(i, 0).data(Qt.UserRole) or "").lower()
            row_arch = self.table.item(i, 1).text()
            hide = (text and text not in name) or \
                (arch != "All types" and row_arch != arch)
            self.table.setRowHidden(i, hide)
        self._fit_table_height()
        # Hiding the selected row drops it out of selectedItems(), so
        # selected() goes empty — but nothing RE-ASKED, and the three action
        # buttons stayed enabled with the detail line still describing a
        # scenario no longer on screen. Play sat filled and clickable over an
        # empty table and emitted nothing. `_rebuild` already re-asks; the
        # typing and archetype-filter path did not.
        self._selection_changed()

    # -------------------------------------------------------------- actions
    def selected(self) -> str:
        items = self.table.selectedItems()
        return items[0].data(Qt.UserRole) if items else ""

    def _selection_changed(self) -> None:
        name = self.selected()
        on = bool(name)
        for btn in (self.play_btn, self.watch_btn, self.gen_btn):
            btn.setEnabled(on)
        if not on:
            self.detail.setText("select a scenario")
            return
        row = next((r for r in self._rows if r.name == name), None)
        if row is None:
            return
        if row.runs:
            self.detail.setText(
                f"{row.archetype} · {row.runs} runs · accuracy "
                f"{row.accuracy:.1%} · calibration {row.calibration:.0%}")
        else:
            self.detail.setText(
                f"{row.archetype} · never trained — Play creates the adaptive "
                "variant and starts learning from run 1")

    def _emit_play(self) -> None:
        if self.selected():
            self.play_requested.emit(self.selected())

    def _emit_watch(self) -> None:
        if self.selected():
            self.watch_requested.emit(self.selected())

    def _generate(self) -> None:
        """One-shot variant refresh (same as `kovadapt generate`)."""
        name = self.selected()
        if not name:
            return
        from ..adapt.archetype import detect_archetype as detect
        from ..adapt.engine import AdaptationEngine, settle_focus
        from ..scenario.generator import generate_adaptive_variant

        adaptive = name + ADAPTIVE_SUFFIX
        profile = PlayerProfile.load(adaptive, self.s.profile_path)
        profile.scenario = adaptive
        if not profile.archetype:
            profile.archetype = detect(name)
        try:
            plan = AdaptationEngine(self.s).plan(profile, None)
            out = generate_adaptive_variant(
                self.s.scenarios_dir / f"{name}.sce", plan, self.s,
                self.s.scenarios_dir / f"{adaptive}.sce")
            settle_focus(profile, plan)
            profile.save(self.s.profile_path)
            msg = f"wrote {out.name} — {plan.describe()}"
        except OSError as exc:
            msg = f"could not generate: {exc}"
        # refresh() rewrites the detail line from the (re-)selected row, so
        # the outcome has to be written after it or it is never seen.
        self.refresh()
        self.detail.setText(msg)

    # ------------------------------------------------------------------
    def restyle(self, *_pal) -> None:
        self._rebuild()


def theme_color(hex_color: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(hex_color))
