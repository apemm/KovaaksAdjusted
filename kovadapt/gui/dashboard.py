"""Dashboard tab: the session hub — play adaptive tasks, launch the game,
start/stop the adaptation loop, live profile stats, overlay controls, log."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import launcher
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from . import theme
from .ascii_art import AsciiProgress, CatSlider
from .onboarding import HintBar
from .overlay import OverlayWindow
from .workers import WatcherWorker


class Dashboard(QWidget):
    report_ready = Signal(object)   # re-emitted RunReport for the analysis tab

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.worker: WatcherWorker | None = None
        self._watching: str = ""
        self._stopping = False
        self._pending: tuple[str, str] | None = None   # (scenario, "play"|"watch")
        self._last_log = ""
        self._last_profile: PlayerProfile | None = None
        self._install: launcher.InstallStatus | None = None
        self.overlay = OverlayWindow(settings)

        hint = HintBar(settings, (
            "Pick a scenario, then <b>Play adaptive task</b> — kovadapt watches "
            "your runs and regenerates the <b>[Adaptive]</b> variant between "
            "them. Mouse telemetry records automatically while a session runs "
            "(the <b>REC</b> dot). <b>Start adapting</b> does the same without "
            "launching the game. The overlay shows only in Borderless/Windowed "
            "mode."))

        # ---------------------------------------------------- play controls
        self.install_lbl = QLabel("checking install…")
        self.install_lbl.setProperty("dim", True)
        self.launch_btn = QPushButton("Launch KovaaK's")
        self.launch_btn.clicked.connect(self._launch_game)
        self.play_btn = QPushButton("▶  Play adaptive task")
        self.play_btn.setProperty("accent", True)
        self.play_btn.setToolTip(
            "Start watching, queue the adaptive playlist, and launch KovaaK's "
            "— in-game, open Playlists → kovadapt adaptive to play")
        self.play_btn.clicked.connect(self.play)

        self.scenario = QComboBox()
        self.scenario.setEditable(True)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_scenarios)
        self.start_btn = QPushButton("Start adapting")
        self.start_btn.clicked.connect(self.toggle)

        # full-width Play panel: the column's lead panel, generous rows
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(QLabel("Scenario:"))
        row1.addWidget(self.scenario, 1)
        row1.addWidget(self.refresh_btn)
        row1.addWidget(self.start_btn)
        self.rec_lbl = QLabel("")
        self.rec_lbl.setTextFormat(Qt.RichText)
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(self.install_lbl, 1)
        row2.addWidget(self.rec_lbl)
        row2.addWidget(self.launch_btn)
        row2.addWidget(self.play_btn)
        play_box = QGroupBox("Play")
        pv = QVBoxLayout(play_box)
        pv.setContentsMargins(14, 12, 14, 14)
        pv.setSpacing(12)
        pv.addLayout(row1)
        pv.addLayout(row2)

        # -------------------------------------------------- overlay controls
        self.ov_toggle = QPushButton("Overlay")
        self.ov_toggle.setCheckable(True)
        self.ov_toggle.setToolTip(
            "Always-on-top session card over the game (Borderless/Windowed only)")
        self.ov_toggle.toggled.connect(self._toggle_overlay)
        self.ov_unlock = QPushButton("Unlock")
        self.ov_unlock.setCheckable(True)
        self.ov_unlock.setToolTip("Unlock to drag the overlay into place; lock "
                                  "to make it click-through again")
        self.ov_unlock.toggled.connect(self._unlock_overlay)
        self.ov_opacity = CatSlider(30, 100)
        self.ov_opacity.setValue(int(settings.overlay_opacity * 100))
        self.ov_opacity.setToolTip("Overlay opacity — the cat walks it there")
        self.ov_opacity.valueChanged.connect(
            lambda v: self.overlay.set_opacity(v / 100))
        # Debounced persist: keyboard/wheel changes never fire sliderReleased.
        self._opacity_save = QTimer(self)
        self._opacity_save.setSingleShot(True)
        self._opacity_save.setInterval(800)
        self._opacity_save.timeout.connect(self._save_settings)
        self.ov_opacity.valueChanged.connect(
            lambda _v: self._opacity_save.start())
        self.ov_auto = QCheckBox("Show when a session starts")
        self.ov_auto.setChecked(settings.overlay_autoshow)
        self.ov_auto.toggled.connect(self._set_autoshow)

        # overlay controls: their own full-width row in the column
        ov_box = QGroupBox("Overlay")
        ov = QHBoxLayout(ov_box)
        ov.setContentsMargins(14, 12, 14, 14)
        ov.setSpacing(12)
        ov.addWidget(self.ov_toggle)
        ov.addWidget(self.ov_unlock)
        ov.addWidget(self.ov_opacity, 1)
        ov.addWidget(self.ov_auto)

        # ------------------------------- profile stats: two-column grid
        self.stat_labels: dict[str, QLabel] = {}
        stats_box = QGroupBox("Learned profile")
        grid = QGridLayout(stats_box)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setHorizontalSpacing(36)
        grid.setVerticalSpacing(14)
        for i, (key, cap) in enumerate([
            ("runs", "Runs"), ("accuracy", "Accuracy EWMA"),
            ("scale", "Target scale"), ("movement", "Movement"),
            ("focus", "Focus region"), ("pace", "Pace (kills/s)"),
            ("archetype", "Archetype"), ("fatigue", "Session fatigue"),
        ]):
            grid.addWidget(QLabel(cap), i // 2, (i % 2) * 2)
            lab = QLabel("—")
            lab.setProperty("stat", True)
            grid.addWidget(lab, i // 2, (i % 2) * 2 + 1)
            self.stat_labels[key] = lab
        grid.setColumnStretch(1, 1)         # value columns breathe
        grid.setColumnStretch(3, 1)

        # calibration readiness: the iris motif filling through the rainbow
        self.readiness = AsciiProgress()
        self.readiness.set_progress(0.0, "calibration 0%")
        self.readiness_msg = QLabel("play runs to calibrate the adaptive model")
        self.readiness_msg.setProperty("dim", True)
        self.readiness_msg.setWordWrap(True)
        grid.addWidget(self.readiness, 4, 0, 1, 2)
        grid.addWidget(self.readiness_msg, 4, 2, 1, 2)

        # accuracy history sparkline — tall enough to actually read
        self.trend = pg.PlotWidget(title="Accuracy per run")
        self.trend.setMinimumHeight(240)
        self.trend.setMaximumHeight(340)
        self.trend.showGrid(y=True, alpha=0.2)
        self.trend_curve = self.trend.plot([], [], symbol="o", symbolSize=5,
                                           symbolPen=None)

        # log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(180)

        lay = QVBoxLayout(self)
        lay.setSpacing(16)
        lay.addWidget(hint)
        lay.addWidget(play_box)
        lay.addWidget(ov_box)
        lay.addWidget(stats_box)
        lay.addWidget(self.trend)
        lay.addWidget(self.log, 1)

        self.refresh_scenarios()
        self.restyle(theme.current())
        self._refresh_install()

    @property
    def last_profile(self) -> PlayerProfile | None:
        """Freshest loaded profile (updated before report_ready re-emits)."""
        return self._last_profile

    # ------------------------------------------------------------- theming
    def restyle(self, pal=None) -> None:
        pal = pal or theme.current()
        from PySide6.QtGui import QColor

        trend_bg = QColor(pal.bg_alt)
        trend_bg.setAlpha(205)          # the backdrop eye ghosts through
        self.trend.setBackground(trend_bg)
        self.trend_curve.setPen(pg.mkPen(pal.accent, width=2))
        self.trend_curve.setSymbolBrush(pal.accent)
        self._render_install()
        self.overlay.restyle()

    # ------------------------------------------------------------- install
    def _refresh_install(self) -> None:
        self._install = launcher.check_install(self.s)
        self._render_install()

    def _render_rec(self, watching: bool) -> None:
        """Make the automatic telemetry recorder visible: it starts with
        every session — nobody should have to wonder how to turn it on."""
        pal = theme.current()
        if not watching:
            self.rec_lbl.setText("")
            return
        if self.s.telemetry_enabled:
            self.rec_lbl.setText(
                f"<span style='color:{pal.bad}'>●</span> REC mouse telemetry")
            self.rec_lbl.setToolTip(
                "Raw Input recording started automatically with this session; "
                "flick analysis appears with each run's report")
        else:
            self.rec_lbl.setText(
                f"<span style='color:{pal.warn}'>○</span> telemetry off")
            self.rec_lbl.setToolTip(
                "Enable 'Record raw mouse telemetry' in Adaptability to get "
                "flick analysis and region evidence")

    def _render_install(self) -> None:
        st = self._install
        if st is None:
            return
        pal = theme.current()
        color = pal.good if st.ok else pal.bad
        self.install_lbl.setText(
            f"<span style='color:{color}'>●</span> {st.describe()}")
        self.install_lbl.setTextFormat(Qt.RichText)
        self.launch_btn.setEnabled(st.ok)
        self.play_btn.setEnabled(st.ok)

    # --------------------------------------------------------------- play
    def _launch_game(self) -> None:
        self._refresh_install()
        msg, _ok = launcher.launch_game()
        self.append_log(msg)

    def _picked_scenario(self) -> str:
        """Current selection, normalized: typing the adaptive name directly
        must not create '[Adaptive] [Adaptive]' compounding variants."""
        return self.scenario.currentText().strip().removesuffix(ADAPTIVE_SUFFIX)

    def play_scenario(self, name: str) -> None:
        """Entry point for the scenario browser: select + play."""
        self.scenario.setCurrentText(name)
        self.play()

    def watch_scenario(self, name: str) -> None:
        if self.worker is not None and self._watching != name:
            self._switch_to(name, "watch")
            return
        self.scenario.setCurrentText(name)
        if self.worker is None:
            self.toggle()

    def _switch_to(self, name: str, kind: str) -> None:
        """Stop the running session and start `name` once it is down —
        changing scenarios must never require a manual stop first."""
        self._pending = (name, kind)
        self.append_log(f"switching from {self._watching!r} to {name!r}…")
        if not self._stopping:
            self.toggle()               # the stop branch

    def play(self) -> None:
        """Watch + queue playlist + deep-link into the adaptive scenario."""
        name = self._picked_scenario()
        if not name:
            self.append_log("pick a scenario first")
            return
        if self._stopping:
            self._pending = (name, "play")   # start this once the stop lands
            return
        if self.worker is not None and self._watching != name:
            self._switch_to(name, "play")
            return
        if self.worker is None and not self._start_watch(name):
            return
        self._refresh_install()
        msg, _ok = launcher.play_adaptive(self.s, name)
        self.append_log(msg)

    # ------------------------------------------------------------- watching
    def toggle(self) -> None:
        if self.worker is not None:
            self._stopping = True
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Stopping…")
            self.play_btn.setEnabled(False)
            return
        name = self._picked_scenario()
        if not name:
            self.append_log("pick a scenario first")
            return
        self._start_watch(name)

    def _start_watch(self, name: str) -> bool:
        """Create the worker, bootstrap the adaptive .sce synchronously (so
        Play can deep-link immediately), and start watching."""
        w = WatcherWorker(self.s, name, parent=self)
        if not w.watcher.base_sce_path().is_file():
            self.append_log(f"scenario file not found: {w.watcher.base_sce_path()}")
            w.deleteLater()
            return False
        # Connect before bootstrap so its log lines actually land in the log.
        w.message.connect(self.append_log)
        w.report_ready.connect(self._on_report)
        w.stopped.connect(self._on_stopped)
        w.finished.connect(w.deleteLater)   # don't retain dead QThreads
        if not w.watcher.adaptive_sce_path().is_file():
            try:
                w.watcher.bootstrap()
            except Exception as exc:
                self.append_log(f"could not create adaptive variant: {exc}")
                w.deleteLater()
                return False
        self.worker = w
        self._watching = name
        w.start()
        self.start_btn.setText("Stop")
        self.scenario.setEnabled(False)
        self._render_rec(True)
        self.refresh_profile(name)
        self.overlay.start_session(name + ADAPTIVE_SUFFIX)
        if self.s.overlay_autoshow and not self.ov_toggle.isChecked():
            self.ov_toggle.setChecked(True)
        return True

    def _on_stopped(self) -> None:
        self.worker = None
        self._watching = ""
        self._stopping = False
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start adapting")
        self.scenario.setEnabled(True)
        self._render_install()          # re-enable Play per install status
        self._render_rec(False)
        self.overlay.stop_session()
        self.append_log("stopped")
        if self._pending is not None:
            name, kind = self._pending
            self._pending = None
            self.scenario.setCurrentText(name)
            if kind == "play":
                self.play()
            else:
                self.toggle()

    def _on_report(self, rep) -> None:
        self.refresh_profile(self._watching or self._picked_scenario())
        fat = getattr(rep, "fatigue", None) or {}
        if fat.get("runs", 0) > 0:
            self.stat_labels["fatigue"].setText(
                f"{fat.get('level', 'fresh')} ({fat.get('score', 0.0):.0%})"
            )
        self.overlay.on_report(rep, self._last_profile)
        self.report_ready.emit(rep)

    # -------------------------------------------------------------- overlay
    def _toggle_overlay(self, on: bool) -> None:
        if on:
            self.overlay.show_overlay()
        else:
            if self.ov_unlock.isChecked():
                self.ov_unlock.setChecked(False)
            self.overlay.hide()

    def _unlock_overlay(self, on: bool) -> None:
        if on and not self.ov_toggle.isChecked():
            self.ov_toggle.setChecked(True)
        self.overlay.set_unlocked(on)
        self.ov_unlock.setText("Lock" if on else "Unlock")

    def _set_autoshow(self, on: bool) -> None:
        self.s.overlay_autoshow = on
        self._save_settings()

    def _save_settings(self) -> None:
        try:
            self.s.save()
        except OSError:
            pass

    def shutdown(self) -> None:
        self.overlay.close()

    # ------------------------------------------------------------------
    def refresh_scenarios(self) -> None:
        cur = self.scenario.currentText()
        self.scenario.clear()
        if self.s.scenarios_dir.is_dir():
            names = sorted(
                p.stem for p in self.s.scenarios_dir.glob("*.sce")
                if not p.stem.endswith(ADAPTIVE_SUFFIX)
            )
            self.scenario.addItems(names)
        if cur:
            self.scenario.setCurrentText(cur)

    def append_log(self, line: str) -> None:
        if line == self._last_log:      # never spam identical lines
            return
        self._last_log = line
        self.log.appendPlainText(line)

    @staticmethod
    def _ascii_bar(frac: float, width: int = 8) -> str:
        filled = round(max(0.0, min(1.0, frac)) * width)
        return "▓" * filled + "░" * (width - filled)

    def refresh_profile(self, base_name: str) -> None:
        prof = PlayerProfile.load(base_name + ADAPTIVE_SUFFIX, self.s.profile_path)
        self._last_profile = prof
        sl = self.stat_labels
        ready = prof.readiness(self.s.region_cols * self.s.region_rows)
        self.readiness.set_progress(
            ready["score"],
            f"calibration {ready['score']:.0%} — {ready.get('stage', '')}")
        parts = ready.get("detail", [])
        fracs = (ready["baseline"], ready["regions"], ready["bias"])
        bars = "   ".join(f"{self._ascii_bar(f)} {txt}"
                          for f, txt in zip(fracs, parts))
        self.readiness_msg.setText(bars or ready["message"])
        if prof.run_count == 0:
            for lab in sl.values():
                lab.setText("—")
            self.trend_curve.setData([], [])
            return
        sl["runs"].setText(str(prof.run_count))
        sl["accuracy"].setText(f"{prof.ewma_accuracy:.1%}")
        sl["scale"].setText(f"{prof.target_scale:.2f}x")
        sl["movement"].setText(f"{prof.movement:.2f}")
        sl["focus"].setText(prof.last_focus or "—")
        sl["pace"].setText(f"{prof.ewma_kps:.2f}")
        sl["archetype"].setText(prof.archetype or "—")
        accs = [h.get("accuracy", 0.0) for h in prof.history[-60:]]
        self.trend_curve.setData(list(range(len(accs))), accs)
