"""Optimizer window: the free Process Lasso alternative for KovaaK's.

A separate top-level window (dark, simple) with four sections:

    Hardware      detected CPU / GPU / RAM / refresh / Windows version
    Checkup       scan -> per-item status + Fix buttons + "Fix all safe"
    Watchdog      auto-tune every game launch; optional start with Windows
    Advice        hardware-matched launch options and settings

Scans run on a QThread (the GPU probe shells out to PowerShell, ~1s);
fixes are individually fast and run inline on click. Watchdog events
arrive on its polling thread and are bridged into Qt via a signal.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..optimize.checkup import CheckResult, SystemCheckup
from ..optimize.hardware import HardwareInfo, detect_hardware
from ..optimize.recommend import (
    recommended_settings,
    skipped_launch_options,
    steam_launch_options,
)
from ..optimize.watchdog import (
    GameWatchdog,
    register_startup,
    startup_registered,
    unregister_startup,
)
from .theme import ACCENT, BAD, FG_DIM, GOOD, WARN

_STATUS_COLOR = {"ok": GOOD, "warn": WARN, "bad": BAD, "info": ACCENT,
                 "unknown": FG_DIM}
_STATUS_DOT = {"ok": "●", "warn": "●", "bad": "●", "info": "○", "unknown": "○"}


class _ScanWorker(QThread):
    """Hardware detection + all checkup probes off the UI thread."""

    done = Signal(object, object)   # (HardwareInfo, list[CheckResult])

    def __init__(self, kovaaks_root: str, parent=None) -> None:
        super().__init__(parent)
        self.kovaaks_root = kovaaks_root

    def run(self) -> None:
        hw = detect_hardware()
        results = SystemCheckup(self.kovaaks_root, hw).run_all()
        self.done.emit(hw, results)


class _WatchdogBridge(QObject):
    event = Signal(str)


class _CheckRow(QFrame):
    """One checkup line: colored status dot, title, detail, optional Fix."""

    def __init__(self, result: CheckResult, on_fix, parent=None) -> None:
        super().__init__(parent)
        self.result = result
        dot = QLabel(_STATUS_DOT.get(result.status, "○"))
        dot.setStyleSheet(
            f"color: {_STATUS_COLOR.get(result.status, FG_DIM)}; font-size: 15px;")
        dot.setFixedWidth(18)
        title = QLabel(f"<b>{result.title}</b>")
        title.setTextFormat(Qt.RichText)
        self.detail = QLabel(result.detail)
        self.detail.setWordWrap(True)
        self.detail.setProperty("dim", True)

        grid = QGridLayout(self)
        grid.setContentsMargins(4, 6, 4, 6)
        grid.addWidget(dot, 0, 0, Qt.AlignTop)
        grid.addWidget(title, 0, 1)
        grid.addWidget(self.detail, 1, 1)
        if result.can_fix:
            self.fix_btn = QPushButton(result.fix_label or "Fix")
            if result.safe:
                self.fix_btn.setProperty("accent", True)
            self.fix_btn.clicked.connect(lambda: on_fix(self))
            grid.addWidget(self.fix_btn, 0, 2, 2, 1, Qt.AlignVCenter)
        grid.setColumnStretch(1, 1)

    def show_outcome(self, msg: str) -> None:
        self.detail.setText(msg)
        if hasattr(self, "fix_btn"):
            self.fix_btn.setEnabled(False)
            self.fix_btn.setText("Applied")


class OptimizerWindow(QWidget):
    """Top-level optimizer window (create with parent=None)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(None)
        self.setWindowTitle("kovadapt — optimizer")
        self.resize(860, 720)
        self.s = settings
        self.hw: HardwareInfo | None = None
        self.checkup: SystemCheckup | None = None
        self._scan: _ScanWorker | None = None

        # --- hardware summary -------------------------------------------
        self.hw_label = QLabel("Scanning hardware…")
        self.hw_label.setProperty("headline", True)
        self.hw_sub = QLabel("")
        self.hw_sub.setProperty("dim", True)
        hw_box = QGroupBox("Detected hardware")
        v = QVBoxLayout(hw_box)
        v.addWidget(self.hw_label)
        v.addWidget(self.hw_sub)

        # --- checkup ------------------------------------------------------
        self.scan_btn = QPushButton("Re-scan")
        self.scan_btn.clicked.connect(self.rescan)
        self.fix_safe_btn = QPushButton("Fix all safe items")
        self.fix_safe_btn.setProperty("accent", True)
        self.fix_safe_btn.setToolTip(
            "Runs every fix that is per-user, reversible, and admin-free. "
            "Invasive fixes (power plan, config deletion) keep their own button.")
        self.fix_safe_btn.clicked.connect(self.fix_all_safe)
        self.fix_safe_btn.setEnabled(False)
        head = QHBoxLayout()
        head.addWidget(self.scan_btn)
        head.addWidget(self.fix_safe_btn)
        head.addStretch(1)

        self.rows_holder = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_holder)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.addStretch(1)
        rows_scroll = QScrollArea()
        rows_scroll.setWidget(self.rows_holder)
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QScrollArea.NoFrame)
        check_box = QGroupBox("System checkup")
        cv = QVBoxLayout(check_box)
        cv.addLayout(head)
        cv.addWidget(rows_scroll, 1)

        # --- watchdog -------------------------------------------------------
        self._bridge = _WatchdogBridge()
        self._bridge.event.connect(self._log)
        self.watchdog = GameWatchdog(on_event=self._bridge.event.emit)
        self.wd_toggle = QCheckBox(
            "Auto-tune on every game launch (High priority + free the input core)")
        self.wd_toggle.setToolTip(
            "Exactly what Process Lasso's persistent rules do, free. Runs while "
            "this app is open; enable the startup option to cover every session.")
        self.wd_toggle.toggled.connect(self._toggle_watchdog)
        self.wd_startup = QCheckBox("Start the watchdog with Windows (background, no window)")
        self.wd_startup.setChecked(startup_registered())
        self.wd_startup.toggled.connect(self._toggle_startup)
        self.wd_log = QPlainTextEdit()
        self.wd_log.setReadOnly(True)
        self.wd_log.setMaximumBlockCount(200)
        self.wd_log.setMaximumHeight(110)
        wd_box = QGroupBox("Watchdog (free Process Lasso replacement)")
        wv = QVBoxLayout(wd_box)
        wv.addWidget(self.wd_toggle)
        wv.addWidget(self.wd_startup)
        wv.addWidget(self.wd_log)

        # --- advice -----------------------------------------------------------
        self.launch_label = QLabel(steam_launch_options(HardwareInfo()))
        self.launch_label.setStyleSheet(
            f"font-family: Consolas, monospace; color: {ACCENT};")
        self.launch_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(70)
        copy_btn.clicked.connect(self._copy_launch)
        lrow = QHBoxLayout()
        lrow.addWidget(self.launch_label, 1)
        lrow.addWidget(copy_btn)
        skip_lines = "".join(
            f"<li><code>{flag}</code> — {why}</li>"
            for flag, why in skipped_launch_options())
        skips = QLabel(f"<b>Skip these (myths):</b><ul>{skip_lines}</ul>")
        skips.setTextFormat(Qt.RichText)
        skips.setWordWrap(True)
        skips.setProperty("dim", True)
        self.recs_label = QLabel("")
        self.recs_label.setTextFormat(Qt.RichText)
        self.recs_label.setWordWrap(True)
        adv_inner = QWidget()
        av = QVBoxLayout(adv_inner)
        av.addWidget(QLabel("Steam launch options (right-click KovaaK's > Properties):"))
        av.addLayout(lrow)
        av.addWidget(skips)
        av.addWidget(self.recs_label)
        av.addStretch(1)
        adv_scroll = QScrollArea()
        adv_scroll.setWidget(adv_inner)
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QScrollArea.NoFrame)
        adv_box = QGroupBox("Recommended for your hardware")
        bv = QVBoxLayout(adv_box)
        bv.addWidget(adv_scroll)

        lay = QVBoxLayout(self)
        lay.addWidget(hw_box)
        lay.addWidget(check_box, 3)
        lay.addWidget(wd_box)
        lay.addWidget(adv_box, 2)

        self.rescan()

    # ------------------------------------------------------------- scanning
    def rescan(self) -> None:
        if self._scan is not None and self._scan.isRunning():
            return
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning…")
        self._scan = _ScanWorker(self.s.kovaaks_root, parent=self)
        self._scan.done.connect(self._on_scan)
        self._scan.start()

    def _on_scan(self, hw: HardwareInfo, results: list[CheckResult]) -> None:
        self.hw = hw
        self.checkup = SystemCheckup(self.s.kovaaks_root, hw)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Re-scan")

        if hw.cpu_name or hw.gpu_name:
            self.hw_label.setText(f"{hw.cpu_name or 'unknown CPU'}  ·  "
                                  f"{hw.gpu_name or 'unknown GPU'}")
            bits = []
            if hw.ram_gb:
                bits.append(f"{hw.ram_gb:.0f} GB RAM")
            if hw.logical_cores:
                bits.append(f"{hw.logical_cores} threads")
            if hw.monitor_hz:
                bits.append(f"{hw.monitor_hz} Hz display")
            bits.append("Windows 11" if hw.is_windows_11 else "Windows 10")
            self.hw_sub.setText("  ·  ".join(bits))
        else:
            self.hw_label.setText("Hardware detection unavailable")
            self.hw_sub.setText("; ".join(hw.notes))

        # rebuild check rows
        while self.rows_layout.count() > 1:
            item = self.rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        fixable_safe = 0
        for r in results:
            self.rows_layout.insertWidget(self.rows_layout.count() - 1,
                                          _CheckRow(r, self._fix_row))
            if r.can_fix and r.safe:
                fixable_safe += 1
        self.fix_safe_btn.setEnabled(fixable_safe > 0)
        self.fix_safe_btn.setText(
            f"Fix all safe items ({fixable_safe})" if fixable_safe
            else "Fix all safe items")

        # advice
        self.launch_label.setText(steam_launch_options(hw))
        recs = recommended_settings(hw)
        cat_names = {"launch": "Launch", "video": "In-game video",
                     "driver": "GPU driver", "windows": "Windows"}
        html = ""
        for pr, badge in ((1, "do this"), (2, "worth trying"), (3, "situational")):
            group = [r for r in recs if r.priority == pr]
            if not group:
                continue
            html += f"<p><b>{badge.capitalize()}:</b></p><ul>"
            for r in group:
                html += (f"<li><b>{r.title}</b> <i>({cat_names.get(r.category, '')})</i>"
                         f"<br>{r.detail}</li>")
            html += "</ul>"
        self.recs_label.setText(html)

    # --------------------------------------------------------------- fixes
    def _fix_row(self, row: _CheckRow) -> None:
        if self.checkup is None:
            return
        row.show_outcome(self.checkup.fix(row.result.check_id))

    def fix_all_safe(self) -> None:
        for i in range(self.rows_layout.count() - 1):
            w = self.rows_layout.itemAt(i).widget()
            if (isinstance(w, _CheckRow) and w.result.can_fix and w.result.safe
                    and hasattr(w, "fix_btn") and w.fix_btn.isEnabled()):
                self._fix_row(w)

    # ------------------------------------------------------------ watchdog
    def _toggle_watchdog(self, on: bool) -> None:
        if on:
            self.watchdog.start()
        else:
            self.watchdog.stop()

    def _toggle_startup(self, on: bool) -> None:
        self._log(register_startup() if on else unregister_startup())

    def _log(self, msg: str) -> None:
        self.wd_log.appendPlainText(msg)

    def _copy_launch(self) -> None:
        QApplication.clipboard().setText(self.launch_label.text())
        self._log("launch options copied to clipboard")

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        # Closing the window hides it; the watchdog keeps running (that is
        # its point). It stops when the whole app exits (daemon thread).
        event.ignore()
        self.hide()
