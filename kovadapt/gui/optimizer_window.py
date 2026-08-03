"""Optimizer window: the free Process Lasso alternative for KovaaK's.

A separate top-level window (dark, simple) with four sections:

    Hardware      detected CPU / GPU / RAM / refresh / Windows version
    Checkup       scan -> per-item status + Fix buttons + "Fix all safe"
    Watchdog      auto-tune every game launch; optional start with Windows
    Advice        hardware-matched launch options and settings

Scans run on a QThread (the GPU probe shells out to PowerShell, ~1s) and so
do fixes — several shell out too, and the power-plan one chains four
powercfg calls with a 15 s timeout each. Watchdog events arrive on its
polling thread and are bridged into Qt via a signal.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRect, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
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
from . import theme

_STATUS_DOT = {"ok": "●", "warn": "●", "bad": "●", "info": "○", "unknown": "○"}

# How tall the watchdog log is allowed to get once it HAS something to show.
# It stays hidden until then — see the note where it is built. Four lines:
# at 110 it took so much off the top that opening it dropped the checkup
# from 197px to 81px on the smallest window the app will open. The log is a
# scrolling record and scrolls; the checkup is a list you act on.
_LOG_H = 72


def _status_color(status: str) -> str:
    pal = theme.current()
    return {"ok": pal.good, "warn": pal.warn, "bad": pal.bad,
            "info": pal.accent}.get(status, pal.fg_dim)


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


class _FixWorker(QThread):
    """Checkup fixes off the UI thread, one after another.

    A fix is not "individually fast": most shell out or hit the registry, and
    _f_power alone chains four powercfg calls whose subprocess timeout is 15 s
    apiece — run inline on click that froze the window for up to a minute.
    SystemCheckup.fix() converts its own exceptions into a message, so this
    loop cannot die half way through a batch.
    """

    one_done = Signal(str, str)     # (check_id, outcome message)

    def __init__(self, checkup: SystemCheckup, check_ids: list[str],
                 parent=None) -> None:
        super().__init__(parent)
        self.checkup = checkup
        self.check_ids = list(check_ids)

    def run(self) -> None:
        for cid in self.check_ids:
            self.one_done.emit(cid, self.checkup.fix(cid))


# Worker threads that were still running at shutdown(). Qt 6 aborts the
# process when a running QThread is destroyed, and a fix can be mid registry
# write, so parking one here (out of the window's child tree, with a live
# Python reference) is safer than terminate()ing it: the interpreter is on
# its way out anyway.
_PARKED: list[QThread] = []


class _WatchdogBridge(QObject):
    event = Signal(str)


class _CheckRow(QFrame):
    """One checkup line: colored status dot, title, detail, optional Fix."""

    def __init__(self, result: CheckResult, on_fix, parent=None) -> None:
        super().__init__(parent)
        self.result = result
        self._dot = dot = QLabel(_STATUS_DOT.get(result.status, "○"))
        dot.setFixedWidth(18)
        self.restyle()
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

    def restyle(self) -> None:
        self._dot.setStyleSheet(
            f"color: {_status_color(self.result.status)}; font-size: 15px;")

    def mark_pending(self) -> None:
        """Queued for the fix worker: the button can't be clicked twice, and
        the row says why nothing has happened yet."""
        if hasattr(self, "fix_btn"):
            self.fix_btn.setEnabled(False)
            self.fix_btn.setText("Applying…")

    # `SystemCheckup.apply_fix` turns any exception into "fix failed: …" and
    # hands it back through the SAME channel as a success, so the outcome
    # string is the only thing that knows which happened.
    FAILED_PREFIX = "fix failed"

    def show_outcome(self, msg: str) -> None:
        """Report what the fix actually did — including that it did not work.

        This used to set the detail text, disable the button and label it
        "Applied" unconditionally, and never touch `result.status` or the
        status dot: a pixel-exact diff of the dot before and after a real fix
        showed 0 of 360 pixels changing. So a FAILED fix rendered as a green
        "Applied" over an unchanged amber dot, and success and failure
        differed by one sentence of body text. On an app whose whole rule is
        never to claim something it has not measured, that is the wrong thing
        to put in a release.
        """
        ok = not msg.lower().startswith(self.FAILED_PREFIX)
        self.detail.setText(msg)
        self.result.status = "ok" if ok else "bad"
        self._dot.setText(_STATUS_DOT.get(self.result.status, "○"))
        self.restyle()
        if hasattr(self, "fix_btn"):
            # A failure leaves the button live: the cause is usually
            # transient (a permission prompt declined, the game running) and
            # re-running is the obvious next move.
            self.fix_btn.setEnabled(not ok)
            self.fix_btn.setText("Applied" if ok else "Retry")


class OptimizerWindow(QWidget):
    """Top-level optimizer window (create with parent=None)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(None)
        self.setWindowTitle("kovadapt — optimizer")
        # 720 tall regardless of the panel, on a window whose main content is
        # a twelve-row list ~860px long. Take the height the screen actually
        # has, capped so this never opens taller than the desktop it is on.
        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None \
            else QRect(0, 0, 1280, 800)
        self.resize(860, max(640, min(940, int(geo.height() * 0.88))))
        self.s = settings
        self.hw: HardwareInfo | None = None
        self.checkup: SystemCheckup | None = None
        self._scan: _ScanWorker | None = None
        self._fix: _FixWorker | None = None
        self._suspended: list[_CheckRow] = []   # buttons parked during a batch

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
        self.wd_log.setMaximumHeight(_LOG_H)
        # Hidden until the watchdog actually says something. This box has no
        # stretch, so its sizeHint comes off the top of the window before the
        # stretch factors divide anything up — and 110px of that was an empty
        # read-only pane. It made the un-stretched watchdog box TALLER than
        # the stretch-3 System checkup beside it, which opened as a 79px slot
        # over a twelve-row list. The "Evidence:" label above already says
        # what will appear here, so an empty pane was not even carrying the
        # explanation.
        self.wd_log.hide()
        self.jitter_lbl = QLabel(
            "Evidence: watch a session with telemetry on and per-run input "
            "jitter shows up here, before vs after each auto-tune.")
        self.jitter_lbl.setProperty("dim", True)
        self.jitter_lbl.setWordWrap(True)
        self._jitter_runs: list[tuple[float, float]] = []   # (epoch, jitter_ms)
        wd_box = QGroupBox("Watchdog (free Process Lasso replacement)")
        wv = QVBoxLayout(wd_box)
        wv.addWidget(self.wd_toggle)
        wv.addWidget(self.wd_startup)
        wv.addWidget(self.jitter_lbl)
        wv.addWidget(self.wd_log)

        # --- advice -----------------------------------------------------------
        self.launch_label = QLabel(steam_launch_options(HardwareInfo()))
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
        # The checkup is what this window is for: a twelve-row list, each row
        # a status and a Fix button. Advice is reference prose that reads fine
        # a few lines at a time. 3:2 gave the list a 79px slot; the two boxes
        # with no stretch at all took their full sizeHint off the top first.
        lay.addWidget(check_box, 5)
        lay.addWidget(wd_box)
        lay.addWidget(adv_box, 2)

        self.restyle()
        self.rescan()

    # ------------------------------------------------------------- theming
    def restyle(self, *_pal) -> None:
        pal = theme.current()
        self.launch_label.setStyleSheet(
            f"font-family: Consolas, monospace; color: {pal.accent};")
        for row in self._rows():
            row.restyle()

    # ------------------------------------------------- input-health evidence
    def note_input_health(self, ih: dict) -> None:
        """Per-run input-health from the watcher: turn watchdog tweaks into
        before/after jitter evidence instead of folklore."""
        import time

        jit = ih.get("jitter_ms")
        if not jit:
            return
        self._jitter_runs.append((time.time(), float(jit)))
        del self._jitter_runs[:-50]
        tunes = self.watchdog.tune_times
        if tunes:
            t = tunes[-1]
            pre = [j for ts, j in self._jitter_runs if ts < t]
            post = [j for ts, j in self._jitter_runs if ts >= t]
            if pre and post:
                self.jitter_lbl.setText(
                    f"Evidence: input jitter {sum(post) / len(post):.2f} ms across "
                    f"{len(post)} run(s) since the last auto-tune, vs "
                    f"{sum(pre) / len(pre):.2f} ms across {len(pre)} before it.")
                return
            if post:
                self.jitter_lbl.setText(
                    f"Evidence: input jitter {sum(post) / len(post):.2f} ms across "
                    f"{len(post)} run(s) since the auto-tune (no pre-tune runs "
                    "this session to compare yet).")
                return
        js = [j for _, j in self._jitter_runs]
        self.jitter_lbl.setText(
            f"Evidence: input jitter {sum(js) / len(js):.2f} ms across "
            f"{len(js)} run(s) — no watchdog tune this session yet.")

    # ------------------------------------------------------------- scanning
    def rescan(self) -> None:
        if self._scan is not None and self._scan.isRunning():
            return
        if self._fix is not None and self._fix.isRunning():
            return   # rebuilding the rows now would orphan the pending fixes
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
        for r in results:
            self.rows_layout.insertWidget(self.rows_layout.count() - 1,
                                          _CheckRow(r, self._fix_row))
        self._refresh_fix_all()

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
    def _rows(self) -> list[_CheckRow]:
        out = []
        for i in range(self.rows_layout.count() - 1):
            w = self.rows_layout.itemAt(i).widget()
            if isinstance(w, _CheckRow):
                out.append(w)
        return out

    def _pending_safe(self) -> list[_CheckRow]:
        """Safe rows whose fix has not been applied (or queued) yet."""
        return [w for w in self._rows()
                if w.result.can_fix and w.result.safe
                and hasattr(w, "fix_btn") and w.fix_btn.isEnabled()]

    def _refresh_fix_all(self) -> None:
        n = len(self._pending_safe())
        self.fix_safe_btn.setEnabled(n > 0)
        self.fix_safe_btn.setText(
            f"Fix all safe items ({n})" if n else "Fix all safe items")

    def _fix_row(self, row: _CheckRow) -> None:
        self._start_fixes([row])

    def fix_all_safe(self) -> None:
        self._start_fixes(self._pending_safe())

    def _start_fixes(self, rows: list[_CheckRow]) -> None:
        """Hand a batch to the fix worker. Never runs on the UI thread: the
        power-plan fix can take the better part of a minute."""
        if self.checkup is None or not rows:
            return
        if self._fix is not None and self._fix.isRunning():
            return
        for row in rows:
            row.mark_pending()
        # Only one batch runs at a time, so every other Fix button goes quiet
        # for the duration — clicking one would otherwise be a silent no-op.
        self._suspended = [r for r in self._rows() if r not in rows
                           and hasattr(r, "fix_btn") and r.fix_btn.isEnabled()]
        for r in self._suspended:
            r.fix_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)      # a re-scan would replace the rows
        self.fix_safe_btn.setEnabled(False)
        self._fix = _FixWorker(self.checkup, [r.result.check_id for r in rows],
                               parent=self)
        self._fix.one_done.connect(self._on_fixed)
        self._fix.finished.connect(self._on_fixes_done)
        self._fix.start()

    def _on_fixed(self, check_id: str, msg: str) -> None:
        # The row can be gone (a scan rebuilt them) — then there is nothing to
        # report the outcome on, and dropping it is the right answer.
        for row in self._rows():
            if row.result.check_id == check_id:
                row.show_outcome(msg)
                return

    def _on_fixes_done(self) -> None:
        # Re-derive the live rows rather than trusting the list captured when
        # the batch started. _on_scan takes every row out of the layout and
        # deleteLater()s it, so a scan landing mid-batch leaves these as freed
        # C++ objects and touching one raises "Internal C++ object already
        # deleted" — the same reason _on_fixed above re-queries instead of
        # holding a reference.
        live = set(self._rows())
        for row in self._suspended:
            if row in live:
                row.fix_btn.setEnabled(True)
        self._suspended = []
        self.scan_btn.setEnabled(True)
        self._refresh_fix_all()

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
        self.wd_log.show()      # first line is what earns it the space

    def _copy_launch(self) -> None:
        QApplication.clipboard().setText(self.launch_label.text())
        self._log("launch options copied to clipboard")

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """App is exiting: wait for the worker threads before Qt teardown (a
        QThread destroyed while running is a fatal abort in Qt 6), then close
        for real so this window can't keep the app alive."""
        self._closing = True
        for th in (self._scan, self._fix):
            if th is None or not th.isRunning():
                continue
            if not th.wait(20000):
                # Still going (a wedged powercfg is the realistic case). Take
                # it out of the child tree and park it so Qt cannot destroy it
                # with this window; see _PARKED.
                th.setParent(None)
                _PARKED.append(th)
        self.close()

    def closeEvent(self, event) -> None:
        # User-closing the window hides it; the watchdog keeps running (that
        # is its point). Real teardown happens via shutdown() at app exit.
        if getattr(self, "_closing", False):
            super().closeEvent(event)
            return
        event.ignore()
        self.hide()
