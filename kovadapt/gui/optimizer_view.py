"""Optimizer tab (v0.2: the free Process Lasso replacement basics).

Live controls: find the FPSAimTrainer process, set High priority, and mask
CPU 0/1 off its affinity — the two things KovaaK's own FAQ recommends
(mouse input is processed on the first core; keeping the game off it
reduces input-pipeline contention). Applied per-launch: Windows resets
priority/affinity when the game restarts, so click Apply after launching.

The checklist below documents the rest of the researched optimizations;
automatic hardware detection + one-click checkup land in v0.3.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .theme import ACCENT, FG_DIM, GOOD

PROCESS_NAME = "FPSAimTrainer"

# (title, detail) — sourced from the KovaaK's official FAQ, PCGamingWiki,
# and community tuning guides. See FEATURES.md for links.
CHECKLIST: list[tuple[str, str]] = [
    ("High process priority + free CPU 0/1",
     "Use the buttons above (per game launch). This is exactly what Process "
     "Lasso automates behind its paywall."),
    ("NVIDIA Reflex / Low Latency Mode",
     "In KovaaK's video settings enable NVIDIA Reflex (or Ultra Low Latency in "
     "the driver for older GPUs). Biggest single input-lag win."),
    ("Fullscreen, not borderless",
     "Exclusive fullscreen + disable 'Fullscreen optimizations' on "
     "FPSAimTrainer.exe (Properties > Compatibility)."),
    ("Ultimate Performance power plan",
     "powercfg -duplicatescheme e9a42b02-d5df-448d-aa66-ad3f9edeb1c9, then "
     "select it in Power Options. Stops core parking/downclocking."),
    ("Hardware-Accelerated GPU Scheduling (HAGS)",
     "Enable only on RTX 3000+/RX 6000+ GPUs; disable on older cards."),
    ("Close Chromium apps while training",
     "Discord/Spotify/browser overlays steal timer resolution and cause "
     "frame-time spikes. Disable overlays at minimum."),
    ("Cap FPS just below monitor refresh x2",
     "Uncapped FPS causes frame-time variance; a stable cap beats a higher "
     "unstable average."),
    ("Reset corrupt GameUserSettings.ini if stuttering",
     r"Delete %localappdata%\FPSAimTrainer\Saved\Config\WindowsNoEditor\ "
     "GameUserSettings.ini (KovaaK's rebuilds it) — known stutter cause."),
    ("1ms timer resolution",
     "Windows 10: keep a timer-resolution tool running. Windows 11 manages "
     "this per-process automatically."),
]


class OptimizerView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # live controls -------------------------------------------------
        self.status = QLabel("Game not detected — launch KovaaK's, then Apply.")
        self.status.setProperty("dim", True)
        detect = QPushButton("Detect game")
        detect.clicked.connect(self.detect)
        self.apply_btn = QPushButton("Apply: High priority + free CPU 0/1")
        self.apply_btn.setProperty("accent", True)
        self.apply_btn.clicked.connect(self.apply)

        live = QGroupBox("Live tuning (free Process Lasso replacement)")
        row = QHBoxLayout(live)
        row.addWidget(detect)
        row.addWidget(self.apply_btn)
        row.addWidget(self.status, 1)

        # checklist ------------------------------------------------------
        inner = QWidget()
        col = QVBoxLayout(inner)
        for title, detail in CHECKLIST:
            t = QLabel(f"•  <b>{title}</b>")
            t.setTextFormat(Qt.RichText)
            d = QLabel(detail)
            d.setWordWrap(True)
            d.setProperty("dim", True)
            d.setContentsMargins(16, 0, 0, 8)
            col.addWidget(t)
            col.addWidget(d)
        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        checklist = QGroupBox("Manual checklist (auto-checkup coming in v0.3)")
        cl = QVBoxLayout(checklist)
        cl.addWidget(scroll)

        lay = QVBoxLayout(self)
        lay.addWidget(live)
        lay.addWidget(checklist, 1)

    # ------------------------------------------------------------------
    def _find_proc(self):
        try:
            import psutil
        except ImportError:
            self.status.setText("psutil not installed — pip install kovadapt[gui]")
            return None
        for p in psutil.process_iter(["name"]):
            if PROCESS_NAME.lower() in (p.info["name"] or "").lower():
                return p
        return None

    def detect(self) -> None:
        p = self._find_proc()
        if p is None:
            self.status.setText("Game not detected — launch KovaaK's first.")
            self.status.setStyleSheet(f"color: {FG_DIM};")
        else:
            self.status.setText(f"Found {p.name()} (pid {p.pid}).")
            self.status.setStyleSheet(f"color: {ACCENT};")

    def apply(self) -> None:
        p = self._find_proc()
        if p is None:
            self.detect()
            return
        import psutil

        try:
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            n = psutil.cpu_count(logical=True) or 2
            if n > 2:
                p.cpu_affinity(list(range(2, n)))
            self.status.setText(
                f"Applied to pid {p.pid}: High priority, CPUs 2-{n - 1}. "
                "Re-apply after each game restart."
            )
            self.status.setStyleSheet(f"color: {GOOD};")
        except psutil.AccessDenied:
            self.status.setText("Access denied — run kovadapt as administrator to apply.")
            self.status.setStyleSheet(f"color: {FG_DIM};")
        except Exception as exc:
            self.status.setText(f"Failed: {exc}")
