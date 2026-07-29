"""Optimizer tab: compact launcher for the full optimizer window.

The real experience lives in optimizer_window.OptimizerWindow (a separate
top-level window — the free Process Lasso alternative). This tab shows a
one-line status and opens/raises that window; the window owns the watchdog
and keeps running when hidden.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..config import Settings


class OptimizerView(QWidget):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.window = None   # created lazily; import stays off the startup path

        head = QLabel("Performance optimizer")
        head.setProperty("headline", True)
        blurb = QLabel(
            "Hardware detection, a one-click system checkup with per-item fixes, "
            "a watchdog that applies High priority and frees the input-processing "
            "core on every game launch (what Process Lasso charges for; CPU 0+1 "
            "with hyperthreading, CPU 0 without), and launch options + settings "
            "matched to your hardware.\n\n"
            "Everything is opt-in and minimally invasive: fixes are per-user and "
            "reversible, nothing runs until you click it, and the only process "
            "kovadapt ever touches is the game's."
        )
        blurb.setWordWrap(True)
        blurb.setProperty("dim", True)
        open_btn = QPushButton("Open optimizer window")
        open_btn.setProperty("accent", True)
        open_btn.clicked.connect(self.open_window)
        btn_row = QHBoxLayout()          # natural-width CTA, not a 950px slab
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setSpacing(16)
        lay.addWidget(head)
        lay.addWidget(blurb)
        lay.addLayout(btn_row)
        lay.addStretch(1)

    def open_window(self) -> None:
        if self.window is None:
            from .optimizer_window import OptimizerWindow

            self.window = OptimizerWindow(self.s)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def restyle(self, *_pal) -> None:
        if self.window is not None:
            self.window.restyle()

    def note_report(self, rep) -> None:
        """Feed per-run input health into the watchdog evidence panel."""
        if self.window is not None:
            self.window.note_input_health(getattr(rep, "input_health", None) or {})

    def shutdown(self) -> None:
        if self.window is not None:
            self.window.shutdown()
