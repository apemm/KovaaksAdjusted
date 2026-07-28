"""First-run startup guide + dismissible contextual hints.

WelcomeDialog: a short paged guide shown on first launch (and on demand from
the Help menu). HintBar: a one-line contextual tip used across tabs; the ×
on any bar tucks ALL hints away (settings.show_hints=False, persisted), and
the Help menu brings them back — instructions are there for new users and
gone in one click for everyone else.
"""

from __future__ import annotations

import weakref

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings

# Every live HintBar, so one dismiss / re-enable reaches all tabs.
_hint_bars: "weakref.WeakSet[HintBar]" = weakref.WeakSet()


class HintBar(QFrame):
    """One-line dismissible tip. Create it under any toolbar/header row."""

    def __init__(self, settings: Settings, text: str, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.setProperty("hint", True)
        tag = QLabel("TIP")
        tag.setStyleSheet("font-weight: 700; font-size: 11px;")
        tag.setProperty("dim", True)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setProperty("dim", True)
        close = QPushButton("×")
        close.setProperty("flat", True)
        close.setFixedWidth(24)
        close.setToolTip("Hide hints everywhere (Help menu brings them back)")
        close.clicked.connect(self._dismiss_all)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.addWidget(tag)
        lay.addWidget(body, 1)
        lay.addWidget(close, 0, Qt.AlignTop)

        _hint_bars.add(self)
        self.setVisible(settings.show_hints)

    def _dismiss_all(self) -> None:
        set_hints_visible(self.s, False)


def set_hints_visible(settings: Settings, visible: bool) -> None:
    """Show/hide every hint bar in the app and persist the choice."""
    settings.show_hints = visible
    try:
        settings.save()
    except OSError:
        pass
    for bar in list(_hint_bars):
        bar.setVisible(visible)


# --------------------------------------------------------------------- guide
_PAGES = [
    (
        "Welcome to kovadapt",
        "kovadapt makes KovaaK's adapt to <i>you</i>. After every run it:"
        "<ol>"
        "<li>reads the run's stats (and your raw mouse movement, if enabled),</li>"
        "<li>updates a per-scenario model of your strengths and weaknesses,</li>"
        "<li>rewrites a <b>[Adaptive]</b> copy of the scenario — resized targets, "
        "spawns shifted toward your weak regions, movement tuned to your pace.</li>"
        "</ol>"
        "The base scenario is never touched, and the game itself is never modified — "
        "only its own scenario files.",
    ),
    (
        "Your first session",
        "<ol>"
        "<li>On the <b>Dashboard</b>, pick a scenario and press <b>Play adaptive "
        "task</b> — kovadapt starts watching, queues the adaptive playlist, and "
        "launches KovaaK's through Steam.</li>"
        "<li>In the game, open <b>Playlists → kovadapt adaptive</b> and play.</li>"
        "<li>Between runs the scenario silently gets harder, easier, or shifts "
        "toward what you miss. Runs of the base scenario count too.</li>"
        "</ol>"
        "The calibration bar fills as the model learns — adaptation works from run 1 "
        "and sharpens over ~10 runs.",
    ),
    (
        "Overlay & optimizer",
        "<b>Overlay</b> — a small always-on-top card with your live session: last "
        "run vs baseline, fatigue, difficulty, input health. Toggle it on the "
        "Dashboard, drag it anywhere with <b>Unlock</b>, tune its opacity. The game "
        "must be Borderless or Windowed for overlays to show."
        "<br><br>"
        "<b>Optimizer</b> — hardware-matched performance checkup with one-click "
        "fixes, plus a watchdog that gives the game High priority and frees the "
        "input-processing core on every launch (the free Process Lasso).",
    ),
    (
        "Your data",
        "Everything lives locally in <code>~/.kovadapt</code> — profiles, mouse "
        "traces, run reports, clips. Nothing is uploaded anywhere."
        "<br><br>"
        "Every tab has short <b>TIP</b> bars while you learn the app; the × on any "
        "of them tucks them all away, and <b>Help → Show hints</b> brings them "
        "back. This guide stays available under <b>Help → Startup guide</b>.",
    ),
]


class WelcomeDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.setWindowTitle("kovadapt — startup guide")
        self.setModal(True)
        self.resize(520, 400)

        self.pages = QStackedWidget()
        for title, body in _PAGES:
            page = QWidget()
            head = QLabel(title)
            head.setProperty("headline", True)
            text = QLabel(body)
            text.setTextFormat(Qt.RichText)
            text.setWordWrap(True)
            text.setAlignment(Qt.AlignTop)
            v = QVBoxLayout(page)
            v.addWidget(head)
            v.addSpacing(6)
            v.addWidget(text, 1)
            self.pages.addWidget(page)

        self.progress = QLabel("")
        self.progress.setProperty("dim", True)
        self.again = QCheckBox("Show this guide on the next start")
        self.again.setChecked(False)     # finishing the guide dismisses it
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(lambda: self._go(-1))
        self.next_btn = QPushButton("Next")
        self.next_btn.setProperty("accent", True)
        self.next_btn.clicked.connect(self._next)

        bar = QHBoxLayout()
        bar.addWidget(self.progress)
        bar.addStretch(1)
        bar.addWidget(self.back_btn)
        bar.addWidget(self.next_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.pages, 1)
        lay.addWidget(self.again)
        lay.addLayout(bar)
        self._sync()

    def _go(self, step: int) -> None:
        self.pages.setCurrentIndex(
            max(0, min(self.pages.count() - 1, self.pages.currentIndex() + step)))
        self._sync()

    def _next(self) -> None:
        if self.pages.currentIndex() == self.pages.count() - 1:
            self.accept()
        else:
            self._go(+1)

    def _sync(self) -> None:
        i, n = self.pages.currentIndex(), self.pages.count()
        self.progress.setText(f"{i + 1} / {n}")
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setText("Get started" if i == n - 1 else "Next")

    def done(self, result: int) -> None:
        # Only finishing the guide dismisses it; closing it mid-read keeps
        # onboarding_done as-is so an unread guide returns next launch.
        if result == QDialog.Accepted:
            self.s.onboarding_done = not self.again.isChecked()
            try:
                self.s.save()
            except OSError:
                pass
        super().done(result)
