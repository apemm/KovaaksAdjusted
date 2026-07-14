"""Qt worker threads bridging the blocking watcher loop into the GUI."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..config import Settings
from ..watcher import SessionWatcher


class WatcherWorker(QThread):
    """Runs SessionWatcher.watch() off the UI thread.

    Signals:
        message(str)         log lines
        report_ready(object) RunReport after each processed run
        stopped()            loop exited
    """

    message = Signal(str)
    report_ready = Signal(object)
    stopped = Signal()

    def __init__(self, settings: Settings, base_scenario: str, parent=None) -> None:
        super().__init__(parent)
        self.watcher = SessionWatcher(
            settings,
            base_scenario,
            on_update=self.message.emit,
            on_report=self.report_ready.emit,
        )

    def run(self) -> None:  # QThread entry point
        try:
            self.watcher.watch()
        except Exception as exc:
            self.message.emit(f"watcher error: {exc}")
        finally:
            self.stopped.emit()

    def stop(self) -> None:
        """Non-blocking stop request; thread exits within ~1s."""
        self.watcher.request_stop()
