"""MouseRecorder start/stop lifecycle: a failed pump must never masquerade
as a running one.

Cross-platform by construction: `RAW_INPUT_AVAILABLE` is forced on and
`_run` is replaced by a fake, so no message-only window, no window class and
no Win32 call is ever made — only the Python-side lifecycle is exercised.
"""

from __future__ import annotations

import threading

import pytest

from kovadapt.telemetry import raw_input
from kovadapt.telemetry.raw_input import MouseRecorder


class _NeverReady(threading.Event):
    """Real set/clear semantics, but wait() reports timeout immediately —
    the production 5 s init timeout would otherwise stall the suite."""

    def wait(self, timeout: float | None = None) -> bool:  # noqa: ARG002
        return False


@pytest.fixture()
def win32(monkeypatch):
    """Pretend we are on Windows without touching any Win32 API."""
    monkeypatch.setattr(raw_input, "RAW_INPUT_AVAILABLE", True)


def _fake_pump(runs: list, ready: bool):
    def _run(self) -> None:
        runs.append(self)
        if ready:
            self._ready.set()
    return _run


def test_failed_init_does_not_poison_later_starts(win32, monkeypatch):
    """A pump that never signals ready must leave the recorder startable."""
    runs: list = []
    monkeypatch.setattr(MouseRecorder, "_run", _fake_pump(runs, ready=False))
    rec = MouseRecorder()
    rec._ready = _NeverReady()

    with pytest.raises(RuntimeError):
        rec.start()
    assert rec._thread is None
    assert not rec.running

    # The retry must really retry: silently returning here is the bug —
    # the caller would believe telemetry is live with no pump behind it.
    with pytest.raises(RuntimeError):
        rec.start()
    assert len(runs) == 2


def test_start_rearms_ready_before_spawning_pump(win32, monkeypatch):
    """A previous session's ready flag must not satisfy the next start()."""
    seen: list[bool] = []

    def _run(self) -> None:
        seen.append(self._ready.is_set())
        self._ready.set()          # let start() return without the 5 s wait

    monkeypatch.setattr(MouseRecorder, "_run", _run)
    rec = MouseRecorder()
    rec._ready.set()               # leftover flag from an earlier session
    rec.start()
    rec._thread.join(timeout=2.0)

    assert seen == [False], "start() must clear _ready before spawning the pump"


def test_stop_clears_thread_when_pump_already_exited(win32, monkeypatch):
    """The pump nulls _hwnd in its finally block; stop() must still release
    the thread handle, or the next start() is a silent no-op."""
    runs: list = []
    monkeypatch.setattr(MouseRecorder, "_run", _fake_pump(runs, ready=True))
    rec = MouseRecorder()

    rec.start()
    rec._thread.join(timeout=2.0)  # fake pump exits at once, leaving _hwnd None
    trace = rec.stop()
    assert trace.t.size == 0
    assert rec._thread is None

    rec.start()                    # must actually spawn a pump again
    rec._thread.join(timeout=2.0)
    assert len(runs) == 2


def test_start_off_windows_still_raises(monkeypatch):
    monkeypatch.setattr(raw_input, "RAW_INPUT_AVAILABLE", False)
    rec = MouseRecorder()
    with pytest.raises(RuntimeError, match="requires Windows"):
        rec.start()
    assert rec._thread is None
