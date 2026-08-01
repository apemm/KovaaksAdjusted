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


def test_stop_reports_a_pump_it_could_not_kill(win32, monkeypatch):
    """stop() used to DISCARD _teardown()'s result.

    start() has always honoured it — a pump that outlives the join is marked
    orphaned so the next start() refuses loudly. stop() dropped it, so a
    pump that would not die was reported as a clean stop while _thread still
    pointed at it, its message-only window alive and the process-wide class
    still registered. watcher._stop_capture then set `self.recorder = None`,
    putting that pump beyond any future PostMessageW for the life of the
    process.
    """
    stuck = threading.Event()

    def _run(self) -> None:
        self._ready.set()
        stuck.wait(10.0)            # ignores WM_CLOSE, like a wedged pump

    monkeypatch.setattr(MouseRecorder, "_run", _run)
    rec = MouseRecorder()
    rec.start()
    try:
        trace = rec.stop()          # must not pretend this worked
        assert trace is not None
        assert rec._orphaned is True, "a pump that outlived the join went unrecorded"
        assert rec._thread is not None
        with pytest.raises(RuntimeError, match="did not shut down"):
            rec.start()
    finally:
        stuck.set()
        if rec._thread is not None:
            rec._thread.join(timeout=2.0)


def test_a_null_window_never_reports_a_live_capture(win32, monkeypatch):
    """_run set _ready straight after CreateWindowExW without checking it.

    On a NULL HWND, RIDEV_INPUTSINK has no target so not one packet is ever
    recorded — yet start() succeeded. Worse, _teardown then skipped
    PostMessageW (`if self._hwnd` is falsy) and the join timed out against a
    thread parked in GetMessageW forever: a leaked daemon thread pinning the
    recorder and its entire retention buffer, silently.
    """
    def _run(self) -> None:
        self._hwnd = None           # CreateWindowExW returned 0
        if not self._hwnd:
            return
        self._ready.set()           # unreachable, and that is the fix

    monkeypatch.setattr(MouseRecorder, "_run", _run)
    rec = MouseRecorder()
    rec._ready = _NeverReady()
    with pytest.raises(RuntimeError, match="failed to initialize"):
        rec.start()
    assert rec._thread is None


def test_the_real_run_guards_the_window_handle():
    """The pins above use a fake pump, so they cannot see the production
    code drift. Assert the guard is in _run itself."""
    import inspect

    src = inspect.getsource(MouseRecorder._run)
    body = src.split("CreateWindowExW", 1)[1]
    guard = body.index("if not self._hwnd")
    ready = body.index("self._ready.set()")
    assert guard < ready, "_ready is set before the HWND is checked"


def test_start_off_windows_still_raises(monkeypatch):
    monkeypatch.setattr(raw_input, "RAW_INPUT_AVAILABLE", False)
    rec = MouseRecorder()
    with pytest.raises(RuntimeError, match="requires Windows"):
        rec.start()
    assert rec._thread is None
