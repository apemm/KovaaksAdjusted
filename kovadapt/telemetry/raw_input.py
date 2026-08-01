"""Windows Raw Input mouse capture (ctypes, no dependencies).

Registers for WM_INPUT on a message-only window running in a background
thread, so we receive *relative* mouse deltas exactly as the game does —
unaffected by pointer ballistics, cursor clipping, or the game confining the
cursor. This is the only correct way to observe flicks while a shooter has
raw input focus.

Overhead: one tiny WndProc callback per packet; buffers are preallocated
numpy chunks. At 8 kHz polling this is well under 1% of a modern core.

Cross-platform note: importable everywhere (analysis/tests run on any OS);
`MouseRecorder.start()` raises unless Windows Raw Input is available.
"""

from __future__ import annotations

import bisect
import sys
import threading
import time

import numpy as np

from .trace import MouseTrace

RAW_INPUT_AVAILABLE = sys.platform == "win32"

_CHUNK = 1 << 16  # packets per buffer chunk (65536)

if RAW_INPUT_AVAILABLE:  # pragma: no cover - exercised only on Windows
    import ctypes
    from ctypes import wintypes

    # Private DLL handles (not the shared ctypes.windll cache) so the exact
    # prototypes declared below can't perturb other modules' ctypes use.
    user32 = ctypes.WinDLL("user32")
    kernel32 = ctypes.WinDLL("kernel32")

    WM_INPUT = 0x00FF
    WM_CLOSE = 0x0010
    RIDEV_INPUTSINK = 0x00000100
    RID_INPUT = 0x10000003
    RIM_TYPEMOUSE = 0
    RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
    RI_MOUSE_LEFT_BUTTON_UP = 0x0002
    HWND_MESSAGE = -3

    class RAWINPUTDEVICE(ctypes.Structure):
        _fields_ = [
            ("usUsagePage", wintypes.USHORT),
            ("usUsage", wintypes.USHORT),
            ("dwFlags", wintypes.DWORD),
            ("hwndTarget", wintypes.HWND),
        ]

    class RAWINPUTHEADER(ctypes.Structure):
        _fields_ = [
            ("dwType", wintypes.DWORD),
            ("dwSize", wintypes.DWORD),
            ("hDevice", wintypes.HANDLE),
            ("wParam", wintypes.WPARAM),
        ]

    class RAWMOUSE(ctypes.Structure):
        _fields_ = [
            ("usFlags", wintypes.USHORT),
            ("ulButtons", wintypes.ULONG),
            ("ulRawButtons", wintypes.ULONG),
            ("lLastX", wintypes.LONG),
            ("lLastY", wintypes.LONG),
            ("ulExtraInformation", wintypes.ULONG),
        ]

    class RAWINPUT(ctypes.Structure):
        _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]

    LRESULT = wintypes.LPARAM  # pointer-sized signed int, same layout as LRESULT

    WNDPROC = ctypes.WINFUNCTYPE(
        LRESULT, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM
    )

    class WNDCLASS(ctypes.Structure):
        _fields_ = [
            ("style", ctypes.c_uint),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HANDLE),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HANDLE),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    # Exact prototypes for every Win32 call this module makes. Without them,
    # ctypes falls back to c_int for arguments and results, truncating
    # pointer-sized values (module handle, HWND_MESSAGE, the WndProc's
    # lparam forwarded to DefWindowProcW). Whether the truncated values
    # still work depends on where ASLR loaded things that boot — the failure
    # mode is an ASLR-lottery OverflowError or a silently dead capture
    # window, so pin every signature.
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
    user32.UnregisterClassW.restype = wintypes.BOOL
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [
        wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM,
    ]
    user32.PostQuitMessage.restype = None
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM,
    ]
    user32.RegisterRawInputDevices.restype = wintypes.BOOL
    user32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICE), ctypes.c_uint, ctypes.c_uint,
    ]
    user32.GetRawInputData.restype = ctypes.c_uint
    user32.GetRawInputData.argtypes = [
        wintypes.LPARAM, ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
    ]
    user32.GetMessageW.restype = ctypes.c_int
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, ctypes.c_uint, ctypes.c_uint,
    ]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]


class _Buffers:
    """Chunked append-only packet storage (lock-free single-writer).

    `retention_s` (None = keep everything) bounds memory over long sessions:
    each time a chunk fills, whole chunks whose newest packet is older than
    `now - retention_s` are dropped, so at least `retention_s` of history is
    always retained (chunk-granular, like the windowed `to_trace` cut).
    Unbounded, a session accrues 16 B/packet — ~460 MB/hour at 8 kHz polling.
    """

    def __init__(self, retention_s: float | None = None) -> None:
        self.retention_s = retention_s
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._new_chunk()
        self.clicks: list[float] = []
        self.clicks_up: list[float] = []

    def _new_chunk(self) -> None:
        self.t = np.empty(_CHUNK, dtype=np.float64)
        self.dx = np.empty(_CHUNK, dtype=np.int32)
        self.dy = np.empty(_CHUNK, dtype=np.int32)
        self.n = 0

    def add(self, t: float, dx: int, dy: int) -> None:
        if self.n == _CHUNK:
            self.chunks.append((self.t, self.dx, self.dy))
            self._new_chunk()
            if self.retention_s is not None:
                self._prune(t - self.retention_s)
        self.t[self.n] = t
        self.dx[self.n] = dx
        self.dy[self.n] = dy
        self.n += 1

    def _prune(self, cutoff: float) -> None:
        """Drop data entirely older than `cutoff`. Called only on chunk
        rollover (~every 65k packets), so the amortized per-packet cost in
        the WndProc hot path is negligible."""
        while self.chunks and self.chunks[0][0][-1] < cutoff:
            del self.chunks[0]
        if self.clicks and self.clicks[0] < cutoff:
            del self.clicks[: bisect.bisect_left(self.clicks, cutoff)]
        if self.clicks_up and self.clicks_up[0] < cutoff:
            del self.clicks_up[: bisect.bisect_left(self.clicks_up, cutoff)]

    def to_trace(self, t0: float | None = None, t1: float | None = None) -> MouseTrace:
        parts = self.chunks + [(self.t[: self.n], self.dx[: self.n], self.dy[: self.n])]
        if t0 is not None or t1 is not None:
            # Chunks are time-ordered, so keep only the ones overlapping
            # [t0, t1]: a run's snapshot then costs O(run length) instead of
            # O(session length). Coarse cut — callers still window() exactly.
            lo = -np.inf if t0 is None else t0
            hi = np.inf if t1 is None else t1
            parts = [p for p in parts if p[0].size and p[0][-1] >= lo and p[0][0] <= hi]
        if not parts:
            parts = [(np.empty(0), np.empty(0, np.int32), np.empty(0, np.int32))]
        return MouseTrace(
            t=np.concatenate([p[0] for p in parts]),
            dx=np.concatenate([p[1] for p in parts]),
            dy=np.concatenate([p[2] for p in parts]),
            clicks=np.asarray(self.clicks, dtype=np.float64),
            clicks_up=np.asarray(self.clicks_up, dtype=np.float64),
        )


class MouseRecorder:
    """Background Raw Input capture. Usage:

        rec = MouseRecorder()
        rec.start()
        ... play ...
        trace = rec.stop()

    While running, `rec.snapshot()` returns a copy of everything so far
    (used to slice out a run without stopping capture).

    `retention_s` bounds the live buffer to a rolling window (None = keep the
    whole session, the pre-v0.4 behavior). Safe for the watcher, which slices
    each run out of the recording within seconds of the run ending; direct
    users that rely on `stop()` returning the full session should leave it
    unset.
    """

    def __init__(self, retention_s: float | None = None) -> None:
        self._buf = _Buffers(retention_s)
        self._thread: threading.Thread | None = None
        self._hwnd = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._orphaned = False      # a pump we failed to shut down

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not RAW_INPUT_AVAILABLE:
            raise RuntimeError("Raw Input capture requires Windows")
        if self._thread is not None:
            if self._orphaned:
                # A previous start() timed out on a pump that would not die.
                # Returning "already running" here would be exactly the silent
                # no-op the re-arm below exists to prevent.
                raise RuntimeError(
                    "a previous Raw Input pump did not shut down; "
                    "restart the app before recording again")
            return
        # Re-arm before spawning: `_ready` outlives stop(), so a second start()
        # on the same recorder (GUI Stop -> Start) would otherwise see the
        # *first* session's flag and return happily even when the new pump died
        # immediately on the stale-window-class path in _run() — capture dead,
        # nothing raised.
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="kovadapt-rawinput", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            # Tear it down BEFORE dropping the reference. Simply clearing
            # _thread orphaned a pump that was merely slow: once it is
            # unreachable, stop() can no longer post WM_CLOSE or join it, so
            # the message-only window and its registered class outlive the
            # recorder — and a stale class dispatching into a freed WNDPROC
            # thunk is a native crash that has shipped before.
            if not self._teardown():
                self._orphaned = True
                raise RuntimeError(
                    "Raw Input window failed to initialize and its pump "
                    "would not stop")
            raise RuntimeError("Raw Input window failed to initialize")

    def _teardown(self, timeout: float = 3.0) -> bool:
        """Ask the pump to close and wait for it. True once it is really gone.

        Shared by stop() and by start()'s timeout path, because both have the
        same obligation: the window and its class must be destroyed on the
        pump thread (see _run's finally), so the only safe way to end a pump
        is to let it end itself. _hwnd may still be unset if the pump has not
        got that far, in which case there is nothing to post to and the join
        is all we can do.
        """
        thread = self._thread
        if thread is None:
            return True
        if self._hwnd:
            # A failed post means the pump will never be asked to quit, so the
            # join below is guaranteed to time out. Report it rather than
            # spending the full timeout pretending otherwise.
            if not user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0):
                return False
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False            # caller decides how loud to be
        # Cleared even when the pump already exited on its own (its finally
        # block nulls _hwnd): a lingering handle would make the next start()
        # a silent no-op.
        self._thread = None
        self._orphaned = False
        return True

    def stop(self) -> MouseTrace:
        """End the recording and return what was captured.

        The teardown result is HONOURED, not discarded. It used to be
        dropped, so a pump that outlived the join was reported as a clean
        stop while `_thread` still pointed at a running pump with its
        message-only window alive and the process-wide class still
        registered — and `watcher._stop_capture` then dropped the last
        reference to it, putting it beyond any future PostMessageW.

        It records the failure rather than raising: stop() runs from
        `watch()`'s finally, where raising would mask whatever ended the
        session. `_orphaned` is the existing channel for exactly this — the
        next start() on this recorder refuses loudly instead of returning a
        silent "already running".
        """
        if not self._teardown():
            self._orphaned = True
        with self._lock:
            return self._buf.to_trace()

    def snapshot(self, t0: float | None = None, t1: float | None = None) -> MouseTrace:
        """Copy of the recording so far; pass a window to copy only the
        chunks overlapping it (chunk-granular — window() the result for an
        exact cut)."""
        with self._lock:
            return self._buf.to_trace(t0, t1)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - Windows-only thread
        buf = self._buf
        lock = self._lock
        raw = RAWINPUT()
        size = wintypes.UINT(ctypes.sizeof(RAWINPUT))
        header_size = ctypes.sizeof(RAWINPUTHEADER)

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                got = wintypes.UINT(size.value)
                if (
                    user32.GetRawInputData(
                        lparam, RID_INPUT, ctypes.byref(raw), ctypes.byref(got), header_size
                    )
                    != wintypes.UINT(-1).value
                    and raw.header.dwType == RIM_TYPEMOUSE
                ):
                    now = time.time()
                    m = raw.mouse
                    with lock:
                        if m.lLastX or m.lLastY:
                            buf.add(now, m.lLastX, m.lLastY)
                        if m.ulButtons & RI_MOUSE_LEFT_BUTTON_DOWN:
                            buf.clicks.append(now)
                        if m.ulButtons & RI_MOUSE_LEFT_BUTTON_UP:
                            buf.clicks_up.append(now)
                return 0
            if msg == WM_CLOSE:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        # The WNDPROC thunk (`proc`) lives on this stack frame and is freed
        # when _run() returns, but window-class registration is process-wide
        # and would otherwise outlive it. The finally block below tears the
        # window and the class down before this frame dies, so the registered
        # class can never point at a freed thunk — without that, a second
        # start() in the same process (GUI Stop -> Start) binds a new window
        # to the stale class and dispatches WM_CREATE into freed memory: a
        # native crash with no Python traceback.
        proc = WNDPROC(wndproc)
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASS()
        wc.lpfnWndProc = proc
        wc.lpszClassName = "KovadaptRawInput"
        wc.hInstance = hinst
        if not user32.RegisterClassW(ctypes.byref(wc)):
            # Stale registration left by a pump that died without cleanup —
            # its thunk pointer is already freed. Drop it and register our
            # own; never create a window against the old pointer.
            user32.UnregisterClassW(wc.lpszClassName, hinst)
            if not user32.RegisterClassW(ctypes.byref(wc)):
                return  # start() times out and raises
        try:
            self._hwnd = user32.CreateWindowExW(
                0, wc.lpszClassName, None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None
            )

            if not self._hwnd:
                # NULL window: RIDEV_INPUTSINK needs a real hwndTarget, so
                # registration fails and NOT ONE PACKET is ever recorded.
                # Setting _ready here anyway made start() succeed, and then
                # _teardown could not post WM_CLOSE (`if self._hwnd` is
                # falsy) so the join timed out against a thread parked in
                # GetMessageW forever — a leaked daemon thread pinning the
                # recorder and its whole retention buffer (hundreds of MB at
                # 30 min and 8 kHz) for the life of the process, silently.
                # Leaving _ready unset makes start() raise, which is the
                # honest outcome: capture did not start.
                return
            rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, self._hwnd)  # generic mouse
            user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE))
            self._ready.set()

            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            # Same-thread teardown: DestroyWindow must run on the creating
            # thread, and UnregisterClassW requires no live windows.
            if self._hwnd:
                user32.DestroyWindow(self._hwnd)
                self._hwnd = None
            user32.UnregisterClassW(wc.lpszClassName, hinst)
