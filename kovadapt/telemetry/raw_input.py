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

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    WM_INPUT = 0x00FF
    WM_CLOSE = 0x0010
    RIDEV_INPUTSINK = 0x00000100
    RID_INPUT = 0x10000003
    RIM_TYPEMOUSE = 0
    RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
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

    WNDPROC = ctypes.WINFUNCTYPE(
        ctypes.c_long, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM
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


class _Buffers:
    """Chunked append-only packet storage (lock-free single-writer)."""

    def __init__(self) -> None:
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._new_chunk()
        self.clicks: list[float] = []

    def _new_chunk(self) -> None:
        self.t = np.empty(_CHUNK, dtype=np.float64)
        self.dx = np.empty(_CHUNK, dtype=np.int32)
        self.dy = np.empty(_CHUNK, dtype=np.int32)
        self.n = 0

    def add(self, t: float, dx: int, dy: int) -> None:
        if self.n == _CHUNK:
            self.chunks.append((self.t, self.dx, self.dy))
            self._new_chunk()
        self.t[self.n] = t
        self.dx[self.n] = dx
        self.dy[self.n] = dy
        self.n += 1

    def to_trace(self) -> MouseTrace:
        parts = self.chunks + [(self.t[: self.n], self.dx[: self.n], self.dy[: self.n])]
        return MouseTrace(
            t=np.concatenate([p[0] for p in parts]),
            dx=np.concatenate([p[1] for p in parts]),
            dy=np.concatenate([p[2] for p in parts]),
            clicks=np.asarray(self.clicks, dtype=np.float64),
        )


class MouseRecorder:
    """Background Raw Input capture. Usage:

        rec = MouseRecorder()
        rec.start()
        ... play ...
        trace = rec.stop()

    While running, `rec.snapshot()` returns a copy of everything so far
    (used to slice out a run without stopping capture).
    """

    def __init__(self) -> None:
        self._buf = _Buffers()
        self._thread: threading.Thread | None = None
        self._hwnd = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not RAW_INPUT_AVAILABLE:
            raise RuntimeError("Raw Input capture requires Windows")
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="kovadapt-rawinput", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Raw Input window failed to initialize")

    def stop(self) -> MouseTrace:
        if self._thread is not None and self._hwnd:
            user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            return self._buf.to_trace()

    def snapshot(self) -> MouseTrace:
        with self._lock:
            return self._buf.to_trace()

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
                return 0
            if msg == WM_CLOSE:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        proc = WNDPROC(wndproc)
        wc = WNDCLASS()
        wc.lpfnWndProc = proc
        wc.lpszClassName = "KovadaptRawInput"
        wc.hInstance = kernel32.GetModuleHandleW(None)
        user32.RegisterClassW(ctypes.byref(wc))
        self._hwnd = user32.CreateWindowExW(
            0, wc.lpszClassName, None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, wc.hInstance, None
        )

        rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, self._hwnd)  # generic mouse
        user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE))
        self._ready.set()

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
