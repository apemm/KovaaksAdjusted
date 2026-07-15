"""Game watchdog: Process Lasso's persistent rules, free.

Windows resets priority and affinity on every process start, so Process
Lasso's paid value is simply re-applying them per launch. GameWatchdog polls
for FPSAimTrainer on a daemon thread (2 s cadence — a game launch takes far
longer than that) and applies High priority + CPU 0/1 masking exactly once
per PID; a new launch means a new PID, which re-triggers.

Also here: per-user startup registration (HKCU Run key) so the watchdog can
run without manually opening kovadapt — no admin, no services, one registry
value that Windows itself documents for this purpose.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable

WINDOWS = sys.platform == "win32"

GAME_PROCESS = "FPSAimTrainer"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "kovadapt-watchdog"


def cpus_to_free() -> list[int]:
    """Logical CPUs to keep the game OFF (where Windows processes mouse input).

    The KovaaK's FAQ advice targets the first *physical* core. With SMT
    (hyperthreading) logical 0 and 1 are that core's two threads, so both are
    freed; without SMT logical 1 is a separate physical core — masking it
    would just donate a core for nothing.
    """
    try:
        import psutil
    except ImportError:
        return [0, 1]
    logical = psutil.cpu_count(logical=True) or 0
    physical = psutil.cpu_count(logical=False) or logical
    return [0, 1] if logical > physical else [0]


def apply_game_tuning() -> tuple[str, bool]:
    """Set High priority + free the input core(s) on the running game."""
    try:
        import psutil
    except ImportError:
        return "psutil not installed — pip install kovadapt[gui]", False
    proc = None
    for p in psutil.process_iter(["name"]):
        if GAME_PROCESS.lower() in (p.info["name"] or "").lower():
            proc = p
            break
    if proc is None:
        return "game not running", False
    try:
        proc.nice(psutil.HIGH_PRIORITY_CLASS)
        n = psutil.cpu_count(logical=True) or 2
        freed = cpus_to_free()
        remaining = [c for c in range(n) if c not in freed]
        if len(remaining) >= 2:
            proc.cpu_affinity(remaining)
            freed_txt = "/".join(str(c) for c in freed)
            smt = " (SMT twin included)" if len(freed) > 1 else ""
            return (f"pid {proc.pid}: High priority, CPU {freed_txt} freed for "
                    f"input processing{smt}", True)
        return f"pid {proc.pid}: High priority ({n} CPUs — affinity unchanged)", True
    except psutil.AccessDenied:
        return "access denied — run kovadapt as administrator", False
    except psutil.NoSuchProcess:
        return "game exited while applying", False


class GameWatchdog:
    """Poll for the game and auto-apply tuning once per launch (PID)."""

    def __init__(self, on_event: Callable[[str], None] | None = None,
                 poll_interval: float = 2.0) -> None:
        self.on_event = on_event or (lambda msg: None)
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tuned_pid: int | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kovadapt-watchdog",
                                        daemon=True)
        self._thread.start()
        self.on_event("watchdog on — will tune the game on every launch")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 1.0)
            self._thread = None
        self.on_event("watchdog off")

    # ------------------------------------------------------------------
    def _find_pid(self) -> int | None:
        import psutil  # ImportError surfaces in _run — never swallow it here

        for p in psutil.process_iter(["name"]):
            if GAME_PROCESS.lower() in (p.info["name"] or "").lower():
                return p.pid
        return None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                pid = self._find_pid()
            except ImportError:
                # Without psutil every poll would silently see "no game";
                # say so once and stop instead.
                self.on_event("watchdog stopped — psutil missing "
                              "(pip install kovadapt[gui])")
                return
            if pid is None:
                self._tuned_pid = None       # game closed; re-arm
                continue
            if pid == self._tuned_pid:
                continue
            msg, ok = apply_game_tuning()
            if ok:
                self._tuned_pid = pid        # once per launch
                self.on_event(f"game detected — {msg}")
            else:
                self._tuned_pid = pid        # don't spam retries on AccessDenied
                self.on_event(f"game detected but not tuned: {msg}")


# ----------------------------------------------------------- startup entry
def _launch_command() -> str:
    """Command Windows runs at login: pythonw (no console) when available."""
    exe = sys.executable
    if getattr(sys, "frozen", False):        # PyInstaller build
        return f'"{exe}" --watchdog'
    pyw = exe.replace("python.exe", "pythonw.exe")
    from pathlib import Path

    if not Path(pyw).is_file():
        pyw = exe
    return f'"{pyw}" -m kovadapt watchdog'


def startup_registered() -> bool:
    if not WINDOWS:
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, RUN_VALUE)
        return True
    except FileNotFoundError:
        return False


def register_startup() -> str:
    """Add the per-user Run value (HKCU: no admin, easily removable)."""
    if not WINDOWS:
        return "not Windows"
    import winreg

    cmd = _launch_command()
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
        winreg.SetValueEx(k, RUN_VALUE, 0, winreg.REG_SZ, cmd)
    return f"registered: {cmd}"


def unregister_startup() -> str:
    if not WINDOWS:
        return "not Windows"
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                            winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, RUN_VALUE)
        return "startup entry removed"
    except FileNotFoundError:
        return "was not registered"
