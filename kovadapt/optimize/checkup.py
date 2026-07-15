"""One-click system checkup: detect everything, fix per item on request.

Each check probes read-only and reports a CheckResult; fixes run only when
explicitly invoked (per-item button or "fix all safe"). `safe=True` marks
fixes that are per-user, reversible, and admin-free (registry HKCU writes,
process tweaks); invasive fixes (power plan, file deletion, HKLM) always
need their own click and say what they will do first.

Windows-only at runtime; importable everywhere.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .hardware import HardwareInfo

WINDOWS = sys.platform == "win32"

GAME_EXE = "FPSAimTrainer.exe"
GAME_PROCESS = "FPSAimTrainer"

# Ultimate Performance scheme (hidden on most consumer SKUs until duplicated).
ULTIMATE_GUID = "e9a42b02-d5df-448d-aa66-ad3f9edeb1c9"
HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

# Chromium-engine apps that degrade timer resolution / spike frame times.
CHROMIUM_PROCS = ("discord", "spotify", "chrome", "msedge", "brave", "opera",
                  "vivaldi", "slack", "obs64", "wallpaper64", "steamwebhelper")
# steamwebhelper is unavoidable while Steam runs; report it, don't count it.
UNAVOIDABLE = ("steamwebhelper",)


@dataclass
class CheckResult:
    check_id: str
    title: str
    status: str            # ok | warn | bad | info | unknown
    detail: str
    can_fix: bool = False
    safe: bool = False     # eligible for "Fix all safe items"
    fix_label: str = ""


@dataclass
class Check:
    check_id: str
    title: str
    probe: Callable[[], CheckResult]
    fix: Callable[[], str] | None = None   # returns an outcome message


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    out = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return (out.stdout or "") + (out.stderr or "")


def game_config_path() -> Path:
    import os

    base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    return (Path(base) / "FPSAimTrainer" / "Saved" / "Config" / "WindowsNoEditor"
            / "GameUserSettings.ini")


def find_game_exes(kovaaks_root: str) -> list[Path]:
    """Game executables, most important first. kovaaks_root is the inner
    FPSAimTrainer game-data dir; the UE4 shipping exe (the process that
    actually renders) lives under it, the Steam bootstrap exe one level up."""
    if not kovaaks_root:
        return []
    root = Path(kovaaks_root)
    cands = (
        root / "Binaries" / "Win64" / "FPSAimTrainer-Win64-Shipping.exe",
        root.parent / GAME_EXE,
        root / GAME_EXE,
    )
    return [p for p in cands if p.is_file()]


class SystemCheckup:
    """Builds the check list against current Settings + detected hardware."""

    def __init__(self, kovaaks_root: str, hw: HardwareInfo) -> None:
        self.kovaaks_root = kovaaks_root
        self.hw = hw
        self.checks: list[Check] = [
            Check("power_plan", "High-performance power plan", self._c_power, self._f_power),
            Check("hags", "Hardware-accelerated GPU scheduling", self._c_hags),
            Check("fso", "Fullscreen optimizations disabled on the game",
                  self._c_fso, self._f_fso),
            Check("mouse_accel", "Windows mouse acceleration off",
                  self._c_mouse, self._f_mouse),
            Check("chromium", "Background Chromium apps", self._c_chromium),
            Check("game_config", "KovaaK's config file health", self._c_config,
                  self._f_config),
            Check("game_process", "Game priority & CPU affinity",
                  self._c_process, self._f_process),
        ]

    # ---------------------------------------------------------------- api
    def run_all(self) -> list[CheckResult]:
        out = []
        for c in self.checks:
            try:
                out.append(c.probe())
            except Exception as exc:
                out.append(CheckResult(c.check_id, c.title, "unknown",
                                       f"probe failed: {exc}"))
        return out

    def fix(self, check_id: str) -> str:
        for c in self.checks:
            if c.check_id == check_id and c.fix is not None:
                try:
                    return c.fix()
                except Exception as exc:
                    return f"fix failed: {exc}"
        return "no automated fix for this item"

    # ------------------------------------------------------------- power
    def _c_power(self) -> CheckResult:
        cid, title = "power_plan", "High-performance power plan"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        out = _run(["powercfg", "/getactivescheme"]).lower()
        if ULTIMATE_GUID in out:
            return CheckResult(cid, title, "ok", "Ultimate Performance is active.")
        if HIGH_PERF_GUID in out:
            return CheckResult(
                cid, title, "ok",
                "High Performance is active (Ultimate adds marginal gains: no core "
                "parking and fixed timer coalescing).",
            )
        return CheckResult(
            cid, title, "warn",
            "A balanced/power-saver plan is active — cores can park and downclock "
            "mid-run, which shows up as frame-time spikes.",
            can_fix=True, safe=False,
            fix_label="Activate Ultimate Performance plan",
        )

    def _f_power(self) -> str:
        # Duplicate the hidden Ultimate scheme (no-op if it already exists),
        # then activate it. Reversible from Windows Power Options at any time.
        _run(["powercfg", "-duplicatescheme", ULTIMATE_GUID])
        schemes = _run(["powercfg", "/list"]).lower()
        guid = ULTIMATE_GUID if ULTIMATE_GUID in schemes else HIGH_PERF_GUID
        _run(["powercfg", "/setactive", guid])
        after = _run(["powercfg", "/getactivescheme"]).lower()
        if guid in after:
            name = "Ultimate" if guid == ULTIMATE_GUID else "High"
            return f"{name} Performance plan activated."
        return "could not activate — change it in Windows Power Options"

    # -------------------------------------------------------------- hags
    def _c_hags(self) -> CheckResult:
        cid, title = "hags", "Hardware-accelerated GPU scheduling"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
            ) as k:
                mode = int(winreg.QueryValueEx(k, "HwSchMode")[0])
        except FileNotFoundError:
            # Value absent = driver default (not "unsupported"); modern
            # NVIDIA/AMD drivers default HAGS on where they support it.
            return CheckResult(
                cid, title, "info",
                "HAGS is at its driver default. If you see stutter, toggle it in "
                "Settings > System > Display > Graphics > Default settings.",
            )
        enabled = mode == 2
        rec = self.hw.hags_recommended
        if rec is None:
            return CheckResult(cid, title, "info",
                               f"HAGS is {'on' if enabled else 'off'}; no strong "
                               "guidance for this GPU.")
        if enabled == rec:
            return CheckResult(cid, title, "ok",
                               f"HAGS is {'on' if enabled else 'off'} — "
                               f"recommended for {self.hw.gpu_name}.")
        want = "on" if rec else "off"
        return CheckResult(
            cid, title, "warn",
            f"HAGS is {'on' if enabled else 'off'} but {want} is recommended for "
            f"{self.hw.gpu_name}. Change it in Settings > System > Display > "
            "Graphics > Default settings (needs admin + reboot), so kovadapt "
            "won't flip it automatically.",
        )

    # --------------------------------------------------------------- fso
    def _fso_key(self):
        import winreg

        return (winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers")

    def _c_fso(self) -> CheckResult:
        cid, title = "fso", "Fullscreen optimizations disabled on the game"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        exes = find_game_exes(self.kovaaks_root)
        if not exes:
            return CheckResult(cid, title, "unknown",
                               "game exe not found — set KovaaK's path first")
        try:
            import winreg

            root, path = self._fso_key()
            with winreg.OpenKey(root, path) as k:
                flags = str(winreg.QueryValueEx(k, str(exes[0]))[0])
        except FileNotFoundError:
            flags = ""
        if "DISABLEDXMAXIMIZEDWINDOWEDMODE" in flags:
            return CheckResult(cid, title, "ok",
                               "Fullscreen optimizations are disabled for the game exe.")
        return CheckResult(
            cid, title, "warn",
            "Windows may run the game through the DWM compositor even in "
            "fullscreen, adding presentation latency.",
            can_fix=True, safe=True,
            fix_label="Disable fullscreen optimizations for the game",
        )

    def _f_fso(self) -> str:
        import winreg

        exes = find_game_exes(self.kovaaks_root)
        if not exes:
            return "game exe not found"
        root, path = self._fso_key()
        with winreg.CreateKey(root, path) as k:
            for exe in exes:
                try:
                    existing = str(winreg.QueryValueEx(k, str(exe))[0])
                except FileNotFoundError:
                    existing = "~"
                if "DISABLEDXMAXIMIZEDWINDOWEDMODE" not in existing:
                    newval = (existing.rstrip() + " DISABLEDXMAXIMIZEDWINDOWEDMODE").strip()
                    winreg.SetValueEx(k, str(exe), 0, winreg.REG_SZ, newval)
        return (f"fullscreen optimizations disabled on {len(exes)} exe(s) "
                "(per-user, reversible in exe Properties > Compatibility)")

    # ------------------------------------------------------------- mouse
    def _c_mouse(self) -> CheckResult:
        cid, title = "mouse_accel", "Windows mouse acceleration off"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        SPI_GETMOUSE = 0x0003
        params = (ctypes.c_int * 3)()
        ctypes.windll.user32.SystemParametersInfoW(SPI_GETMOUSE, 0,
                                                   ctypes.byref(params), 0)
        if params[2] == 0:
            return CheckResult(
                cid, title, "ok",
                "'Enhance pointer precision' is off. (KovaaK's uses Raw Input, so "
                "this mainly guards desktop/menu consistency and other games.)",
            )
        return CheckResult(
            cid, title, "warn",
            "'Enhance pointer precision' (acceleration) is on. Raw Input games "
            "bypass it, but consistency across desktop and non-raw games matters "
            "for muscle memory.",
            can_fix=True, safe=True,
            fix_label="Turn off pointer acceleration",
        )

    def _f_mouse(self) -> str:
        SPI_SETMOUSE = 0x0004
        SPIF_UPDATEINIFILE, SPIF_SENDCHANGE = 0x01, 0x02
        params = (ctypes.c_int * 3)(0, 0, 0)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETMOUSE, 0, ctypes.byref(params),
            SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
        )
        return ("pointer acceleration off (persisted)" if ok
                else "could not change the setting")

    # ---------------------------------------------------------- chromium
    def _c_chromium(self) -> CheckResult:
        cid, title = "chromium", "Background Chromium apps"
        try:
            import psutil
        except ImportError:
            return CheckResult(cid, title, "unknown",
                               "psutil not installed (pip install kovadapt[gui])")
        running: set[str] = set()
        for p in psutil.process_iter(["name"]):
            n = (p.info["name"] or "").lower().removesuffix(".exe")
            if n in CHROMIUM_PROCS:
                running.add(n)
        blocking = sorted(running - set(UNAVOIDABLE))
        if not blocking:
            return CheckResult(cid, title, "ok",
                               "No timer-degrading background apps detected.")
        return CheckResult(
            cid, title, "warn",
            f"Running: {', '.join(blocking)}. Chromium apps request coarse "
            "timers and cause frame-time spikes; close them (or at least their "
            "overlays) while training. Not auto-closed — you may have unsaved work.",
        )

    # ------------------------------------------------------------ config
    def _c_config(self) -> CheckResult:
        cid, title = "game_config", "KovaaK's config file health"
        ini = game_config_path()
        if not ini.is_file():
            return CheckResult(cid, title, "info",
                               "GameUserSettings.ini not found (fresh install or "
                               "custom location).")
        try:
            text = ini.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            return CheckResult(
                cid, title, "bad",
                f"{ini} is unreadable/corrupt — a documented stutter cause.",
                can_fix=True, safe=False,
                fix_label="Back up + delete (game rebuilds it)",
            )
        issues = []
        if "[/Script/FPSAimTrainer.FPSGameUserSettings]" not in text \
                and "[ScalabilityGroups]" not in text:
            issues.append("missing expected sections")
        if text.count("\x00"):
            issues.append("contains NUL bytes")
        if issues:
            return CheckResult(
                cid, title, "bad",
                f"{ini.name}: {'; '.join(issues)} — likely corrupt.",
                can_fix=True, safe=False,
                fix_label="Back up + delete (game rebuilds it)",
            )
        return CheckResult(cid, title, "ok", f"{ini.name} looks healthy.")

    def _f_config(self) -> str:
        ini = game_config_path()
        if not ini.is_file():
            return "nothing to do"
        backup = ini.with_suffix(".ini.kovadapt-backup")
        backup.write_bytes(ini.read_bytes())
        ini.unlink()
        return f"deleted (backup at {backup.name}); launch KovaaK's to rebuild it"

    # ----------------------------------------------------------- process
    def _c_process(self) -> CheckResult:
        cid, title = "game_process", "Game priority & CPU affinity"
        try:
            import psutil
        except ImportError:
            return CheckResult(cid, title, "unknown",
                               "psutil not installed (pip install kovadapt[gui])")
        proc = None
        for p in psutil.process_iter(["name"]):
            if GAME_PROCESS.lower() in (p.info["name"] or "").lower():
                proc = p
                break
        if proc is None:
            return CheckResult(cid, title, "info",
                               "Game not running — the watchdog applies this "
                               "automatically on every launch.")
        from .watchdog import cpus_to_free

        try:
            high = proc.nice() == psutil.HIGH_PRIORITY_CLASS
            n = psutil.cpu_count(logical=True) or 2
            freed = cpus_to_free()
            affinity = set(proc.cpu_affinity())
            off_input = n <= 2 or not (affinity & set(freed))
        except psutil.AccessDenied:
            return CheckResult(cid, title, "unknown",
                               "access denied reading the game process (run as admin)")
        freed_txt = "/".join(str(c) for c in freed)
        if high and off_input:
            return CheckResult(cid, title, "ok",
                               f"High priority set and CPU {freed_txt} freed.")
        return CheckResult(
            cid, title, "warn",
            "Game running without the KovaaK's-FAQ tuning "
            f"(high priority: {high}, off input CPU {freed_txt}: {off_input}).",
            can_fix=True, safe=True,
            fix_label=f"Apply high priority + free CPU {freed_txt}",
        )

    def _f_process(self) -> str:
        from .watchdog import apply_game_tuning

        msg, ok = apply_game_tuning()
        return msg
