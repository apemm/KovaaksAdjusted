"""One-click system checkup: detect everything, fix per item on request.

Each check probes read-only and reports a CheckResult; fixes run only when
explicitly invoked (per-item button or "fix all safe"). `safe=True` marks
fixes that are per-user, reversible, and admin-free (registry HKCU writes,
process tweaks); invasive fixes (power plan, file deletion, HKLM) always
need their own click and say what they will do first.

Windows-only at runtime; importable everywhere.
"""

from __future__ import annotations

import codecs
import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .hardware import HardwareInfo, hags_live_state

WINDOWS = sys.platform == "win32"

GAME_EXE = "FPSAimTrainer.exe"
GAME_PROCESS = "FPSAimTrainer"

# Game Bar / Game DVR background-capture switches (both must be 0 for "off").
GAMEDVR_APP_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\GameDVR"
GAMEDVR_STORE_KEY = r"System\GameConfigStore"
GAMEBAR_KEY = r"Software\Microsoft\GameBar"

# Ultimate Performance scheme (hidden on most consumer SKUs until duplicated).
ULTIMATE_GUID = "e9a42b02-d5df-448d-aa66-ad3f9edeb1c9"
HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
_GUID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)

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


def _read_hkcu_dword(path: str, name: str) -> int | None:
    """One HKCU DWORD, None when the key/value is absent or unreadable."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as k:
            return int(winreg.QueryValueEx(k, name)[0])
    except OSError:
        return None


def _hags_registry_mode() -> int | None:
    """HwSchMode registry value (2 = on, 1 = off), None when absent —
    the after-reboot INTENT, not necessarily what the driver runs with."""
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
        ) as k:
            return int(winreg.QueryValueEx(k, "HwSchMode")[0])
    except FileNotFoundError:
        return None


_KOVADAPT_KEY = r"Software\kovadapt"


def _stored_scheme_guid() -> str | None:
    """GUID of the performance scheme a previous fix activated. Name matching
    is locale-dependent and duplicated schemes get fresh random GUIDs, so
    this HKCU note is the only way to recognize our own scheme again on
    non-English Windows."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KOVADAPT_KEY) as k:
            return str(winreg.QueryValueEx(k, "PowerSchemeGuid")[0]).lower()
    except OSError:
        return None


def _store_scheme_guid(guid: str) -> None:
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _KOVADAPT_KEY) as k:
        winreg.SetValueEx(k, "PowerSchemeGuid", 0, winreg.REG_SZ, guid.lower())


def parse_powercfg_ac_index(output: str) -> int | None:
    """Current AC value from `powercfg /q` output (hex like 0x00000064),
    None when the setting is missing from the listing."""
    m = re.search(r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+|\d+)", output)
    if not m:
        return None
    raw = m.group(1)
    return int(raw, 16) if raw.lower().startswith("0x") else int(raw)


def _query_timer_resolution() -> tuple[float, float] | None:
    """(current_ms, finest_ms) via NtQueryTimerResolution (100 ns units),
    None on failure. Confusingly, ntdll's "maximum" is the finest resolution."""
    try:
        ntdll = ctypes.WinDLL("ntdll")
        coarsest = ctypes.c_ulong()
        finest = ctypes.c_ulong()
        current = ctypes.c_ulong()
        if ntdll.NtQueryTimerResolution(ctypes.byref(coarsest), ctypes.byref(finest),
                                        ctypes.byref(current)) != 0:
            return None
        return current.value / 10000.0, finest.value / 10000.0
    except Exception:
        return None


def _game_running() -> bool | None:
    """True/False when psutil can tell us, None when it isn't installed."""
    try:
        import psutil
    except ImportError:
        return None
    for p in psutil.process_iter(["name"]):
        if GAME_PROCESS.lower() in (p.info["name"] or "").lower():
            return True
    return False


def game_config_path() -> Path:
    import os

    base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    return (Path(base) / "FPSAimTrainer" / "Saved" / "Config" / "WindowsNoEditor"
            / "GameUserSettings.ini")


def read_game_config(path: Path) -> str | None:
    """GameUserSettings.ini as text, None when no plausible encoding fits.

    Path.read_text() with no encoding= decodes with the process ANSI codepage,
    and that made healthy configs look corrupt — the verdict this file offers a
    DELETE for. Unreal writes this ini as UTF-8 (with a BOM once a non-ASCII
    string is stored, UTF-16 on its AutoDetect save path), so on a cp1252
    machine one accented character in a player/scenario/crosshair name was
    enough: "A-acute" is C3 81 in UTF-8 and 0x81 is an undefined cp1252 slot,
    so the read raised UnicodeDecodeError. UTF-16 was worse — it decoded to
    mojibake full of NULs and tripped the corruption heuristics below.

    Tolerate a BOM the way Settings.load already does for settings.json, and
    keep a legacy single-byte codepage only as the last candidate, for files
    Unreal wrote down its pure-ANSI path. None (nothing decodes) is the honest
    corrupt signal.

    The legacy fallbacks are NAMED, not taken from the locale. Deriving the
    last candidate from locale.getpreferredencoding() made the outcome depend
    on the machine: where that is already UTF-8 (Linux/macOS, PYTHONUTF8=1, or
    Windows' "Use Unicode UTF-8" option) the list collapsed to two UTF-8
    entries, a genuine legacy-ANSI config decoded under neither, and a healthy
    file was offered for DELETE again — the exact bug this function exists to
    kill, reintroduced on a different machine. cp1252 then latin-1 covers it;
    latin-1 decodes any byte sequence, so this now only returns None for a
    UTF-16 file whose BOM was stripped, which the corruption heuristics catch.
    """
    raw = path.read_bytes()
    if raw[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        candidates = ("utf-16",)   # the BOM itself picks the endianness
    else:
        candidates = ("utf-8-sig", "cp1252", "latin-1")
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


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
            Check("core_parking", "Core parking off on the active plan",
                  self._c_parking),
            Check("hags", "Hardware-accelerated GPU scheduling", self._c_hags),
            Check("hags_live", "GPU scheduling: registry vs live driver",
                  self._c_hags_live),
            Check("fso", "Fullscreen optimizations disabled on the game",
                  self._c_fso, self._f_fso),
            Check("mouse_accel", "Windows mouse acceleration off",
                  self._c_mouse, self._f_mouse),
            Check("gamedvr", "Game Bar background capture off",
                  self._c_gamedvr, self._f_gamedvr),
            Check("game_mode", "Windows Game Mode", self._c_game_mode,
                  self._f_game_mode),
            Check("timer_res", "System timer resolution", self._c_timer),
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
        # Duplicated schemes get a fresh random GUID and localized names, so
        # recognize: canonical GUIDs, English names, and the GUID a previous
        # kovadapt fix recorded (the locale-independent path).
        stored = _stored_scheme_guid()
        if stored and stored in out:
            return CheckResult(cid, title, "ok",
                               "The performance plan kovadapt activated is active.")
        if ULTIMATE_GUID in out or "ultimate performance" in out:
            return CheckResult(cid, title, "ok", "Ultimate Performance is active.")
        if HIGH_PERF_GUID in out or "high performance" in out:
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
        # The Ultimate scheme is hidden on consumer SKUs: -duplicatescheme
        # creates a VISIBLE copy under a NEW random GUID every call. Reuse an
        # existing copy before creating one, and activate the GUID that
        # actually exists. Reversible from Windows Power Options at any time.
        def named_guid(listing: str, name: str) -> str | None:
            for line in listing.lower().splitlines():
                m = _GUID_RE.search(line)
                if m and name in line:
                    return m.group(0)
            return None

        listing = _run(["powercfg", "/list"])
        # Reuse, in order: the copy we made before (GUID note in HKCU — the
        # only locale-independent handle), then an English-named copy.
        stored = _stored_scheme_guid()
        guid = stored if (stored and stored in listing.lower()) else None
        guid = guid or named_guid(listing, "ultimate performance")
        name = "Ultimate"
        if guid is None:
            m = _GUID_RE.search(_run(["powercfg", "-duplicatescheme", ULTIMATE_GUID]))
            guid = m.group(0).lower() if m else None
        if guid is None:  # duplication failed: High Perf fallback
            guid, name = named_guid(listing, "high performance"), "High"
        if guid is not None:
            _run(["powercfg", "/setactive", guid])
            if guid in _run(["powercfg", "/getactivescheme"]).lower():
                try:
                    _store_scheme_guid(guid)   # recognize it next scan, any locale
                except OSError:
                    pass
                return f"{name} Performance plan activated."
        return "could not activate — change it in Windows Power Options"

    # ------------------------------------------------------------ parking
    def _c_parking(self) -> CheckResult:
        cid, title = "core_parking", "Core parking off on the active plan"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        # /qh, not /q: CPMINCORES is a hidden setting on stock plans and /q
        # omits hidden settings entirely (verified on Win11 24H2).
        out = _run(["powercfg", "/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", "CPMINCORES"])
        pct = parse_powercfg_ac_index(out)
        if pct is None:
            return CheckResult(cid, title, "unknown",
                               "could not read CPMINCORES from the active plan "
                               "(powercfg output unrecognized)")
        if pct >= 100:
            return CheckResult(cid, title, "ok",
                               f"Min unparked cores is {pct}% — no core can park "
                               "mid-run on this plan.")
        return CheckResult(
            cid, title, "warn",
            f"The active plan keeps only {pct}% of cores unparked; a parked core "
            "waking mid-flick is a frame-time spike. Changing it can need admin, "
            "so kovadapt won't write it — run: powercfg /setacvalueindex "
            "SCHEME_CURRENT SUB_PROCESSOR CPMINCORES 100 && powercfg /setactive "
            "SCHEME_CURRENT",
        )

    # -------------------------------------------------------------- hags
    def _c_hags(self) -> CheckResult:
        cid, title = "hags", "Hardware-accelerated GPU scheduling"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        mode = _hags_registry_mode()
        if mode is None:
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

    def _c_hags_live(self) -> CheckResult:
        """The registry probe above is only the after-reboot intent; this asks
        the graphics kernel (D3DKMTQueryAdapterInfo) what is running NOW."""
        cid, title = "hags_live", "GPU scheduling: registry vs live driver"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        supported, enabled = hags_live_state()
        if supported is None:
            return CheckResult(cid, title, "unknown",
                               "WDDM 2.7 caps query failed — live HAGS state "
                               "unavailable (pre-2004 Windows or an odd driver)")
        if not supported:
            return CheckResult(
                cid, title, "info",
                "The driver reports no hardware-scheduling support, so the HAGS "
                "toggle has no effect on this GPU/driver.",
            )
        live = "on" if enabled else "off"
        try:
            mode = _hags_registry_mode()
        except Exception:
            mode = None
        if mode is None:
            return CheckResult(cid, title, "info",
                               f"HAGS is live {live} (driver default — no registry "
                               "override set).")
        if (mode == 2) == bool(enabled):
            return CheckResult(cid, title, "ok",
                               f"Registry intent and the live driver agree: HAGS "
                               f"is {live}.")
        want = "on" if mode == 2 else "off"
        return CheckResult(
            cid, title, "warn",
            f"The registry says HAGS {want} but the driver is running with it "
            f"{live} — a reboot is pending, or the driver overrode the toggle. "
            "The live state is what your frames actually get.",
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

    # ----------------------------------------------------------- gamedvr
    def _c_gamedvr(self) -> CheckResult:
        cid, title = "gamedvr", "Game Bar background capture off"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        app = _read_hkcu_dword(GAMEDVR_APP_KEY, "AppCaptureEnabled")
        store = _read_hkcu_dword(GAMEDVR_STORE_KEY, "GameDVR_Enabled")
        # Absent value = Windows default = capture enabled.
        if app == 0 and store == 0:
            return CheckResult(cid, title, "ok",
                               "Game DVR background capture is off — no capture "
                               "process shadowing the game.")

        def show(v: int | None) -> str:
            return "absent (default on)" if v is None else str(v)

        return CheckResult(
            cid, title, "warn",
            "Game DVR keeps a background capture process recording gameplay, "
            "which costs frames and frame-time consistency. Current values: "
            f"AppCaptureEnabled={show(app)}, GameDVR_Enabled={show(store)} — "
            "set them back to restore.",
            can_fix=True, safe=True,
            fix_label="Turn off Game Bar background capture",
        )

    def _f_gamedvr(self) -> str:
        import winreg

        prior = []
        for path, name in ((GAMEDVR_APP_KEY, "AppCaptureEnabled"),
                           (GAMEDVR_STORE_KEY, "GameDVR_Enabled")):
            old = _read_hkcu_dword(path, name)
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path) as k:
                winreg.SetValueEx(k, name, 0, winreg.REG_DWORD, 0)
            prior.append(f"{name} was {'absent' if old is None else old}")
        return ("background capture off (per-user; " + ", ".join(prior) +
                " — set back to restore, or toggle in Settings > Gaming > Captures)")

    # --------------------------------------------------------- game mode
    def _c_game_mode(self) -> CheckResult:
        cid, title = "game_mode", "Windows Game Mode"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        val = _read_hkcu_dword(GAMEBAR_KEY, "AutoGameModeEnabled")
        # Absent value = Windows default = Game Mode on.
        if val is None or val != 0:
            return CheckResult(
                cid, title, "info",
                f"Game Mode is on{' (default)' if val is None else ''}. Current "
                "Microsoft guidance is to keep it on: it steers Windows Update "
                "and driver notifications away from the running game.",
            )
        return CheckResult(
            cid, title, "warn",
            "Game Mode is off. On Windows 11 Microsoft's current guidance is ON — "
            "it defers background work while the game runs; the old advice to "
            "disable it dates from its buggy 2017 debut.",
            can_fix=True, safe=True,
            fix_label="Turn Game Mode on",
        )

    def _f_game_mode(self) -> str:
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, GAMEBAR_KEY) as k:
            winreg.SetValueEx(k, "AutoGameModeEnabled", 0, winreg.REG_DWORD, 1)
        return ("Game Mode on (per-user, reversible in Settings > Gaming > "
                "Game Mode)")

    # ------------------------------------------------------------- timer
    def _c_timer(self) -> CheckResult:
        cid, title = "timer_res", "System timer resolution"
        if not WINDOWS:
            return CheckResult(cid, title, "unknown", "not Windows")
        res = _query_timer_resolution()
        if res is None:
            return CheckResult(cid, title, "unknown",
                               "NtQueryTimerResolution failed")
        cur, finest = res
        state = f"currently {cur:.2f} ms (finest this system supports: {finest:.2f} ms)"
        if self.hw.is_windows_11:
            return CheckResult(
                cid, title, "info",
                f"Timer {state}. Windows 11 raises resolution per-process, so a "
                "game asking for 1 ms gets it even when this desktop-wide "
                "reading looks coarse — nothing to do here.",
            )
        if cur > 1.05 and _game_running():
            return CheckResult(
                cid, title, "warn",
                f"Timer {state} while the game is running — on Windows 10 the "
                "resolution is global and frame pacing suffers above ~1 ms. No "
                "auto-fix: it is owned by whichever app requested it; run a 1 ms "
                "timer tool alongside the game.",
            )
        return CheckResult(
            cid, title, "info",
            f"Timer {state}. On Windows 10 the resolution is global; re-check "
            "with the game running, and keep a 1 ms timer tool handy if it "
            "reads coarse mid-session.",
        )

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
            text = read_game_config(ini)
        except OSError:
            text = None
        if text is None:
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
        # Same resolver the watchdog tunes through: probing the Steam stub
        # while the watchdog tunes the renderer would report on the wrong
        # process either way round.
        from .watchdog import find_game_process

        proc = find_game_process(psutil)
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
