"""Hardware and system-state detection. Read-only; no admin rights needed.

Sources, cheapest first: ctypes Win32 calls (refresh rate, RAM), the registry
(CPU name, HAGS, Windows build), and one PowerShell CIM query for the GPU
(wmic is gone from recent Windows 11 builds). Everything degrades to unknown
values off-Windows or on probe failure — never raises.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict

WINDOWS = sys.platform == "win32"

_NVIDIA_REFLEX_MIN_GEN = 900     # GTX 900+ supports Reflex in-game


@dataclass
class HardwareInfo:
    cpu_name: str = ""
    logical_cores: int = 0
    ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_vendor: str = ""         # nvidia | amd | intel | ""
    monitor_hz: int = 0
    windows_build: int = 0
    is_windows_11: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    # ---------------------------------------------------------- derived
    @property
    def nvidia_gen(self) -> int:
        """Rough NVIDIA series number (e.g. 3070 -> 3000, 1660 -> 1600),
        0 when not NVIDIA/undetected. Used for Reflex/HAGS advice."""
        if self.gpu_vendor != "nvidia":
            return 0
        m = re.search(r"\b(?:RTX|GTX)\s*(\d{3,4})", self.gpu_name, re.IGNORECASE)
        if not m:
            return 0
        # Model number -> series: 980 -> 900, 1660 -> 1600, 3070 -> 3000.
        return (int(m.group(1)) // 100) * 100

    @property
    def supports_reflex(self) -> bool:
        return self.nvidia_gen >= _NVIDIA_REFLEX_MIN_GEN

    @property
    def has_rt_cores(self) -> bool:
        """RTX 2000+ (Turing and newer) — gates the CPU->GPU offload advice."""
        return (self.gpu_vendor == "nvidia" and self.nvidia_gen >= 2000
                and "rtx" in self.gpu_name.lower())

    @property
    def smooth_motion_capable(self) -> bool:
        """NVIDIA Smooth Motion: driver-level frame generation that works on
        DX11 titles like KovaaK's (RTX 50 at launch, extended to RTX 40)."""
        return self.nvidia_gen >= 4000

    @property
    def hags_recommended(self) -> bool | None:
        """True/False when we have evidence for this GPU; None = no guidance.
        HAGS helps on RTX 3000+ / RX 6000+, tends to hurt older cards."""
        if self.gpu_vendor == "nvidia":
            return self.nvidia_gen >= 3000
        if self.gpu_vendor == "amd":
            m = re.search(r"\bRX\s*(\d{4})", self.gpu_name, re.IGNORECASE)
            return bool(m) and int(m.group(1)) >= 6000
        return None


# ------------------------------------------------------------------ probes
def _cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return ""


def _logical_cores() -> int:
    import os

    return os.cpu_count() or 0


def _ram_gb() -> float:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return round(st.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _monitor_hz() -> int:
    """Refresh rate of the primary display via EnumDisplaySettingsW."""
    try:
        ENUM_CURRENT_SETTINGS = -1

        class DEVMODEW(ctypes.Structure):
            _fields_ = [
                ("dmDeviceName", ctypes.c_wchar * 32),
                ("dmSpecVersion", ctypes.c_uint16),
                ("dmDriverVersion", ctypes.c_uint16),
                ("dmSize", ctypes.c_uint16),
                ("dmDriverExtra", ctypes.c_uint16),
                ("dmFields", ctypes.c_uint32),
                ("dmUnion1", ctypes.c_byte * 16),
                ("dmColor", ctypes.c_int16),
                ("dmDuplex", ctypes.c_int16),
                ("dmYResolution", ctypes.c_int16),
                ("dmTTOption", ctypes.c_int16),
                ("dmCollate", ctypes.c_int16),
                ("dmFormName", ctypes.c_wchar * 32),
                ("dmLogPixels", ctypes.c_uint16),
                ("dmBitsPerPel", ctypes.c_uint32),
                ("dmPelsWidth", ctypes.c_uint32),
                ("dmPelsHeight", ctypes.c_uint32),
                ("dmDisplayFlags", ctypes.c_uint32),
                ("dmDisplayFrequency", ctypes.c_uint32),
                # trailing printer/display fields omitted; dmSize covers us
                ("dmICMMethod", ctypes.c_uint32),
                ("dmICMIntent", ctypes.c_uint32),
                ("dmMediaType", ctypes.c_uint32),
                ("dmDitherType", ctypes.c_uint32),
                ("dmReserved1", ctypes.c_uint32),
                ("dmReserved2", ctypes.c_uint32),
                ("dmPanningWidth", ctypes.c_uint32),
                ("dmPanningHeight", ctypes.c_uint32),
            ]

        dm = DEVMODEW()
        dm.dmSize = ctypes.sizeof(DEVMODEW)
        if ctypes.windll.user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS,
                                                     ctypes.byref(dm)):
            return int(dm.dmDisplayFrequency)
    except Exception:
        pass
    return 0


def _gpu() -> tuple[str, str]:
    """(name, vendor) of the most capable GPU via one CIM query."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | "
             "Select-Object -ExpandProperty Name) -join '|'"],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        names = [n.strip() for n in out.stdout.strip().split("|") if n.strip()]
    except Exception:
        names = []
    if not names:
        return "", ""

    def vendor(n: str) -> str:
        ln = n.lower()
        if "nvidia" in ln or "geforce" in ln or "rtx" in ln or "gtx" in ln:
            return "nvidia"
        if "amd" in ln or "radeon" in ln:
            return "amd"
        if "intel" in ln:
            return "intel"
        return ""

    # Prefer a discrete GPU over integrated when both are present.
    ranked = sorted(names, key=lambda n: {"nvidia": 0, "amd": 1}.get(vendor(n), 2))
    return ranked[0], vendor(ranked[0])


def _windows_build() -> int:
    try:
        return sys.getwindowsversion().build
    except Exception:
        return 0


def detect_hardware() -> HardwareInfo:
    """Probe everything; safe to call from any thread. ~1s (one PowerShell)."""
    if not WINDOWS:
        return HardwareInfo(notes=["not Windows — detection skipped"])
    hw = HardwareInfo(
        cpu_name=_cpu_name(),
        logical_cores=_logical_cores(),
        ram_gb=_ram_gb(),
        monitor_hz=_monitor_hz(),
        windows_build=_windows_build(),
    )
    hw.gpu_name, hw.gpu_vendor = _gpu()
    hw.is_windows_11 = hw.windows_build >= 22000
    if not hw.gpu_name:
        hw.notes.append("GPU query failed — vendor-specific advice unavailable")
    return hw
