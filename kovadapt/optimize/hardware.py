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


# ------------------------------------------------- live WDDM driver state
# D3DKMTQueryAdapterInfo(KMTQAITYPE_WDDM_2_7_CAPS) reports what the graphics
# kernel is running with RIGHT NOW, unlike the HwSchMode registry value which
# is only the after-reboot intent. The structs mirror d3dkmthk.h exactly.

_KMTQAITYPE_WDDM_2_7_CAPS = 70

# D3DDDI_WDDM_2_7_CAPS: a UINT bitfield; bit 0 HwSchSupported, bit 1 HwSchEnabled
# (bit 2 HwSchEnabledByDefault, rest reserved).
_HWSCH_SUPPORTED = 0x1
_HWSCH_ENABLED = 0x2


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _D3DKMT_ADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint32),           # D3DKMT_HANDLE
        ("AdapterLuid", _LUID),
        ("NumOfSources", ctypes.c_uint32),
        ("bPrecisePresentRegionsPreferred", ctypes.c_int32),
    ]


class _D3DKMT_ENUMADAPTERS2(ctypes.Structure):
    _fields_ = [
        ("NumAdapters", ctypes.c_uint32),
        ("pAdapters", ctypes.POINTER(_D3DKMT_ADAPTERINFO)),
    ]


class _D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint32),
        ("Type", ctypes.c_int32),                # KMTQUERYADAPTERINFOTYPE
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint32),
    ]


class _D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = [("hAdapter", ctypes.c_uint32)]


def decode_wddm27_caps(value: int) -> tuple[bool, bool]:
    """(HwSchSupported, HwSchEnabled) from the D3DDDI_WDDM_2_7_CAPS bitfield."""
    return bool(value & _HWSCH_SUPPORTED), bool(value & _HWSCH_ENABLED)


def hags_live_state() -> tuple[bool | None, bool | None]:
    """(supported, enabled) for hardware GPU scheduling per the live driver.

    Enumerates the kernel-mode adapters (D3DKMTEnumAdapters2 returns already
    open handles) and returns the caps of the first adapter that supports
    hardware scheduling — that is the real GPU; software adapters like the
    Basic Render Driver report unsupported. (None, None) off-Windows or when
    any step fails. Read-only; every handle is closed before returning.
    """
    if not WINDOWS:
        return None, None
    handles: list[int] = []
    try:
        gdi32 = ctypes.WinDLL("gdi32")
        enum = _D3DKMT_ENUMADAPTERS2()
        if gdi32.D3DKMTEnumAdapters2(ctypes.byref(enum)) != 0 or not enum.NumAdapters:
            return None, None
        arr = (_D3DKMT_ADAPTERINFO * enum.NumAdapters)()
        enum.pAdapters = ctypes.cast(arr, ctypes.POINTER(_D3DKMT_ADAPTERINFO))
        status = gdi32.D3DKMTEnumAdapters2(ctypes.byref(enum))
        handles = [arr[i].hAdapter for i in range(enum.NumAdapters) if arr[i].hAdapter]
        if status != 0:
            return None, None
        fallback: tuple[bool, bool] | None = None
        for h in handles:
            caps = ctypes.c_uint32(0)
            q = _D3DKMT_QUERYADAPTERINFO(
                hAdapter=h,
                Type=_KMTQAITYPE_WDDM_2_7_CAPS,
                pPrivateDriverData=ctypes.cast(ctypes.byref(caps), ctypes.c_void_p),
                PrivateDriverDataSize=ctypes.sizeof(caps),
            )
            if gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(q)) != 0:
                continue                       # pre-WDDM-2.7 driver on this adapter
            supported, enabled = decode_wddm27_caps(caps.value)
            if supported:
                return supported, enabled
            if fallback is None:
                fallback = (supported, enabled)
        return fallback if fallback is not None else (None, None)
    except Exception:
        return None, None
    finally:
        try:
            for h in handles:
                gdi32.D3DKMTCloseAdapter(ctypes.byref(_D3DKMT_CLOSEADAPTER(hAdapter=h)))
        except Exception:
            pass


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
