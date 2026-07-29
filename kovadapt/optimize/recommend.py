"""Hardware-matched recommendations: Steam launch options + settings profile.

Everything here is advice rendering — no system mutation. Recommendations
are keyed off HardwareInfo so a GTX 1060 owner and an RTX 5080 owner see
different guidance. Launch options are deliberately conservative: UE4 flags
that are placebo or harmful for KovaaK's specifically are listed as
"skip" with the reason, because the community pastes them around anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareInfo


@dataclass(frozen=True)
class Recommendation:
    title: str
    detail: str
    category: str        # launch | video | driver | windows
    priority: int = 1    # 1 = do this, 2 = worth trying, 3 = situational


def steam_launch_options(hw: HardwareInfo) -> str:
    """The launch-option string worth using. KovaaK's is a lightweight UE4
    title; most magic flags do nothing. -USEALLAVAILABLECORES is kept because
    it is harmless and occasionally helps on high-core-count CPUs; -NOSPLASH
    just skips the splash screen."""
    return "-USEALLAVAILABLECORES -NOSPLASH"


def skipped_launch_options() -> list[tuple[str, str]]:
    """(flag, why not) — shown so users stop cargo-culting them in."""
    return [
        ("-high", "sets priority once at launch; the kovadapt watchdog does it "
                  "properly and re-applies it every launch"),
        ("-malloc=system", "UE4 allocator advice from 2015; no measurable effect "
                           "in KovaaK's and can fragment memory"),
        ("-notexturestreaming", "KovaaK's textures are tiny; forces higher VRAM "
                                "use for nothing"),
        ("-dx12", "KovaaK's is a DX11 title; forcing other RHIs breaks or slows it"),
        ("-ONETHREAD", "debugging flag; halves performance"),
    ]


def recommended_settings(hw: HardwareInfo) -> list[Recommendation]:
    recs: list[Recommendation] = []

    # --- in-game video ---------------------------------------------------
    if hw.monitor_hz:
        cap = max(hw.monitor_hz * 2, 240)
        recs.append(Recommendation(
            f"Cap FPS near {cap} (2x your {hw.monitor_hz} Hz refresh)",
            "A stable cap beats a higher-but-variable uncapped rate: frame-time "
            "consistency is what your aim feels. Set it in KovaaK's video "
            "settings, not the driver.",
            "video", 1))
    recs.append(Recommendation(
        "Exclusive fullscreen, native resolution, everything else Low",
        "KovaaK's visual settings don't affect target visibility; Low keeps "
        "GPU frame times flat. Resolution scale 100% — never below.",
        "video", 1))
    recs.append(Recommendation(
        "Max the workshop cache lifetime (KovaaK's Settings > Main, set 168)",
        "The 'loading workshop scenario data' stall on every boot can't be "
        "disabled, but the cache refresh can be stretched to weekly — the "
        "game then starts near-instantly between refreshes. kovadapt's own "
        "adaptive scenarios are local files and never wait on this.",
        "video", 1))

    # --- driver ----------------------------------------------------------
    if hw.supports_reflex:
        recs.append(Recommendation(
            "Enable NVIDIA Reflex in KovaaK's (Video > NVIDIA Reflex: On)",
            "Reflex holds the render queue at zero — the single biggest "
            "input-latency reduction available on your GPU.",
            "driver", 1))
    elif hw.gpu_vendor == "nvidia":
        recs.append(Recommendation(
            "NVIDIA Control Panel > Low Latency Mode: Ultra",
            "Your GPU predates in-game Reflex; Ultra low-latency in the driver "
            "is the equivalent queue-limiting fallback.",
            "driver", 1))
    elif hw.gpu_vendor == "amd":
        recs.append(Recommendation(
            "AMD Adrenalin > Radeon Anti-Lag: On",
            "AMD's render-queue limiter; same idea as NVIDIA Reflex.",
            "driver", 1))
    if hw.gpu_vendor == "nvidia":
        recs.append(Recommendation(
            "NVIDIA Control Panel > Power management: Prefer maximum performance",
            "Stops the GPU downclocking between the light frames KovaaK's "
            "renders; set it per-game on FPSAimTrainer.exe to avoid idle drain.",
            "driver", 2))
    recs.append(Recommendation(
        "VSync off everywhere (game + driver)",
        "VSync adds up to a frame of latency. With a 2x-refresh cap, tearing "
        "is minimal; if it bothers you, G-Sync/FreeSync + a cap 3 fps below "
        "refresh is the low-latency compromise.",
        "driver", 1))

    # --- CPU -> GPU load shifting (RTX / RT-core GPUs) ---------------------
    # KovaaK's at high fps is CPU-bound (game thread + input processing), so
    # smoothness comes from taking work OFF the CPU, not adding GPU tricks.
    if hw.has_rt_cores:
        if hw.hags_recommended is True:
            # RTX 30+ only: on 20-series HAGS tends to cost more in stutter
            # than it gains, and the windows section below says so instead —
            # each GPU gets exactly one HAGS recommendation, never both.
            recs.append(Recommendation(
                "HAGS ON — move frame scheduling from CPU to GPU",
                "Hardware-accelerated GPU scheduling hands the scheduling queue to "
                "the GPU's dedicated engine, freeing CPU time exactly where KovaaK's "
                "bottlenecks. On RTX 30+ this is the one real CPU→GPU shift with no "
                "latency cost. (Settings > Display > Graphics > Default settings; "
                "needs a reboot.)",
                "windows", 1))
        if hw.smooth_motion_capable:
            recs.append(Recommendation(
                "Frame generation (NVIDIA Smooth Motion): OFF for training",
                "Your GPU supports driver-level frame gen on DX11 titles like "
                "KovaaK's, and with Reflex the latency cost is small — but "
                "generated frames never sample your mouse, so they add perceived "
                "smoothness without aim information. At the 300+ real fps this "
                "game reaches, real frames win. Save Smooth Motion for "
                "GPU-heavy games where base fps is low.",
                "driver", 3))
    if hw.gpu_vendor == "nvidia":
        recs.append(Recommendation(
            "Let the GPU absorb headroom: raise resolution before raising cap",
            "If your fps cap is met with big margin, extra CPU headroom does "
            "nothing — but higher resolution shifts relative load to the GPU "
            "and keeps frame times flatter than an unreachable cap would.",
            "video", 3))

    # --- windows ----------------------------------------------------------
    hags = hw.hags_recommended
    if hags is True and not hw.has_rt_cores:
        # AMD RX 6000+ — the NVIDIA RTX 30+ case already got the priority-1
        # "HAGS ON" recommendation above; don't repeat it.
        recs.append(Recommendation(
            "Keep hardware-accelerated GPU scheduling ON",
            f"Recommended on {hw.gpu_name or 'your GPU'}; slightly better "
            "latency consistency on recent GPUs.",
            "windows", 2))
    elif hags is False:
        recs.append(Recommendation(
            "Try hardware-accelerated GPU scheduling OFF",
            f"On {hw.gpu_name or 'older GPUs'} HAGS often costs more in stutter "
            "than it gains; A/B it across a session.",
            "windows", 2))
    if not hw.is_windows_11:
        recs.append(Recommendation(
            "Keep a 1 ms timer-resolution tool running (Windows 10)",
            "Windows 10 uses a global timer resolution that background apps can "
            "degrade; Windows 11 manages it per-process. Any open-source timer "
            "tool works.",
            "windows", 2))
    if hw.logical_cores >= 12:
        recs.append(Recommendation(
            "Let the kovadapt watchdog keep the game off the input core",
            f"With {hw.logical_cores} logical cores there is zero downside to "
            "reserving the first physical core (both its threads when "
            "hyperthreading is on) for Windows' input processing — the "
            "KovaaK's FAQ recommendation Process Lasso charges for.",
            "windows", 1))
    return sorted(recs, key=lambda r: r.priority)
