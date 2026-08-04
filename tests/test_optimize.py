"""Optimizer package tests: pure logic + probes that run on any OS.

System-mutating fixes are exercised only through their guards (mocked pids,
tmp config files); nothing here touches real registry/power state.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (keeps parity with the rest of the suite)

from kovadapt.optimize.checkup import SystemCheckup, find_game_exes, game_config_path
from kovadapt.optimize.hardware import HardwareInfo
from kovadapt.optimize.recommend import (
    recommended_settings,
    skipped_launch_options,
    steam_launch_options,
)
from kovadapt.optimize.watchdog import GameWatchdog


# ------------------------------------------------------------- hardware info
def test_nvidia_generation_parsing():
    cases = {
        "NVIDIA GeForce RTX 5080 Laptop GPU": 5000,
        "NVIDIA GeForce RTX 3070": 3000,
        "NVIDIA GeForce RTX 2060 SUPER": 2000,
        "NVIDIA GeForce GTX 1660 Ti": 1600,
        "NVIDIA GeForce GTX 980": 900,
    }
    for name, gen in cases.items():
        hw = HardwareInfo(gpu_name=name, gpu_vendor="nvidia")
        assert hw.nvidia_gen == gen, name
    assert HardwareInfo(gpu_name="AMD Radeon RX 7800 XT", gpu_vendor="amd").nvidia_gen == 0


def test_capability_flags():
    rtx50 = HardwareInfo(gpu_name="NVIDIA GeForce RTX 5080", gpu_vendor="nvidia")
    assert rtx50.supports_reflex and rtx50.has_rt_cores and rtx50.smooth_motion_capable
    assert rtx50.hags_recommended is True

    gtx10 = HardwareInfo(gpu_name="NVIDIA GeForce GTX 1080", gpu_vendor="nvidia")
    assert gtx10.supports_reflex          # GTX 900+ has in-game Reflex
    assert not gtx10.has_rt_cores
    assert not gtx10.smooth_motion_capable
    assert gtx10.hags_recommended is False

    rx68 = HardwareInfo(gpu_name="AMD Radeon RX 6800 XT", gpu_vendor="amd")
    assert rx68.hags_recommended is True
    assert HardwareInfo().hags_recommended is None


# ------------------------------------------------------------ recommendations
def test_recommendations_gate_on_hardware():
    rtx = HardwareInfo(gpu_name="NVIDIA GeForce RTX 4070", gpu_vendor="nvidia",
                       monitor_hz=240, logical_cores=16, is_windows_11=True)
    titles = [r.title for r in recommended_settings(rtx)]
    assert any("Reflex" in t for t in titles)
    assert any("HAGS ON" in t for t in titles)
    assert any("Smooth Motion" in t for t in titles)   # honest OFF-for-training rec
    assert any("240" in t for t in titles)             # refresh-matched fps cap
    assert not any("timer-resolution tool" in t for t in titles)  # win11

    # GTX 750 predates in-game Reflex (GTX 900+) -> driver fallback
    old = HardwareInfo(gpu_name="NVIDIA GeForce GTX 750 Ti", gpu_vendor="nvidia",
                       monitor_hz=144, logical_cores=8, is_windows_11=False)
    titles_old = [r.title for r in recommended_settings(old)]
    assert any("Low Latency Mode: Ultra" in t for t in titles_old)
    assert not any("Smooth Motion" in t for t in titles_old)
    assert any("timer-resolution tool" in t for t in titles_old)  # win10


def test_frame_gen_rec_is_honest():
    """Frame gen must never be recommended as a latency win for training."""
    rtx = HardwareInfo(gpu_name="NVIDIA GeForce RTX 5090", gpu_vendor="nvidia")
    fg = [r for r in recommended_settings(rtx) if "Smooth Motion" in r.title]
    assert len(fg) == 1
    assert "OFF for training" in fg[0].title
    assert fg[0].priority == 3   # situational, never "do this"


def test_hags_advice_is_consistent_per_gpu():
    """Every GPU generation gets exactly one HAGS recommendation — never a
    priority-1 'turn it ON' alongside a 'try OFF' (the RTX 20-series trap)."""
    def hags_recs(hw: HardwareInfo):
        return [r for r in recommended_settings(hw)
                if "HAGS" in r.title or "GPU scheduling" in r.title]

    rtx20 = hags_recs(HardwareInfo(gpu_name="NVIDIA GeForce RTX 2070", gpu_vendor="nvidia"))
    assert len(rtx20) == 1
    assert "OFF" in rtx20[0].title           # matches the checkup probe's advice

    rtx30 = hags_recs(HardwareInfo(gpu_name="NVIDIA GeForce RTX 3070", gpu_vendor="nvidia"))
    assert len(rtx30) == 1
    assert rtx30[0].title.startswith("HAGS ON") and rtx30[0].priority == 1

    gtx10 = hags_recs(HardwareInfo(gpu_name="NVIDIA GeForce GTX 1080", gpu_vendor="nvidia"))
    assert len(gtx10) == 1 and "OFF" in gtx10[0].title

    rx6 = hags_recs(HardwareInfo(gpu_name="AMD Radeon RX 6800 XT", gpu_vendor="amd"))
    assert len(rx6) == 1 and rx6[0].title.endswith("ON")

    assert hags_recs(HardwareInfo()) == []   # unknown GPU: no HAGS guidance


def test_launch_options_and_myths():
    assert "-USEALLAVAILABLECORES" in steam_launch_options(HardwareInfo())
    flags = dict(skipped_launch_options())
    assert "-high" in flags and "-dx12" in flags


# ------------------------------------------------------------------- checkup
def test_find_game_exes_layout(tmp_path):
    root = tmp_path / "FPSAimTrainer" / "FPSAimTrainer"   # nested like Steam
    ship = root / "Binaries" / "Win64" / "FPSAimTrainer-Win64-Shipping.exe"
    boot = root.parent / "FPSAimTrainer.exe"
    ship.parent.mkdir(parents=True)
    ship.write_bytes(b"")
    boot.write_bytes(b"")
    exes = find_game_exes(str(root))
    assert exes[0] == ship          # shipping exe first (the real process)
    assert boot in exes
    assert find_game_exes("") == []


def test_config_corruption_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    ini = game_config_path()
    assert str(tmp_path) in str(ini)
    checkup = SystemCheckup("", HardwareInfo())

    res = checkup._c_config()
    assert res.status == "info"      # missing file is fine

    ini.parent.mkdir(parents=True)
    ini.write_text("[ScalabilityGroups]\nsg.ViewDistanceQuality=0\n")
    assert checkup._c_config().status == "ok"

    ini.write_bytes(b"\x00\x00garbage\x00")
    res = checkup._c_config()
    assert res.status == "bad" and res.can_fix and not res.safe

    out = checkup._f_config()
    assert "backup" in out
    assert not ini.exists()
    assert ini.with_suffix(".ini.kovadapt-backup").exists()


def test_run_all_never_raises():
    results = SystemCheckup("", HardwareInfo()).run_all()
    assert len(results) == 12
    assert all(r.status in ("ok", "warn", "bad", "info", "unknown") for r in results)
    assert SystemCheckup("", HardwareInfo()).fix("nonexistent") \
        == "no automated fix for this item"


# ------------------------------------------------------------------ watchdog
def test_watchdog_applies_once_per_pid(monkeypatch):
    events: list[str] = []
    wd = GameWatchdog(on_event=events.append, poll_interval=0.01)
    applied: list[int] = []
    pid_seq = iter([None, 100, 100, 100, None, 200, 200])
    current = {"pid": None}

    def fake_find(self):
        try:
            current["pid"] = next(pid_seq)
        except StopIteration:
            pass
        return current["pid"]

    monkeypatch.setattr(GameWatchdog, "_find_pid", fake_find)
    monkeypatch.setattr("kovadapt.optimize.watchdog.apply_game_tuning",
                        lambda: (applied.append(current["pid"]) or ("tuned", True)))
    wd.start()
    import time

    # 10s, not 2. The loop exits the moment the second apply lands, so a
    # healthy run still finishes in ~70ms and the ceiling costs nothing —
    # but this waits on a background thread polling every 10ms, and in a full
    # suite run competing with Qt widget teardown that thread gets starved
    # often enough to miss a 2s wall-clock deadline. Observed once in four
    # full runs. A deadline that fails under load is testing the machine.
    deadline = time.time() + 10.0
    while len(applied) < 2 and time.time() < deadline:
        time.sleep(0.01)
    wd.stop()
    assert applied == [100, 200]     # once per launch, re-armed after exit
    assert any("watchdog on" in e for e in events)
    # Jitter-evidence markers: one epoch timestamp per applied tuning, and the
    # timestamp surfaces in the event payload for downstream correlation.
    assert len(wd.tune_times) == 2
    now = time.time()
    assert all(now - 60 < t <= now for t in wd.tune_times)
    assert any("tuned at epoch" in e for e in events)


def test_watchdog_reports_missing_psutil(monkeypatch):
    """A core install (no psutil) must say why it can't tune, not poll
    silently forever."""
    import sys
    import time

    monkeypatch.setitem(sys.modules, "psutil", None)   # forces ImportError
    events: list[str] = []
    wd = GameWatchdog(on_event=events.append, poll_interval=0.02)
    wd.start()
    deadline = time.time() + 2.0
    while wd.running and time.time() < deadline:
        time.sleep(0.01)
    assert not wd.running
    assert any("psutil missing" in e for e in events)


# ------------------------------------------------------- power plan checkup
def test_power_probe_matches_duplicated_scheme_by_name(monkeypatch):
    """Duplicated Ultimate schemes get a fresh random GUID; the probe must
    recognize them by name, not only by the canonical GUIDs."""
    from kovadapt.optimize import checkup as ck

    monkeypatch.setattr(ck, "WINDOWS", True)
    # The GUID note lives in the REAL HKCU — never let tests read it.
    monkeypatch.setattr(ck, "_stored_scheme_guid", lambda: None)
    active = ("Power Scheme GUID: 11111111-2222-3333-4444-555555555555  "
              "(Ultimate Performance)")
    monkeypatch.setattr(ck, "_run", lambda cmd, timeout=15.0: active)
    res = ck.SystemCheckup("", HardwareInfo())._c_power()
    assert res.status == "ok"
    assert "Ultimate" in res.detail


def test_power_fix_reuses_existing_ultimate_copy(monkeypatch):
    """The fix must activate an existing Ultimate copy instead of minting a
    new duplicate on every click."""
    from kovadapt.optimize import checkup as ck

    # The GUID note lives in the REAL HKCU — capture writes, never touch it
    # (an early version of this test overwrote the developer's actual value).
    stored: list[str] = []
    monkeypatch.setattr(ck, "_stored_scheme_guid", lambda: None)
    monkeypatch.setattr(ck, "_store_scheme_guid", stored.append)

    ult = "aaaaaaaa-bbbb-cccc-dddd-eeeeffff0000"
    listing = (
        "Existing Power Schemes (* Active)\n"
        "-----------------------------------\n"
        "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced) *\n"
        f"Power Scheme GUID: {ult}  (Ultimate Performance)\n"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, timeout=15.0):
        calls.append(cmd)
        if cmd[1] == "/list":
            return listing
        if cmd[1] == "/getactivescheme":
            return f"Power Scheme GUID: {ult}  (Ultimate Performance)"
        return ""

    monkeypatch.setattr(ck, "_run", fake_run)
    msg = ck.SystemCheckup("", HardwareInfo())._f_power()
    assert "Ultimate" in msg
    assert not any("-duplicatescheme" in c for call in calls for c in call)
    assert ["powercfg", "/setactive", ult] in calls
    assert stored == [ult]      # the note is recorded, via the mock only


# ------------------------------------------------------ core parking checkup
POWERCFG_CPMINCORES = """\
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)
  Subgroup GUID: 54533251-82be-4824-96c1-47b60b740d00  (Processor power management)
    Power Setting GUID: 0cc5b647-c1df-4637-891a-dec35c318583  (Processor performance core parking min cores)
      Minimum Possible Setting: 0x00000000
      Maximum Possible Setting: 0x00000064
      Possible Settings increment: 0x00000001
      Possible Settings units: %
    Current AC Power Setting Index: {ac}
    Current DC Power Setting Index: 0x00000032
"""


def test_powercfg_ac_index_parsing():
    from kovadapt.optimize.checkup import parse_powercfg_ac_index

    assert parse_powercfg_ac_index(POWERCFG_CPMINCORES.format(ac="0x00000064")) == 100
    assert parse_powercfg_ac_index(POWERCFG_CPMINCORES.format(ac="0x0000000a")) == 10
    # The DC line alone must not satisfy the AC pattern.
    assert parse_powercfg_ac_index("Current DC Power Setting Index: 0x00000064") is None
    assert parse_powercfg_ac_index(
        "The power scheme, subgroup or setting specified does not exist.") is None
    assert parse_powercfg_ac_index("") is None


def test_core_parking_status_mapping(monkeypatch):
    from kovadapt.optimize import checkup as ck

    monkeypatch.setattr(ck, "WINDOWS", True)
    checkup = ck.SystemCheckup("", HardwareInfo())

    monkeypatch.setattr(ck, "_run",
                        lambda cmd, timeout=15.0: POWERCFG_CPMINCORES.format(ac="0x00000064"))
    assert checkup._c_parking().status == "ok"

    monkeypatch.setattr(ck, "_run",
                        lambda cmd, timeout=15.0: POWERCFG_CPMINCORES.format(ac="0x00000032"))
    res = checkup._c_parking()
    assert res.status == "warn" and not res.can_fix     # may need admin: never auto
    assert "powercfg /setacvalueindex" in res.detail    # exact command surfaced
    assert "50" in res.detail

    monkeypatch.setattr(ck, "_run", lambda cmd, timeout=15.0:
                        "The power scheme, subgroup or setting specified does not exist.")
    assert checkup._c_parking().status == "unknown"


# ---------------------------------------------------- gamedvr / game mode
def test_gamedvr_registry_mapping(monkeypatch):
    from kovadapt.optimize import checkup as ck

    monkeypatch.setattr(ck, "WINDOWS", True)
    checkup = ck.SystemCheckup("", HardwareInfo())

    values = {"AppCaptureEnabled": 0, "GameDVR_Enabled": 0}
    monkeypatch.setattr(ck, "_read_hkcu_dword", lambda path, name: values[name])
    assert checkup._c_gamedvr().status == "ok"

    values["GameDVR_Enabled"] = 1
    res = checkup._c_gamedvr()
    assert res.status == "warn" and res.can_fix and res.safe
    assert "GameDVR_Enabled=1" in res.detail        # prior value shown to restore

    monkeypatch.setattr(ck, "_read_hkcu_dword", lambda path, name: None)
    res = checkup._c_gamedvr()                      # absent = Windows default = on
    assert res.status == "warn"
    assert "default on" in res.detail


def test_game_mode_registry_mapping(monkeypatch):
    from kovadapt.optimize import checkup as ck

    monkeypatch.setattr(ck, "WINDOWS", True)
    checkup = ck.SystemCheckup("", HardwareInfo())

    monkeypatch.setattr(ck, "_read_hkcu_dword", lambda path, name: None)
    res = checkup._c_game_mode()
    assert res.status == "info" and not res.can_fix  # absent = default ON = fine

    monkeypatch.setattr(ck, "_read_hkcu_dword", lambda path, name: 1)
    assert checkup._c_game_mode().status == "info"

    monkeypatch.setattr(ck, "_read_hkcu_dword", lambda path, name: 0)
    res = checkup._c_game_mode()
    assert res.status == "warn" and res.can_fix and res.safe


# ------------------------------------------------------------ hags live
def test_wddm27_caps_decoding():
    from kovadapt.optimize.hardware import decode_wddm27_caps

    assert decode_wddm27_caps(0x0) == (False, False)
    assert decode_wddm27_caps(0x1) == (True, False)      # supported, running off
    assert decode_wddm27_caps(0x3) == (True, True)       # supported, running on
    assert decode_wddm27_caps(0x7) == (True, True)       # EnabledByDefault bit ignored
    assert decode_wddm27_caps(0xFFFFFFF9) == (True, False)  # reserved bits ignored


def test_hags_live_vs_registry_mapping(monkeypatch):
    from kovadapt.optimize import checkup as ck

    monkeypatch.setattr(ck, "WINDOWS", True)
    checkup = ck.SystemCheckup("", HardwareInfo())

    monkeypatch.setattr(ck, "hags_live_state", lambda: (True, True))
    monkeypatch.setattr(ck, "_hags_registry_mode", lambda: 2)
    assert checkup._c_hags_live().status == "ok"

    monkeypatch.setattr(ck, "_hags_registry_mode", lambda: 1)  # intent off, live on
    res = checkup._c_hags_live()
    assert res.status == "warn"
    assert "reboot" in res.detail

    monkeypatch.setattr(ck, "_hags_registry_mode", lambda: None)  # no override set
    assert checkup._c_hags_live().status == "info"

    monkeypatch.setattr(ck, "hags_live_state", lambda: (False, False))
    assert checkup._c_hags_live().status == "info"     # driver has no hw scheduling

    monkeypatch.setattr(ck, "hags_live_state", lambda: (None, None))
    assert checkup._c_hags_live().status == "unknown"  # query failed


# --------------------------------------------------------------- timer
def test_timer_resolution_status_mapping(monkeypatch):
    from kovadapt.optimize import checkup as ck

    monkeypatch.setattr(ck, "WINDOWS", True)
    monkeypatch.setattr(ck, "_query_timer_resolution", lambda: (15.63, 0.5))
    monkeypatch.setattr(ck, "_game_running", lambda: True)

    win10 = ck.SystemCheckup("", HardwareInfo(is_windows_11=False))
    res = win10._c_timer()
    assert res.status == "warn" and not res.can_fix  # owned by the requesting app

    monkeypatch.setattr(ck, "_game_running", lambda: False)
    assert win10._c_timer().status == "info"         # coarse but game not running

    monkeypatch.setattr(ck, "_game_running", lambda: None)  # no psutil: never warn
    assert win10._c_timer().status == "info"

    win11 = ck.SystemCheckup("", HardwareInfo(is_windows_11=True))
    monkeypatch.setattr(ck, "_game_running", lambda: True)
    res = win11._c_timer()
    assert res.status == "info"                      # win11 handles it per-process
    assert "per-process" in res.detail

    monkeypatch.setattr(ck, "_query_timer_resolution", lambda: None)
    assert win11._c_timer().status == "unknown"


def test_find_game_process_prefers_the_renderer_over_the_steam_stub():
    """KovaaK's is a UE4 two-exe pair and BOTH match the name substring.

    Steam launches a ~512 KB FPSAimTrainer.exe bootstrap which spawns the
    real ~88 MB FPSAimTrainer-Win64-Shipping.exe. process_iter yields
    roughly in PID order, so 'first match wins' always picked the stub —
    priority and affinity landed on a process that renders nothing and
    handles no input, and the watchdog reported success having tuned
    nothing.
    """
    from kovadapt.optimize.watchdog import find_game_process

    class FakeProc:
        def __init__(self, pid, name):
            self.pid = pid
            self.info = {"name": name}

    class FakePsutil:
        def __init__(self, procs):
            self._procs = procs

        def process_iter(self, _attrs):
            return iter(self._procs)

    stub = FakeProc(1000, "FPSAimTrainer.exe")          # starts first
    renderer = FakeProc(2000, "FPSAimTrainer-Win64-Shipping.exe")

    assert find_game_process(FakePsutil([stub, renderer])) is renderer
    # order must not matter
    assert find_game_process(FakePsutil([renderer, stub])) is renderer
    # a rename degrades to the old behaviour, not to "game not running"
    assert find_game_process(FakePsutil([stub])) is stub
    assert find_game_process(FakePsutil([FakeProc(3, "chrome.exe")])) is None
    assert find_game_process(FakePsutil([])) is None
