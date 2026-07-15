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
    assert len(results) == 7
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

    deadline = time.time() + 2.0
    while len(applied) < 2 and time.time() < deadline:
        time.sleep(0.01)
    wd.stop()
    assert applied == [100, 200]     # once per launch, re-armed after exit
    assert any("watchdog on" in e for e in events)
