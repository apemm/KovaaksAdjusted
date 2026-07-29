"""Restyle + off-thread-work regressions (offscreen QPA).

Two behaviours are pinned here:

* a theme/accent switch rebuilds the Scenarios table, and must not cost the
  user their selection (which is what enables Play / Start adapting /
  Generate variant);
* the optimizer's fixes run off the UI thread — the power-plan fix chains
  four powercfg subprocess calls, each with a 15 s timeout.

Nothing in this file runs a real system fix or probe: the checkup layer is
stubbed wholesale, so no powercfg, no registry write and no game process is
ever touched.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import Settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Anything reaching Path.home() here must land in tmp_path — the suite
    once corrupted the developer's real settings.json."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def settings(tmp_path):
    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    scen = root / "Saved" / "SaveGames" / "Scenarios"
    scen.mkdir(parents=True)
    for name in ("1w6ts reload", "Bounce 180 Tracking", "Pasu Voltaic"):
        (scen / f"{name}.sce").write_text("Name=x\n", encoding="utf-8")
    return Settings(
        kovaaks_root=str(root),
        profile_dir=str(tmp_path / "prof"),
        telemetry_enabled=False,
        onboarding_done=True,
    )


def _wait_until(qapp, pred, timeout_ms: int = 5000) -> bool:
    from PySide6.QtTest import QTest

    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if pred():
            return True
        QTest.qWait(10)
    qapp.processEvents()
    return bool(pred())


# ------------------------------------------------------- scenario browser
def test_theme_switch_keeps_the_selected_scenario(qapp, settings):
    """restyle() is what MainWindow._restyle calls on every theme/accent
    change; it rebuilds the table, and the selection has to survive it."""
    from kovadapt.gui import theme
    from kovadapt.gui.browser import ScenarioBrowser

    b = ScenarioBrowser(settings)
    assert b.table.rowCount() == 3
    b.table.selectRow(1)
    picked = b.selected()
    assert picked
    assert b.play_btn.isEnabled()

    b.restyle(theme.build_palette(dark=False, accent="ember"))

    assert b.selected() == picked
    for btn in (b.play_btn, b.watch_btn, b.gen_btn):
        assert btn.isEnabled(), "theme switch disabled the action buttons"
    assert b.detail.text() != "select a scenario"


def test_refresh_keeps_the_selected_scenario(qapp, settings):
    """Same rebuild path from the Refresh button / a sort change."""
    from kovadapt.gui.browser import ScenarioBrowser

    b = ScenarioBrowser(settings)
    b.table.selectRow(2)
    picked = b.selected()
    b.refresh()
    assert b.selected() == picked

    b.sort_by.setCurrentText("Most runs")   # re-sorts, same rows
    assert b.selected() == picked


def test_generate_outcome_survives_the_refresh(qapp, settings):
    """_generate() refreshes the table afterwards; the outcome line must not
    be overwritten by the reselected row's stats."""
    from kovadapt.gui.browser import ScenarioBrowser

    b = ScenarioBrowser(settings)
    b.table.selectRow(0)
    name = b.selected()
    (settings.scenarios_dir / f"{name}.sce").unlink()   # force the error path
    b._generate()
    assert b.detail.text().startswith("could not generate")


# ------------------------------------------------------- optimizer window
class _FakeCheckup:
    """Stands in for SystemCheckup: no powercfg, no registry, no process
    probing. `fix()` records the thread it ran on and blocks on `gate`."""

    def __init__(self, kovaaks_root, hw) -> None:
        self.kovaaks_root = kovaaks_root
        self.gate = threading.Event()
        self.gate.set()
        self.fixed: list[tuple[str, int]] = []

    def run_all(self):
        from kovadapt.optimize.checkup import CheckResult

        return [
            CheckResult("mouse_accel", "Mouse acceleration", "warn", "on",
                        can_fix=True, safe=True, fix_label="Turn it off"),
            CheckResult("gamedvr", "Game DVR", "warn", "on",
                        can_fix=True, safe=True, fix_label="Turn it off"),
            CheckResult("power_plan", "Power plan", "warn", "balanced",
                        can_fix=True, safe=False, fix_label="Activate"),
            CheckResult("chromium", "Background apps", "ok", "none"),
        ]

    def fix(self, check_id: str) -> str:
        self.fixed.append((check_id, threading.get_ident()))
        self.gate.wait(3.0)
        return f"{check_id} applied"


@pytest.fixture()
def opt(qapp, settings, monkeypatch):
    from kovadapt.gui import optimizer_window as ow
    from kovadapt.optimize.hardware import HardwareInfo

    monkeypatch.setattr(ow, "detect_hardware", lambda: HardwareInfo())
    monkeypatch.setattr(ow, "SystemCheckup", _FakeCheckup)
    # Read-only on the real HKCU, but stub it anyway so this file cannot
    # reach the machine's Run key by any path.
    monkeypatch.setattr(ow, "startup_registered", lambda: False)
    monkeypatch.setattr(ow, "register_startup", lambda: "stubbed")
    monkeypatch.setattr(ow, "unregister_startup", lambda: "stubbed")

    win = ow.OptimizerWindow(settings)
    assert _wait_until(qapp, lambda: win.checkup is not None), "scan never landed"
    yield win
    win.checkup.gate.set()
    win.shutdown()


def _row(win, check_id):
    from kovadapt.gui.optimizer_window import _CheckRow

    for i in range(win.rows_layout.count()):
        w = win.rows_layout.itemAt(i).widget()
        if isinstance(w, _CheckRow) and w.result.check_id == check_id:
            return w
    raise AssertionError(f"no check row for {check_id}")


def test_fix_button_does_not_block_the_ui_thread(qapp, opt):
    row = _row(opt, "mouse_accel")
    opt.checkup.gate.clear()          # the fix now blocks until we release it

    t0 = time.perf_counter()
    row.fix_btn.click()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"click() blocked the UI thread for {elapsed:.2f}s"

    assert _wait_until(qapp, lambda: opt.checkup.fixed)
    check_id, ident = opt.checkup.fixed[0]
    assert check_id == "mouse_accel"
    assert ident != threading.get_ident(), "fix ran on the UI thread"
    # Nothing else is clickable while a batch runs, or the click is a no-op.
    assert not _row(opt, "power_plan").fix_btn.isEnabled()
    assert not opt.scan_btn.isEnabled()

    opt.checkup.gate.set()
    assert _wait_until(qapp, lambda: row.detail.text() == "mouse_accel applied")
    assert not row.fix_btn.isEnabled()
    assert _wait_until(qapp, lambda: _row(opt, "power_plan").fix_btn.isEnabled())
    assert opt.scan_btn.isEnabled()


def test_fix_all_safe_runs_only_safe_items_off_thread(qapp, opt):
    opt.fix_safe_btn.click()
    assert _wait_until(qapp, lambda: len(opt.checkup.fixed) == 2)
    # finished -> _on_fixes_done is a QUEUED slot, so the thread stopping is
    # not the barrier — the UI going live again is.
    assert _wait_until(qapp, lambda: opt.scan_btn.isEnabled()
                       and not opt._fix.isRunning())
    assert [cid for cid, _ in opt.checkup.fixed] == ["mouse_accel", "gamedvr"]
    assert all(ident != threading.get_ident() for _, ident in opt.checkup.fixed)
    assert _row(opt, "power_plan").fix_btn.isEnabled()   # unsafe: untouched
    assert not opt.fix_safe_btn.isEnabled()              # nothing safe left


def test_shutdown_waits_for_a_running_fix(qapp, opt):
    """A QThread destroyed while running is a fatal abort in Qt 6."""
    row = _row(opt, "mouse_accel")
    opt.checkup.gate.clear()
    row.fix_btn.click()
    assert _wait_until(qapp, lambda: opt.checkup.fixed)
    assert opt._fix.isRunning()

    threading.Timer(0.2, opt.checkup.gate.set).start()
    opt.shutdown()
    assert not opt._fix.isRunning()
    assert opt._scan is None or not opt._scan.isRunning()
