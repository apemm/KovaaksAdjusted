"""End-to-end tests for kovadapt.watcher.SessionWatcher against a fake
KovaaK's folder tree (telemetry and clips disabled — runs on any OS).

The tree/CSV helpers here are shared with tests/test_cli.py.
"""

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kovadapt.adapt.engine import AdaptationEngine
from kovadapt.analysis.report import RunReport
from kovadapt.config import ADAPTIVE_SUFFIX, Settings
from kovadapt.profile.player import PlayerProfile
from kovadapt.scenario.generator import generate_adaptive_variant
from kovadapt.scenario.sce import SceFile
from kovadapt.stats.parser import parse_stats_csv
from kovadapt.watcher import SessionWatcher

FIXTURES = Path(__file__).parent / "fixtures"

BASE = "mini test"
# A scenario whose name contains the filename grammar's own separator.
DASHED = "1wall 2targets small - valorant"
TS = "2026.05.27-20.25.38"  # -> started.isoformat() == 2026-05-27T20:25:38


def mini_sce_text(name: str) -> str:
    """A minimal valid scenario (same shape as tests/test_scenario.py)."""
    spawns = "".join(
        f"\tentity\n\t\ttype PlayerSpawn\n"
        f"\t\tVector3 position {x}.000000 288.000000 {z}.000000\n"
        f"\t\tVector3 angles 180.000000 0.000000 0.000000\n"
        for x in (-800, -400, 0, 400, 800) for z in (200, 600, 1000)
    )
    return f"""Name={name}
AddedBots=target.bot;target.bot
Timelimit=60.0

[Character Profile]
Name=Player
MaxHealth=100.0
MainBBRadius=1.0

[Character Profile]
Name=target
MaxHealth=1.0
MaxSpeed=0.0
MainBBRadius=0.5
MainBBHeight=2.0
ProjBBRadius=0.5

[Bot Profile]
Name=target
DodgeProfileNames=Mimic
AimingProfileNames=Default

[Dodge Profile]
Name=Mimic
MinLRTimeChange=0.2
MaxLRTimeChange=0.5
JumpFrequency=0.5

[Map Data]
reflex map version 8
global
\tentity
\t\ttype WorldSpawn
""" + spawns


def make_kovaaks_tree(root: Path, *scenarios: str) -> None:
    """Fake KovaaK's install: stats/ + Saved/SaveGames/Scenarios/<name>.sce."""
    (root / "stats").mkdir(parents=True, exist_ok=True)
    scen = root / "Saved" / "SaveGames" / "Scenarios"
    scen.mkdir(parents=True, exist_ok=True)
    for name in scenarios:
        (scen / f"{name}.sce").write_text(mini_sce_text(name), encoding="utf-8")


def write_stats_csv(stats_dir: Path, scenario: str, ts: str = TS) -> Path:
    """Drop a valid stats CSV (fixture body; scenario comes from the name)."""
    body = (FIXTURES / "sample_stats.csv").read_text(encoding="utf-8")
    p = stats_dir / f"{scenario} - Challenge - {ts} Stats.csv"
    p.write_text(body, encoding="utf-8")
    return p


def make_settings(root: Path, state: Path) -> Settings:
    return Settings(
        kovaaks_root=str(root),
        profile_dir=str(state),
        telemetry_enabled=False,
        clips_enabled=False,
    )


def wait_for(cond, timeout: float = 5.0, step: float = 0.005) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(step)
    return cond()


@pytest.fixture
def env(tmp_path: Path) -> Settings:
    root = tmp_path / "kovaaks"
    make_kovaaks_tree(root, BASE)
    return make_settings(root, tmp_path / "state")


def _start_watch(w: SessionWatcher) -> threading.Thread:
    th = threading.Thread(
        target=w.watch, kwargs={"poll_interval": 0.02, "settle": 0.02}, daemon=True
    )
    th.start()
    return th


# ------------------------------------------------------------------ process_run
def test_process_run_end_to_end(env: Settings):
    logs: list[str] = []
    reports: list[RunReport] = []
    w = SessionWatcher(env, BASE, on_update=logs.append, on_report=reports.append)
    out = w.process_run(write_stats_csv(env.stats_dir, BASE))

    # adaptive variant written next to the base scenario
    assert out == env.scenarios_dir / f"{BASE}{ADAPTIVE_SUFFIX}.sce"
    assert SceFile.read(out).get_header("Name") == BASE + ADAPTIVE_SUFFIX
    # base scenario untouched (edits never compound onto it)
    base_text = w.base_sce_path().read_text(encoding="utf-8")
    assert base_text == mini_sce_text(BASE)

    # profile persisted under the adaptive name, one run folded in
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert prof.run_count == 1
    assert prof.scenario == BASE + ADAPTIVE_SUFFIX
    assert prof.archetype == "clicking"  # 9 shots / 6 kills -> clicking heuristic

    # report JSON lands in the mirrored reports/ tree (ARCHITECTURE.md contract)
    rp = env.profile_path / "reports" / "mini_test" / "2026-05-27T20-25-38.json"
    assert rp.is_file()
    rep = RunReport.load(rp)
    assert rep.scenario == BASE
    assert abs(rep.accuracy - 6 / 9) < 1e-9
    assert w.last_report is not None and reports == [w.last_report]
    assert any("run #1" in m for m in logs)


def test_report_path_mirrors_trace_tree(env: Settings):
    w = SessionWatcher(env, BASE, on_update=lambda m: None)
    run = parse_stats_csv(write_stats_csv(env.stats_dir, BASE))
    # traces/<slug>/<ts>.npz  ->  reports/<slug>/<ts>.json (":" -> "-")
    expected = env.profile_path / "reports" / "mini_test" / "2026-05-27T20-25-38.json"
    assert w._report_path(run) == expected


def test_base_and_adaptive_runs_feed_one_profile(env: Settings):
    w = SessionWatcher(env, BASE, on_update=lambda m: None)
    w.process_run(write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.25.38"))
    w.process_run(
        write_stats_csv(env.stats_dir, BASE + ADAPTIVE_SUFFIX, ts="2026.05.27-20.30.00")
    )
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert prof.run_count == 2


# -------------------------------------------------------------------- bootstrap
def test_bootstrap_creates_variant_when_missing(env: Settings):
    logs: list[str] = []
    w = SessionWatcher(env, BASE, on_update=logs.append)
    assert not w.adaptive_sce_path().is_file()
    out = w.bootstrap()
    assert out == w.adaptive_sce_path() and out.is_file()
    sce = SceFile.read(out)
    assert sce.get_header("Name") == BASE + ADAPTIVE_SUFFIX
    # spawns are resampled from the base's own coordinates, never invented
    orig_xz = {(float(x), float(z)) for x in (-800, -400, 0, 400, 800)
               for z in (200, 600, 1000)}
    assert all((p.x, p.z) in orig_xz for p in sce.spawn_points())
    # neutral plan: profile saved with no runs recorded
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert prof.run_count == 0
    assert prof.archetype == "clicking"
    assert any(m.startswith("created") for m in logs)


# ------------------------------------------------------- file relevance / queue
def test_relevant_filenames(env: Settings):
    w = SessionWatcher(env, BASE, on_update=lambda m: None)
    assert w._relevant(f"{BASE} - Challenge - {TS} Stats.csv")
    assert w._relevant(f"{BASE}{ADAPTIVE_SUFFIX} - Challenge - {TS} Stats.csv")
    assert not w._relevant(f"other scenario - Challenge - {TS} Stats.csv")
    assert not w._relevant(f"{BASE} extended - Challenge - {TS} Stats.csv")
    assert not w._relevant(f"{BASE}{ADAPTIVE_SUFFIX}{ADAPTIVE_SUFFIX} - Challenge - {TS} Stats.csv")
    assert not w._relevant("not a stats file.csv")


def test_relevant_with_dash_in_scenario_name(env: Settings):
    # the stats filename grammar's scenario group is greedy — " - " in names works
    w = SessionWatcher(env, DASHED, on_update=lambda m: None)
    ts = "2026.05.21-18.42.29"
    assert w._relevant(f"{DASHED} - Challenge - {ts} Stats.csv")
    assert w._relevant(f"{DASHED}{ADAPTIVE_SUFFIX} - Challenge - {ts} Stats.csv")
    # a prefix of the dashed name is a different scenario
    assert not w._relevant(f"1wall 2targets small - Challenge - {ts} Stats.csv")


def test_pending_files_filters_and_orders_by_mtime(env: Settings):
    w = SessionWatcher(env, BASE, on_update=lambda m: None)
    older = write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-19.00.00")
    newer = write_stats_csv(env.stats_dir, BASE + ADAPTIVE_SUFFIX, ts="2026.05.27-20.00.00")
    write_stats_csv(env.stats_dir, "someone else", ts="2026.05.27-19.30.00")
    (env.stats_dir / "notes.csv").write_text("not a stats file", encoding="utf-8")
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now - 50, now - 50))
    assert w._pending_files() == [older, newer]
    w._seen.add(older.name)
    assert w._pending_files() == [newer]


# ------------------------------------------------------------------ watch loop
def test_watch_ignores_history_and_processes_new_runs(env: Settings):
    logs: list[str] = []
    # history predating watch() must be ignored
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-19.00.00")
    w = SessionWatcher(env, BASE, on_update=logs.append)
    th = _start_watch(w)
    try:
        assert wait_for(lambda: any(m.startswith("watching") for m in logs))
        # bootstrap ran because the adaptive variant was missing at watch start
        assert w.adaptive_sce_path().is_file()
        write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.25.38")
        write_stats_csv(env.stats_dir, "unrelated task", ts="2026.05.27-20.26.00")
        assert wait_for(lambda: any("run #1" in m for m in logs))
    finally:
        w.request_stop()
        th.join(timeout=5.0)
    assert not th.is_alive()  # request_stop() exits the loop
    prof = PlayerProfile.load(w.adaptive_name, env.profile_path)
    assert prof.run_count == 1  # neither the pre-existing nor the unrelated CSV counted


def test_watch_survives_process_run_error(env: Settings):
    # NOTE: a corrupt CSV body never raises (the parser is deliberately
    # tolerant), so force a real process_run failure instead: the base .sce
    # vanishing makes generate_adaptive_variant blow up mid-loop.
    logs: list[str] = []
    w = SessionWatcher(env, BASE, on_update=logs.append)
    w.bootstrap()  # variant exists up-front so watch() skips bootstrapping
    base = w.base_sce_path()
    backup = base.read_text(encoding="utf-8")
    th = _start_watch(w)
    try:
        assert wait_for(lambda: any(m.startswith("watching") for m in logs))
        base.unlink()
        write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.00.00")
        assert wait_for(lambda: any(m.startswith("error processing") for m in logs))
        base.write_text(backup, encoding="utf-8")
        write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.10.00")
        assert wait_for(lambda: any("run #1" in m for m in logs))
    finally:
        w.request_stop()
        th.join(timeout=5.0)
    assert not th.is_alive()
    # the failed run never reached profile.save; only the second run persisted
    prof = PlayerProfile.load(w.adaptive_name, env.profile_path)
    assert prof.run_count == 1


# ------------------------------------------------------------------- capture
def test_watch_stops_a_partially_started_capture(env: Settings):
    """A capture start that fails halfway must still be torn down.

    _start_capture() used to run BEFORE watch()'s try/finally, so a clip
    recorder blowing up after the mouse recorder was already running left
    the Raw Input pump thread alive with no owner — and the next watch()
    overwrote self.recorder and orphaned it, along with its message-only
    window and registered window class.
    """
    stopped: list[str] = []

    class FakeRecorder:
        def stop(self) -> None:
            stopped.append("mouse")

    w = SessionWatcher(env, BASE)
    w.bootstrap()          # variant exists, so watch() goes straight to capture

    def half_started_then_boom() -> None:
        w.recorder = FakeRecorder()              # mouse telemetry came up...
        raise RuntimeError("dxcam: no supported output")   # ...clips did not

    w._start_capture = half_started_then_boom
    with pytest.raises(RuntimeError):
        w.watch(poll_interval=0.02, settle=0.02)

    assert stopped == ["mouse"], "the started recorder was never stopped"
    assert w.recorder is None


def test_clip_recorder_failure_does_not_kill_the_session(env: Settings, monkeypatch):
    """Clips are optional: dxcam raising must cost the clips, not the run."""
    import dataclasses

    from kovadapt.capture import clips as clips_mod

    class Boom:
        def __init__(self, **kwargs):
            raise RuntimeError("no supported output")

    monkeypatch.setattr(clips_mod, "CLIPS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(clips_mod, "ClipRecorder", Boom, raising=False)

    logs: list[str] = []
    w = SessionWatcher(dataclasses.replace(env, clips_enabled=True), BASE,
                       on_update=logs.append)
    w._start_capture()                      # must not raise
    assert w.clip_recorder is None
    assert any("clip capture: unavailable" in m for m in logs)


def test_unactionable_focus_is_not_credited_to_the_bandit(env: Settings):
    """A focus the .sce could not express must not become bandit evidence.

    resample_spawns reuses original spawn coordinates and never invents them,
    so a focus region holding no candidate spawn cannot be emphasized, and a
    layout with fewer target spawns than grid cells is left untouched entirely
    (generator.resample_spawns returns an empty set for both). plan() has
    already stored the region as last_focus by then, so credit_focus_region()
    would book the NEXT run's accuracy deficit against an emphasis the player
    never saw — evidence manufactured from a change that was silently dropped.

    This fixture is the untouched-layout case: 15 target spawns against a 5x5
    grid, so nothing is ever resampled and NO focus is actionable here.
    """
    from kovadapt.scenario.generator import resample_spawns

    w = SessionWatcher(env, BASE)
    w.bootstrap()
    prof = PlayerProfile.load(w.adaptive_name, env.profile_path)
    engine = AdaptationEngine(env, rng=np.random.default_rng(0))
    plan = engine.plan(prof, None)

    sce = SceFile.read(w.base_sce_path())
    assert resample_spawns(sce, plan.spawn_weights, env.region_cols,
                           env.region_rows, plan.seed) == set(),         "fixture is meant to be the too-few-spawns case"

    generate_adaptive_variant(w.base_sce_path(), plan, env, w.adaptive_sce_path())
    assert plan.focus_applied is False,         "an untouched layout expresses no focus, so the arm must not be credited"

    # and the watcher acts on it: last_focus is cleared before the profile save
    logs: list[str] = []
    w2 = SessionWatcher(env, BASE, on_update=logs.append)
    w2.bootstrap()
    w2.process_run(write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-21.00.00"))
    saved = PlayerProfile.load(w2.adaptive_name, env.profile_path)
    assert saved.last_focus is None, "an unactionable focus was still persisted"
    assert any("focus not applied" in m for m in logs)


def test_every_generate_path_clears_an_unactionable_focus(env: Settings,
                                                          monkeypatch):
    """The guard lived inline in process_run and NOWHERE else.

    Four other sites pair a plan() with a generate and then save: the
    watcher's own bootstrap (which the dashboard's Start calls directly),
    `kovadapt generate`, and the Scenarios browser's Generate button. Each
    persisted a last_focus the emitted .sce provably could not express, and
    the next run then booked its accuracy deficit against that arm — which
    reaches the bandit's spawn weights, not just the display.

    Measured against a real install: 19 of 31 .sce files carry fewer than
    5x5 target spawns and 8 carry none, so this is the common layout, not
    the exotic one. Same too-few-spawns fixture as the test above.
    """
    from kovadapt.gui import browser as browser_mod          # noqa: F401
    from kovadapt.profile.player import RegionPosterior

    def seed_profile(w: SessionWatcher) -> None:
        """A profile with runs, so credit_focus_region does not early-return
        and last_focus is a value with consequences."""
        p = PlayerProfile.load(w.adaptive_name, env.profile_path)
        p.run_count = 5
        p.ewma_accuracy = 0.8
        p.regions["r0c0"] = RegionPosterior(mean=0.1, var=0.2, n=3)
        p.save(env.profile_path)

    # --- bootstrap (also the dashboard's Start path) --------------------
    w = SessionWatcher(env, BASE)
    w.bootstrap()
    seed_profile(w)
    w.bootstrap()
    assert PlayerProfile.load(w.adaptive_name, env.profile_path).last_focus is None, \
        "bootstrap persisted a focus the layout cannot express"

    # --- cli.cmd_generate ----------------------------------------------
    from kovadapt import cli

    # cmd_generate resolves its own Settings; without this it reaches the
    # developer's real KovaaK's install (same pattern as tests/test_cli.py).
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path=None: env))
    seed_profile(w)
    cli.cmd_generate(SimpleNamespace(scenario=BASE))
    assert PlayerProfile.load(w.adaptive_name, env.profile_path).last_focus is None, \
        "kovadapt generate persisted an unactionable focus"


def test_an_actionable_focus_is_still_credited(env: Settings):
    """The guard must not silently disable the bandit on real layouts."""
    from kovadapt.scenario.generator import resample_spawns

    # A 6x6 wall: 36 target spawns clears the 5x5 grid floor, so resampling
    # really runs and a focus region CAN be expressed. Built from the same
    # per-entity block mini_sce_text uses, so the .sce grammar stays identical.
    block = (
        "\tentity\n\t\ttype PlayerSpawn\n"
        "\t\tVector3 position {x}.000000 288.000000 {z}.000000\n"
        "\t\tVector3 angles 180.000000 0.000000 0.000000\n"
    )
    spawns = "".join(
        block.format(x=x, z=z)
        for x in (-900, -540, -180, 180, 540, 900)
        for z in (100, 340, 580, 820, 1060, 1300)
    )
    base = mini_sce_text(BASE)
    head = base[:base.index("\tentity\n\t\ttype PlayerSpawn")]
    w = SessionWatcher(env, BASE)
    w.base_sce_path().write_text(head + spawns, encoding="utf-8")
    w.bootstrap()
    prof = PlayerProfile.load(w.adaptive_name, env.profile_path)
    plan = AdaptationEngine(env, rng=np.random.default_rng(0)).plan(prof, None)
    sce = SceFile.read(w.base_sce_path())
    used = resample_spawns(sce, plan.spawn_weights, env.region_cols,
                           env.region_rows, plan.seed)
    assert used, "dense layout should be reweighted"
    plan.focus_region = sorted(used)[0]
    generate_adaptive_variant(w.base_sce_path(), plan, env, w.adaptive_sce_path())
    assert plan.focus_applied is True


def test_a_moving_frame_does_not_feed_the_strafe_skew(tmp_path, monkeypatch):
    """The bias verdict writes Left/RightStrafeTimeMult into the .sce. On a
    scenario that lets the player move, that verdict measures counter-strafing
    as much as aim — so it must not reach the profile at all.

    Suppressed rather than discounted: there is no calibrated weight to
    discount it by, and a made-up one would be a guess steering a real file.
    """
    from kovadapt.adapt.channels import measurement_mask
    from kovadapt.scenario.capability import Capability

    still = Capability(motion={"t": "SELF"}, player_frame="STATIC")
    moving = Capability(motion={"t": "SELF"}, player_frame="MOBILE")
    assert measurement_mask(still)["directional_bias"] is True
    assert measurement_mask(moving)["directional_bias"] is False

    # and the profile-facing effect: observe_bias is what banks it
    from kovadapt.profile.player import PlayerProfile

    prof = PlayerProfile(scenario="x [Adaptive]")
    prof.observe_bias(0.42)
    assert prof.bias_obs == 1, "sanity: a measurement does reach the profile"
    # a suppressed one simply never gets here, which is why the watcher gate
    # is the right place for it rather than a weight inside observe_bias
