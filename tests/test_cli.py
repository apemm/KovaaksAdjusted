"""Contract tests for kovadapt.cli (no Qt, no Windows-only commands).

Settings.load is monkeypatched so every command sees a fake KovaaK's tree
built by the helpers in tests/test_watcher.py.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kovadapt import cli
from kovadapt.config import ADAPTIVE_SUFFIX, Settings
from kovadapt.profile.player import PlayerProfile

from test_watcher import BASE, make_kovaaks_tree, make_settings, write_stats_csv

OTHER = "Other Task"


@pytest.fixture
def env(tmp_path: Path, monkeypatch) -> Settings:
    root = tmp_path / "kovaaks"
    make_kovaaks_tree(root, BASE, OTHER)
    s = make_settings(root, tmp_path / "state")
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path=None: s))
    return s


# ------------------------------------------------------------------- scenarios
def test_scenarios_lists_sce_stems_sorted(env: Settings, capsys):
    cli.main(["scenarios"])
    assert capsys.readouterr().out.splitlines() == [OTHER, BASE]


def test_scenarios_filter_is_case_insensitive(env: Settings, capsys):
    cli.main(["scenarios", "TASK"])
    assert capsys.readouterr().out.splitlines() == [OTHER]
    cli.main(["scenarios", "mini"])
    assert capsys.readouterr().out.splitlines() == [BASE]


# -------------------------------------------------------------------- generate
def test_generate_writes_variant_and_prints_plan(env: Settings, capsys):
    cli.main(["generate", BASE])
    out = capsys.readouterr().out
    target = env.scenarios_dir / f"{BASE}{ADAPTIVE_SUFFIX}.sce"
    assert target.is_file()
    assert "wrote" in out and "plan: scale=" in out
    # a profile is persisted but no runs are recorded by a bare generate
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert prof.run_count == 0
    assert prof.archetype == "clicking"
    # pins current behavior: the CLI list does not hide adaptive variants
    cli.main(["scenarios"])
    assert f"{BASE}{ADAPTIVE_SUFFIX}" in capsys.readouterr().out.splitlines()


# ---------------------------------------------------------------------- status
def test_status_empty_profile(env: Settings, capsys):
    cli.main(["status", BASE])
    assert capsys.readouterr().out.strip() == "no runs recorded yet"


def test_status_after_replay_shows_heatmap_grid(env: Settings, capsys):
    write_stats_csv(env.stats_dir, BASE)
    cli.main(["replay", BASE])
    capsys.readouterr()
    cli.main(["status", BASE])
    out = capsys.readouterr().out
    assert f"scenario:      {BASE}{ADAPTIVE_SUFFIX}" in out
    assert "runs:          1" in out
    assert "accuracy EWMA: 66.7%" in out
    assert "region deficit heatmap" in out
    grid = out.split("heatmap", 1)[1]
    # v0.4 default grid is 5x5: every cell still unobserved
    assert grid.count("--") == 25


def test_status_heatmap_renders_region_evidence(env: Settings, capsys):
    write_stats_csv(env.stats_dir, BASE)
    cli.main(["replay", BASE])
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    prof.region("r1c1").update(0.5)  # posterior mean 0.4 after one update
    prof.save(env.profile_path)
    capsys.readouterr()
    cli.main(["status", BASE])
    assert "+0.40(1)" in capsys.readouterr().out


# ---------------------------------------------------------------------- replay
def test_replay_folds_base_and_adaptive_history(env: Settings, capsys):
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.25.38")
    write_stats_csv(env.stats_dir, BASE + ADAPTIVE_SUFFIX, ts="2026.05.27-20.30.00")
    write_stats_csv(env.stats_dir, OTHER, ts="2026.05.27-20.35.00")  # not ours
    cli.main(["replay", BASE])
    assert "replayed 2 runs" in capsys.readouterr().out
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert prof.run_count == 2
    assert abs(prof.ewma_accuracy - 6 / 9) < 1e-9


def test_replay_merges_base_and_adaptive_chronologically(env: Settings, capsys):
    # Interleaved history (base, adaptive, base): EWMAs, first-run seeding and
    # bandit credit are order-sensitive, so replay must fold runs in true
    # chronological order — not all base runs first, then all adaptive ones.
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.00.00")
    write_stats_csv(env.stats_dir, BASE + ADAPTIVE_SUFFIX, ts="2026.05.27-20.10.00")
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.20.00")
    cli.main(["replay", BASE])
    assert "replayed 3 runs" in capsys.readouterr().out
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert [h["ts"] for h in prof.history] == [
        "2026-05-27T20:00:00", "2026-05-27T20:10:00", "2026-05-27T20:20:00",
    ]
    assert prof.last_run_ts == "2026-05-27T20:20:00"  # newest run, not last stream


# ------------------------------------------------- adaptive-name normalization
def test_generate_normalizes_adaptive_scenario_arg(env: Settings, capsys):
    # "X [Adaptive]" as input must act on base X — never compound the suffix.
    cli.main(["generate", BASE + ADAPTIVE_SUFFIX])
    out = capsys.readouterr().out
    assert "note:" in out and f'using base scenario "{BASE}"' in out
    assert (env.scenarios_dir / f"{BASE}{ADAPTIVE_SUFFIX}.sce").is_file()
    doubled = f"{BASE}{ADAPTIVE_SUFFIX}{ADAPTIVE_SUFFIX}"
    assert not (env.scenarios_dir / f"{doubled}.sce").exists()
    # learning lands in the single-suffix profile, no doubled fork
    assert PlayerProfile.path_for(BASE + ADAPTIVE_SUFFIX, env.profile_path).is_file()
    assert not PlayerProfile.path_for(doubled, env.profile_path).is_file()


def test_watch_normalizes_adaptive_scenario_arg(env: Settings, monkeypatch, capsys):
    import kovadapt.watcher as watcher_mod

    created: dict = {}

    class FakeWatcher:
        def __init__(self, s, scenario, **kw):
            created["scenario"] = scenario

        def base_sce_path(self):
            return env.scenarios_dir / f"{created['scenario']}.sce"

        def watch(self):
            created["watched"] = True

    monkeypatch.setattr(watcher_mod, "SessionWatcher", FakeWatcher)
    cli.main(["watch", BASE + ADAPTIVE_SUFFIX])
    assert created == {"scenario": BASE, "watched": True}
    assert "note:" in capsys.readouterr().out


def test_replay_and_status_normalize_adaptive_scenario_arg(env: Settings, capsys):
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.25.38")
    cli.main(["replay", BASE + ADAPTIVE_SUFFIX])
    assert "replayed 1 runs" in capsys.readouterr().out
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    assert prof.run_count == 1  # folded into the single-suffix profile
    cli.main(["status", BASE + ADAPTIVE_SUFFIX])
    assert f"scenario:      {BASE}{ADAPTIVE_SUFFIX}" in capsys.readouterr().out


def test_base_scenario_strips_doubled_suffix(capsys):
    doubled = f"X{ADAPTIVE_SUFFIX}{ADAPTIVE_SUFFIX}"
    assert cli._base_scenario(doubled) == "X"
    assert cli._base_scenario("X") == "X"
    assert "note:" in capsys.readouterr().out  # printed only for the stripped name


# ------------------------------------------------------------- error handling
def test_missing_kovaaks_root_exits_with_hint(tmp_path: Path, monkeypatch):
    s = make_settings(tmp_path / "kovaaks", tmp_path / "state")
    s.kovaaks_root = ""  # simulate failed install discovery (post-init)
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path=None: s))
    with pytest.raises(SystemExit) as ei:
        cli.main(["scenarios"])
    assert "KovaaK's install not found" in str(ei.value)


def test_unknown_command_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["frobnicate"])
    assert ei.value.code == 2
    capsys.readouterr()  # swallow the argparse usage message


def test_no_command_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main([])
    assert ei.value.code == 2
    capsys.readouterr()


def test_help_lists_every_subcommand(capsys):
    # gui/watchdog/checkup need Qt/Windows to *run*; only pin that argparse
    # registers them alongside the core commands.
    with pytest.raises(SystemExit) as ei:
        cli.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("scenarios", "gui", "watchdog", "checkup",
                 "watch", "generate", "status", "replay"):
        assert name in out


def test_version_flag_names_the_build(capsys):
    """There was no way to ask a running install what it is: `--version` exited
    2 with an argparse usage error, and the frozen exe carried no version
    resource either, so a bug report could not name a build."""
    from kovadapt import __version__

    for flag in ("--version", "-V"):
        with pytest.raises(SystemExit) as e:
            cli.main([flag])
        assert e.value.code == 0, flag
        assert capsys.readouterr().out.strip() == f"kovadapt {__version__}"


def test_generate_on_a_missing_scenario_says_so_without_a_traceback(env, capsys):
    """cmd_watch has always guarded this; cmd_generate did not, so a typo
    surfaced as an 11-frame FileNotFoundError out of SceFile.read."""
    with pytest.raises(SystemExit) as e:
        cli.cmd_generate(SimpleNamespace(scenario="No Such Scenario"))
    msg = str(e.value)
    assert "scenario file not found" in msg
    assert "No Such Scenario.sce" in msg
    assert "Traceback" not in msg


def test_status_heatmap_shows_a_dash_for_regions_with_no_evidence(env, capsys):
    """`plan()` materializes every region arm the first time it runs, so after
    a single `generate` the map printed "+0.00(0)" for all 25 cells — a
    measured-looking zero for regions nothing has ever observed. The sentinel
    was gated on the arm OBJECT existing rather than on its evidence."""
    from kovadapt.config import ADAPTIVE_SUFFIX
    from kovadapt.profile.player import PlayerProfile

    # status needs a profile with runs before it reaches the heatmap at all
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.25.38")
    cli.cmd_replay(SimpleNamespace(scenario=BASE))
    cli.cmd_generate(SimpleNamespace(scenario=BASE))
    capsys.readouterr()
    cli.cmd_status(SimpleNamespace(scenario=BASE))
    grid = capsys.readouterr().out
    assert "+0.00(0)" not in grid, "an unobserved region reported a measurement"
    assert grid.count("--") >= 25

    # ...and an arm that HAS evidence still prints its number
    prof = PlayerProfile.load(BASE + ADAPTIVE_SUFFIX, env.profile_path)
    prof.region("r1c1").update(0.4)
    prof.save(env.profile_path)
    cli.cmd_status(SimpleNamespace(scenario=BASE))
    out = capsys.readouterr().out
    assert "(1)" in out and out.count("--") < 25


def test_replay_with_no_stats_does_not_destroy_the_learned_profile(env, capsys):
    """DATA LOSS. cmd_replay builds a FRESH empty PlayerProfile — it never
    loads the existing one, because rebuilding from scratch is the whole
    point — and then saved it unconditionally. So a misspelled scenario, or a
    stats folder the user had cleared, wrote an empty profile over everything
    that scenario had learned, printed "replayed 0 runs" and exited 0.
    """
    from kovadapt.config import ADAPTIVE_SUFFIX
    from kovadapt.profile.player import PlayerProfile

    # a profile worth losing
    prof = PlayerProfile(scenario=BASE + ADAPTIVE_SUFFIX)
    prof.run_count = 42
    prof.ewma_accuracy = 0.88
    prof.region("r1c1").update(0.3)
    path = prof.save(env.profile_path)
    before = path.read_bytes()

    with pytest.raises(SystemExit) as e:
        cli.cmd_replay(SimpleNamespace(scenario="Nonexistent Scenario"))
    assert "no stats found" in str(e.value)

    assert path.read_bytes() == before, "the learned profile was overwritten"
    assert PlayerProfile.load(BASE + ADAPTIVE_SUFFIX,
                             env.profile_path).run_count == 42

    # ...and a real replay still rebuilds
    write_stats_csv(env.stats_dir, BASE, ts="2026.05.27-20.25.38")
    cli.cmd_replay(SimpleNamespace(scenario=BASE))
    assert "replayed 1 runs" in capsys.readouterr().out
    assert PlayerProfile.load(BASE + ADAPTIVE_SUFFIX,
                              env.profile_path).run_count == 1


def test_the_headless_watchdog_leaves_an_audit_trail(env, tmp_path, monkeypatch):
    """The start-with-Windows entry runs under pythonw with NO CONSOLE, so
    `print` goes nowhere. Without this the only artefact of a background
    process that changes another process's priority and affinity would be the
    change itself — which is the shape of software people are right to
    distrust. The record is what separates the two.
    """
    import threading
    import time

    log = env.profile_path / "watchdog.log"
    assert not log.exists()

    t = threading.Thread(target=lambda: cli.cmd_watchdog(SimpleNamespace()),
                         daemon=True)
    t.start()
    for _ in range(40):                       # let the first line land
        if log.is_file() and log.read_text(encoding="utf-8").strip():
            break
        time.sleep(0.05)

    text = log.read_text(encoding="utf-8")
    assert "watchdog started" in text
    # it must say what it does and how to remove it, not just that it ran
    assert "High priority" in text
    assert "kovadapt-watchdog" in text and r"CurrentVersion\Run" in text


def test_the_watchdog_log_is_trimmed_not_deleted(env):
    """A log that erases itself is no better than no log, so it trims from the
    FRONT and keeps the newest entries."""
    log = env.profile_path / "watchdog.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"line {i}" for i in range(cli._LOG_LINES + 500))
                   + "\n", encoding="utf-8")

    import threading
    import time
    threading.Thread(target=lambda: cli.cmd_watchdog(SimpleNamespace()),
                     daemon=True).start()
    for _ in range(40):
        if len(log.read_text(encoding="utf-8").splitlines()) <= cli._LOG_LINES:
            break
        time.sleep(0.05)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= cli._LOG_LINES
    assert "watchdog started" in lines[-1] or "watchdog" in lines[-1]
    assert not any(ln == "line 0" for ln in lines), "trimmed from the wrong end"
