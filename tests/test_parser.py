from datetime import datetime
from pathlib import Path

from kovadapt.stats.parser import parse_stats_csv, parse_stats_filename, iter_runs


def test_filename_parsing():
    meta = parse_stats_filename(
        "1w3ts reload - Challenge - 2026.05.27-20.25.38 Stats.csv"
    )
    assert meta is not None
    scenario, mode, ts = meta
    assert scenario == "1w3ts reload"
    assert mode == "Challenge"
    assert ts == datetime(2026, 5, 27, 20, 25, 38)
    # scenario names containing " - " must still parse (greedy scenario group)
    meta2 = parse_stats_filename(
        "1wall 2targets small - valorant - Challenge - 2026.05.21-18.42.29 Stats.csv"
    )
    assert meta2 is not None and meta2[0] == "1wall 2targets small - valorant"


def test_fixture_parse(fixtures: Path):
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    assert len(run.kills) == 6
    assert run.kill_count == 6
    assert run.hit_count == 6 and run.miss_count == 3
    assert abs(run.accuracy - 6 / 9) < 1e-9
    assert run.score == 60.0
    k3 = run.kills[2]
    assert k3.shots == 2 and k3.hits == 1 and abs(k3.ttk - 0.21) < 1e-9
    # relative timestamps monotonic from 0
    assert run.kills[0].t == 0.0
    assert all(b.t >= a.t for a, b in zip(run.kills, run.kills[1:]))
    assert run.kills_per_second() > 0


MIDNIGHT_CSV = """\
Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,Damage Done,Damage Possible,Efficiency,Cheated,OverShots
1,23:59:58.500,target,BB Gun,0.000000s,1,1,1.000000,1.000000,1.000000,1.000000,0,0
2,23:59:59.200,target,BB Gun,0.000000s,1,1,1.000000,1.000000,1.000000,1.000000,0,0
3,00:00:00.100,target,BB Gun,0.210000s,2,1,0.500000,1.000000,2.000000,0.500000,0,0
4,00:00:01.300,target,BB Gun,0.000000s,1,1,1.000000,1.000000,1.000000,1.000000,0,0

Kills:,4
Hit Count:,4
Miss Count:,1
Score:,40.0
"""


def test_midnight_rollover(tmp_path: Path):
    """A run crossing 00:00 must keep relative kill times monotonic from 0
    (no negative t, no degenerate kills_per_second -> corrupted ewma_kps)."""
    p = tmp_path / "wrap - Challenge - 2026.07.26-23.59.55 Stats.csv"
    p.write_text(MIDNIGHT_CSV, encoding="utf-8")
    run = parse_stats_csv(p)
    assert len(run.kills) == 4
    ts = [k.t for k in run.kills]
    assert ts[0] == 0.0
    assert all(b >= a >= 0.0 for a, b in zip(ts, ts[1:]))
    assert abs(ts[2] - 1.6) < 1e-6      # 00:00:00.100 is 1.6 s after 23:59:58.500
    assert abs(ts[3] - 2.8) < 1e-6
    assert run.kills_per_second() > 0.0
    assert (run.interkill_intervals() > 0).all()


def test_real_stats_files(kovaaks_root: Path):
    """Parse every real stats file on this machine without crashing."""
    n_ok = 0
    for run in iter_runs(kovaaks_root / "stats"):
        assert run.scenario
        assert 0.0 <= run.accuracy <= 1.0
        n_ok += 1
    assert n_ok > 0
