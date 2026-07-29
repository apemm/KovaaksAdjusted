"""Cross-session skill trends: report-history loading is bounded and
tolerant, Theil-Sen classifications respect each metric's polarity (a
falling Fitts slope is IMPROVEMENT), zeros mean "not measured" and never
enter a fit, and thin histories say "insufficient data" instead of guessing."""

from __future__ import annotations

import json

from kovadapt.analysis.skill import (
    INSUFFICIENT,
    MIN_OBSERVATIONS,
    NUMERIC_FIELDS,
    fit_skill,
    load_report_history,
)
from kovadapt.profile.player import _slug


def report_fields(**kw) -> dict:
    base = dict(
        scenario="Tile Frenzy - 180", started_iso="2026-07-01T12:00:00",
        score=800.0, accuracy=0.70, kps=0.8, overshoot_rate=0.20,
        mean_flick_ms=180.0, mean_corrections=1.0, fitts_slope_ms=120.0,
        # bulky report fields the loader must drop
        notable=[{"kind": "flick", "t_start": 0.0}], region_deficits={"r0c0": 0.4},
        bias={"bias_score": 0.1}, summary_text="x" * 400,
    )
    base.update(kw)
    return base


def write_report(profile_dir, iso: str, **kw) -> None:
    fields = report_fields(started_iso=iso, **kw)
    d = profile_dir / "reports" / _slug(fields["scenario"])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{iso.replace(':', '-')}.json").write_text(json.dumps(fields))


def entries(n=20, scenario="S", **series) -> list[dict]:
    """n chronological entries; series values are callables of i or constants."""
    out = []
    for i in range(n):
        e = dict(scenario=scenario, started_iso=f"2026-07-01T{i:02d}:00:00",
                 score=800.0, accuracy=0.70, kps=0.8, overshoot_rate=0.20,
                 mean_flick_ms=180.0, mean_corrections=1.0, fitts_slope_ms=120.0)
        for k, v in series.items():
            e[k] = v(i) if callable(v) else v
        out.append(e)
    return out


# ---------------------------------------------------------------- loading
def test_load_keeps_only_slim_fields_and_defaults_missing(tmp_path):
    write_report(tmp_path, "2026-07-01T12:00:00")
    # a sparse report (old schema): only two fields present
    d = tmp_path / "reports" / _slug("Tile Frenzy - 180")
    (d / "2026-07-01T13-00-00.json").write_text(
        json.dumps({"scenario": "Tile Frenzy - 180", "accuracy": 0.9}))
    got = load_report_history(tmp_path)
    assert len(got) == 2
    for e in got:
        assert set(e) == {"scenario", "started_iso", *NUMERIC_FIELDS}
    sparse = got[1]
    assert sparse["accuracy"] == 0.9
    assert sparse["started_iso"] == ""
    assert sparse["kps"] == 0.0 and sparse["fitts_slope_ms"] == 0.0


def test_load_limit_keeps_newest_in_chronological_order(tmp_path):
    for i in range(6):
        write_report(tmp_path, f"2026-07-0{i + 1}T12:00:00", score=float(i))
    got = load_report_history(tmp_path, limit=4)
    assert [e["score"] for e in got] == [2.0, 3.0, 4.0, 5.0]   # newest 4, oldest first


def test_load_scenario_filter_uses_slug(tmp_path):
    write_report(tmp_path, "2026-07-01T12:00:00", scenario="Tile Frenzy - 180")
    write_report(tmp_path, "2026-07-01T13:00:00", scenario="1wall 6targets small")
    got = load_report_history(tmp_path, scenario="Tile Frenzy - 180")
    assert [e["scenario"] for e in got] == ["Tile Frenzy - 180"]


def test_load_skips_corrupt_files_silently(tmp_path):
    write_report(tmp_path, "2026-07-01T12:00:00")
    d = tmp_path / "reports" / _slug("Tile Frenzy - 180")
    (d / "2026-07-01T13-00-00.json").write_text("{not json at all")
    (d / "2026-07-01T14-00-00.json").write_text("[1, 2, 3]")   # JSON but not a report
    got = load_report_history(tmp_path)
    assert len(got) == 1


def test_load_missing_reports_dir_is_empty(tmp_path):
    assert load_report_history(tmp_path) == []
    assert load_report_history(tmp_path, scenario="Nope") == []


# ---------------------------------------------------------------- fitting
def test_improving_player_classifications():
    t = fit_skill(entries(
        n=30,
        fitts_slope_ms=lambda i: 150.0 - 1.5 * i,   # falling = improving
        mean_flick_ms=lambda i: 200.0 - 2.0 * i,    # falling = faster
        kps=lambda i: 0.80 + 0.01 * i,              # rising = improving
        overshoot_rate=lambda i: 0.30 - 0.005 * i,  # falling = better
    ))
    o = t.overall
    assert o["fitts_slope_ms"].classification == "improving"
    assert o["mean_flick_ms"].classification == "improving"
    assert o["kps"].classification == "improving"
    assert o["overshoot_rate"].classification == "improving"
    assert o["accuracy"].classification == "flat"   # constant, in band by design
    assert o["score"].classification == "flat"
    assert t.n_runs == 30


def test_declining_classifications():
    t = fit_skill(entries(
        n=30,
        kps=lambda i: 1.10 - 0.01 * i,
        overshoot_rate=lambda i: 0.15 + 0.01 * i,
    ))
    assert t.overall["kps"].classification == "declining"
    assert t.overall["overshoot_rate"].classification == "declining"


def test_fitts_polarity_is_inverted():
    # A RISING Fitts slope (more ms per bit) is decline, not growth.
    t = fit_skill(entries(n=30, fitts_slope_ms=lambda i: 100.0 + 2.0 * i))
    trend = t.overall["fitts_slope_ms"]
    assert trend.slope > 0
    assert trend.classification == "declining"


def test_fitts_zero_entries_excluded_from_fit():
    # Even runs had too few flicks (fitts 0.0 = not measured) — only the 15
    # measured runs enter the fit, and they are clearly improving.
    t = fit_skill(entries(
        n=30, fitts_slope_ms=lambda i: 0.0 if i % 2 == 0 else 150.0 - 3.0 * i))
    trend = t.overall["fitts_slope_ms"]
    assert trend.n == 15
    assert trend.classification == "improving"


def test_insufficient_data_paths():
    thin = fit_skill(entries(n=MIN_OBSERVATIONS - 1))
    assert all(m.classification == INSUFFICIENT for m in thin.overall.values())
    # per-metric: 20 runs but only 6 with a measured kps
    t = fit_skill(entries(n=20, kps=lambda i: 0.8 if i < 6 else 0.0))
    assert t.overall["kps"].classification == INSUFFICIENT
    assert t.overall["kps"].n == 6
    assert t.overall["accuracy"].classification == "flat"   # others still fit


def test_per_scenario_trends_split():
    ents = entries(n=20, scenario="A", kps=lambda i: 0.80 + 0.01 * i) \
        + entries(n=20, scenario="B", kps=lambda i: 1.00 - 0.01 * i)
    t = fit_skill(ents)
    assert t.per_scenario["A"]["kps"].classification == "improving"
    assert t.per_scenario["B"]["kps"].classification == "declining"
    assert t.n_runs == 40


# ---------------------------------------------------------------- summary
def test_summary_reads_like_evidence():
    t = fit_skill(entries(n=30, fitts_slope_ms=lambda i: 150.0 - 1.5 * i))
    text = t.summary()
    assert text
    assert "30" in text                      # run count
    assert "%" in text                       # actual magnitude
    assert "fell" in text                    # direction, plainly stated
    assert 2 <= text.count(". ") + 1 <= 4    # 2-4 sentences


def test_summary_insufficient_says_so():
    text = fit_skill(entries(n=3)).summary()
    assert text
    assert str(MIN_OBSERVATIONS) in text
    assert fit_skill([]).summary()
