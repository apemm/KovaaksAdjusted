"""PlayerProfile bias cold-start and region-credit unit contracts.

Two claims were investigated here (see the module docstrings below):

  A. observe_bias' cold start was keyed on run_count, not on its own
     observation counter — REAL, fixed, pinned by the tests in this file.
  B. credit_focus_region and credit_observed_regions write incompatible
     units into the same Gaussian posterior — NOT real; the telemetry path
     is damped into accuracy-deficit scale on purpose. The sign-convention
     test below pins the property that actually has to hold.
"""

from pathlib import Path

import json

import numpy as np
import pytest

from kovadapt.adapt.engine import AdaptationEngine
from kovadapt.config import Settings
from kovadapt.profile.player import PlayerProfile
from kovadapt.stats.parser import parse_stats_csv


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Nothing in this file may reach the developer's real ~/.kovadapt."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))


def _settings() -> Settings:
    return Settings(kovaaks_root=".")


def _alpha(half_life: float = 5.0) -> float:
    return PlayerProfile(scenario="_")._alpha(half_life)


# --------------------------------------------------------------- claim A
def test_bias_seeds_on_first_measurement_not_first_run():
    """A bias measurement arriving at run 20 must seed the EWMA, not be
    folded into 0.0 at the EWMA rate.

    Bias needs both flick sides sampled (watcher gates on >= 8 flicks with
    >= 3 per side) and the replay backfill supplies none at all, so the
    first usable measurement routinely lands well after run 1.
    """
    prof = PlayerProfile(scenario="t")
    prof.run_count = 20                 # backfilled runs, none carrying bias
    prof.observe_bias(0.6)
    assert prof.ewma_bias == pytest.approx(0.6)
    assert prof.bias_obs == 1


def test_bias_first_run_still_seeds_directly():
    """The run-1 path (pinned in test_adapt.py) is unchanged."""
    prof = PlayerProfile(scenario="t")
    prof.run_count = 1
    prof.observe_bias(-0.4)
    assert prof.ewma_bias == pytest.approx(-0.4)


def test_bias_second_measurement_blends_at_ewma_rate():
    prof = PlayerProfile(scenario="t")
    prof.run_count = 20
    prof.observe_bias(0.6)
    prof.observe_bias(0.2)
    assert prof.ewma_bias == pytest.approx(0.6 + _alpha() * (0.2 - 0.6))
    assert prof.bias_obs == 2


def test_legacy_profile_with_bias_evidence_is_not_reseeded(tmp_path):
    """Profiles written before bias_obs existed load with bias_obs == 0 but
    a real EWMA; the migration must blend into it, never overwrite it."""
    d = tmp_path / "prof"
    (d / "profiles").mkdir(parents=True)
    legacy = {"scenario": "t", "run_count": 30, "ewma_bias": 0.4}
    PlayerProfile.path_for("t", d).write_text(json.dumps(legacy))

    prof = PlayerProfile.load("t", d)
    assert prof.bias_obs == 0
    prof.observe_bias(-0.2)
    assert prof.ewma_bias == pytest.approx(0.4 + _alpha() * (-0.2 - 0.4))


def test_bias_obs_round_trips_through_json(tmp_path):
    d = tmp_path / "prof"
    prof = PlayerProfile(scenario="t")
    prof.run_count = 3
    prof.observe_bias(0.5)
    prof.observe_bias(0.5)
    prof.save(d)
    assert PlayerProfile.load("t", d).bias_obs == 2


def test_engine_bias_after_backfill_drives_the_dodge(fixtures):
    """End-to-end consequence: a 0.30 bias measured after a replay backfill
    has to clear the engine's 0.05 dodge gate on the run it was measured.
    Attenuated by the EWMA rate it lands at 0.039 and the strafe skew stays
    neutral for another two runs."""
    prof = PlayerProfile(scenario="t")
    engine = AdaptationEngine(_settings(), rng=np.random.default_rng(0))
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    for _ in range(12):                 # replay backfill: no telemetry at all
        engine.observe(prof, run)
    assert prof.ewma_bias == 0.0

    engine.observe(prof, run, bias_score=0.3)
    assert prof.ewma_bias == pytest.approx(0.3)
    assert engine.plan(prof, run).dodge_bias > 0


# --------------------------------------------------------------- claim B
def test_both_credit_paths_agree_on_the_deficit_sign(fixtures):
    """Claim B (incompatible units) does not hold, but the property that
    genuinely must: both bandit credit paths write *positive = weaker* into
    RegionPosterior, and the telemetry path's damping keeps its writes on
    the same order of magnitude as a run-level accuracy deficit — so mixing
    regimes across runs cannot flip the bandit's ranking.
    """
    s = _settings()
    run = parse_stats_csv(fixtures / "sample_stats.csv")   # accuracy 6/9

    run_level = PlayerProfile(scenario="a")
    run_level.run_count = 5
    run_level.ewma_accuracy = 0.85      # played well below baseline this run
    run_level.last_focus = "r1c1"
    run_level.credit_focus_region(
        run, prior_var=s.bandit_prior_var, obs_noise=s.bandit_obs_noise)

    telemetry = PlayerProfile(scenario="b")
    telemetry.credit_observed_regions(
        {"r1c1": 1.0, "r0c0": -1.0}, s.telemetry_blend,
        prior_var=s.bandit_prior_var, obs_noise=s.bandit_obs_noise)

    assert run_level.region("r1c1").mean > 0
    assert telemetry.region("r1c1").mean > 0
    assert telemetry.region("r0c0").mean < 0
    # Same order of magnitude: a 1-sigma telemetry deficit is worth between
    # a third and three times a run-level accuracy miss, not 10x either way.
    ratio = telemetry.region("r1c1").mean / run_level.region("r1c1").mean
    assert 0.33 < ratio < 3.0
