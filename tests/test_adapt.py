import numpy as np

from kovadapt.adapt.stochastic import OrnsteinUhlenbeck, sample_dodge_params, squash
from kovadapt.adapt.bandit import ThompsonRegionBandit, region_keys
from kovadapt.adapt.engine import AdaptationEngine
from kovadapt.config import Settings
from kovadapt.profile.player import PlayerProfile
from kovadapt.stats.parser import parse_stats_csv


def _settings() -> Settings:
    return Settings(kovaaks_root=".")


def test_ou_stationary_moments():
    ou = OrnsteinUhlenbeck(theta=0.5, mu=0.0, sigma=0.3)
    path = ou.path(0.0, 20000, seed=1)
    burn = path[2000:]
    assert abs(burn.mean()) < 0.05
    assert abs(burn.std() - ou.stationary_std()) < 0.05


def test_ou_mean_reversion():
    ou = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=1e-9)
    x = 5.0
    for _ in range(20):
        x = ou.step(x)
    assert abs(x) < 0.01


def test_squash_and_dodge_params():
    assert 0.0 <= squash(-100) <= squash(0) <= squash(100) <= 1.0
    calm = sample_dodge_params(0.0, np.random.default_rng(0))
    fast = sample_dodge_params(1.0, np.random.default_rng(0))
    assert fast["MinLRTimeChange"] < calm["MinLRTimeChange"]
    assert fast["JumpFrequency"] > calm["JumpFrequency"]
    for p in (calm, fast):
        assert p["MaxLRTimeChange"] > p["MinLRTimeChange"] > 0


def test_bandit_converges_to_weak_region():
    prof = PlayerProfile(scenario="t")
    rng = np.random.default_rng(0)
    bandit = ThompsonRegionBandit(prof, 3, 3, rng=rng)
    weak = "r1c0"
    picks = []
    for _ in range(200):
        focus = bandit.choose_focus()
        picks.append(focus)
        deficit = 0.15 if focus == weak else -0.05  # weak region yields deficit
        deficit += rng.normal(0, 0.02)
        prof.region(focus).update(deficit)
    # Converged to mostly exploiting the weakness (Thompson keeps ~15-25%
    # exploration by design, so require a strong plurality, not a monopoly).
    assert picks[-50:].count(weak) >= 25
    assert picks.count(weak) > max(picks.count(k) for k in set(picks) - {weak})


def test_spawn_weights_sum_to_one():
    prof = PlayerProfile(scenario="t")
    bandit = ThompsonRegionBandit(prof, 3, 3, rng=np.random.default_rng(0))
    w = bandit.spawn_weights("r0c0", 0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["r0c0"] == 0.5
    assert set(w) == set(region_keys(3, 3))


def test_size_controller_direction(fixtures):
    s = _settings()
    engine = AdaptationEngine(s, rng=np.random.default_rng(0))
    run = parse_stats_csv(fixtures / "sample_stats.csv")  # 66.7% acc: inside band
    prof = PlayerProfile(scenario="t")
    # get enough shots for the controller to act
    run.summary["Hit Count:"] = "90"
    run.summary["Miss Count:"] = "10"   # 90%: too easy -> shrink
    engine.observe(prof, run)
    base = prof.target_scale
    plan = engine.plan(prof, run)
    assert prof.target_scale < base or plan.target_scale <= base * (1 + 0.35)

    run.summary["Hit Count:"] = "40"
    run.summary["Miss Count:"] = "60"   # 40%: too hard -> grow
    prof2 = PlayerProfile(scenario="t")
    engine.observe(prof2, run)
    engine.plan(prof2, run)
    assert prof2.target_scale > 1.0


def test_engine_plan_structure(fixtures):
    s = _settings()
    engine = AdaptationEngine(s, rng=np.random.default_rng(42))
    prof = PlayerProfile(scenario="t")
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    engine.observe(prof, run)
    plan = engine.plan(prof, run)
    assert s.min_target_scale <= plan.target_scale <= s.max_target_scale
    assert 0.0 <= plan.movement <= 1.0
    assert plan.focus_region in region_keys(s.region_cols, s.region_rows)
    assert abs(sum(plan.spawn_weights.values()) - 1.0) < 1e-9
    assert prof.last_focus == plan.focus_region


def test_profile_roundtrip(tmp_path, fixtures):
    prof = PlayerProfile(scenario="rt test")
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    prof.last_focus = "r0c0"
    prof.observe_run(run)
    prof.credit_focus_region(run)
    prof