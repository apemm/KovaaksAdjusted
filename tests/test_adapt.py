import numpy as np

from kovadapt.adapt.archetype import detect_archetype
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
    prof.credit_focus_region(run)  # run_count > 0 now, so the posterior updates
    prof.region("r2c1").update(0.3)
    prof.target_scale = 1.23
    prof.ou_state = -0.4

    path = prof.save(tmp_path)
    assert path.is_file()
    loaded = PlayerProfile.load("rt test", tmp_path)

    assert loaded.scenario == "rt test"
    assert loaded.run_count == 1
    assert loaded.ewma_accuracy == prof.ewma_accuracy
    assert loaded.target_scale == prof.target_scale
    assert loaded.ou_state == prof.ou_state
    assert loaded.last_focus == "r0c0"
    assert set(loaded.regions) == set(prof.regions)
    for key, post in prof.regions.items():
        assert loaded.regions[key].mean == post.mean
        assert loaded.regions[key].var == post.var
        assert loaded.regions[key].n == post.n
    assert loaded.history == prof.history


# --------------------------------------------------------------- v0.3 features
def test_dodge_direction_bias():
    rng = np.random.default_rng(0)
    neutral = sample_dodge_params(0.5, np.random.default_rng(0))
    left = sample_dodge_params(0.5, np.random.default_rng(0), direction_bias=0.8)
    right = sample_dodge_params(0.5, np.random.default_rng(0), direction_bias=-0.8)
    assert abs(neutral["LeftStrafeTimeMult"] - neutral["RightStrafeTimeMult"]) < 0.25
    assert left["LeftStrafeTimeMult"] > left["RightStrafeTimeMult"]
    assert right["RightStrafeTimeMult"] > right["LeftStrafeTimeMult"]
    for p in (left, right):  # clamped to the sane KovaaK's range
        assert 0.5 <= p["LeftStrafeTimeMult"] <= 2.0
        assert 0.5 <= p["RightStrafeTimeMult"] <= 2.0
    # extreme bias stays clamped
    extreme = sample_dodge_params(0.5, rng, direction_bias=5.0)
    assert 0.5 <= extreme["RightStrafeTimeMult"] <= 2.0


def test_engine_wires_dodge_bias(fixtures):
    s = _settings()
    engine = AdaptationEngine(s, rng=np.random.default_rng(7))
    prof = PlayerProfile(scenario="t")
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    engine.observe(prof, run, bias_score=0.6)   # left much weaker
    assert prof.ewma_bias == 0.6                # first run sets directly
    plan = engine.plan(prof, run)
    assert plan.dodge_bias > 0
    assert plan.dodge_params["LeftStrafeTimeMult"] > plan.dodge_params["RightStrafeTimeMult"]


def test_posterior_decay():
    prof = PlayerProfile(scenario="t")
    post = prof.region("r0c0")
    for _ in range(10):
        post.update(0.5)
    mean_before, var_before = post.mean, post.var
    prof.decay_regions(0.3, prior_var=1.0)
    assert 0 < post.mean < mean_before          # shrinks toward 0
    assert var_before < post.var < 1.0          # relaxes toward the prior


def test_fatigue_easing_direction(fixtures):
    s = _settings()
    run = parse_stats_csv(fixtures / "sample_stats.csv")

    def plan_with(fatigue):
        engine = AdaptationEngine(s, rng=np.random.default_rng(11))
        prof = PlayerProfile(scenario="t")
        engine.observe(prof, run)
        return engine.plan(prof, run, fatigue=fatigue), prof

    fresh, prof_fresh = plan_with(0.0)
    tired, prof_tired = plan_with(1.0)
    assert tired.target_scale > fresh.target_scale     # bigger targets when tired
    assert tired.movement < fresh.movement             # calmer targets when tired
    # persisted state is un-eased: identical across the two runs
    assert prof_tired.target_scale == prof_fresh.target_scale
    assert prof_tired.movement == prof_fresh.movement


def test_archetype_detection(fixtures):
    assert detect_archetype("1wall 6targets small") == "clicking"
    assert detect_archetype("Smoothbot Voltaic") == "tracking"
    assert detect_archetype("popcorn voltaic switch") == "switching"
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    # fixture: 9 shots / 6 kills — nowhere near tracking's shots-per-kill
    assert detect_archetype("unnamed task", run) == "clicking"


def test_size_controller_is_accuracy_biased(fixtures):
    """Below-band recovery outweighs above-band pushing: the same excess
    grows targets more than it shrinks them (accuracy-first design)."""
    s = _settings()
    prof_lo = PlayerProfile(scenario="lo")
    prof_hi = PlayerProfile(scenario="hi")
    run = parse_stats_csv(fixtures / "sample_stats.csv")
    run.summary["Hit Count:"] = "90"
    mid = 0.5 * (s.target_accuracy_low + s.target_accuracy_high)
    below = s.target_accuracy_low - 0.05
    above = s.target_accuracy_high + 0.05
    run.summary["Miss Count:"] = str(round(90 * (1 - below) / below))
    engine = AdaptationEngine(s, rng=np.random.default_rng(0))
    engine.plan(prof_lo, run)
    run.summary["Miss Count:"] = str(round(90 * (1 - above) / above))
    engine2 = AdaptationEngine(s, rng=np.random.default_rng(0))
    engine2.plan(prof_hi, run)
    grow = prof_lo.target_scale - 1.0
    shrink = 1.0 - prof_hi.target_scale
    assert grow > 0 and shrink > 0
    assert grow > shrink * 1.3          # same excess, asymmetric response
    _ = mid


def test_settings_for_archetype():
    s = Settings(kovaaks_root=".")
    t = s.for_archetype("tracking")
    assert t is not s
    assert t.target_accuracy_low == 0.70 and t.target_accuracy_high == 0.88
    assert t.region_cols == s.region_cols            # untouched fields inherited
    assert s.target_accuracy_low == 0.85             # doctrine default unchanged
    assert s.for_archetype("clicking") is s          # empty overrides -> same object
    assert s.for_archetype("") is s
    s.archetype_enabled = False
    assert s.for_archetype("tracking") is s          # feature off -> baseline
    # unknown override keys are ignored rather than crashing dataclasses.replace
    s.archetype_enabled = True
    s.archetype_overrides["tracking"]["not_a_field"] = 1.0
    assert s.for_archetype("tracking").target_accuracy_low == 0.70


def test_ou_params_follow_archetype_overrides():
    """Contract: engine tunables resolve via _effective(), including
    ou_theta/ou_sigma — archetype overrides of them must not silently vanish."""
    s = _settings()
    s.archetype_overrides["tracking"]["ou_theta"] = 25.0   # near-instant mean reversion
    s.archetype_overrides["tracking"]["ou_sigma"] = 1e-12  # ~deterministic step

    def state_after_plan(archetype: str) -> float:
        engine = AdaptationEngine(s, rng=np.random.default_rng(3))
        prof = PlayerProfile(scenario="t")
        prof.archetype = archetype
        prof.ou_state = 5.0
        engine.plan(prof, None)          # bootstrap path must stay valid
        return prof.ou_state

    # Override applied: e^-25 collapses the state to ~0 in one step.
    assert abs(state_after_plan("tracking")) < 1e-3
    # Base path unchanged: theta 0.35 keeps most of the state (5 * e^-0.35 ~ 3.5).
    assert state_after_plan("clicking") > 2.0


def test_settings_roundtrip_new_fields(tmp_path):
    s = Settings(kovaaks_root=".")
    s.bandit_posterior_decay = 0.05
    s.archetype_overrides["tracking"]["size_learning_rate"] = 0.42
    p = s.save(tmp_path / "settings.json")
    loaded = Settings.load(p)
    assert loaded.bandit_posterior_decay == 0.05
    assert loaded.archetype_overrides["tracking"]["size_learning_rate"] == 0.42
    assert loaded.dodge_bias_enabled == s.dodge_bias_enabled


def test_settings_save_defaults_to_canonical_load_path(tmp_path, monkeypatch):
    """save() must default to the bootstrap file load() reads, even when
    profile_dir is customized — otherwise saved settings are never loaded."""
    from pathlib import Path

    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    custom = tmp_path / "elsewhere"
    s = Settings(kovaaks_root=".", profile_dir=str(custom))
    s.target_accuracy_high = 0.91
    p = s.save()
    assert p == fake_home / ".kovadapt" / "settings.json" and p.is_file()
    assert not (custom / "settings.json").exists()
    loaded = Settings.load()  # default path round-trips the customization
    assert loaded.target_accuracy_high == 0.91
    assert loaded.profile_dir == str(custom)