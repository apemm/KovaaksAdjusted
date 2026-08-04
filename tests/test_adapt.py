import json

import numpy as np
import pytest

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


def test_invincible_target_tracking_is_not_stamped_clicking():
    """Pure-tracking scenarios kill nothing, so shots-per-kill is undefined.

    Guarding the ratio on kill_count > 0 skipped the heuristic for exactly
    the scenarios it exists to catch: on the maintainer's real stats folder
    23 of 95 scenarios (the whole Whisphere/cloverRawControl/PGTI family)
    were stamped "clicking" and then scored against the clicking accuracy
    band they can never reach, growing targets every single run.
    """
    from kovadapt.stats.models import Run

    def run(kills: int, hits: int, misses: int) -> Run:
        return Run(scenario="x", started=None, kills=[],
                   summary={"Kills:": float(kills), "Hit Count:": float(hits),
                            "Miss Count:": float(misses)})

    # invincible targets: thousands of damage ticks, nothing ever dies
    assert detect_archetype("Whisphere", run(0, 2500, 3000)) == "tracking"
    assert detect_archetype("PGTI Voltaic Easy", run(0, 1140, 4832)) == "tracking"
    # a genuine no-shot (AFK) run carries no evidence -> stays the safe default
    assert detect_archetype("unnamed task", run(0, 0, 0)) == "clicking"
    # and a clicking run is still clicking: a hit kills, so hits stay low
    assert detect_archetype("unnamed task", run(6, 6, 3)) == "clicking"


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


def test_settings_load_tolerates_utf8_bom(tmp_path):
    """PowerShell 5.1's -Encoding utf8 writes a BOM; load() must not choke."""
    p = tmp_path / "settings.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"target_accuracy_high": 0.91}).encode())
    assert Settings.load(p).target_accuracy_high == 0.91


def test_settings_load_survives_corrupt_file(tmp_path):
    """A broken settings.json boots on defaults and is set aside, not fatal."""
    p = tmp_path / "settings.json"
    p.write_text("{not json", encoding="utf-8")
    s = Settings.load(p)
    assert s.target_accuracy_high == Settings().target_accuracy_high
    assert not p.exists() and (tmp_path / "settings.json.bad").is_file()

    p.write_text('["a", "list"]', encoding="utf-8")  # valid JSON, wrong shape
    assert Settings.load(p).theme == Settings().theme
    assert not p.exists()

def test_zero_shot_run_is_dropped_whole():
    """An AFK run carries no evidence and must not reach the EWMA or bandit.

    Run.accuracy is 0.0 for zero shots (a guard, not a measurement), so
    folding one in dragged the accuracy EWMA down AND, since
    credit_focus_region scores `ewma_accuracy - run.accuracy`, booked the
    whole baseline as a deficit against the last focus region — inventing a
    strong weakness the bandit would then chase.
    """
    from kovadapt.stats.models import Run

    s = _settings()
    eng = AdaptationEngine(s, rng=np.random.default_rng(1))
    prof = PlayerProfile(scenario="x")

    import datetime as _dt

    def run(kills, hits, misses):
        return Run(scenario="x", started=_dt.datetime(2026, 7, 29, 12, 0, 0),
                   kills=[],
                   summary={"Kills:": float(kills), "Hit Count:": float(hits),
                            "Miss Count:": float(misses)})

    eng.observe(prof, run(10, 90, 10))          # a real run: 90% accuracy
    prof.last_focus = "r1c1"
    before_acc, before_runs = prof.ewma_accuracy, prof.run_count
    before_region = prof.region("r1c1").mean

    eng.observe(prof, run(0, 0, 0))             # the alt-tab

    assert prof.ewma_accuracy == before_acc, "zero-shot run moved the EWMA"
    assert prof.run_count == before_runs, "zero-shot run counted as training"
    assert prof.region("r1c1").mean == before_region, "fabricated a weakness"


def test_hitting_the_size_ceiling_does_not_give_back_earned_scale():
    """plan() must persist the UN-coupled base scale.

    Dividing the clipped value back out only round-trips while nothing
    clips; at max_target_scale it wrote back a strictly smaller base, so
    touching the ceiling permanently shrank targets for the very player the
    size controller had been growing them for.
    """
    s = _settings()
    s.size_speed_coupling = 0.35
    s.max_target_scale = 2.50
    eng = AdaptationEngine(s, rng=np.random.default_rng(3))

    prof = PlayerProfile(scenario="x")
    prof.target_scale = s.max_target_scale       # already earned the ceiling
    prof.run_count = 5
    eng.plan(prof, None)

    assert prof.target_scale == pytest.approx(s.max_target_scale), (
        f"earned scale ratcheted down to {prof.target_scale}")


def test_one_bad_run_cannot_cross_the_whole_scale_range(fixtures):
    """A single unrepresentative run used to traverse the entire clamp range.

    Measured before the cap: a 15-shot run at 6.7% — wrong sensitivity after a
    config change, wrong scenario loaded, spraying to end a run — gives
    excess -0.783, so scale *= exp(0.987) = 2.68x and clips straight to
    max_target_scale in ONE step. The error term is structurally asymmetric:
    accuracy is bounded in [0,1] with the band at 0.85-0.95, so excess below
    can reach -0.85 while above it caps at +0.05, and the 1.4/0.8 gain split
    compounds that rather than compensating. The best possible correction
    afterwards is 3.5% per run, so the excursion costs ~25-30 runs — a whole
    session training against targets that are trivially easy.

    (It does self-heal: a player facing 2.5x targets really does hit ~100%,
    which drives the controller back down. So this bounds the step rather
    than rejecting the run.)
    """
    from kovadapt.adapt.engine import _MAX_SIZE_STEP, _MIN_SIZE_STEP

    s = _settings()
    engine = AdaptationEngine(s, rng=np.random.default_rng(0))
    run = parse_stats_csv(fixtures / "sample_stats.csv")

    def step(hits: int, misses: int, start: float = 1.0) -> float:
        run.summary["Hit Count:"] = str(hits)
        run.summary["Miss Count:"] = str(misses)
        prof = PlayerProfile(scenario="t")
        prof.target_scale = start
        engine.plan(prof, run)
        return prof.target_scale

    # the reported case: nowhere near the 2.5 clamp any more
    aborted = step(1, 14)
    assert aborted == pytest.approx(_MAX_SIZE_STEP, abs=1e-6)
    assert aborted < s.max_target_scale * 0.6

    # the commoner case: one wrong-sensitivity run
    assert step(16, 24) <= _MAX_SIZE_STEP + 1e-9

    # no single run may leave the band in either direction
    for hits, misses in ((0, 12), (1, 15), (16, 40), (40, 0), (100, 0), (0, 100)):
        moved = step(hits, misses)
        assert _MIN_SIZE_STEP - 1e-9 <= moved <= _MAX_SIZE_STEP + 1e-9, (
            f"{hits}/{hits + misses} moved scale to {moved}")

    # ...but a genuinely struggling player still REACHES the ceiling, over
    # several runs rather than one — the cap must not disable the controller
    prof = PlayerProfile(scenario="t")
    run.summary["Hit Count:"], run.summary["Miss Count:"] = "2", "38"
    for _ in range(8):
        engine.plan(prof, run)
    assert prof.target_scale == pytest.approx(s.max_target_scale, abs=1e-6)

    # and an in-band run still moves nothing at all
    assert step(36, 4) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------- archetype evidence, not latching
def test_a_name_only_guess_is_corrected_by_the_first_real_run():
    """Two of the four stamping sites — the browser's Start adapting and
    `kovadapt generate` — run BEFORE the scenario has ever been played, and
    every site latched on `if not profile.archetype`. So a pre-run click wrote
    a name-only guess that no amount of later evidence could correct.

    Measured on the real 95-scenario library: 31 scenarios take `clicking`
    from their name and `tracking` from their own stats — Whisphere,
    WhisphereRawControl, waldoTS, cloverRawControl, the Polarized Hell set.
    Each would have been scored forever against an accuracy band its
    invincible targets can never reach.
    """
    from datetime import datetime

    from kovadapt.adapt.archetype import classify_archetype, stamp_archetype
    from kovadapt.profile.player import PlayerProfile
    from kovadapt.stats.models import Run

    # a tracking scenario whose name says nothing: invincible targets, so the
    # CSV reports hits and no kills at all
    name = "WhisphereRawControl"
    tracking_run = Run(scenario=name, started=datetime(2026, 8, 3, 10, 0),
                       summary={"Kills:": "0", "Hit Count:": "412",
                                "Miss Count:": "180", "Score:": "900"})
    assert classify_archetype(name) == ("clicking", "default")
    assert classify_archetype(name, tracking_run) == ("tracking", "stats")
    # ...and a run that does NOT trip the heuristic is evidence of clicking,
    # not an absence of evidence. Without that the clicking case never
    # reaches "stats" and stays re-evaluatable forever.
    clicking_run = Run(scenario=name, started=datetime(2026, 8, 3, 10, 0),
                       summary={"Kills:": "30", "Hit Count:": "45",
                                "Miss Count:": "20", "Score:": "800"})
    assert classify_archetype(name, clicking_run) == ("clicking", "stats")

    prof = PlayerProfile(scenario=name + " [Adaptive]")
    stamp_archetype(prof, name)                       # the pre-run click
    assert (prof.archetype, prof.archetype_source) == ("clicking", "default")

    changed = stamp_archetype(prof, name, tracking_run)   # the first real run
    assert changed == ("clicking", "tracking"), changed
    assert prof.archetype_source == "stats"


def test_a_stats_stamp_is_not_re_litigated_by_later_runs():
    """One correction, then it holds — otherwise an odd run flips a scenario
    back and forth and the adaptation parameters move with it."""
    from datetime import datetime

    from kovadapt.adapt.archetype import stamp_archetype
    from kovadapt.profile.player import PlayerProfile
    from kovadapt.stats.models import Run

    prof = PlayerProfile(scenario="X [Adaptive]")
    prof.archetype, prof.archetype_source = "tracking", "stats"
    odd = Run(scenario="X", started=datetime(2026, 8, 3, 10, 0),
              summary={"Kills:": "30", "Hit Count:": "40", "Miss Count:": "5"})
    assert stamp_archetype(prof, "X", odd) is None
    assert prof.archetype == "tracking"


def test_a_profile_written_before_the_field_existed_self_heals():
    """`archetype_source` is "" on every profile already on disk, which sorts
    below every real level — so the next run re-evaluates it. Those are
    exactly the profiles that may be carrying a pre-run guess."""
    from datetime import datetime

    from kovadapt.adapt.archetype import stamp_archetype
    from kovadapt.profile.player import PlayerProfile
    from kovadapt.stats.models import Run

    old = PlayerProfile(scenario="Whisphere [Adaptive]")
    old.archetype = "clicking"                 # no source: an old file
    assert old.archetype_source == ""
    run = Run(scenario="Whisphere", started=datetime(2026, 8, 3, 10, 0),
              summary={"Kills:": "0", "Hit Count:": "300", "Miss Count:": "90"})
    assert stamp_archetype(old, "Whisphere", run) == ("clicking", "tracking")


def test_a_named_archetype_still_beats_no_evidence():
    """A keyword match is real evidence and must not be overwritten by a
    later call that has nothing — the order the sites run in is not fixed."""
    from kovadapt.adapt.archetype import stamp_archetype
    from kovadapt.profile.player import PlayerProfile

    prof = PlayerProfile(scenario="Close Strafes Invincible [Adaptive]")
    stamp_archetype(prof, "Close Strafes Invincible")
    assert (prof.archetype, prof.archetype_source) == ("tracking", "name")
    assert stamp_archetype(prof, "Close Strafes Invincible") is None
    assert prof.archetype == "tracking"


def test_the_switching_keyword_could_not_match_the_naming_convention():
    """`" ts "` was space-delimited on both sides. The community writes target
    switching as a SUFFIX — waldoTS, beanTS, FloatTS, devTS — or the Voltaic
    numeric wall form, 1w2ts / 1w3ts. On the real 95-scenario library that
    keyword matched exactly none of the 16 scenarios carrying the convention.

    Nine landed on `clicking` and were scored against an 0.85-0.95 accuracy
    band instead of switching's 0.65-0.85, so the size controller shrank
    targets every run chasing a number a switching task does not produce.

    The trap in fixing it: a case-insensitive `ts` word match also catches
    "6targets", "2targets" and "4 Targets", which would have relabelled
    `1wall 6targets small [Adaptive]` — the main adaptive scenario on this
    machine — as target-switching. Case is what separates them, and
    lowercasing the name first is what threw it away.
    """
    from kovadapt.adapt.archetype import classify_archetype

    for name in ("waldoTS Novice", "beanTS", "beanTS Larger", "devTS Goated",
                 "FloatTS Angelic Easy", "1w2ts Pasu Perfected",
                 "1w3ts reload Larger", "psalm TS", "Voltaic TS Hard"):
        assert classify_archetype(name) == ("switching", "name"), name

    # ...and every one of these is NOT switching, however much "ts" they carry
    for name in ("1wall 6targets small [Adaptive]", "Wide Wall 4 Targets Small",
                 "1wall 2targets small - valorant", "6 targets TE",
                 "TARGETS ONLY", "Ground Plaza Sixshot"):
        arch, _ = classify_archetype(name)
        assert arch != "switching", name


def test_stats_may_not_overturn_a_switching_stamp():
    """The stats heuristic can only ever answer tracking or clicking, so its
    disagreement with `switching` is the heuristic reporting from outside its
    own range — not evidence.

    Eight scenarios here prove it: domiSwitch, voxTargetSwitch and
    tamTargetSwitch read 30-216 shots per kill, because you TRACK a target and
    then switch to the next. The name is the better authority, and no stats
    signature could ever say otherwise.
    """
    from datetime import datetime

    from kovadapt.adapt.archetype import classify_archetype
    from kovadapt.stats.models import Run

    # 167 shots per kill — far past the tracking threshold, and decisively so
    trackingish = Run(scenario="domiSwitch Easy",
                      started=datetime(2026, 8, 3, 10, 0),
                      summary={"Kills:": "30", "Hit Count:": "4200",
                               "Miss Count:": "819", "Score:": "900"})
    assert classify_archetype("domiSwitch Easy", trackingish) == ("switching", "name")
    assert classify_archetype("waldoTS Novice", trackingish) == ("switching", "name")

    # a click-based switching scenario is still switching, from the other side
    clicky = Run(scenario="voxTargetSwitch Click",
                 started=datetime(2026, 8, 3, 10, 0),
                 summary={"Kills:": "60", "Hit Count:": "62",
                          "Miss Count:": "22", "Score:": "800"})
    assert classify_archetype("voxTargetSwitch Click", clicky) == ("switching", "name")


def test_decisive_stats_overturn_a_name_keyword_but_a_marginal_run_does_not():
    """`Controlsphere Click Easy` takes "tracking" from the `controlsphere`
    keyword and reads 1.8 shots per kill against a threshold of 20. No reading
    of 1.8 is tracking, and the name is simply wrong about that scenario.

    But a name keyword is usually right, so only DECISIVE evidence may
    overturn it. Measured across the 55 real scenarios that record kills: 39
    sit below 10 shots per kill, 6 above 40, and every one of the 10 in
    between is a TS or Switch scenario — a genuine hybrid. A factor-of-two
    band therefore fires on the one real error and on none of the hybrids.
    """
    from datetime import datetime

    from kovadapt.adapt.archetype import classify_archetype
    from kovadapt.stats.models import Run

    def run(kills, hits, misses, name="Controlsphere Click Easy"):
        return Run(scenario=name, started=datetime(2026, 8, 3, 10, 0),
                   summary={"Kills:": str(kills), "Hit Count:": str(hits),
                            "Miss Count:": str(misses), "Score:": "800"})

    name = "Controlsphere Click Easy"
    assert classify_archetype(name) == ("tracking", "name")     # name alone
    # 42 kills, 76 shots -> 1.8 per kill. Decisive, 11x clear of the threshold.
    assert classify_archetype(name, run(42, 60, 16)) == ("clicking", "stats")

    # 15 per kill: under the threshold, so it reads clicking — but not by the
    # factor of two the override needs, so the keyword holds.
    assert classify_archetype(name, run(10, 100, 50)) == ("tracking", "name")
    # 30 per kill: over the threshold and agrees with the name anyway
    assert classify_archetype(name, run(10, 200, 100)) == ("tracking", "name")

    # an unnamed scenario still takes the stats answer outright, decisive or not
    assert classify_archetype("unnamed", run(10, 100, 50, "unnamed")) == \
        ("clicking", "stats")
