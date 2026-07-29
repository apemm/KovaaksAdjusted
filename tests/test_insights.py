"""Insight engine: each diagnostic fires on its pattern, every insight
carries sources + live-number reasoning (the cite-everything rule), and the
input-health gate suppresses microstructure claims on noisy data."""

from __future__ import annotations

from kovadapt.analysis.insights import Insight, generate_insights
from kovadapt.analysis.report import RunReport
from kovadapt.config import Settings
from kovadapt.profile.player import PlayerProfile


def make_rep(**kw) -> RunReport:
    # accuracy defaults INSIDE the v0.4 doctrine band (0.85-0.95)
    base = dict(
        scenario="X [Adaptive]", started_iso="2026-07-28T12:00:00",
        score=800.0, accuracy=0.90, avg_ttk=0.9, kills=40, kps=0.8,
        n_flicks=40, mean_flick_ms=180.0, overshoot_rate=0.1,
        mean_corrections=0.8,
        input_health={"polling_hz_est": 1000.0, "jitter_ms": 0.4},
    )
    base.update(kw)
    return RunReport(**base)


def make_prof(**kw) -> PlayerProfile:
    p = PlayerProfile(scenario="X [Adaptive]")
    p.run_count = 12
    p.ewma_accuracy = 0.70
    p.archetype = kw.pop("archetype", "clicking")
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def settings() -> Settings:
    return Settings(kovaaks_root=".", profile_dir=".")


def ids(insights: list[Insight]) -> set[str]:
    return {i.id for i in insights}


def hist(acc=0.90, score=800.0, kps=0.8, n=12) -> list[dict]:
    return [{"accuracy": acc, "score": score, "kps": kps} for _ in range(n)]


# ------------------------------------------------------------------- rules
def test_overshoot_control_failure_fires():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    prof = make_prof(history=hist())
    got = generate_insights(rep, prof, settings())
    assert "dx-overshoot-control" in ids(got)
    assert "dx-overshoot-strategic" not in ids(got)


def test_overshoot_strategic_is_positive_not_a_fix():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=0.5, accuracy=0.90)
    prof = make_prof(history=hist())
    got = generate_insights(rep, prof, settings())
    (ins,) = [i for i in got if i.id == "dx-overshoot-strategic"]
    assert ins.kind == "positive" and ins.severity == "info"
    assert "approximation" in ins.reasoning   # shot timing not yet measured


def test_input_health_gates_microstructure():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5,
                   input_health={"polling_hz_est": 1000.0, "jitter_ms": 5.0})
    prof = make_prof(history=hist())
    got = generate_insights(rep, prof, settings())
    assert "dx-input-health" in ids(got)
    assert "dx-overshoot-control" not in ids(got)   # suppressed, not reported


def test_accuracy_band_needs_sustained_evidence():
    prof = make_prof(history=hist(acc=0.97))
    got = generate_insights(make_rep(accuracy=0.97), prof, settings())
    assert "dx-acc-above-band" in ids(got)
    # one high run among normal ones must NOT fire
    prof2 = make_prof(history=hist(acc=0.70) + [{"accuracy": 0.97, "score": 800, "kps": 0.8}])
    got2 = generate_insights(make_rep(accuracy=0.97), prof2, settings())
    assert "dx-acc-above-band" not in ids(got2)


def test_archetype_correction_profiles():
    prof_t = make_prof(archetype="tracking", history=hist())
    got_t = generate_insights(make_rep(mean_corrections=2.4), prof_t, settings())
    assert "dx-tracking-jitter" in ids(got_t)
    prof_s = make_prof(archetype="switching", history=hist())
    got_s = generate_insights(make_rep(mean_corrections=1.6), prof_s, settings())
    assert "dx-switch-corrections" in ids(got_s)


def test_bias_and_region_and_fatigue():
    prof = make_prof(ewma_bias=0.3, history=hist())
    prof.region("r4c0").update(0.9)      # top-left corner of the 5x5 grid
    prof.region("r4c0").update(0.9)
    rep = make_rep(fatigue={"level": "declining", "score": 0.4, "runs": 7})
    got = generate_insights(rep, prof, settings())
    assert {"dx-bias", "dx-region-deficit", "dx-fatigue"} <= ids(got)
    (region,) = [i for i in got if i.id == "dx-region-deficit"]
    assert "upper left" in region.title


def test_progress_framing_score_flat_speed_up():
    h = hist(score=800.0, kps=0.8, n=5) + hist(score=805.0, kps=1.0, n=5)
    prof = make_prof(history=h)
    got = generate_insights(make_rep(), prof, settings())
    assert "dx-fitts-progress" in ids(got)


# -------------------------------------------------------------- invariants
def test_every_insight_carries_citations_and_numbers():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5,
                   fatigue={"level": "fatigued", "score": 0.8, "runs": 9})
    prof = make_prof(ewma_bias=0.3, history=hist(acc=0.97))
    got = generate_insights(rep, prof, settings())
    assert got
    for ins in got:
        assert ins.sources, ins.id
        assert ins.body and ins.prescription and ins.reasoning, ins.id
        assert any(ch.isdigit() for ch in ins.reasoning), ins.id
        assert ins.confidence


def test_deterministic():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    prof_a = make_prof(history=hist())
    prof_b = make_prof(history=hist())
    assert generate_insights(rep, prof_a, settings()) == \
        generate_insights(rep, prof_b, settings())


def test_quiet_on_a_clean_run():
    got = generate_insights(make_rep(), make_prof(history=hist()), settings())
    assert not [i for i in got if i.severity != "info"]


# ------------------------------------------------------- cross-session trends
def test_cross_session_trends_trigger_cards():
    from kovadapt.analysis.skill import fit_skill

    def ents(**series):
        out = []
        for i in range(20):
            e = dict(scenario="X", started_iso=f"2026-07-01T{i:02d}:00:00",
                     score=800.0 + (i % 3), accuracy=0.70, kps=0.8,
                     overshoot_rate=0.2, mean_flick_ms=180.0,
                     mean_corrections=1.0, fitts_slope_ms=120.0)
            for k, v in series.items():
                e[k] = v(i)
            out.append(e)
        return out

    # falling Fitts slope over 20 runs + flat score -> cross-session progress card
    trends = fit_skill(ents(fitts_slope_ms=lambda i: 150.0 - 2.0 * i))
    got = generate_insights(make_rep(), make_prof(history=hist()), settings(),
                            trends=trends)
    (card,) = [i for i in got if i.id == "dx-fitts-progress"]
    assert card.kind == "progress" and card.sources
    assert "20" in card.reasoning            # run count
    assert "%" in card.reasoning             # actual magnitude of the change

    # overshoot rising across sessions -> exactly the one extra card, nothing else
    rising = fit_skill(ents(overshoot_rate=lambda i: 0.15 + 0.01 * i))
    got2 = generate_insights(make_rep(), make_prof(history=hist()), settings(),
                             trends=rising)
    (over,) = [i for i in got2 if i.id == "dx-overshoot-control"]
    assert "sessions" in over.title and "20" in over.reasoning and over.sources
    assert "dx-fitts-progress" not in ids(got2)
