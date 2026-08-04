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


# ------------------------------------------------------------- sensitivity
def test_sens_insight_argues_both_ways_never_a_directive():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    got = generate_insights(rep, make_prof(history=hist()), settings())
    (card,) = [i for i in got if i.id == "p-sensitivity-doctrine"]
    assert card.kind == "info" and card.severity == "info"
    body = card.body.lower()
    # the no-directive invariant: BOTH cases are always rendered together
    assert "case for lower" in body and "case for higher" in body
    # never a one-way imperative
    for banned in ("lower your sens", "raise your sens",
                   "increase your sens", "decrease your sens"):
        assert banned not in body
    assert "no direction" in card.prescription.lower()
    # reasoning = the cm/360 math with the live dpi/sens numbers
    assert "cm/360" in card.reasoning and "800" in card.reasoning \
        and "0.022" in card.reasoning
    assert card.sources and card.confidence


def test_sens_insight_needs_configured_dpi_and_sens():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    prof = make_prof(history=hist())
    s0 = settings()
    s0.mouse_dpi = 0.0
    assert "p-sensitivity-doctrine" not in ids(generate_insights(rep, prof, s0))
    s1 = settings()
    s1.game_sens = 0.0
    assert "p-sensitivity-doctrine" not in ids(generate_insights(rep, prof, s1))


def test_sens_insight_silent_on_thin_evidence():
    # clean run: no substantive side -> the card must not appear at all
    got = generate_insights(make_rep(), make_prof(history=hist()), settings())
    assert "p-sensitivity-doctrine" not in ids(got)


def test_a_multi_run_card_never_states_its_verdict_as_if_it_were_this_run():
    """The KPI tile at the top of the Analysis page reads THIS run; these
    cards fire off the last three. Both are correct and they can legitimately
    disagree — open a saved report from yesterday and the tile reads "90% /
    in-band" in green while the card, unqualified, read "Accuracy parked above
    the band" in red and prescribed deliberately slowing down. Its own
    evidence line said "...this run 90%".

    The region card already carries "across runs" for exactly this reason and
    says so in a comment. This pins the same rule for the band cards: a card
    whose condition is evaluated over a window has to name that window in its
    TITLE, where the reader meets it.
    """
    prof = make_prof(history=hist(acc=0.99))
    # this run sits comfortably INSIDE the band the last three ran above
    rep = make_rep(accuracy=0.90)
    got = generate_insights(rep, prof, settings())
    cards = [i for i in got if i.id in ("dx-acc-above-band", "dx-acc-below-band")]
    assert cards, "the band card did not fire on three above-ceiling runs"

    for card in cards:
        assert "recent" in card.title.lower(), (
            f"{card.title!r} states a three-run finding with no window named, "
            "on a page whose headline number is this run alone")
        # and the evidence still carries both, so the card stays checkable
        assert "Last 3 runs" in card.reasoning
        assert "this run 90%" in card.reasoning


# ------------------------------------------- the gate, and what it blames
def test_a_125hz_mouse_is_not_disqualified_from_every_finding():
    """The threshold was 490Hz — "below any competitive polling class", which
    is a judgement about hardware tier, not about whether the measurement
    survives. 125Hz is the USB default and enormously common, and that gate
    withheld every overshoot, correction, bias and moment claim on the page
    from anyone using one.

    Measured before changing it: synthetic flicks with known geometry sampled
    at 1000/500/250/125/62 Hz, through the real segment_flicks. Overshoot came
    back 0.319/0.319/0.320/0.318 and corrections 2.00/2.00/2.00/2.00 —
    indistinguishable from 1000Hz down to 125. At 62Hz both break (+9.4% and
    +55%), because a corrective submovement lasts ~25-50ms and a 16ms sampling
    period leaves 2-3 samples to see it with.
    """
    from kovadapt.analysis.report import input_degraded

    clean_125 = make_rep(input_health={"polling_hz_est": 125.0, "jitter_ms": 0.9})
    assert not input_degraded(clean_125), (
        "a 125Hz device with clean timing is still being disqualified")

    # ...and the rate where the measurement genuinely breaks still is
    assert input_degraded(make_rep(
        input_health={"polling_hz_est": 62.0, "jitter_ms": 0.9}))
    # jitter remains a gate in its own right: that IS contention
    assert input_degraded(make_rep(
        input_health={"polling_hz_est": 1000.0, "jitter_ms": 8.0}))


def test_the_page_names_the_real_cause_not_a_guess():
    """Jitter and a low report rate have different causes and different
    fixes, and the message gave one answer for both — it told a 125Hz mouse
    to go check for background apps. Jitter IS contention: packets that
    arrived were delayed. A low report rate is a device setting, and closing
    Chrome will not change it."""
    from kovadapt.analysis.report import _summary_text

    slow = make_rep(input_health={"polling_hz_est": 60.0, "jitter_ms": 0.5},
                    n_flicks=40, overshoot_rate=0.4, mean_corrections=2.0)
    text = _summary_text(slow, True).lower()
    assert "60hz" in text.replace(" ", "") or "60 hz" in text
    assert "device or driver setting" in text
    # it may SAY "not background load"; what it must not do is send you off
    # to hunt for background apps, which is the fix for the other cause
    assert "background apps" not in text and "optimizer checkup" not in text, (
        f"a slow-reporting mouse is still being sent to chase contention: {text!r}")
    assert "polling rate" in text, "it does not say what would actually fix it"

    noisy = make_rep(input_health={"polling_hz_est": 1000.0, "jitter_ms": 9.0},
                     n_flicks=40, overshoot_rate=0.4, mean_corrections=2.0)
    text2 = _summary_text(noisy, True).lower()
    assert "jitter" in text2 and "background apps" in text2, (
        "real contention should still point at contention")


def test_when_both_causes_are_wrong_the_page_names_both():
    """Branching on jitter alone meant a device that was slow AND contended
    got told only about the contention — so you fix one thing and wonder why
    nothing changed. They have different fixes; both have to be said."""
    from kovadapt.analysis.report import _summary_text

    both = make_rep(input_health={"polling_hz_est": 60.0, "jitter_ms": 9.0},
                    n_flicks=40, overshoot_rate=0.4, mean_corrections=2.0)
    text = _summary_text(both, True).lower()
    assert "60hz" in text.replace(" ", "") or "60 hz" in text, text
    assert "9.0ms" in text.replace(" ", "") or "jitter" in text, text
    assert "device or driver setting" in text
    assert "delaying input" in text
