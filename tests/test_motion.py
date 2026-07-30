"""The motion system's rules, pinned.

gui/motion.py is the single source of timing for every animated surface, so
these are the guarantees the rest of the app is allowed to rely on. Pure
timing math over Settings — no Qt, no display.
"""

from __future__ import annotations

import pytest

from kovadapt.config import Settings
from kovadapt.gui import motion


def _s(level: str, **over) -> Settings:
    return Settings(kovaaks_root=".", motion=level, **over)


def test_the_three_levels_scale_as_documented():
    assert motion.scale(_s("full")) == 1.0
    assert 0.0 < motion.scale(_s("reduced")) < 1.0
    assert motion.scale(_s("off")) == 0.0


def test_an_unknown_or_missing_level_falls_back_to_full():
    """The setting is user-editable text in a JSON file on disk, so a typo or
    an older/newer file must not disable motion or crash a paint path."""
    assert motion.level(_s("nonsense")) == motion.FULL
    assert motion.level(_s("")) == motion.FULL

    class Bare:                      # no `motion` attribute at all
        pass

    assert motion.level(Bare()) == motion.FULL
    assert motion.scale(Bare()) == 1.0


def test_off_means_no_animation_at_all():
    """`animates` False is the signal to paint the END STATE directly. A
    zero-length animation still schedules a timer and still repaints."""
    off = _s("off")
    assert motion.animates(off) is False
    assert motion.ms(off, motion.CEREMONY) == 0
    assert motion.stagger(off, 12) == 0
    assert motion.ambient(off) is False


def test_reduced_keeps_reveals_but_drops_ambient():
    """The whole point of the middle setting: a run landing still animates,
    the backdrop eye and every idle loop do not."""
    red = _s("reduced")
    assert motion.animates(red) is True
    assert motion.ms(red, motion.BASE) > 0
    assert motion.ambient(red) is False
    assert motion.ambient(_s("full")) is True


def test_the_duration_ladder_is_ordered_and_short():
    rungs = [motion.INSTANT, motion.FAST, motion.BASE, motion.SLOW,
             motion.SLOWER, motion.CEREMONY]
    assert rungs == sorted(rungs), "the ladder must be monotonic"
    # exactly one sub-300ms rung is the "default foreground move"
    assert motion.BASE < 300
    # ambient never resolves fast enough to read as twitch
    assert motion.AMBIENT_MIN >= 800


def test_glyph_clock_stays_in_the_flicker_safe_band():
    """Character-quantized animation runs at ~15 Hz: between-frames produce no
    visible change while costing a full repaint, and above this a glyph grid
    reads as flicker because adjacent cells cross ramp thresholds apart."""
    assert 10 <= motion.GLYPH_HZ <= 15
    assert motion.GLYPH_MS == pytest.approx(1000 / motion.GLYPH_HZ, abs=1)


def test_exits_run_faster_than_entrances():
    full = _s("full")
    assert motion.exit_ms(full, motion.BASE) < motion.ms(full, motion.BASE)


def test_stagger_grows_with_distance_but_is_capped():
    full = _s("full")
    near, far = motion.stagger(full, 1), motion.stagger(full, 6)
    assert near < far
    # a huge distance must not leave the last cell waiting seconds
    assert motion.stagger(full, 500) <= motion.STAGGER_CAP


def test_stagger_is_by_distance_not_index():
    """Index order looks like a grid being typed; distance from an origin looks
    like one event propagating outward."""
    origin = (2, 2)
    assert motion.grid_distance(2, 2, origin) == 0.0
    d_near = motion.grid_distance(2, 3, origin)
    d_far = motion.grid_distance(0, 0, origin)
    assert d_near < d_far
    assert motion.stagger(_s("full"), d_near) < motion.stagger(_s("full"), d_far)


def test_easing_curves_are_sane_and_clamped():
    for fn in (motion.ease_out, motion.ease_in, motion.ease_in_out):
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(1.0) == pytest.approx(1.0)
        assert fn(-5.0) == pytest.approx(0.0)      # clamped, never negative
        assert fn(5.0) == pytest.approx(1.0)
    # out decelerates (ahead at the midpoint), in accelerates (behind)
    assert motion.ease_out(0.5) > 0.5
    assert motion.ease_in(0.5) < 0.5


def test_motion_survives_a_settings_json_round_trip(tmp_path, monkeypatch):
    """New Settings fields need defaults or every existing file breaks
    (CLAUDE.md). Pass profile_dir explicitly: config.py evaluates its default
    at import time, so patching Path.home afterwards does not move it."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    s = Settings(kovaaks_root=".", profile_dir=str(tmp_path), motion="reduced")
    s.save(tmp_path / "settings.json")
    again = Settings.load(tmp_path / "settings.json")
    assert again.motion == "reduced"

    # a file written before the field existed still loads, at the default
    import json

    raw = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    raw.pop("motion")
    (tmp_path / "old.json").write_text(json.dumps(raw), encoding="utf-8")
    assert Settings.load(tmp_path / "old.json").motion == motion.FULL
