"""Every generated palette must clear its contrast floors.

Palettes used to be 56 hand-picked hex values with nothing checking them,
and three of the four accent presets shipped under the WCAG 4.5:1 floor on
the cream background — as text AND as white-on-fill for the primary button.
Now that colors are derived (gui/color.py), this walks every role pair of
every theme x accent combination so a regression fails the suite instead of
shipping.

Pure math over gui/theme.py + gui/color.py: no Qt, no display.
"""

from __future__ import annotations

import pytest

from kovadapt.gui import color
from kovadapt.gui.theme import (ACCENTS, CONTROL_CONTRAST, DIM_CONTRAST,
                                TEXT_CONTRAST, build_palette)

# (dark, midnight, rgb) for every theme the app can show
MODES = [
    ("light", dict(dark=False)),
    ("dark", dict(dark=True)),
    ("midnight", dict(dark=True, midnight=True)),
    ("rgb", dict(dark=True, rgb=True)),
]


def _palettes():
    for mode_name, kwargs in MODES:
        for accent in ACCENTS:
            yield mode_name, accent, build_palette(accent=accent, **kwargs)


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_body_text_is_readable(mode, accent):
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    for surface in (pal.bg, pal.bg_alt, pal.bg_raised):
        ratio = color.contrast_ratio(pal.fg, surface)
        assert ratio >= TEXT_CONTRAST, (
            f"{mode}/{accent}: body text {pal.fg} on {surface} is {ratio:.2f}:1")


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_dim_text_is_readable(mode, accent):
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    ratio = color.contrast_ratio(pal.fg_dim, pal.bg)
    assert ratio >= DIM_CONTRAST, (
        f"{mode}/{accent}: dim text {pal.fg_dim} is {ratio:.2f}:1 on {pal.bg}")


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_accent_clears_both_of_its_pairs(mode, accent):
    """The exact bug that shipped: an accent has TWO jobs.

    It is link/emphasis text ON the page, and it is the fill UNDER the
    primary button's label. mint, ocean and rose failed both on cream.
    """
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    as_text = color.contrast_ratio(pal.accent, pal.bg)
    on_fill = color.contrast_ratio(pal.accent_fg, pal.accent)
    assert as_text >= TEXT_CONTRAST, (
        f"{mode}/{accent}: accent as text is {as_text:.2f}:1")
    assert on_fill >= TEXT_CONTRAST, (
        f"{mode}/{accent}: label on accent fill is {on_fill:.2f}:1")


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_status_colors_are_readable(mode, accent):
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    for role in ("good", "warn", "bad"):
        value = getattr(pal, role)
        ratio = color.contrast_ratio(value, pal.bg)
        assert ratio >= TEXT_CONTRAST, (
            f"{mode}/{accent}: {role} {value} is {ratio:.2f}:1")


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_control_edges_clear_three_to_one(mode, accent):
    """An input's outline is its affordance; a section rule is not.

    They shared one `border` token, so every control on cream was outlined
    at 1.29:1. border_control carries the real floor.
    """
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    ratio = color.contrast_ratio(pal.border_control, pal.bg)
    assert ratio >= CONTROL_CONTRAST, (
        f"{mode}/{accent}: control edge {pal.border_control} is {ratio:.2f}:1")


def test_light_theme_is_actually_warm_paper():
    """Cream needs chroma. #f6f4ee sat at 0.008 and read as warm grey."""
    pal = build_palette(dark=False)
    r, g, b = color.hex_to_rgb(pal.bg)
    assert r > b, f"light background {pal.bg} is not warm"
    assert (r - b) * 255 >= 8, (
        f"light background {pal.bg} has too little warmth to read as paper")


def test_surfaces_stay_distinguishable():
    """Panels must separate from the page without going near-white."""
    for _mode, _accent, pal in _palettes():
        assert pal.bg != pal.bg_alt
        if not pal.is_dark:
            # the failure that made cream read as a generic white app
            assert color.relative_luminance(pal.bg_alt) < 0.97, (
                f"{pal.name}: panel {pal.bg_alt} is effectively white")


def test_rainbow_holds_perceptual_lightness():
    """HSV swings ~4.4:1 across hue; the iris uses this instead."""
    lums = [color.relative_luminance(color.rainbow_hex(i / 24)) for i in range(24)]
    assert max(lums) / min(lums) < 1.35, (
        f"rainbow luminance varies {max(lums) / min(lums):.2f}x across hue")


def test_accent_name_survives_on_the_palette():
    """gui/ascii_art._cat_coat needs the preset key, not a hex match."""
    for accent in ACCENTS:
        assert build_palette(dark=True, accent=accent).accent_name == accent
    assert build_palette(dark=False, accent="nonexistent").accent_name == "indigo"
