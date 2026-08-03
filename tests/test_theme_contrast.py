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
                                TEXT_CONTRAST, build_palette, build_qss)

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


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_text_stays_readable_on_a_selected_row(mode, accent):
    """`selection` had NO contrast test, and it is a text background: the
    sheet puts p.fg on it for selected list rows, table rows, menu items and
    the combo popup."""
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    ratio = color.contrast_ratio(pal.fg, pal.selection)
    assert ratio >= TEXT_CONTRAST, (
        f"{mode}/{accent}: text {pal.fg} on selection {pal.selection} "
        f"is {ratio:.2f}:1")


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_a_selected_row_separates_from_the_surface_it_sits_on(mode, accent):
    """Both ends of this have been wrong, in opposite directions.

    Mixing bg toward the accent in linear light gave dark 3.11:1 against
    bg_alt but made light a 1.21:1 cold grey. Fixing the hue with a fixed
    LIGHTNESS offset fixed light's colour and dropped dark to 1.21:1 — a
    selected row you could not pick out of a list. Only fitting against the
    surface holds both, so this asserts against `bg_alt`: that is what
    selections are painted on (table rows, list items, the combo popup),
    not `bg`.
    """
    from kovadapt.gui.theme import SELECTION_CONTRAST

    pal = build_palette(accent=accent, **dict(MODES)[mode])
    ratio = color.contrast_ratio(pal.selection, pal.bg_alt)
    assert ratio >= SELECTION_CONTRAST - 0.01, (
        f"{mode}/{accent}: selection {pal.selection} is {ratio:.2f}:1 from "
        f"the panel {pal.bg_alt} — a selected row would not read as selected")


def test_the_selection_floor_leaves_room_for_the_text_on_top():
    """SELECTION_CONTRAST is 2.6, not the 3.0 control floor, and that is a
    measured trade rather than a soft target: at 3.0 the fit pushes the
    selection far enough from the page that `fg` on top of it lands at
    4.08:1, under the text floor. Raising it must fail here, not on screen.
    """
    from kovadapt.gui.theme import SELECTION_CONTRAST

    assert SELECTION_CONTRAST < CONTROL_CONTRAST
    for _mode, _accent, pal in _palettes():
        assert color.contrast_ratio(pal.fg, pal.selection) >= TEXT_CONTRAST


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_the_selection_is_a_tint_of_the_accent_not_a_grey(mode, accent):
    """It used to be mix(bg, accent) in LINEAR light, which on warm cream
    landed at #e8e2e1 — a cold neutral. On paper that reads as the Windows
    system highlight, the exact thing the table rules were written to kill,
    and it is what Arjun saw in the theme picker.

    So: the selection must carry more chroma than the page it sits on, and
    must lean the same way as its own accent rather than toward grey.
    """
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    sel = color.hex_to_rgb(pal.selection)
    bg = color.hex_to_rgb(pal.bg)
    # chroma proxy: spread between the extreme channels
    spread = max(sel) - min(sel)
    assert spread > max(bg) - min(bg), (
        f"{mode}/{accent}: selection {pal.selection} is flatter than the "
        f"page {pal.bg} — that is a grey, not a tint")
    # and it must be visibly a different surface from the page
    assert pal.selection != pal.bg


def test_accent_name_survives_on_the_palette():
    """gui/ascii_art._cat_coat needs the preset key, not a hex match."""
    for accent in ACCENTS:
        assert build_palette(dark=True, accent=accent).accent_name == accent
    assert build_palette(dark=False, accent="nonexistent").accent_name == "indigo"


@pytest.mark.parametrize("mode,accent", [(m, a) for m, _ in MODES for a in ACCENTS])
def test_the_overlay_sliders_cat_is_visible_on_its_panel(mode, accent):
    """CatSlider replaced QSlider as the overlay-opacity control, so the cat
    IS the handle — the one element showing where the value sits.

    Its coat came from a hand-picked hex table keyed on the accent NAME that
    never consulted pal.is_dark, so the default black cat sat on the dark
    themes' near-black page at 1.06:1 and mint's cream-white cat on cream at
    1.17:1. On three of the four themes the only visible part was its 2px
    eyes. theme.py's own control floor is 3.0.
    """
    pytest.importorskip("PySide6")
    from kovadapt.gui.ascii_art import _cat_coat

    pal = build_palette(accent=accent, **dict(MODES)[mode])
    body, _edge, _eye = _cat_coat(pal)
    ratio = color.contrast_ratio(body.name(), pal.bg_alt)
    assert ratio >= CONTROL_CONTRAST, (
        f"{mode}/{accent}: cat {body.name()} on panel {pal.bg_alt} "
        f"is {ratio:.2f}:1 — the slider handle is invisible")


def test_no_caller_asks_for_a_mono_size_off_the_grid():
    """`theme.mono()` SNAPS to CELL_SIZES rather than honouring its argument,
    and the snap is silent — so `mono(13)` returned 12px, a size smaller than
    the 13px body row it sat in, with its baseline 3.0px high. Measured
    against Segoe UI 13: mono12 is cap -0.8px / baseline -3.0px, mono14 is
    cap +0.6px / baseline -1.0px.

    Ties round DOWN (`min` keeps the first minimum), so an off-grid size
    always lands on the smaller neighbour — the wrong direction for text that
    has to sit level with body copy. Callers must name a real cell size, so
    that choice is visible in the source instead of happening silently.
    """
    import re
    from pathlib import Path

    from kovadapt.gui.theme import CELL_SIZES

    gui = Path(__file__).resolve().parents[1] / "kovadapt" / "gui"
    pattern = re.compile(r"\b(?:theme\.)?(?:mono|_mono_css)\((\d+)")
    offenders = []
    for path in sorted(gui.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                px = int(m.group(1))
                if px not in CELL_SIZES:
                    offenders.append(f"{path.name}:{n} asks for {px}px")
    assert not offenders, (
        "mono sizes must be on theme.CELL_SIZES "
        f"{CELL_SIZES}; these snap silently: {offenders}")


# ------------------------------------------- what is drawn is meant to be seen
def test_border_is_never_the_ink_a_chart_reads_with():
    """`border` is documented as a hairline that "may be low contrast", and it
    is: 1.07:1 against bg_alt on midnight, 1.14 on dark, 1.37 on light. That
    is fine for a widget edge and wrong for anything a reader has to see.

    It was the ink for the bar charts' unfilled remainder — the dotted track
    whose own docstring said it "carries the scale" — for the trend and Fitts
    axes, for the hairline that marks an occupied-but-untouched spawn cell,
    and for the progress meter's remaining run. `border_control` exists for
    exactly this: 2.79-5.65:1, still quiet, actually present.

    The rule this pins: if it is drawn to be seen it clears 3:1, and if it is
    not meant to be seen it should not be drawn at all.
    """
    from pathlib import Path

    gui = Path(__file__).resolve().parent.parent / "kovadapt" / "gui"
    offenders = []
    for f in sorted(gui.glob("*.py")):
        if f.name == "overlay.py":
            # The overlay paints over the GAME, not over bg_alt, at an alpha
            # the author tunes deliberately ("the edge stays findable when
            # faint"). Its contrast question is a different one.
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "pal.border" in line and "border_control" not in line:
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, (
        "pal.border used as painted ink — promote to border_control:\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("name,kw", MODES)
def test_border_control_clears_the_floor_border_does_not(name, kw):
    """The measurement behind the rule above, so it fails if a palette change
    ever makes border_control as faint as border."""
    pal = build_palette(accent="indigo", **kw)
    assert color.contrast_ratio(pal.border_control, pal.bg_alt) >= 2.7, (
        f"{name}: border_control has drifted down to a decorative hairline")
    assert color.contrast_ratio(pal.border, pal.bg_alt) < 2.0, (
        f"{name}: border is no longer the faint hairline its docstring "
        "promises — if it has been promoted, the two tokens have collapsed "
        "into one and border_control has nothing to distinguish it")


# ------------------------------------------- paint-site alpha is invisible here
@pytest.mark.parametrize("name,kw", MODES)
def test_empty_state_text_is_not_knocked_under_the_dim_floor(name, kw):
    """A rendered pin, because this file is structurally blind to it: it walks
    ROLE PAIRS, and the defect was a hard-coded `setAlphaF(0.75)` at the paint
    site. `fg_dim` is already bisected to DIM_CONTRAST; throwing 25% away put
    every empty state at 3.10-3.53:1 — 22-31% under the floor, at normal-text
    size, on the two screens a new user sees first.

    Sampled BELOW the panel title: the title is full-alpha `fg_dim` and reads
    5.12:1, so "darkest ink in the widget" passes while the empty text fails.
    """
    import os
    import sys

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    import numpy as np
    from PySide6.QtGui import QColor, QImage, QPixmap
    from PySide6.QtWidgets import QApplication

    from kovadapt.gui import theme, viz

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    previous = app.styleSheet()
    pal = build_palette(accent="indigo", **kw)
    theme._apply(app, pal)
    try:
        w = viz.AsciiBars(title="flick cost by direction")
        w.set_data([], [])                       # the empty state under test
        w.resize(640, 240)
        pm = QPixmap(640, 240)
        pm.fill(QColor(pal.bg_alt))
        w.render(pm)
        img = pm.toImage().convertToFormat(QImage.Format_RGB32)
        buf = np.frombuffer(img.constBits(), dtype=np.uint8)
        arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :640, :3][:, :, ::-1]

        panel = np.array([int(pal.bg_alt[i:i + 2], 16) for i in (1, 3, 5)])
        body = arr[40:, :]                       # below the 26px title band
        dist = np.abs(body.astype(int) - panel).sum(axis=-1)
        assert dist.max() > 30, "no empty-state ink found below the title"
        ink = body.reshape(-1, 3)[int(np.argmax(dist))]
        hexs = "#{:02x}{:02x}{:02x}".format(*(int(c) for c in ink))
        ratio = color.contrast_ratio(hexs, pal.bg_alt)
        assert ratio >= DIM_CONTRAST - 0.15, (
            f"{name}: empty-state text is {ratio:.2f}:1 against the panel, "
            f"under the {DIM_CONTRAST} floor its own colour was fitted to")
        w.deleteLater()
        app.processEvents()
    finally:
        app.setStyleSheet(previous)


@pytest.mark.parametrize("name,kw", MODES)
def test_the_disabled_primary_rule_restates_every_role_it_needs(name, kw):
    """Structural, over all five accents, because it is cheap: the enabled
    rule sets background, border-color and color, so the disabled rule has to
    set all three or whichever it omits survives from the rule above. It
    omitted border-color."""
    import re

    for accent in ACCENTS:
        pal = build_palette(accent=accent, **kw)
        sheet = build_qss(pal)
        rule = re.search(
            r'QPushButton\[accent="true"\]:disabled\s*\{([^}]*)\}', sheet)
        assert rule, f"{name}/{accent}: the disabled accent rule is gone"
        body = rule.group(1)
        for prop in ("background", "border-color", "color"):
            assert prop in body, (
                f"{name}/{accent}: the disabled accent button does not set "
                f"{prop}, so it keeps the enabled value")
        assert pal.accent not in body, (
            f"{name}/{accent}: the disabled button is painted in the accent")


@pytest.mark.parametrize("name,kw", MODES)
def test_a_disabled_primary_button_looks_dead(name, kw, accent="indigo"):
    """It set background and color but not border-color, so the accent ring
    from the enabled rule survived — byte-identical to the live ring in 20/20.
    On midnight and rgb that ring reads ~10:1 against its fill while the greyed
    label reads 5.26, making the DEAD primary the most saturated thing in the
    row. Its fill was bg_raised too, the fill an ENABLED plain button gets."""
    import os
    import sys

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    import numpy as np
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QPushButton

    from kovadapt.gui import theme

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    previous = app.styleSheet()
    pal = build_palette(accent=accent, **kw)
    theme._apply(app, pal)
    try:
        btn = QPushButton("Play adaptive task")
        btn.setProperty("accent", True)
        btn.setEnabled(False)
        btn.resize(200, 34)
        btn.show()
        app.processEvents()
        img = btn.grab().toImage().convertToFormat(QImage.Format_RGB32)
        buf = np.frombuffer(img.constBits(), dtype=np.uint8)
        arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :200, :3][:, :, ::-1]

        acc = tuple(int(pal.accent[i:i + 2], 16) for i in (1, 3, 5))
        alt = tuple(int(pal.bg_alt[i:i + 2], 16) for i in (1, 3, 5))
        border = tuple(int(x) for x in arr[0, 100])
        fill = tuple(int(x) for x in arr[17, 6])       # inside the border, left of the label

        assert border != acc, (
            f"{name}/{accent}: the disabled button keeps its live accent ring")
        assert all(abs(a - b) <= 3 for a, b in zip(fill, alt)), (
            f"{name}/{accent}: disabled fill is {fill}, not bg_alt {alt} — it "
            "wears the fill an enabled plain button gets")
        btn.deleteLater()
        app.processEvents()
    finally:
        app.setStyleSheet(previous)
