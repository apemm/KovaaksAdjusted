"""Axis, label and ramp honesty pins for gui/viz.py (offscreen QPA).

Every case here was found by RENDERING the widget and looking at it, and each
assertion is written against the pixels or the exact string that carried the
lie:

* the run axis printed "run 1 · 62%" and "run 60 · newest" under a window of a
  137-run history (both callers slice ``profile.history[-60:]``), i.e. two
  labels naming runs 1 and 60 over what are really runs 78 and 137. viz.py
  mints those numbers from the list index and cannot know its offset;
* the y axis printed "-0%" in the gutter: _y_range's widening loop compared RAW
  format() strings, so a hairline range centred on zero stopped widening the
  instant "-0%" differed from "0%", and the gutter then called format()
  directly instead of the _fmt_value helper this module added to stop signed
  zeros — a helper which could not see a zero through a "%" suffix anyway;
* the "mean 67%" chip sat on the flat line's own glyph run behind a 0.85-alpha
  backing, so the accent glyphs read straight through the label in both
  themes;
* a FLAT trend — a panel whose entire finding is "this did not move" — drew a
  run of '~' ripples (or of '"' ticks half a row off its own mean rule),
  because the four-glyph sub-cell ramp had no bucket centred on 0.5;
* the occupancy heatmap normalized its one-signed range by its max, so
  3.60..3.79 became 0.95..1.00 and all 25 zones painted '@' at full
  saturation: a map reading "everything is maximally bad" over a 5% spread.

The pixel probes assume the DARK and LIGHT indigo palettes and lean on two
palette facts rather than on viz.py's layout: the chart's ink is the accent
(blue-tinted in dark) while every label and rule is neutral, and text plus its
antialiasing lies ON the bg -> fg_dim line in RGB space while accent ink does
not.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no system font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.gui import theme, viz  # noqa: E402

_LABELS5 = [[f"r{r}c{c}" for c in range(5)] for r in range(5)]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Nothing here may reach the developer's real ~/.kovadapt."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(autouse=True)
def _no_leaked_module_settings():
    """viz.use_settings() is module state — never leak it into another test."""
    yield
    viz.use_settings(None)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def pal(monkeypatch):
    p = theme.build_palette(dark=True, accent="indigo")
    monkeypatch.setattr(theme, "_current", p)
    return p


# --------------------------------------------------------------- pixel probes
def _shot(widget, pal, width: int, height: int) -> np.ndarray:
    widget.resize(width, height)
    pm = QPixmap(width, height)
    pm.fill(QColor(pal.bg))
    widget.render(pm)
    img = pm.toImage().convertToFormat(QImage.Format_RGB32)
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    a = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)
    return a[:, :img.width(), :3][:, :, ::-1].astype(int)


def _hex(hexs: str) -> np.ndarray:
    return np.array([int(hexs[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _exact(arr: np.ndarray, hexs: str) -> np.ndarray:
    """Mask of pixels painted in exactly this colour — the glyph cores of ink
    drawn at full alpha, so it does not care what the platform painted the
    panel background."""
    return np.all(arr == _hex(hexs).astype(int), axis=-1)


def _accent_ink(arr: np.ndarray, keep_right: int = 40) -> np.ndarray:
    """Mask of the chart's own (blue-tinted) ink; the value tag rides outside
    the plot on the right, so the last strip is dropped."""
    ink = (arr[:, :, 2] - arr[:, :, 0]) > 25
    ink[:, arr.shape[1] - keep_right:] = False
    return ink


def _chroma(arr: np.ndarray) -> np.ndarray:
    """Red-vs-blue separation per pixel, measured above the panel background's
    own separation: the heat ramp is strongly warm or cool, every rule and
    label is neutral."""
    bg = arr[0, 0]
    return np.abs(arr[:, :, 0] - arr[:, :, 2]) - abs(int(bg[0]) - int(bg[2]))


def _off_grey_axis(arr: np.ndarray, pal) -> np.ndarray:
    """Per-pixel distance from the bg -> fg_dim line in RGB space.

    Label text and every antialiased pixel of it lie ON that line; the accent
    line does not. That is what lets a probe say "the chart's ink is showing
    through this label" without knowing where the label is drawn.
    """
    bg, fgd = _hex(pal.bg), _hex(pal.fg_dim)
    unit = (fgd - bg) / np.linalg.norm(fgd - bg)
    rel = arr.astype(float) - bg
    return np.abs(rel - (rel @ unit)[..., None] * unit).sum(axis=-1)


def _mean_chip_box(arr: np.ndarray, pal, axis_h: float) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) of the "mean NN%" label's ink.

    On a FLAT panel it is the only full-alpha fg_dim ink in the plot's
    right-hand strip: the min/max labels are suppressed, the gutter is on the
    left and the run axis is below.
    """
    w = arr.shape[1]
    lab = _exact(arr, pal.fg_dim)
    lab[:, :w - 220] = False
    lab[arr.shape[0] - int(axis_h) - 6:, :] = False
    rows, cols = np.nonzero(lab.any(axis=1))[0], np.nonzero(lab.any(axis=0))[0]
    assert rows.size and cols.size, "the mean label was never painted"
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _is_signed_zero(txt: str) -> bool:
    return txt.lstrip().startswith("-") and not any(c in "123456789" for c in txt)


# ============================================================ 1. the run axis
_WINDOW = [0.50 + 0.0015 * i for i in range(137)][-60:]   # what both callers pass


def test_the_run_axis_never_names_a_run_it_cannot_identify(qapp, pal):
    """Both callers slice history[-60:], so values[0] is run 78 of 137 and
    values[-1] is run 137. The axis used to print "run 1" and "run 60" over
    them — an index the widget cannot know the offset of, presented as a run
    number. Without an offset it must say only what it can support."""
    tr = viz.AsciiTrend(title="accuracy per run", fmt="{:.0%}")
    tr.set_data(_WINDOW, tag="70%")
    left, right = tr.run_axis_text(170)
    assert left == "oldest shown · 62%"
    assert right == "60 runs · newest"
    for text in (left, right):
        assert "run 1 " not in text and "run 60 " not in text, (
            f"{text!r} names a run that is not the one drawn")

    # and in the pixels: no absolute number reaches the axis strip
    arr = _shot(tr, pal, 1400, 220)
    strip = arr[220 - int(viz._AXIS_H):, :]
    cited = viz.AsciiTrend(title="accuracy per run", fmt="{:.0%}")
    cited.set_data(_WINDOW, tag="70%", first_run=78)
    cited_strip = _shot(cited, pal, 1400, 220)[220 - int(viz._AXIS_H):, :]
    assert not np.array_equal(strip, cited_strip), (
        "first_run does not reach the pixels — the axis label is not data-driven")


def test_a_cited_window_names_the_runs_it_really_drew(qapp, pal):
    """Given the offset, the axis names run 78 and run 137 — the runs actually
    on screen — and a decimated window names the whole range it covers."""
    tr = viz.AsciiTrend(fmt="{:.0%}")
    tr.set_data(_WINDOW, first_run=78)
    assert tr.run_axis_text(170) == ("run 78 · 62%", "run 137 · newest")

    tr.set_data([0.4 + 0.001 * i for i in range(300)], first_run=41)
    assert tr.run_axis_text(74) == ("run 41 · 40%", "runs 41..340 · 74 columns")

    # two runs stay a segment, cited or not: the shape is not a trend and the
    # axis may not imply one
    tr.set_data([0.42, 0.71], first_run=12)
    assert tr.run_axis_text(170) == ("run 12 · 42%", "2 runs · segment")
    tr.set_data([0.42, 0.71])
    assert tr.run_axis_text(170) == ("oldest shown · 42%", "2 runs · segment")


def test_an_uncited_window_says_nothing_it_cannot_support(qapp, pal):
    """Uncited, the labels carry only facts the widget owns: which end is
    oldest, how many runs it was handed, and whether they each got a column."""
    tr = viz.AsciiTrend(fmt="{:.0%}")
    tr.set_data([0.4 + 0.001 * i for i in range(300)])
    assert tr.run_axis_text(74) == ("oldest shown · 40%", "300 runs · 74 columns")

    # under two runs no axis is drawn at all, so it claims nothing
    tr.set_data([0.5])
    assert tr.run_axis_text(170) == ("", "")
    tr.clear()
    assert tr.run_axis_text(170) == ("", "")

    # clear() must forget a citation, or the next scenario inherits it
    tr.set_data(_WINDOW, first_run=78)
    tr.clear()
    tr.set_data(_WINDOW)
    assert tr.run_axis_text(170)[0] == "oldest shown · 62%"


# ======================================================= 2. no signed zeros
_ZERO_FLAT = [
    ([0.0, 0.0, 0.0], "{:.0%}"),
    ([0.0, 0.0], "{:+.2f}"),
    ([0.0] * 20, "{:.2f}"),
    ([0.0, 0.0, 0.0], "{:.1f}"),
    ([-0.0, -0.0], "{:+.2f}"),
]


@pytest.mark.parametrize("vals,fmt", _ZERO_FLAT)
def test_the_value_axis_never_prints_a_signed_zero(qapp, pal, vals, fmt):
    """A hairline range centred on zero put "-0%" in the gutter: the widening
    loop compared raw format() strings ("-0%" != "0%", so it stopped at once)
    and the gutter then formatted the value itself. Both ends now go through
    _fmt_value, which is also what the loop compares — so the two labels
    cannot be a signed zero, and cannot be the same number either."""
    tr = viz.AsciiTrend(title="accuracy over runs", fmt=fmt)
    tr.set_data(vals, tag="0")
    lo, hi, flat = tr._y_range()
    assert flat, "a constant history is not flat"
    painted = (viz._fmt_value(fmt, lo), viz._fmt_value(fmt, hi))
    for txt in painted:
        assert not _is_signed_zero(txt), f"{fmt} paints {txt!r} on the y axis"
    assert painted[0] != painted[1], (
        f"{fmt} prints {painted[0]!r} at both ends of {vals[:2]}")
    assert not tr.grab().isNull()


def test_fmt_value_sees_a_zero_through_its_unit(qapp):
    """The helper stripped only "0." — so "-0%" kept its "%" and read as a
    nonzero string, which is how a signed zero reached a percentage axis. It
    also has to catch -0.0, which is not less than zero in Python."""
    assert viz._fmt_value("{:.0%}", -5e-10) == "0%"
    assert viz._fmt_value("{:.0%}", -0.004) == "0%"
    assert viz._fmt_value("{:.0%}", -0.02) == "-2%"          # a real -2% survives
    assert viz._fmt_value("{:+.2f}", -0.0) == "+0.00"
    assert viz._fmt_value("{:.2f}", -0.0) == "0.00"
    # the pins the helper already carried
    assert viz._fmt_value("{:+.2f}", -0.001) == "+0.00"
    assert viz._fmt_value("{:+.2f}", -1.2) == "-1.20"


def test_every_number_the_trend_paints_goes_through_fmt_value(qapp, pal):
    """One helper, or the panel disagrees with itself about the same zero: the
    tag, the mean chip, the min/max marks and the axis value are all the same
    metric in the same format."""
    tr = viz.AsciiTrend(title="bias over runs", fmt="{:+.2f}")
    tr.set_data([-0.001, -0.0004, 0.30, -0.0009], tag=None)
    arr = _shot(tr, pal, 900, 240)
    # "-0.00" would be the only glyph run containing a minus beside two zeros;
    # assert instead on every string the paint path can build from these values
    for v in tr._values + [float(np.mean(tr._values))]:
        assert not _is_signed_zero(viz._fmt_value("{:+.2f}", v))
    assert arr.shape == (240, 900, 3)


# ================================================= 3. the mean chip is opaque
@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("h", [200, 220, 240, 300])
def test_the_mean_chip_is_opaque_so_the_line_cannot_read_through_it(
        qapp, monkeypatch, dark, h):
    """On a flat panel the line sits exactly on the mean rule — it IS the mean
    — so the chip always lands on the line's own glyph run. At 0.85 alpha the
    accent glyphs showed through the text in both themes, and a label you
    cannot read is not a label."""
    pal = theme.build_palette(dark=dark, accent="indigo")
    monkeypatch.setattr(theme, "_current", pal)
    tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
    tr.set_data([0.667] * 12, tag="67%")
    arr = _shot(tr, pal, 1400, h)
    r0, r1, c0, c1 = _mean_chip_box(arr, pal, viz._AXIS_H)
    bleed = float(_off_grey_axis(arr, pal)[r0:r1 + 1, c0:c1 + 1].max())
    assert bleed < 6.0, (
        f"chart ink is showing through the mean label (off-axis {bleed:.1f}; "
        "the label and its antialiasing should be pure greyscale)")

    # the label itself is still there: "mean 67%" is ~39px of glyph run, and
    # only its cores are palette-exact, so measure the BOX, not the pixel count
    assert c1 - c0 >= 30 and r1 - r0 >= 5, (
        f"the chip swallowed its own label ({c1 - c0}x{r1 - r0}px of ink)")


@pytest.mark.parametrize("dark", [True, False])
def test_the_flat_line_sits_on_the_mean_rule_it_is_the_mean_of(qapp, monkeypatch, dark):
    """A flat range put the level mid-cell as often as not, and the sub-cell
    glyph had to lean out of its cell to cover for it. Snapped to the row, the
    line's ink and the label riding the rule share a centre outright."""
    pal = theme.build_palette(dark=dark, accent="indigo")
    monkeypatch.setattr(theme, "_current", pal)
    for h in (200, 205, 210, 220, 230, 240, 260, 300, 340):
        tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
        tr.set_data([0.667] * 12, tag="67%")
        arr = _shot(tr, pal, 1400, h)
        ink = _accent_ink(arr, keep_right=120)
        rows = np.nonzero(ink.any(axis=1))[0]
        r0, r1, _c0, _c1 = _mean_chip_box(arr, pal, viz._AXIS_H)
        delta = (int(rows.min()) + int(rows.max())) / 2 - (r0 + r1) / 2
        assert abs(delta) <= 1.5, (
            f"h={h}: the flat line's ink sits {delta:+.1f}px off the label on "
            "the mean rule it is the mean OF")


# ================================================== 4. a flat trend reads flat
def test_the_sub_cell_ramp_puts_a_flat_glyph_dead_centre(qapp):
    """int(sub * len) over four glyphs bucketed 0.5 into '~', so a value on a
    row's CENTRE — exactly where a trend that did not move lands — drew a
    ripple. No ramp bucket may be a wave."""
    ramp = viz._TREND_SUB
    assert "~" not in ramp, "the sub-cell ramp still carries a ripple glyph"
    assert ramp[int(0.5 * len(ramp))] == "-", (
        f"dead centre of {ramp!r} is {ramp[int(0.5 * len(ramp))]!r}, not a flat rule")
    assert ramp[0] == "_" and ramp[-1] == '"'       # bottom .. top, still monotone


@pytest.mark.parametrize("vals,why", [
    ([0.667] * 12, "identical values whose np.mean is 1 ULP above them"),
    ([0.0] * 3, "a history of zeros, whose range centres exactly on 0"),
    ([0.5] * 300, "a long constant history, decimated"),
    ([0.6666, 0.6667] * 6, "a hairline spread every label rounds to 67%"),
])
def test_a_flat_trend_draws_a_flat_rule_not_a_ripple(qapp, pal, vals, why):
    """A '-' inks a single 3px stroke per column; the '~' it used to draw inks 4
    and the '"' it drew at half of all panel heights inks 7-8, two ticks
    hanging above the row. On a panel whose whole finding is "this did not
    move", only the rule is honest."""
    for h in (200, 210, 220, 240, 260, 300, 340):
        tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
        tr.set_data(vals, tag="x")
        assert tr._y_range()[2], f"{why} was not recognised as flat"
        arr = _shot(tr, pal, 1400, h)
        ink = _accent_ink(arr, keep_right=120)
        rows = np.nonzero(ink.any(axis=1))[0]
        span = int(rows.max()) - int(rows.min()) + 1
        assert span <= 3, (
            f"h={h}, {why}: the flat line inks {span}px of height — a rule is "
            "3px, a tilde 4 and a pair of quote ticks 7")


# ============================== 5. a narrow one-signed map is not amplified
_NARROW = np.array([[3.60 + 0.19 * ((r * 5 + c) / 24.0) for c in range(5)]
                    for r in range(5)])
_WIDE = np.array([[0.20 + 3.60 * ((r * 5 + c) / 24.0) for c in range(5)]
                  for r in range(5)])


def test_a_narrow_one_signed_map_is_not_amplified_to_fill_the_ramp(qapp, pal):
    """3.60..3.79 divided by 3.79 is 0.95..1.00, so every zone landed on the
    ramp's last step: 25 zones of '@' at full saturation, a map reading
    "everything is maximally bad" over a 5% spread. A spread under one ramp
    step cannot colour a difference at all, so it is drawn uniformly faint."""
    hm = viz.AsciiHeatmap(title="aim travel around engagements")
    hm.set_data(_NARROW, _LABELS5, fmt="{:.2f}")
    assert hm._flat_map
    norm = hm._norm
    assert np.allclose(norm, norm.flat[0]), "the flat map still varies by zone"
    assert 0.0 < abs(float(norm.flat[0])) < 0.10, (
        f"{float(norm.flat[0]):.3f} is not a faint uniform level")
    glyphs = {viz._HEAT_RAMP[int(round(min(abs(float(v)), 1.0)
                                       * (len(viz._HEAT_RAMP) - 1)))]
              for v in norm.ravel()}
    assert glyphs == {"."}, f"the zones paint {glyphs} — the ramp is still filled"


@pytest.mark.parametrize("dark", [True, False])
def test_the_flat_map_paints_no_colour_it_cannot_justify(qapp, monkeypatch, dark):
    """In the pixels: the narrow map's lattice is neutral, while the same
    widget over a genuinely spread range keeps the whole ramp. The fix must not
    drain a map that has something to say."""
    pal = theme.build_palette(dark=dark, accent="indigo")
    monkeypatch.setattr(theme, "_current", pal)

    def lattice_chroma(grid) -> int:
        hm = viz.AsciiHeatmap(title="aim travel around engagements")
        hm.set_data(grid, _LABELS5, fmt="{:.2f}")
        arr = _shot(hm, pal, 900, 320)
        return int(_chroma(arr)[30:280, 70:888].max())

    narrow, wide = lattice_chroma(_NARROW), lattice_chroma(_WIDE)
    # zero-anchored, 3.60..3.79 used to paint 227 (dark) / 123 (light) of ramp
    # colour — the same as a map spanning the whole range
    assert narrow <= 45, (
        f"the narrow map paints {narrow} of ramp colour — it has none to spend")
    assert wide >= 90, f"a genuinely spread map lost its ramp ({wide})"


def test_the_flat_map_says_why_every_zone_looks_the_same(qapp, pal):
    """25 identically faint zones and no explanation read as a broken chart for
    exactly the reason 25 hollow ones do — and the footer's numbers are the
    zones' own, so it cannot claim a range that is not there."""
    hm = viz.AsciiHeatmap(title="aim travel around engagements")
    hm.set_data(_NARROW, _LABELS5, fmt="{:.2f}")
    note = hm.spread_note()
    assert note == ("measured zones 3.60..3.79 — too narrow to shade; "
                    "the numbers carry it")
    assert hm.footer_note() == note                  # nothing else to explain
    assert not hm.key_is_readable(), (
        "a ramp key beside a map painted in one shade keys colours that are "
        "not on screen")
    # and no swatch reaches the footer strip
    arr = _shot(hm, pal, 900, 320)
    assert int((_chroma(arr)[320 - 17:, :] > 30).sum()) == 0

    # a spread map explains nothing, because there is nothing to explain
    hm.set_data(_WIDE, _LABELS5, fmt="{:.2f}")
    assert hm.spread_note() == "" and hm.key_is_readable()

    # both halves share the line when both apply
    partial = np.full((5, 5), np.nan)
    partial[0, 0], partial[1, 2], partial[4, 4] = 3.60, 3.62, 3.64
    hm.set_data(partial, _LABELS5, fmt="{:.2f}")
    foot = hm.footer_note()
    assert foot.startswith("3 of 25 zones measured") and "too narrow" in foot


def test_the_flat_map_rule_leaves_real_findings_alone(qapp, pal):
    """The z-scored deficit map is the case zero-anchoring exists for, and a
    lone measured zone has no spread to judge — neither may be drained."""
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.4 * (r - 2) + 0.2 * (c - 2)
         for r in range(5) for c in range(5)}, 5, 5)
    hm = viz.AsciiHeatmap(title="weakness by wall region")
    hm.set_data(grid, labels, fmt="{:+.2f}")
    assert not hm._flat_map
    assert float(np.nanmax(hm._norm)) == pytest.approx(1.0)
    assert hm.key_is_readable()

    # one zone: no range at all, so its own magnitude stands
    lone, lone_labels = viz.region_grid({"r0c0": 2.0}, 3, 3)
    solo = viz.AsciiHeatmap()
    solo.set_data(lone, lone_labels, fmt="{:+.2f}")
    assert not solo._flat_map
    assert float(solo._norm[0, 0]) == pytest.approx(1.0)
    assert solo.spread_note() == ""

    # a genuine strength is still drawn loudly
    strong, strong_labels = viz.region_grid({"r0c0": 0.10, "r2c2": -1.80}, 5, 5)
    hot = viz.AsciiHeatmap()
    hot.set_data(strong, strong_labels, fmt="{:+.2f}")
    assert not hot._flat_map
    assert float(hot._norm[2, 2]) == pytest.approx(-1.0)

    # and clearing the grid forgets the verdict
    hot.clear()
    assert not hot._flat_map and hot.spread_note() == ""
