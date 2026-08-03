"""Honesty pins for gui/viz.py (offscreen QPA): no chart may draw a claim its
data cannot carry, and no two parts of one panel may disagree.

Every case here was found by LOOKING at a render, and each assertion is
written against the pixels or the exact sentence that carried the lie:

* the bars' footer spelled out "top two: left 0.42 / right 0.31 = 1.35x"
  underneath a title reading "INPUT TIMING TOO NOISY TO COMPARE DIRECTIONS
  THIS RUN", and printed "4.00x" off one flick per side. Flick cost is
  overshoot plus corrections — pure microstructure — so the sentence needs the
  caller's input-health verdict (viz.py cannot compute it) and a sample;
* AsciiTrend drew a *perfectly flat* history nine grid rows away from its own
  mean rule (np.mean over identical values lands one ULP off them and the mean
  is folded into the range), and drew [0.6666, 0.6667] as a full-panel
  sawtooth while all six labels on the panel printed "67%";
* two runs were interpolated into ~170 glyph marks of steady climb, which is
  the exact thing the run ticks exist to stop a viewer counting;
* the heatmap's colour key rendered ten swatches of #f6571c..#ff5414 between
  "3.60" and "3.79" — one colour, presented as a scale;
* the lattice axes spoke a FOURTH region vocabulary (bottom/low/middle/high/
  top x far left/left/centre/right/far right) while the Coach, on the same
  screen, called the same zone "upper left";
* "every bar is 0.00 — no cost recorded to compare" was returned for a set
  whose value column printed -0.20 / -0.10 / -0.30.

The pixel probes assume the DARK indigo palette: its accent is blue-tinted ink
(B-R >> 0) while every label, rule and axis is neutral grey, which is what
lets a probe tell chart ink from furniture without reimplementing the layout.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no system font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.analysis.insights import _region_words  # noqa: E402
from kovadapt.gui import theme, viz  # noqa: E402


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
def _image(widget, pal, width: int, height: int) -> QImage:
    widget.resize(width, height)
    pm = QPixmap(width, height)
    pm.fill(QColor(pal.bg))
    widget.render(pm)
    return pm.toImage().convertToFormat(QImage.Format_RGB32)


def _rgb(img: QImage) -> np.ndarray:
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    a = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)
    return a[:, :img.width(), :3][:, :, ::-1].astype(int)


def _shot(widget, pal, width: int, height: int) -> np.ndarray:
    return _rgb(_image(widget, pal, width, height))


def _colours(band: np.ndarray) -> int:
    """How many distinct colours a strip holds — 1 means nothing was drawn on
    it, whatever the platform painted the panel's background."""
    return len(np.unique(band.reshape(-1, 3), axis=0))


def _exact(arr: np.ndarray, hexs: str) -> np.ndarray:
    """Mask of pixels painted in exactly this colour — the glyph cores of ink
    drawn at full alpha. Palette-exact, so it does not care what the app-wide
    stylesheet painted the panel's background (other suites install one)."""
    want = np.array([int(hexs[i:i + 2], 16) for i in (1, 3, 5)])
    return np.all(arr == want, axis=-1)


def _chroma(arr: np.ndarray) -> np.ndarray:
    """Red-vs-blue separation per pixel: the heat ramp is strongly warm or
    cool, every rule and label is neutral. Measured ABOVE the panel
    background's own separation, which is not zero in every theme."""
    bg = arr[0, 0]
    base = abs(int(bg[0]) - int(bg[2]))
    return np.abs(arr[:, :, 0] - arr[:, :, 2]) - base


def _accent_ink(arr: np.ndarray, keep_right: int = 40) -> np.ndarray:
    """Mask of the chart's own (blue-tinted) ink; the value tag rides outside
    the plot on the right, so the last strip is dropped."""
    ink = (arr[:, :, 2] - arr[:, :, 0]) > 25
    ink[:, arr.shape[1] - keep_right:] = False
    return ink


def _cell_heights(ink: np.ndarray, cell: int = 8) -> dict[int, int]:
    """Per glyph-cell column, the vertical extent of the ink in it."""
    out: dict[int, tuple[int, int]] = {}
    for x in np.nonzero(ink.any(axis=0))[0]:
        ys = np.nonzero(ink[:, x])[0]
        lo, hi = out.get(int(x) // cell, (10 ** 9, -1))
        out[int(x) // cell] = (min(lo, int(ys.min())), max(hi, int(ys.max())))
    return {k: hi - lo for k, (lo, hi) in out.items()}


def _middle_half(heights: dict[int, int]) -> list[int]:
    """The ink heights of the middle half of the inked columns — clear of the
    end marks, so it measures the LINE itself."""
    keys = sorted(heights)
    q = max(len(keys) // 4, 1)
    return [heights[k] for k in keys[q:-q]]


def _row_span(mask: np.ndarray) -> tuple[int, int]:
    rows = np.nonzero(mask.any(axis=1))[0]
    assert rows.size, "nothing was drawn at all"
    return int(rows.min()), int(rows.max())


def _clusters(rows: np.ndarray, gap: int = 3) -> list[np.ndarray]:
    return np.split(rows, np.nonzero(np.diff(rows) > gap)[0] + 1)


# ============================================================== the bars' ratio
def _bars(pal, values, title="flick cost by direction", **kw):
    bars = viz.AsciiBars(title=title)
    bars.set_data(["left", "vertical", "right"][:len(values)], values,
                  [f"{n} flicks" for n in kw.get("ratio_counts", [])] or None,
                  **kw)
    return bars


def test_the_bars_print_no_ratio_until_the_caller_vouches_for_one(qapp, pal):
    """The rendered lie: a panel titled "INPUT TIMING TOO NOISY TO COMPARE
    DIRECTIONS THIS RUN" with a precise ratio spelled out underneath. viz.py
    cannot check input health (analysis.report.input_degraded is the app's one
    definition and it needs a RunReport), so the footer is OFF unless the
    caller hands over the per-bar counts that justify it."""
    h = 300
    noisy = viz.AsciiBars(
        title="input timing too noisy to compare directions this run")
    noisy.set_data(["left", "vertical", "right"], [0.42, 0.11, 0.31],
                   ["7 flicks", "3 flicks", "9 flicks"])
    assert noisy.ratio_footer() == ""
    arr = _shot(noisy, pal, 690, h)
    band = arr[h - int(viz._FOOTER_H) + 1:, 8:]
    assert _colours(band) == 1, "a sentence was painted in the footer band"

    # ... and the same data WITH the caller's permission does print it
    ok = viz.AsciiBars(title="your left flicks cost 1.4x more than your right")
    ok.set_data(["left", "vertical", "right"], [0.42, 0.11, 0.31],
                ["7 flicks", "3 flicks", "9 flicks"], ratio_counts=[7, 3, 9])
    assert ok.ratio_footer() == "top two: left 0.42 / right 0.31 = 1.35x"
    arr = _shot(ok, pal, 690, h)
    assert _colours(arr[h - int(viz._FOOTER_H) + 1:, 8:]) > 1, "the footer vanished"


def test_a_ratio_needs_three_samples_in_every_bar_it_names(qapp, pal):
    """"top two: left 0.40 / right 0.10 = 4.00x" was printed under a title
    reading "only 1 left / 1 right flicks — too few to call a side". The floor
    is analysis.movement.directional_bias's own: three flicks a side."""
    h = 300
    thin = viz.AsciiBars(title="only 1 left / 1 right flicks — too few to call a side")
    thin.set_data(["left", "right"], [0.40, 0.10], ["1 flick", "1 flick"],
                  ratio_counts=[1, 1])
    assert thin.ratio_footer() == ""
    assert _colours(_shot(thin, pal, 690, h)[h - int(viz._FOOTER_H) + 1:, 8:]) == 1

    thin.set_data(["left", "right"], [0.40, 0.10], ratio_counts=[9, 2])
    assert thin.ratio_footer() == "", "the second bar carries only 2 flicks"
    thin.set_data(["left", "right"], [0.40, 0.10], ratio_counts=[3, 3])
    assert thin.ratio_footer().endswith("4.00x")
    assert viz._RATIO_MIN_N == 3


def test_no_footer_sentence_contradicts_the_value_column(qapp, pal):
    """"every bar is 0.00 — no cost recorded to compare" was returned for a set
    whose value column printed -0.20 / -0.10 / -0.30."""
    bars = viz.AsciiBars()
    bars.set_data(["left", "vertical", "right"], [-0.20, -0.10, -0.30],
                  ratio_counts=[9, 6, 8])
    note = bars.ratio_footer()
    printed = [f"{v:.2f}" for v in (-0.20, -0.10, -0.30)]
    assert "0.00" not in note, f"{note!r} claims a zero the chart never prints"
    assert "-0.10" in note and note.startswith("no bar is above zero")
    assert printed == ["-0.20", "-0.10", "-0.30"]      # what the column shows

    # a genuine all-zero set keeps its sentence, and a bar that PRINTS 0.00 is
    # never cited as "the only bar above zero"
    bars.set_data(["left", "right"], [0.0, 0.0], ratio_counts=[40, 38])
    assert bars.ratio_footer() == "every bar is 0.00 — no cost recorded to compare"
    bars.set_data(["left", "right"], [0.004, 0.0], ratio_counts=[40, 38])
    assert bars.ratio_footer() == "every bar is 0.00 — no cost recorded to compare"
    bars.set_data(["left", "right"], [0.40, 0.001], ratio_counts=[40, 38])
    assert bars.ratio_footer() == "left 0.40 is the only bar above zero"


def test_the_analysis_page_never_pairs_a_noisy_title_with_a_ratio(qapp, pal, tmp_path):
    """End to end through the real caller: on a run whose input timing is too
    noisy to read microstructure from, the bias panel's title says so and the
    footer says nothing."""
    pytest.importorskip("pyqtgraph")
    from kovadapt.analysis.report import RunReport
    from kovadapt.config import Settings
    from kovadapt.gui.analysis_view import AnalysisView
    from kovadapt.profile.player import PlayerProfile

    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    settings = Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof"),
                        telemetry_enabled=False, onboarding_done=True, motion="off")
    prof = PlayerProfile(scenario="Beta 1wall Click [Adaptive]")
    prof.run_count = 4
    prof.history = [{"accuracy": 0.6, "kps": 1.0, "score": 400.0} for _ in range(4)]
    rep = RunReport(
        scenario="Beta 1wall Click", started_iso="2026-07-28T10:00:00",
        score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
        summary_text="30 kills at 61% accuracy.",
        n_flicks=22, mean_flick_ms=190.0, overshoot_rate=0.42, mean_corrections=2.6,
        input_health={"jitter_ms": 9.0, "polling_hz": 120.0, "gaps": 4},
        bias={"left": {"n": 9, "overshoot": 0.38, "corrections": 2.6},
              "right": {"n": 8, "overshoot": 0.27, "corrections": 1.9},
              "vertical": {"n": 5, "overshoot": 0.10, "corrections": 0.8},
              "bias_score": 0.2})

    view = AnalysisView(settings)
    view.show_report(rep, profile=prof)
    assert "too noisy" in view.bias_bars._title.lower()
    assert view.bias_bars.ratio_footer() == "", (
        "the panel spells out a ratio its own title calls impossible")
    arr = _shot(view.bias_bars, pal, 690, 300)
    assert _colours(arr[300 - int(viz._FOOTER_H) + 1:, 8:]) == 1
    view.deleteLater()


# =============================================================== flat trends
_FLAT_CASES = [
    ([0.667] * 3, "identical values whose np.mean is 1 ULP above them"),
    ([0.6666, 0.6667] * 6, "a hairline spread every label rounds to 67%"),
    ([0.5] * 300, "a long constant history"),
]


@pytest.mark.parametrize("vals,why", _FLAT_CASES)
def test_a_flat_history_draws_flat_on_its_own_mean_rule(qapp, pal, vals, why):
    """The line used to be drawn nine grid rows from the mean rule it is the
    mean OF, or thrown into a full-panel oscillation over a spread of 0.0001.
    Flat data is one row, sitting on its own reference."""
    w, h = 1400, 220
    row_h = QFontMetricsF(theme.mono(14)).height()
    tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
    tr.set_data(vals, tag="x")
    _lo, _hi, flat = tr._y_range()
    assert flat, f"{why} was not recognised as flat"

    arr = _shot(tr, pal, w, h)
    lo_y, hi_y = _row_span(_accent_ink(arr, keep_right=90))
    assert hi_y - lo_y <= row_h, (
        f"{why} drew a {hi_y - lo_y}px shape (one row is {row_h}px)")

    # The "mean 67%" label rides the reference rule and is the only full-alpha
    # fg_dim ink in the plot's right-hand strip (the min/max labels are gone on
    # a flat panel, the gutter is on the left, the run axis is below). The line
    # has to sit on it, because it IS the mean of these runs.
    label = _exact(arr, pal.fg_dim)[:h - int(viz._AXIS_H) - 20, w - 130:w - 30]
    c_lo, c_hi = _row_span(label)
    assert not (hi_y < c_lo or c_hi < lo_y), (
        f"the flat line at rows {lo_y}..{hi_y} is off its own mean rule "
        f"at rows {c_lo}..{c_hi}")


def test_the_value_axis_never_prints_one_number_at_both_ends(qapp, pal):
    """Six labels reading "67%" over a dramatic sawtooth was the panel's whole
    story. An axis with the same number at both ends is not an axis."""
    for vals, fmt in (([0.6666, 0.6667] * 6, "{:.0%}"),
                      ([0.667] * 3, "{:.0%}"),
                      ([0.5] * 300, "{:.0%}"),
                      ([12.0, 12.00001], "{:.2f}"),
                      ([0.0, 0.0], "{:+.2f}")):
        tr = viz.AsciiTrend(fmt=fmt)
        tr.set_data(vals)
        lo, hi, _flat = tr._y_range()
        assert fmt.format(lo) != fmt.format(hi), (
            f"{fmt} prints {fmt.format(lo)!r} at both ends of {vals[:2]}")

    # and in the pixels: the two gutter labels are different glyph runs
    tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
    tr.set_data([0.6666, 0.6667] * 6, tag="x")
    arr = _shot(tr, pal, 1400, 220)
    gutter = arr[34:, :40]                     # below the title, left of the plot
    rows = np.nonzero((gutter != arr[0, 0]).any(axis=-1).any(axis=1))[0] + 34
    groups = _clusters(rows)
    assert len(groups) >= 2, "the axis printed fewer than two labels"
    top, bot = groups[0], groups[-1]
    n = min(len(top), len(bot))
    assert not np.array_equal(arr[top[0]:top[0] + n, :40],
                              arr[bot[0]:bot[0] + n, :40]), (
        "the top and bottom axis labels render identical pixels")


# ================================================================ two runs
def test_two_runs_draw_a_marked_segment_not_a_field_of_glyphs(qapp, pal):
    """n=2 was ~170 interpolated glyph columns of steady climb — a long history
    to any eye that counts marks. Two runs are two points and a rule."""
    w, h = 1400, 220
    seg = viz.AsciiTrend(title="accuracy over 2 runs", fmt="{:.0%}")
    seg.set_data([0.42, 0.71], tag="71%")
    ink = _accent_ink(_shot(seg, pal, w, h))
    heights = _cell_heights(ink)
    keys = sorted(heights)
    assert len(keys) > 20, "the two runs are not joined up at all"

    # a RULE inks every column it crosses; a row of glyph marks leaves the gap
    # between one glyph and the next (the old n=2 inked 69% of its span)
    xs = np.nonzero(ink.any(axis=0))[0]
    span = int(xs.max()) - int(xs.min()) + 1
    assert len(xs) / span >= 0.98, (
        f"only {len(xs) / span:.0%} of the columns carry ink — this is a row of "
        "marks, not one segment")
    interior = _middle_half(heights)               # the marks live at the ends
    assert max(interior) <= 4, f"a glyph-sized mark sits in the segment: {max(interior)}"

    # both ends carry a MARK: more ink in a box around the endpoint than around
    # the middle of the rule
    arr = _shot(seg, pal, w, h)
    nonbg = (arr != arr[0, 0]).any(axis=-1)
    xs = np.nonzero(ink.any(axis=0))[0]

    def box(x: int) -> int:
        ys = np.nonzero(ink[:, x])[0]
        y = int(ys.mean())
        return int(nonbg[y - 4:y + 5, x - 4:x + 5].sum())

    # indices into the inked columns, so every probe column really has ink
    first, mid, last = int(xs[1]), int(xs[len(xs) // 2]), int(xs[-2])
    assert box(first) >= 2 * box(mid), "the first run is not marked"
    assert box(last) >= 2 * box(mid), "the newest run is not marked"

    # a real history keeps the glyph line — the pin above has teeth
    many = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
    many.set_data([0.30, 0.72, 0.35, 0.68, 0.40, 0.75, 0.44, 0.80], tag="80%")
    inner = sorted(_middle_half(_cell_heights(_accent_ink(_shot(many, pal, w, h)))))
    assert inner[len(inner) // 2] >= 5, "the glyph line lost its glyphs"


# ============================================================ the colour key
_LABELS5 = [[f"r{r}c{c}" for c in range(5)] for r in range(5)]


def _key_swatch_pixels(arr: np.ndarray, foot_h: int = 17) -> int:
    """Ramp-coloured pixels in the footer strip, i.e. colour-key swatches. The
    coverage note beside them is neutral grey, so it never counts."""
    return int((_chroma(arr)[arr.shape[0] - foot_h:, :] > 30).sum())


def test_the_colour_key_appears_only_when_the_colours_differ(qapp, pal):
    """Ten swatches of #f6571c..#ff5414 captioned "3.60" and "3.79" are one
    colour presented as a scale. The zones' own numbers carry the values when
    the ramp cannot."""
    narrow = np.array([[3.60 + 0.19 * ((r * 5 + c) / 24.0) for c in range(5)]
                       for r in range(5)])
    hm = viz.AsciiHeatmap(title="aim travel around engagements")
    hm.set_data(narrow, _LABELS5, fmt="{:.2f}")
    assert not hm.key_is_readable()
    assert _key_swatch_pixels(_shot(hm, pal, 900, 320)) == 0, (
        "a colour key was painted over a range with no colour in it")

    # a genuinely diverging map keeps its key: cool to warm across zero
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.4 * (r - 2) + 0.2 * (c - 2)
         for r in range(5) for c in range(5)}, 5, 5)
    div = viz.AsciiHeatmap(title="weakness by wall region")
    div.set_data(grid, labels, fmt="{:+.2f}")
    assert div.key_is_readable()
    assert _key_swatch_pixels(_shot(div, pal, 900, 320)) > 20, "the real key vanished"

    # one measured zone is not a range at all
    lone, lone_labels = viz.region_grid({"r0c0": 2.0}, 5, 5)
    solo = viz.AsciiHeatmap(title="weakness by wall region")
    solo.set_data(lone, lone_labels, fmt="{:+.2f}")
    assert not solo.key_is_readable()
    assert solo._key_ends() is None


# ======================================================= one region vocabulary
def test_the_lattice_speaks_the_same_region_words_as_the_coach(qapp, pal):
    """The axes said "top / far right" while the Coach card on the same screen
    said "upper right" — one grid read as two findings."""
    for rows, cols in ((5, 5), (3, 3), (4, 6), (1, 5), (5, 1)):
        row_words, col_words = viz._axis_words(rows, cols)
        dims = SimpleNamespace(region_rows=rows, region_cols=cols)
        assert len(row_words) == rows and len(col_words) == cols
        for r in range(rows):
            for c in range(cols):
                coach = _region_words(f"r{r}c{c}", dims)
                axis = f"{row_words[r]} {col_words[c]}"
                # _region_words collapses its one middle-centre zone to
                # "center"; every other zone must match word for word
                assert axis == coach or (coach == "center"
                                         and axis == "middle center"), (
                    f"{rows}x{cols} zone r{r}c{c}: axis {axis!r} vs coach {coach!r}")

    # none of viz.py's own five-band words survive on an axis
    row_words, col_words = viz._axis_words(5, 5)
    legacy = {"bottom", "top", "low", "high", "centre", "far left", "far right"}
    assert not (set(row_words) | set(col_words)) & legacy


def test_the_axis_prints_one_word_per_band_the_vocabulary_distinguishes(qapp, pal):
    """The shared vocabulary has three bands, so a 5-zone axis has one "center"
    spanning three zones — printed once, over the band it actually covers."""
    assert viz._label_groups(["a", "a", "b", "b", "b", "c"]) == [
        ("a", 0, 1), ("b", 2, 4), ("c", 5, 5)]
    assert viz._label_groups([]) == []
    _rows, cols = viz._axis_words(5, 5)
    assert [g[0] for g in viz._label_groups(cols)] == ["left", "center", "right"]
    assert viz._label_groups(cols)[1] == ("center", 1, 3)


def test_the_renderer_reads_the_shared_words(qapp, pal, monkeypatch):
    """A rendered pin, because the words are only a lie once they are on
    screen: patching the shared vocabulary must change the picture.

    This test used to also prove that viz.py's OWN second vocabulary never
    reached the pixels — the lattice axes and the Coach called the same zone
    two different things, one finding presented as two. That vocabulary is
    now deleted rather than merely unused, so the half of this test that
    patched it is gone: a table that does not exist cannot be rendered, which
    is a stronger guarantee than a test that it is not.
    """
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.4 * (r - 2) for r in range(5) for c in range(5)}, 5, 5)

    def render() -> QImage:
        hm = viz.AsciiHeatmap(title="weakness by wall region")
        hm.set_data(grid, labels, fmt="{:+.2f}")
        return _image(hm, pal, 900, 340)

    before = render()
    assert not hasattr(viz, "zone_words"), "the legacy vocabulary came back"
    assert not hasattr(viz, "_band_words")
    monkeypatch.setattr(viz, "_axis_words",
                        lambda rows, cols: (("A",) * rows, ("B",) * cols))
    assert render() != before, "the axis does not read the shared vocabulary"


def test_the_exact_region_key_stays_on_hover(qapp, pal):
    """The words are for the person; r{row}c{col} is the cross-module contract
    and hover is where it lives."""
    grid, labels = viz.region_grid({"r4c0": 1.9, "r3c1": -0.4}, 5, 5)
    hm = viz.AsciiHeatmap(title="weakness by wall region")
    hm.set_data(grid, labels, fmt="{:+.2f}")
    hm.resize(690, 270)
    x0, y0, zw, zh, gap, _rows, _cols = hm._geom()
    assert hm.zone_info(x0 + zw / 2, y0 + zh / 2) == "r4c0 · +1.90"
    assert "not measured" in hm.zone_info(x0 + zw / 2, y0 + 2 * (zh + gap) + zh / 2)


def test_a_worst_bar_is_a_claim_and_answers_to_the_gate(qapp, pal):
    """Marking a worst bar IS the side-vs-side claim, in colour instead of
    words, so it cannot outlive the sentence version of it.

    The input-health gate held for the title and for the footer ratio and
    LEAKED THROUGH THE PAINT: on a degraded run the headline read "input
    timing too noisy to compare directions this run" while a red bar
    underneath named the worst direction anyway.
    """
    vals = [0.55, 0.20, 0.18]        # left is clearly worst

    def render(counts):
        b = viz.AsciiBars(title="flick quality by direction")
        b.set_data(["left", "vertical", "right"], vals,
                   ["14 flicks", "9 flicks", "13 flicks"], ratio_counts=counts)
        return _image(b, pal, 720, 300)

    permitted = render([14, 9, 13])     # comparison allowed
    withheld = render(None)             # gate closed
    assert permitted != withheld, "the gate does not change what is drawn"

    bad = QColor(pal.bad).rgb()
    def bad_px(img):
        return sum(1 for x in range(img.width()) for y in range(img.height())
                   if img.pixel(x, y) == bad)

    assert bad_px(permitted) > 0, "the worst bar was never highlighted"
    assert bad_px(withheld) == 0, (
        "a worst bar is still painted with the comparison withheld — "
        "the same verdict the title refuses to give")
