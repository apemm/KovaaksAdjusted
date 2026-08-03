"""Legibility and motion pins for gui/viz.py (offscreen QPA).

These assert what a person can READ off the three charts, by measuring the
rendered pixels rather than by trusting that a paint call happened:

* a bar's length is strictly proportional to its value, and 0.22 vs 0.24 is a
  visible distance apart (they used to differ by one dissolving character);
* the numerals never sit inside the empty track;
* the trend is one CONNECTED line — every column carries ink and adjacent
  columns never leave a vertical gap bigger than a row, which is exactly what
  a field of unconnected marks would fail;
* the heatmap paints position words and values, never the r{row}c{col} region
  keys (the keys stay on hover, where the engine contract belongs), and it
  says how many zones were measured;
* every animation comes from gui/motion.py: it runs at GLYPH_HZ, is skipped
  entirely when motion is off, and its timer never ticks while the widget is
  hidden or idle.

The pixel probes assume the DARK indigo palette, whose accent is blue-tinted
ink (B-R >> 0) and whose fg is neutral grey — that is what lets a probe tell
chart ink from label text without reimplementing the layout.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no system font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import Settings  # noqa: E402
from kovadapt.gui import motion, theme, viz  # noqa: E402


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


def _settings(tmp_path, motion_level: str) -> Settings:
    # profile_dir is a CLASS-level default evaluated at import, so a patched
    # Path.home does NOT move it: pass one explicitly, always.
    return Settings(profile_dir=str(tmp_path / "prof"), motion=motion_level,
                    telemetry_enabled=False)


def _image(widget, pal, width: int, height: int) -> QImage:
    """The widget composited over the theme background, as the app shows it
    (these widgets are transparent; a bare grab() lands on Fusion grey)."""
    widget.resize(width, height)
    pm = QPixmap(width, height)
    pm.fill(QColor(pal.bg))
    widget.render(pm)
    return pm.toImage().convertToFormat(QImage.Format_RGB32)


def _rgb(img: QImage) -> np.ndarray:
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    a = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)
    return a[:, :img.width(), :3][:, :, ::-1].astype(int)   # H, W, RGB


def _hexrgb(hexs: str) -> np.ndarray:
    return np.array([int(hexs[i:i + 2], 16) for i in (1, 3, 5)])


def _exact(arr: np.ndarray, hexs: str) -> np.ndarray:
    return np.all(arr == _hexrgb(hexs), axis=-1)


# --------------------------------------------------------------------- bars
def _bar_bands(height: int, n: int) -> list[tuple[int, int]]:
    """The y band each bar row occupies — the same split paintEvent uses,
    expressed in the module's own furniture constants."""
    top = 26.0
    body = height - top - viz._RULER_H - viz._FOOTER_H - 4
    return [(int(top + i * body / n) + 2, int(top + (i + 1) * body / n) - 2)
            for i in range(n)]


def test_bar_length_is_proportional_and_close_values_stay_distinct(qapp, pal):
    """0.22 and 0.24 used to render as glyph runs one character apart with a
    dissolving tip, so the chart could not be compared at all. Length must be
    strictly proportional to value and the tips visibly separated."""
    w, h = 690, 270
    bars = viz.AsciiBars(title="flick cost by direction")
    # ratio_counts is the caller's permission to make a side-vs-side claim,
    # and the worst-bar highlight IS that claim in colour — without them no
    # bar paints pal.bad and this probe has nothing to find. Proportionality
    # is what's under test either way; the counts just buy the highlight.
    bars.set_data(["left", "vertical", "right"], [0.22, 0.24, 0.60],
                  ["37 flicks", "8 flicks", "41 flicks"], ratio_counts=[37, 8, 41])
    arr = _rgb(_image(bars, pal, w, h))
    accent = _exact(arr, pal.accent)
    worst = _exact(arr, pal.bad)                   # the max bar paints pal.bad

    bands = _bar_bands(h, 3)
    x0 = min(int(np.nonzero(accent[a:b].any(axis=0))[0].min()) for a, b in bands[:2])
    tips = [int(np.nonzero(accent[a:b].any(axis=0))[0].max()) for a, b in bands[:2]]
    # the max bar's tip: pal.bad ink inside the track, not its value column
    row = worst[bands[2][0]:bands[2][1], :int(w * 0.85)]
    tips.append(int(np.nonzero(row.any(axis=0))[0].max()))

    assert tips[0] < tips[1] < tips[2], "bars are not ordered by value"
    assert tips[1] - tips[0] >= 8, (
        f"0.22 and 0.24 differ by only {tips[1] - tips[0]}px — not comparable")
    lengths = [t - x0 for t in tips]
    for i, v in enumerate((0.22, 0.24, 0.60)):
        assert lengths[i] / lengths[2] == pytest.approx(v / 0.60, abs=0.04), (
            f"bar {i} length is not proportional to its value: {lengths}")


def test_bars_never_print_the_number_inside_the_track(qapp, pal):
    """The value used to be drawn at the run's tip, which put "0.10" sitting
    among the dots of the empty track. It lives in its own right-hand column
    now, so no bright numeral may appear in the middle of the panel."""
    w, h = 690, 270
    bars = viz.AsciiBars(title="flick cost by direction")
    bars.set_data(["left", "vertical", "right"], [0.10, 0.22, 0.24],
                  ["37 flicks", "8 flicks", "41 flicks"])
    arr = _rgb(_image(bars, pal, w, h))
    text = _exact(arr, pal.fg)                     # labels + numerals only
    ys, xs = np.nonzero(text)
    middle = [x for x in xs if 110 <= x <= 0.85 * w]
    assert not middle, f"numerals drawn over the track at x={sorted(set(middle))[:6]}"
    assert xs.max() > 0.9 * w, "the value column never rendered"


def test_bars_ratio_note_cites_the_top_two_bars(qapp, pal):
    """'These two are even' is itself the finding, and a 1.09x gap cannot be
    eyeballed off any bar chart — so the ratio is spelled out, from the values
    that are drawn and nothing else."""
    bars = viz.AsciiBars()
    bars.set_data(["left", "vertical", "right"], [0.22, 0.10, 0.24])
    note = bars.ratio_note()
    assert "right 0.24" in note and "left 0.22" in note and "1.09x" in note

    bars.set_data(["left", "right"], [0.0, 0.0])
    assert bars.ratio_note() == "every bar is 0.00 — no cost recorded to compare"

    bars.set_data(["left", "right"], [0.4, 0.0])
    assert bars.ratio_note() == "left 0.40 is the only bar above zero"

    bars.set_data(["vertical"], [0.4])
    assert bars.ratio_note() == "vertical 0.40 — one direction measured"
    assert viz.AsciiBars().ratio_note() == ""


def test_bars_with_no_cost_draw_no_bar_at_all(qapp, pal):
    """A zero must stay a zero: no tip, no fill, and no ruler ticks implying a
    range the data does not have."""
    w, h = 690, 270
    bars = viz.AsciiBars(title="no cost in either direction")
    bars.set_data(["left", "vertical", "right"], [0.0, 0.0, 0.0], ["3", "0", "2"])
    arr = _rgb(_image(bars, pal, w, h))
    assert not _exact(arr, pal.accent).any(), "an empty bar painted accent ink"
    assert not _exact(arr, pal.bad).any(), "an empty bar was flagged worst"


# -------------------------------------------------------------------- trend
def _line_spans(arr: np.ndarray, width: int, cell: int = 8) -> dict[int, tuple[int, int]]:
    """Per glyph-cell column, the (top, bottom) y of the trend's own ink.

    The line is accent-coloured, i.e. blue-tinted; every label, rule and axis
    on the panel is neutral grey. Blue-minus-red therefore isolates the line
    including its antialiased edges, without the test knowing the layout."""
    ink = (arr[:, :, 2] - arr[:, :, 0]) > 25
    ink[:, width - 80:] = False                    # the value tag rides outside
    spans: dict[int, tuple[int, int]] = {}
    for x in np.nonzero(ink.any(axis=0))[0]:
        ys = np.nonzero(ink[:, x])[0]
        lo, hi = spans.get(x // cell, (10 ** 9, -1))
        spans[x // cell] = (min(lo, int(ys.min())), max(hi, int(ys.max())))
    return spans


@pytest.mark.parametrize("vals,width", [
    ([0.30, 0.72, 0.35, 0.68, 0.40, 0.75, 0.44, 0.80, 0.38, 0.66], 1400),
    ([0.30, 0.72, 0.35, 0.68, 0.40, 0.75, 0.44, 0.80, 0.38, 0.66], 690),
    ([0.10, 0.95], 1400),
    (list(np.linspace(0.3, 0.8, 300)), 1400),
    ([0.52, 0.49, 0.55, 0.47, 0.58, 0.51, 0.62, 0.48], 400),   # dashboard widths
    ([0.52, 0.49, 0.55, 0.47, 0.58, 0.51, 0.62, 0.48], 950),
    # the riser cases: alternating runs about one column apart, so the level
    # swings most of the grid between neighbouring columns
    ([0.12, 0.93] * 35, 690),
    ([0.12, 0.93] * 80, 1400),
])
def test_trend_reads_as_one_connected_line(qapp, pal, vals, width):
    """At 1400px the old sparkline was a field of unconnected '= - + *' marks.
    Every cell column must carry ink and no two neighbours may leave a gap
    wider than one glyph row, which is what makes the eye follow a line."""
    from PySide6.QtGui import QFontMetricsF

    row_h = QFontMetricsF(theme.mono(14)).height()
    tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
    tr.set_data(vals, tag="x")
    arr = _rgb(_image(tr, pal, width, 220))
    spans = _line_spans(arr, width)
    keys = sorted(spans)
    assert len(keys) > 20
    covered = len(keys) / (keys[-1] - keys[0] + 1)
    assert covered >= 0.95, f"the line has empty columns: {covered:.2f} coverage"
    worst = 0
    for k in keys[:-1]:
        if k + 1 not in spans:
            continue
        (a1, b1), (a2, b2) = spans[k], spans[k + 1]
        worst = max(worst, max(0, max(a1, a2) - min(b1, b2)))
    assert worst <= row_h + 4, (
        f"a {worst}px vertical gap between adjacent columns (row is {row_h}px)")


def test_one_outlier_run_never_vanishes_into_the_resampling(qapp, pal):
    """300 runs into ~76 glyph columns: linear resampling on its own dropped a
    lone 0.95 run completely and drew a flat line through a history containing
    a personal best. Every column carries the min..max envelope of the runs
    that fell in it, so the outlier survives the decimation. A chart that
    understates its data is worse than no chart."""
    from PySide6.QtGui import QFontMetricsF

    row_h = QFontMetricsF(theme.mono(14)).height()
    width = 690
    flat_vals = [0.5] * 300
    spike_vals = list(flat_vals)
    spike_vals[137] = 0.95

    def extent(vals) -> float:
        tr = viz.AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
        tr.set_data(vals, tag="x")
        spans = _line_spans(_rgb(_image(tr, pal, width, 220)), width)
        return (max(hi for _lo, hi in spans.values())
                - min(lo for lo, _hi in spans.values()))

    assert extent(flat_vals) <= row_h, "a constant history is not a flat line"
    assert extent(spike_vals) >= 4 * row_h, (
        "a single outlier run disappeared into the resampled line")


def test_trend_baseline_is_the_mean_of_the_runs_shown(qapp, pal):
    """The reference rule is the only claim the widget makes on its own, so it
    is the mean of exactly the runs it drew — or whatever the caller supplied."""
    tr = viz.AsciiTrend(fmt="{:.0%}")
    vals = [0.40, 0.55, 0.61, 0.52]
    tr.set_data(vals)
    assert tr.baseline() == pytest.approx(float(np.mean(vals)))
    tr.set_data(vals, baseline=0.5)
    assert tr.baseline() == pytest.approx(0.5)
    tr.clear()
    assert tr.baseline() is None                   # nothing shown, nothing claimed


def test_trend_survives_a_panel_too_small_to_draw(qapp, pal):
    tr = viz.AsciiTrend(title="accuracy", fmt="{:.0%}")
    tr.set_data([0.4, 0.6, 0.5])
    assert not _image(tr, pal, 90, 60).isNull()
    assert not _image(tr, pal, 1400, 40).isNull()


# ------------------------------------------------------------------ heatmap
def test_heatmap_paints_no_region_keys_but_keeps_them_on_hover(qapp, pal):
    """r4c0 is the contract between adapt/, analysis/ and scenario/ — not text
    for a person reading a chart. Changing the labels must not change a single
    pixel (they are hover-only now), while zone_info still reports the key."""
    grid, labels = viz.region_grid({"r4c0": 1.9, "r3c1": -0.4}, 5, 5)
    keyed = viz.AsciiHeatmap(title="weakness by wall region")
    keyed.set_data(grid, labels, fmt="{:+.2f}")
    other = viz.AsciiHeatmap(title="weakness by wall region")
    other.set_data(grid, [["ZZZZZZ"] * 5 for _ in range(5)], fmt="{:+.2f}")
    assert _image(keyed, pal, 690, 270) == _image(other, pal, 690, 270), (
        "the zone labels reach the pixels — a region key is on screen")

    keyed.resize(690, 270)
    x0, y0, zw, zh, gap, rows, cols = keyed._geom()
    top_left = keyed.zone_info(x0 + zw / 2, y0 + zh / 2)
    assert top_left == "r4c0 · +1.90"              # exact key, exact value
    assert "not measured" in keyed.zone_info(x0 + zw / 2,
                                             y0 + 2 * (zh + gap) + zh / 2)


def test_heatmap_prints_each_measured_value(qapp, pal):
    """A value has to be readable off the picture, not only on hover: the
    number is drawn on the zone, so changing its format changes the render."""
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.4 * (r - 2) for r in range(5) for c in range(5)}, 5, 5)
    a = viz.AsciiHeatmap(title="weakness by wall region")
    a.set_data(grid, labels, fmt="{:+.2f}")
    b = viz.AsciiHeatmap(title="weakness by wall region")
    b.set_data(grid, labels, fmt="{:+.1f}")
    assert _image(a, pal, 690, 300) != _image(b, pal, 690, 300)


def test_heatmap_counts_the_zones_it_measured(qapp, pal):
    """25 hollow outlines with no explanation read as a broken chart."""
    hm = viz.AsciiHeatmap(title="weakness by wall region")
    grid, labels = viz.region_grid({"r4c0": 1.9, "r3c1": -0.4}, 5, 5)
    hm.set_data(grid, labels, fmt="{:+.2f}")
    assert hm.coverage_note().startswith("2 of 25 zones measured")

    hm.set_data(np.zeros((5, 5)))
    assert hm.coverage_note() == ""                 # nothing to explain
    hm.set_data(np.full((3, 3), np.nan))
    assert hm.coverage_note() == "no zone measured this run — all 9 drawn hollow"
    assert not _image(hm, pal, 690, 270).isNull()
    hm.clear()
    assert hm.coverage_note() == ""


def test_unmeasured_zones_are_never_coloured_like_data(qapp, pal):
    """NaN is on purpose. The heat ramp is strongly warm (R >> B) or cool
    (B >> R), so an unmeasured zone must stay neutral: no tint, no glyph
    texture, no number — it is drawn as a hollow outline and nothing else."""
    grid, labels = viz.region_grid({"r0c0": 2.0}, 3, 3)
    hm = viz.AsciiHeatmap()
    hm.set_data(grid, labels)
    assert np.isnan(hm._norm[1, 1])
    arr = _rgb(_image(hm, pal, 420, 320))
    x0, y0, zw, zh, gap, rows, cols = hm._geom()

    def chroma(disp_r: int, c: int) -> int:
        x = int(x0 + c * (zw + gap))
        y = int(y0 + disp_r * (zh + gap))
        cell = arr[y + 2:int(y + zh) - 2, x + 2:int(x + zw) - 2]
        return int(np.abs(cell[:, :, 0] - cell[:, :, 2]).max())

    assert chroma(2, 0) > 40, "the measured zone lost its ramp colour"
    for disp_r, c in ((0, 0), (1, 1), (2, 2), (0, 2)):
        assert chroma(disp_r, c) <= 20, (
            f"unmeasured zone at display {disp_r},{c} was painted like data")


def test_signed_zero_is_never_printed(qapp, pal):
    assert viz._fmt_value("{:+.2f}", -0.001) == "+0.00"
    assert viz._fmt_value("{:+.2f}", -1.2) == "-1.20"
    assert viz._fmt_value("{:.2f}", -0.0001) == "0.00"


# ------------------------------------------------------------------- motion
def _shown(widget, pal, width: int, height: int):
    widget.resize(width, height)
    widget.show()
    QTest.qWait(20)
    return widget


def test_motion_off_paints_the_end_state_and_schedules_nothing(qapp, pal, tmp_path):
    """With motion off nothing animates: the chart jumps to its final frame
    and no timer is ever started (a zero-length animation is still frames)."""
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.3 * (c - 2) for r in range(5) for c in range(5)}, 5, 5)
    cases = [
        (lambda: viz.AsciiBars(title="bars"), lambda w: w.set_data(
            ["left", "vertical", "right"], [0.42, 0.08, 0.91], ["9", "3", "7"]),
         690, 270),
        (lambda: viz.AsciiHeatmap(title="zones"),
         lambda w: w.set_data(grid, labels, fmt="{:+.2f}"), 690, 270),
        (lambda: viz.AsciiTrend(title="acc", fmt="{:.0%}"),
         lambda w: w.set_data([0.4, 0.5, 0.45, 0.6], tag="60%"), 1400, 220),
    ]
    for make, fill, w, h in cases:
        widget = make()
        widget.settings = _settings(tmp_path, "off")
        _shown(widget, pal, w, h)
        fill(widget)
        assert not widget._timer.isActive(), f"{type(widget).__name__} timed a no-op"
        assert widget._ign_span == 0.0
        off_frame = _image(widget, pal, w, h)

        settled = make()                           # never shown: end state too
        fill(settled)
        assert off_frame == _image(settled, pal, w, h), (
            f"{type(widget).__name__} with motion off is not its end state")
        widget.hide()


def test_the_reveal_is_actually_visible_then_lands(qapp, pal, tmp_path):
    """A staggered reveal that changed no pixels would be a lie: the mid-flight
    frame must differ from the settled one, and the timer must retire itself."""
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.3 * (c - 2) for r in range(5) for c in range(5)}, 5, 5)
    hm = viz.AsciiHeatmap(title="zones")
    hm.settings = _settings(tmp_path, "full")
    _shown(hm, pal, 690, 270)
    hm.set_data(grid, labels, fmt="{:+.2f}")
    assert hm._timer.isActive()
    assert hm._timer.interval() == motion.GLYPH_MS   # glyph art, not 60 Hz
    mid = _image(hm, pal, 690, 270)

    QTest.qWait(int(hm._ign_span) + 200)
    assert not hm._timer.isActive(), "an idle chart is still scheduling frames"
    assert mid != _image(hm, pal, 690, 270), "the reveal painted nothing"
    hm.hide()


def test_no_timer_ever_ticks_behind_a_hidden_widget(qapp, pal, tmp_path):
    bars = viz.AsciiBars(title="bars")
    bars.settings = _settings(tmp_path, "full")
    _shown(bars, pal, 690, 270)
    bars.set_data(["left", "right"], [0.4, 0.9], ["9", "7"])
    assert bars._timer.isActive()
    bars.hide()
    assert not bars._timer.isActive(), "the reveal kept ticking while hidden"

    # data arriving while hidden is held, not animated behind the user's back,
    # and the hidden widget still paints its FINISHED chart
    bars.set_data(["left", "right"], [0.9, 0.2], ["9", "7"])
    assert not bars._timer.isActive()
    settled = viz.AsciiBars(title="bars")
    settled.set_data(["left", "right"], [0.9, 0.2], ["9", "7"])
    assert _image(bars, pal, 690, 270) == _image(settled, pal, 690, 270)
    bars.show()                                    # ... and ignites on arrival
    QTest.qWait(20)
    assert bars._timer.isActive()
    bars.hide()


def test_reveal_staggers_by_distance_from_the_weakest_zone(qapp, pal, tmp_path):
    """The origin is the weakest measured zone, in display coordinates, and a
    zone next to it lights before one across the lattice."""
    grid, _labels = viz.region_grid({"r0c4": 2.4, "r4c0": 0.2, "r2c2": -0.5}, 5, 5)
    origin, far = viz.AsciiHeatmap._ignite_from(grid)
    assert origin == (4.0, 4.0)                    # data row 0 = display row 4
    assert far == pytest.approx(motion.grid_distance(0, 0, origin))

    hm = viz.AsciiHeatmap()
    hm.settings = _settings(tmp_path, "full")
    _shown(hm, pal, 690, 270)
    hm.set_data(grid, _labels, fmt="{:+.2f}")
    hm._ign_t0 = time.monotonic() - 0.09           # mid-flight, deterministic
    lit = hm._ignite_lit()
    assert lit is not None
    assert lit(0.0) > lit(2.0) > lit(5.6), "the wave is not ordered by distance"
    hm.hide()

    # all-NaN maps have no weakest zone to start from: light from the middle
    empty, _ = viz.region_grid({}, 5, 5)
    assert viz.AsciiHeatmap._ignite_from(empty)[0] == (2.0, 2.0)


def test_hover_ripple_is_ambient_gated_and_expires(qapp, pal, tmp_path):
    grid, labels = viz.region_grid(
        {f"r{r}c{c}": 0.3 * (c - 2) for r in range(5) for c in range(5)}, 5, 5)

    def hover(level: str):
        hm = viz.AsciiHeatmap(title="zones")
        hm.settings = _settings(tmp_path, level)
        _shown(hm, pal, 690, 270)
        hm.set_data(grid, labels, fmt="{:+.2f}")
        QTest.qWait(int(hm._ign_span) + 120)       # let the reveal land first
        x0, y0, zw, zh, gap, _rows, _cols = hm._geom()
        pos = QPointF(x0 + 2 * (zw + gap) + zw / 2, y0 + 2 * (zh + gap) + zh / 2)
        hm.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, pos, Qt.NoButton,
                                      Qt.NoButton, Qt.NoModifier))
        return hm

    full = hover("full")
    assert full._rip_span > 0.0 and full._timer.isActive()
    assert full.toolTip() == "r2c2 · +0.00"        # the key, on hover, exactly
    QTest.qWait(int(full._rip_span) + 150)
    assert not full._timer.isActive(), "the ripple never retired its timer"
    full.hide()

    for level in ("reduced", "off"):
        quiet = hover(level)
        assert quiet._rip_span == 0.0, f"{level} motion still rippled"
        assert not quiet._timer.isActive()
        quiet.hide()


def test_an_empty_chart_schedules_nothing(qapp, pal, tmp_path):
    """clear() has nothing to reveal, so it must not spend 400 ms of frames
    animating a placeholder."""
    for widget, empty in ((viz.AsciiBars(title="b"), lambda w: w.clear()),
                          (viz.AsciiHeatmap(title="h"), lambda w: w.clear()),
                          (viz.AsciiTrend(title="t"), lambda w: w.set_data([0.5]))):
        widget.settings = _settings(tmp_path, "full")
        _shown(widget, pal, 690, 270)
        empty(widget)
        assert not widget._timer.isActive(), f"{type(widget).__name__} animated nothing"
        assert widget._ign_span == 0.0
        widget.hide()


def test_module_settings_are_read_at_use_time(qapp, pal, tmp_path):
    """use_settings() registers the live object; the level is read per call, so
    flipping motion while the app runs takes effect without a reconstruction."""
    s = _settings(tmp_path, "off")
    viz.use_settings(s)
    bars = viz.AsciiBars(title="bars")
    _shown(bars, pal, 690, 270)
    bars.set_data(["left", "right"], [0.4, 0.9])
    assert not bars._timer.isActive()

    s.motion = "full"                              # the user changes the dial
    bars.set_data(["left", "right"], [0.5, 0.8])
    assert bars._timer.isActive()
    bars.hide()


def test_public_api_still_answers_every_caller(qapp, pal):
    """analysis_view.py and dashboard.py call exactly these."""
    assert viz.NOISE_FLOOR > 0
    assert viz.pool(np.ones((8, 8)), 2, 2).shape == (2, 2)
    grid, labels = viz.region_grid({"r0c0": 1.0}, 5, 5)
    assert grid.shape == (5, 5) and labels[0][0] == "r0c0"

    bars = viz.AsciiBars(title="t")
    bars.set_title("t2")
    bars.set_data(["a"], [1.0], ["1 flick"])
    bars.restyle(pal)
    bars.clear()
    hm = viz.AsciiHeatmap(title="t")
    hm.set_title("t2")
    hm.set_data(grid, labels, fmt="{:+.2f}")
    hm.restyle(pal)
    assert hm.zone_info(-5, -5) is None
    hm.clear()
    assert hm.zone_info(50, 50) is None            # no data, no answer
    tr = viz.AsciiTrend(title="t", fmt="{:.0%}")
    tr.set_title("t2")
    tr.set_data([0.4, 0.5], tag="50%")
    tr.restyle(pal)
    tr.clear()
    for widget, size in ((bars, (690, 270)), (hm, (690, 270)), (tr, (1400, 220))):
        assert not _image(widget, pal, *size).isNull()
