"""ASCII viz widgets (offscreen QPA): construct, accept data, render to
nonblank pixmaps, restyle across palettes. Skipped wholesale without PySide6."""

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

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.gui import theme  # noqa: E402
from kovadapt.gui.viz import (  # noqa: E402
    AsciiBars,
    AsciiHeatmap,
    AsciiTrend,
    pool,
    region_grid,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never touch the developer's real ~/.kovadapt (same guard as the GUI
    smoke tests — theme/settings code paths save under Path.home())."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


_PALETTES = {
    "dark": theme.build_palette(dark=True),
    "light": theme.build_palette(dark=False),
    "midnight": theme.build_palette(dark=True, midnight=True),
    "rgb": theme.build_palette(dark=True, rgb=True),
}


def _render_colors(widget, w=440, h=280):
    """Grab the widget offscreen and sample its pixels."""
    widget.resize(w, h)
    img = widget.grab().toImage()
    return {img.pixel(x, y)
            for x in range(0, img.width(), 6) for y in range(0, img.height(), 6)}


# ------------------------------------------------------------------- bars
def test_bars_render_nonblank_and_restyle_every_palette(qapp, monkeypatch):
    bars = AsciiBars(title="flick quality by direction")
    bars.set_data(["left", "vertical", "right"], [0.42, 0.08, 0.91],
                  ["12 flicks", "3 flicks", "9 flicks"])
    for pal in _PALETTES.values():
        monkeypatch.setattr(theme, "_current", pal)
        bars.restyle()
        assert len(_render_colors(bars)) > 1, f"blank render under {pal.name}"


def test_bars_degenerate_data_never_crashes(qapp):
    bars = AsciiBars()
    assert not bars.grab().isNull()          # constructed, no data
    bars.set_data(["a", "b"], [0.0, 0.0])    # all-zero: zero-anchored, no bars
    assert not bars.grab().isNull()
    bars.clear()
    assert not bars.grab().isNull()
    bars.set_data(["only"], [1.0])           # no sublabels path
    assert not bars.grab().isNull()


# ---------------------------------------------------------------- heatmap
def test_heatmap_renders_grid_with_ramp(qapp):
    hm = AsciiHeatmap(title="aim travel")
    grid = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    labels = [[f"r{r}c{c}" for c in range(5)] for r in range(5)]
    hm.set_data(grid, labels)
    colors = _render_colors(hm)
    assert len(colors) > 3                   # ramp spans many colors, not two


def test_heatmap_zone_hover_reports_label_and_value(qapp):
    hm = AsciiHeatmap(title="zones")
    grid = np.zeros((5, 5))
    grid[0, 0] = 1.0                         # r0c0 = BOTTOM-left (aim, +y up)
    grid[4, 4] = -1.0
    labels = [[f"r{r}c{c}" for c in range(5)] for r in range(5)]
    hm.set_data(grid, labels, fmt="{:+.2f}")
    hm.resize(420, 320)
    x0, y0, zw, zh, gap, rows, cols = hm._geom()
    bottom_left = hm.zone_info(x0 + zw / 2, y0 + (rows - 1) * (zh + gap) + zh / 2)
    assert bottom_left == "r0c0 · +1.00"
    top_right = hm.zone_info(x0 + (cols - 1) * (zw + gap) + zw / 2, y0 + zh / 2)
    assert top_right == "r4c4 · -1.00"
    assert hm.zone_info(1.0, 1.0) is None    # title band is not a zone


def test_heatmap_empty_and_flat_grids(qapp, monkeypatch):
    hm = AsciiHeatmap()
    assert not hm.grab().isNull()            # no data: placeholder paints
    hm.set_data(np.zeros((3, 3)))            # flat zero grid
    assert not hm.grab().isNull()
    hm.set_data(np.full((3, 3), 2.5))        # flat nonzero grid
    assert not hm.grab().isNull()
    monkeypatch.setattr(theme, "_current", _PALETTES["rgb"])
    hm.set_data(np.linspace(0, 1, 9).reshape(3, 3))
    hm.restyle()                             # rainbow ramp path
    assert len(_render_colors(hm)) > 3
    hm.clear()
    assert not hm.grab().isNull()


def test_region_grid_maps_keys_bottom_up():
    grid, labels = region_grid({"r0c0": 1.5, "r4c4": -2.0, "r2c1": 0.3}, 5, 5)
    assert grid.shape == (5, 5)
    assert grid[0, 0] == 1.5                 # row 0 = bottom row
    assert grid[4, 4] == -2.0
    assert grid[2, 1] == 0.3
    assert grid[1, 1] == 0.0                 # missing regions read the mean
    assert labels[0][0] == "r0c0" and labels[4][4] == "r4c4"


def test_pool_downsamples_preserving_row_order():
    field = np.zeros((64, 64))
    field[:13, :] = 9.0                      # hot band at row 0 (= bottom)
    zones = pool(field, 5, 5)
    assert zones.shape == (5, 5)
    assert zones[0].mean() > zones[4].mean()
    assert pool(np.empty((0, 0)), 3, 4).shape == (3, 4)


# ------------------------------------------------------------------ trend
def test_trend_renders_with_tag_and_restyles(qapp, monkeypatch):
    tr = AsciiTrend(title="accuracy over runs", fmt="{:.0%}")
    tr.set_data([0.42, 0.5, 0.47, 0.55, 0.61], tag="61%")
    assert tr.minimumHeight() >= 200
    assert len(_render_colors(tr)) > 1
    monkeypatch.setattr(theme, "_current", _PALETTES["rgb"])
    tr.restyle()                             # rainbow-column path
    assert len(_render_colors(tr)) > 1


def test_trend_degenerate_data_never_crashes(qapp):
    tr = AsciiTrend()
    assert not tr.grab().isNull()            # no data: placeholder
    tr.set_data([0.5])                       # one run is not a trend
    assert not tr.grab().isNull()
    tr.set_data([0.5, 0.5, 0.5])             # flat line: padded range
    assert not tr.grab().isNull()
    tr.set_data(list(np.linspace(0.3, 0.8, 500)))   # more runs than columns
    assert not tr.grab().isNull()
    tr.clear()
    assert not tr.grab().isNull()
