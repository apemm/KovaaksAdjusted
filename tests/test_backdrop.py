"""Live backdrop loop tests (offscreen QPA): the per-cell loop colors are
exactly periodic in LOOP_T, the blink erase-wedges match the almond
geometry, the shared CHARACTER lids (blink_cells — lid-skin rows + lash
silhouettes, used by both the opening and the backdrop) cover the iris at
peak blink and are pure in k, and a composed frame at phase LOOP_T is
byte-identical to the frame at phase 0. Skipped wholesale without PySide6."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from kovadapt.gui import ascii_art  # noqa: E402
from kovadapt.gui import backdrop as bkd  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Same guard as test_gui_smoke: nothing here may ever write the
    developer's real ~/.kovadapt."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


INK = QColor("#dcdee6")


def _live_cells():
    cells = [c for c in ascii_art.stencil() if c.role in bkd._LIVE_ROLES]
    assert cells, "stencil must contain live cells"
    return cells


# ----------------------------------------------------------- loop colors
def test_loop_colors_exactly_periodic_in_T():
    """frame(LOOP_T) == frame(0) at the color level, for both hue modes
    and both theme brightnesses — the heart of the perfect loop."""
    T = ascii_art.LOOP_T
    for iris_hue in (0.62, None):          # accent-locked / Gamer rainbow
        for is_dark in (True, False):
            for cell in _live_cells()[::5]:
                a = ascii_art.loop_cell_color(
                    cell, 0.0, is_dark=is_dark, iris_hue=iris_hue, ink=INK)
                b = ascii_art.loop_cell_color(
                    cell, T, is_dark=is_dark, iris_hue=iris_hue, ink=INK)
                assert a == b and a.alphaF() == b.alphaF()


def test_loop_colors_actually_animate():
    cells = _live_cells()
    T = ascii_art.LOOP_T
    changed = sum(
        ascii_art.loop_cell_color(c, 0.0, is_dark=True, iris_hue=0.62, ink=INK)
        != ascii_art.loop_cell_color(c, T / 3, is_dark=True, iris_hue=0.62,
                                     ink=INK)
        for c in cells)
    assert changed > len(cells) * 0.5      # most of the iris shimmers


def test_iris_hue_stays_locked_to_accent_outside_gamer():
    """Normal modes: the hue never strays beyond the static +-0.05
    iridescence band around the theme accent, at any loop phase."""
    iris = [c for c in _live_cells() if c.role == "iris"]
    for phase in (0.0, 1.7, 4.2, 6.9):
        for cell in iris[::7]:
            col = ascii_art.loop_cell_color(
                cell, phase, is_dark=True, iris_hue=0.62, ink=INK)
            assert abs(col.hueF() - 0.62) < 0.051


def test_gamer_rainbow_rotates_exactly_one_cycle_per_loop():
    T = ascii_art.LOOP_T
    cell = next(c for c in _live_cells() if c.role == "iris")
    h0 = ascii_art.loop_cell_color(
        cell, 0.0, is_dark=True, iris_hue=None, ink=INK).hueF()
    h_half = ascii_art.loop_cell_color(
        cell, T / 2, is_dark=True, iris_hue=None, ink=INK).hueF()
    # half a loop rotates the rainbow by exactly half a hue cycle
    assert abs((h_half - h0) % 1.0 - 0.5) < 1e-3


def test_the_backdrop_has_no_highlight_cells_left_to_animate():
    """The twin glints used to twinkle out of phase here. The glint role is
    gone from the art (one iris everywhere), so what is pinned now is its
    absence — including from the live set, which is what the loop animates."""
    assert not [c for c in ascii_art.stencil() if c.role == "glint"]
    assert bkd._LIVE_ROLES == ("iris",)
    assert all(c.role == "iris" for c in _live_cells())


# ----------------------------------------------------------------- blink
def test_blink_amount_shape_and_period():
    T = ascii_art.LOOP_T
    b0, bl = ascii_art.BLINK_PHASE, ascii_art.BLINK_LEN
    assert ascii_art.blink_amount(0.0) == 0.0
    assert ascii_art.blink_amount(b0) == 0.0           # closed exactly once
    assert ascii_art.blink_amount(b0 + bl) == 0.0
    assert ascii_art.blink_amount(b0 + bl / 2) == pytest.approx(1.0)
    assert 0.0 < ascii_art.blink_amount(b0 + bl / 4) < 1.0
    for phase in (0.0, b0 + bl / 2, b0 + bl / 4, 3.3):
        assert ascii_art.blink_amount(phase + T) == pytest.approx(
            ascii_art.blink_amount(phase), abs=1e-9)


def test_blink_wedges_cover_iris_at_peak():
    """The eyelid wedges must swallow every live iris/glint cell when the
    lids are fully shut, cover only part of them mid-blink, and vanish
    when the eye is open."""
    rect = QRectF(0, 0, ascii_art.COLS * 8.0, ascii_art.ROWS * 16.0)
    cw = rect.width() / ascii_art.COLS
    ch = rect.height() / ascii_art.ROWS
    centers = [QPointF((c.col + 0.5) * cw, (c.row + 0.5) * ch)
               for c in _live_cells()]

    assert ascii_art.blink_lid_paths(rect, 0.0) is None

    wedges, edges = ascii_art.blink_lid_paths(rect, 1.0)
    covered = sum(wedges.contains(pt) for pt in centers)
    assert covered == len(centers)
    assert not edges.isEmpty()             # the lid margin is strokable

    half_wedges, _ = ascii_art.blink_lid_paths(rect, 0.5)
    part = sum(half_wedges.contains(pt) for pt in centers)
    assert 0 < part < len(centers)


def test_blink_cells_character_lids_cover_live_cells():
    """The shared character-lid helper (opening + backdrop blink): at full
    closure every live iris/glint cell position is replaced by a lid
    glyph; mid-blink only part of them; an open eye has no lid cells."""
    live = {(c.col, c.row) for c in _live_cells()}
    assert ascii_art.blink_cells(0.0) == []

    shut = ascii_art.blink_cells(1.0)
    shut_pos = {(c, r) for c, r, _ch, _kind, _sh in shut}
    assert live <= shut_pos                    # full coverage when shut
    assert {kind for _c, _r, _ch, kind, _sh in shut} == {"skin", "lash"}

    half_pos = {(c, r) for c, r, _ch, _kind, _sh in ascii_art.blink_cells(0.5)}
    covered = live & half_pos
    assert 0 < len(covered) < len(live)        # the lids sweep, not switch


def test_blink_cells_lash_silhouettes_and_purity():
    """Mid-blink the moving margin carries directional lash strokes, the
    shades are sane alphas, and the helper is pure in k — the backdrop's
    byte-identical loop depends on that determinism."""
    mid = ascii_art.blink_cells(0.6)
    assert mid == ascii_art.blink_cells(0.6)
    lash_chars = {ch for _c, _r, ch, kind, _sh in mid if kind == "lash"}
    assert lash_chars & set("/|\\")            # lash silhouettes, not a wipe
    assert all(0.0 <= sh <= 1.0 for _c, _r, _ch, _k, sh in mid)


# ----------------------------------------------- composed frame identity
def test_frame_at_T_is_byte_identical_to_frame_at_zero(qapp):
    from kovadapt.gui.backdrop import Backdrop

    win = QWidget()
    win.resize(900, 700)
    bd = Backdrop(win)
    bd._timer.stop()                       # deterministic: no live ticks

    canvas = QPixmap(900, 700)
    canvas.fill(QColor("#101116"))
    p = QPainter(canvas)
    bd.paint(p)                            # _ensure(): frame at loop phase 0
    p.end()

    frame0 = bytes(bd._frame.toImage().constBits())
    bd._loop = ascii_art.BLINK_PHASE + ascii_art.BLINK_LEN / 2
    bd._render_frame()                     # peak blink: wedges erase pixels
    mid = bytes(bd._frame.toImage().constBits())
    bd._loop = ascii_art.LOOP_T
    bd._render_frame()
    wrapped = bytes(bd._frame.toImage().constBits())

    assert mid != frame0                   # the animation moves
    assert wrapped == frame0               # ...and the loop is perfect
    win.deleteLater()


def test_backdrop_rebuilds_on_theme_and_resize_notify(qapp):
    from kovadapt.gui.backdrop import Backdrop

    win = QWidget()
    win.resize(800, 600)
    bd = Backdrop(win)
    bd._timer.stop()
    canvas = QPixmap(800, 600)
    canvas.fill(QColor("#101116"))
    p = QPainter(canvas)
    bd.paint(p)
    p.end()
    assert bd._frame is not None
    bd.notify_theme()
    assert bd._frame is None               # dropped; next paint re-renders
    p = QPainter(canvas)
    bd.paint(p)
    p.end()
    assert bd._frame is not None
    bd.notify_resize()
    assert bd._frame is None
    win.deleteLater()


def test_exclude_roles_strips_everything(qapp):
    pm = ascii_art.render_pixmap(
        200, exclude_roles=("iris", "outline", "lash", "shade",
                            "reticle", "hub"))
    assert not any(bytes(pm.toImage().constBits()))    # fully transparent
