"""Character-art data viz: the Analysis charts drawn as glyph matrices in
the ascii_art visual language (monospace glyphs, density ramps, LED color)
instead of pyqtgraph. pyqtgraph survives only inside TrajectoryReplay.

Three widgets share the conventions:

    AsciiBars     horizontal glyph-run bars (flick cost by direction):
                  '@#*+=-:.' density fading along each run, value labels,
                  the worst bar in pal.bad, always anchored at zero
    AsciiHeatmap  a zone grid (Settings.region_cols x region_rows): each
                  zone a block of glyphs whose density and color (an
                  inferno-like ramp built from QColor.fromHsvF; the full
                  rainbow in RGB gamer mode) map its value; hover a zone
                  for its label + raw value
    AsciiTrend    a tall glyph sparkline ('.:-=+*#' gives sub-cell vertical
                  resolution) with a current-value tag on the newest column

All three are pure QPainter and keep only their data: they read
theme.current() at paint time, so restyle() is nothing but update() —
never cache a palette. Grid row 0 is the BOTTOM row everywhere (aim
convention, +y up), matching the r{row}c{col} region-key contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from . import theme
from .ascii_art import _mono

_BAR_RAMP = "@#*+=-:."           # dense at the zero anchor -> light at the tip
_TREND_RAMP = ".:-=+*#"          # light -> dense: fraction of a cell filled
_HEAT_RAMP = " .:-=+*#%@"        # light -> dense zone texture


def _seed(r: int, c: int) -> float:
    """Stable per-cell noise in [0, 1) (same hash family as ascii_art)."""
    return (math.sin(c * 12.9898 + r * 78.233) * 43758.5453) % 1.0


def _heat_color(v: float, pal) -> QColor:
    """Value 0..1 -> ramp color. Inferno-like on dark themes (deep purple
    through red to yellow), a cream-safe violet-to-ember on light ones, and
    the full rainbow in RGB gamer mode."""
    v = min(max(v, 0.0), 1.0)
    if pal.rgb:
        return QColor.fromHsvF((1.0 - v) * 0.83, 0.85, 0.35 + 0.65 * v)
    if pal.is_dark:
        return QColor.fromHsvF((0.78 + 0.38 * v) % 1.0,
                               0.92 - 0.25 * v * v,
                               0.22 + 0.78 * v)
    return QColor.fromHsvF((0.78 + 0.30 * v) % 1.0,
                           0.20 + 0.72 * v,
                           0.82 - 0.10 * v)


def pool(field: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean-pool a 2D field down to (rows, cols) zone means. Row order is
    preserved: feed row 0 = bottom and it stays the bottom row."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((rows, cols))
    return np.array([[chunk.mean() for chunk in np.array_split(band, cols, axis=1)]
                     for band in np.array_split(arr, rows, axis=0)])


def region_grid(deficits: dict[str, float], cols: int,
                rows: int) -> tuple[np.ndarray, list[list[str]]]:
    """r{row}c{col} dict (aim convention: higher row = higher on the wall)
    -> (grid with row 0 at the bottom, matching labels). Regions without an
    entry read 0.0 — the z-scored mean."""
    labels = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
    grid = np.array([[float(deficits.get(labels[r][c], 0.0)) for c in range(cols)]
                     for r in range(rows)])
    return grid, labels


def _paint_title(p: QPainter, pal, title: str, width: int) -> float:
    """Dim uppercase mono header line; returns the content top y."""
    if not title:
        return 8.0
    f = _mono()
    f.setPixelSize(11)
    p.setFont(f)
    p.setPen(QColor(pal.fg_dim))
    p.drawText(QRectF(10, 4, width - 20, 16), Qt.AlignLeft | Qt.AlignVCenter,
               title.upper())
    return 26.0


def _paint_empty(p: QPainter, pal, rect: QRectF, text: str) -> None:
    f = _mono()
    f.setPixelSize(13)
    p.setFont(f)
    col = QColor(pal.fg_dim)
    col.setAlphaF(0.75)
    p.setPen(col)
    p.drawText(rect, Qt.AlignCenter, f"· {text} ·")


class AsciiBars(QWidget):
    """Horizontal bar chart as glyph runs, zero-anchored: each bar is a run
    of ramp characters fading '@' -> '.' toward the tip, its value printed
    at the tip; every bar sharing the max (when positive) paints pal.bad."""

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._labels: list[str] = []
        self._sublabels: list[str] = []
        self._values: list[float] = []
        self.setMinimumHeight(220)

    # ------------------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, labels: Sequence[str], values: Sequence[float],
                 sublabels: Sequence[str] | None = None) -> None:
        self._labels = [str(x) for x in labels]
        self._values = [float(v) for v in values]
        self._sublabels = [str(s) for s in (sublabels or [])]
        self.update()

    def clear(self) -> None:
        self.set_data([], [])

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        top = _paint_title(p, pal, self._title, w)
        if not self._values:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "waiting for flick data")
            return

        n = len(self._values)
        row_h = (h - top - 8) / n
        gh = max(13, min(20, int(row_h * 0.38)))
        font = _mono()
        font.setPixelSize(gh)
        p.setFont(font)
        fm = QFontMetricsF(font)
        cw = fm.horizontalAdvance("@")

        lab_font = _mono()
        lab_font.setPixelSize(13)
        lab_fm = QFontMetricsF(lab_font)
        label_w = max([lab_fm.horizontalAdvance(s) for s in
                       self._labels + self._sublabels] + [56.0]) + 10
        value_w = 64.0
        bar_x0 = 12 + label_w
        ncells = max(int((w - bar_x0 - value_w - 14) // cw), 4)
        vmax = max(self._values)
        scale = vmax if vmax > 0 else 1.0

        # the zero axis every bar grows from
        axis = QColor(pal.fg_dim)
        axis.setAlphaF(0.55)
        p.setPen(axis)
        p.drawLine(int(bar_x0 - cw * 0.6), int(top + 2),
                   int(bar_x0 - cw * 0.6), int(h - 8))

        for i, v in enumerate(self._values):
            cy = top + i * row_h + row_h / 2
            color = QColor(pal.bad if (v == vmax and vmax > 0) else pal.accent)

            # direction label (+ dim sublabel under it)
            p.setFont(lab_font)
            p.setPen(QColor(pal.fg))
            if i < len(self._sublabels):
                p.drawText(QRectF(6, cy - 18, label_w, 17),
                           Qt.AlignRight | Qt.AlignBottom, self._labels[i])
                sub_f = _mono()
                sub_f.setPixelSize(10)
                p.setFont(sub_f)
                p.setPen(QColor(pal.fg_dim))
                p.drawText(QRectF(6, cy + 1, label_w, 13),
                           Qt.AlignRight | Qt.AlignTop, self._sublabels[i])
            else:
                p.drawText(QRectF(6, cy - row_h / 2, label_w, row_h),
                           Qt.AlignRight | Qt.AlignVCenter, self._labels[i])

            # the glyph run: dense at zero, fading toward the tip
            p.setFont(font)
            filled = int(round(min(v / scale, 1.0) * ncells))
            for j in range(ncells):
                x = bar_x0 + j * cw
                cell = QRectF(x, cy - gh * 0.75, cw * 2, gh * 1.5)
                if j < filled:
                    frac = j / max(filled - 1, 1)
                    ch = _BAR_RAMP[int(frac * (len(_BAR_RAMP) - 1))]
                    col = QColor(color)
                    col.setAlphaF(1.0 - 0.35 * frac)
                else:
                    ch = "."
                    col = QColor(pal.border)
                p.setPen(col)
                p.drawText(cell, Qt.AlignLeft | Qt.AlignVCenter, ch)

            # value at the tip
            vx = min(bar_x0 + filled * cw + 8, w - value_w - 2)
            p.setPen(QColor(pal.bad) if (v == vmax and vmax > 0) else QColor(pal.fg))
            p.drawText(QRectF(vx, cy - row_h / 2, value_w, row_h),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{v:.2f}")


class AsciiHeatmap(QWidget):
    """Zone-grid heatmap: each zone a block of glyphs whose density and
    ramp color carry the value; grid row 0 = bottom (aim convention).
    Hovering a zone tooltips its label and raw value."""

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._grid: np.ndarray | None = None
        self._norm: np.ndarray | None = None
        self._labels: list[list[str]] | None = None
        self._fmt = "{:.2f}"
        self.setMinimumHeight(220)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, grid: np.ndarray | None,
                 labels: Sequence[Sequence[str]] | None = None,
                 fmt: str = "{:.2f}") -> None:
        """grid: 2D array, row 0 = bottom. labels: same shape, shown in the
        hover tooltip (defaults to the r{row}c{col} region keys)."""
        if grid is None:
            self._grid = self._norm = self._labels = None
        else:
            g = np.asarray(grid, dtype=float)
            self._grid = g
            lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
            if hi > lo:
                self._norm = (g - lo) / (hi - lo)
            else:
                self._norm = np.full_like(g, 0.5 if hi != 0.0 else 0.0)
            self._labels = ([[str(s) for s in row] for row in labels]
                            if labels is not None else None)
            self._fmt = fmt
        self.update()

    def clear(self) -> None:
        self.set_data(None)

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------------
    def _geom(self) -> tuple[float, float, float, float, float, int, int] | None:
        """(x0, y0, zone_w, zone_h, gap, rows, cols) of the zone lattice —
        screen row 0 at y0 is the TOP row (data row rows-1)."""
        if self._grid is None:
            return None
        rows, cols = self._grid.shape
        top = 26.0 if self._title else 8.0
        gap = 3.0
        x0, y0 = 10.0, top
        zw = (self.width() - x0 * 2 - gap * (cols - 1)) / cols
        zh = (self.height() - y0 - 10.0 - gap * (rows - 1)) / rows
        if zw <= 4 or zh <= 4:
            return None
        return x0, y0, zw, zh, gap, rows, cols

    def zone_info(self, x: float, y: float) -> str | None:
        """'label · value' for the zone under widget coords (x, y), else None."""
        geom = self._geom()
        if geom is None:
            return None
        x0, y0, zw, zh, gap, rows, cols = geom
        c = int((x - x0) // (zw + gap))
        disp_r = int((y - y0) // (zh + gap))
        if not (0 <= c < cols and 0 <= disp_r < rows):
            return None
        if (x - x0) - c * (zw + gap) > zw or (y - y0) - disp_r * (zh + gap) > zh:
            return None                                   # in the gutter
        r = rows - 1 - disp_r                             # data row (bottom = 0)
        label = self._labels[r][c] if self._labels else f"r{r}c{c}"
        return f"{label} · {self._fmt.format(float(self._grid[r, c]))}"

    def mouseMoveEvent(self, event) -> None:
        info = self.zone_info(event.position().x(), event.position().y())
        self.setToolTip(info or "")
        if info:
            QToolTip.showText(event.globalPosition().toPoint(), info, self)
        super().mouseMoveEvent(event)

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        top = _paint_title(p, pal, self._title, w)
        geom = self._geom()
        if geom is None:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "no movement data")
            return
        x0, y0, zw, zh, gap, rows, cols = geom

        # ramp legend in the title band: low ..dense.. high
        if self._title and w > 300:
            f = _mono()
            f.setPixelSize(11)
            p.setFont(f)
            fm = QFontMetricsF(f)
            lx = w - 10 - fm.horizontalAdvance("#") * 10 - fm.horizontalAdvance("low  high  ")
            p.setPen(QColor(pal.fg_dim))
            p.drawText(QPointF(lx, 15), "low ")
            lx += fm.horizontalAdvance("low ")
            for k in range(10):
                p.setPen(_heat_color(k / 9.0, pal))
                p.drawText(QPointF(lx, 15), "#")
                lx += fm.horizontalAdvance("#")
            p.setPen(QColor(pal.fg_dim))
            p.drawText(QPointF(lx, 15), " high")

        font = _mono()
        font.setPixelSize(13)
        p.setFont(font)
        fm = QFontMetricsF(font)
        cw = max(fm.horizontalAdvance("@"), 4.0)
        chh = max(fm.height() * 0.92, 6.0)

        for disp_r in range(rows):
            r = rows - 1 - disp_r
            for c in range(cols):
                v = float(self._norm[r, c])
                zx = x0 + c * (zw + gap)
                zy = y0 + disp_r * (zh + gap)
                color = _heat_color(v, pal)

                # soft backing tint so a zone reads even between glyphs
                back = QColor(color)
                back.setAlphaF(0.07 + 0.20 * v)
                p.setPen(Qt.NoPen)
                p.setBrush(back)
                p.drawRoundedRect(QRectF(zx, zy, zw, zh), 4, 4)

                # the glyph block: density carries the value, seeded jitter
                # keeps it organic instead of a flat stamp
                p.setFont(font)
                gcols = max(int(zw // cw), 1)
                grows = max(int(zh // chh), 1)
                mx = zx + (zw - gcols * cw) / 2
                my = zy + (zh - grows * chh) / 2
                for gr in range(grows):
                    for gc in range(gcols):
                        d = v + (_seed(disp_r * 31 + gr, c * 17 + gc) - 0.5) * 0.22
                        d = min(max(d, 0.0), 1.0)
                        ch = _HEAT_RAMP[int(round(d * (len(_HEAT_RAMP) - 1)))]
                        if ch == " ":
                            continue
                        col = QColor(color)
                        col.setAlphaF(0.55 + 0.45 * d)
                        p.setPen(col)
                        p.drawText(QRectF(mx + gc * cw, my + gr * chh, cw * 2, chh * 1.4),
                                   Qt.AlignLeft | Qt.AlignVCenter, ch)

                # dim zone key in the corner (the r{row}c{col} contract, visible)
                if self._labels is not None and zw > 44 and zh > 30:
                    lf = _mono()
                    lf.setPixelSize(9)
                    p.setFont(lf)
                    lab = QColor(pal.fg_dim)
                    lab.setAlphaF(0.5)
                    p.setPen(lab)
                    p.drawText(QRectF(zx + 4, zy + 2, zw - 8, 12),
                               Qt.AlignLeft | Qt.AlignTop, self._labels[r][c])


class AsciiTrend(QWidget):
    """Metric-over-runs sparkline as a glyph matrix: each column is one run,
    the '.:-=+*#' ramp resolves the value's position inside its cell, a dim
    dotted area fills below the line, and the newest column carries a value
    tag. In RGB mode the columns run the rainbow."""

    def __init__(self, title: str = "", fmt: str = "{:.0%}", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._fmt = fmt
        self._values: list[float] = []
        self._tag: str | None = None
        self.setMinimumHeight(200)

    # ------------------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, values: Sequence[float], tag: str | None = None) -> None:
        self._values = [float(v) for v in values]
        self._tag = tag
        self.update()

    def clear(self) -> None:
        self.set_data([])

    def restyle(self, *_pal) -> None:
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        top = _paint_title(p, pal, self._title, w)
        if len(self._values) < 2:
            _paint_empty(p, pal, QRectF(0, top, w, h - top), "not enough runs yet")
            return

        font = _mono()
        font.setPixelSize(13)
        p.setFont(font)
        fm = QFontMetricsF(font)
        cw = fm.horizontalAdvance("@")
        chh = fm.height()

        tag_text = self._tag if self._tag is not None else self._fmt.format(self._values[-1])
        tag_w = fm.horizontalAdvance(tag_text) + 16
        x_left, x_right = 12.0, w - tag_w - 10
        y_top, y_bot = top + 6, h - 12.0

        # resample to exactly the columns available (up- and down-sampling),
        # so the line always spans the full panel width
        src = self._values
        ncols = max(int((x_right - x_left) // cw), 4)
        idx = np.linspace(0, len(src) - 1, ncols).round().astype(int)
        vals = [src[i] for i in idx]
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        grid_rows = max(int((y_bot - y_top) // chh), 4)
        x0 = x_left

        # dotted floor the columns stand on
        pen = QPen(QColor(pal.border), 1, Qt.DotLine)
        p.setPen(pen)
        p.drawLine(int(x_left), int(y_bot), int(x_right), int(y_bot))

        head_y = y_bot
        for j, v in enumerate(vals):
            yfrac = (v - lo) / (hi - lo)
            level = yfrac * (grid_rows - 1)
            cell = int(level)
            sub = level - cell
            if pal.rgb:
                col_base = QColor.fromHsvF((j / max(len(vals) - 1, 1)) * 0.83, 0.8, 1.0)
            else:
                col_base = QColor(pal.accent)
            x = x0 + j * cw
            # the line glyph: ramp char picks the height inside the cell
            ch = _TREND_RAMP[int(sub * (len(_TREND_RAMP) - 1))]
            y = y_bot - (cell + 1) * chh
            col = QColor(col_base)
            col.setAlphaF(0.72 + 0.28 * (j / max(len(vals) - 1, 1)))
            p.setPen(col)
            p.drawText(QRectF(x, y, cw * 2, chh * 1.2), Qt.AlignLeft | Qt.AlignVCenter, ch)
            # dim area fill below the line
            for k in range(cell):
                fill = QColor(col_base)
                fill.setAlphaF(0.14 + 0.26 * (k / max(cell, 1)))
                p.setPen(fill)
                p.drawText(QRectF(x, y_bot - (k + 1) * chh, cw * 2, chh * 1.2),
                           Qt.AlignLeft | Qt.AlignVCenter, ":")
            if j == len(vals) - 1:
                head_y = y_bot - (level + 0.5) * chh
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(col_base))
                p.drawEllipse(QPointF(x + cw * 0.5, head_y), 3.0, 3.0)

        # current-value tag riding the line's end
        tag_col = QColor(pal.accent)
        p.setPen(tag_col)
        p.setFont(font)
        ty = min(max(head_y, y_top + chh / 2), y_bot - chh / 2)
        p.drawText(QRectF(x_right + 6, ty - chh / 2, tag_w, chh),
                   Qt.AlignLeft | Qt.AlignVCenter, tag_text)

        # dim min/max scale marks
        sf = _mono()
        sf.setPixelSize(10)
        p.setFont(sf)
        p.setPen(QColor(pal.fg_dim))
        p.drawText(QRectF(x_left, y_top - 4, 90, 13), Qt.AlignLeft | Qt.AlignTop,
                   self._fmt.format(hi))
        p.drawText(QRectF(x_left, y_bot - 13, 90, 13), Qt.AlignLeft | Qt.AlignBottom,
                   self._fmt.format(lo))
