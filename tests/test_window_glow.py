"""The window's corner glow must be a colour, not a hole.

`QMainWindow`'s background brush is the BOTTOM of the paint stack. A
translucent gradient stop there has nothing behind it to blend into, so Qt
fills the alpha from the cleared backing store — black — and the "subtle 7%
accent glow" in the top-right corner rendered as a near-black blob instead.

It shipped for two versions because the dark themes hide it: the blob lands
within a few luminance points of a #16171d page. On cream it measured
luminance 20 against the page's 244, a charcoal box in the corner, and that
is how it was finally caught — by looking at the running app, not by a test.
So there are two pins here: the structural one that says no window-level
gradient stop may carry alpha, and the RENDERED one that says the corner
pixels look like the page. The rendered pin is the one that would have
caught it.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMainWindow

from kovadapt.gui import color
from kovadapt.gui import theme
from kovadapt.gui.theme import ACCENTS, build_palette, build_qss

MODES = [
    ("light", dict(dark=False)),
    ("dark", dict(dark=True)),
    ("midnight", dict(dark=True, midnight=True)),
    ("rgb", dict(dark=True, rgb=True)),
]
CASES = [(m, a) for m, _ in MODES for a in ACCENTS]


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _lum(c: QColor) -> float:
    return 0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()


# The QMainWindow rule, and only it: child widgets legitimately use rgba()
# because they DO have a parent painted behind them.
_WINDOW_RULE = re.compile(r"QMainWindow\s*\{(.*?)\}", re.S)


@pytest.mark.parametrize("mode,accent", CASES)
def test_no_window_gradient_stop_carries_alpha(mode, accent):
    """Structural: the window's own background may not ask the compositor
    for a colour it cannot supply."""
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    body = "\n".join(m.group(1) for m in _WINDOW_RULE.finditer(build_qss(pal)))
    assert body, "the QMainWindow background rule vanished"
    assert "rgba(" not in body, (
        f"{mode}/{accent}: a translucent stop in the window background — "
        f"there is nothing behind it, so this renders as black:\n{body}")


@pytest.mark.parametrize("mode,accent", CASES)
def test_the_corner_glow_stays_near_the_page(qapp, mode, accent):
    """Rendered, because the bug was only ever visible in pixels.

    The glow is meant to be a whisper of the accent. Anywhere in the window
    — corner included — must stay close to the page in luminance; the shipped
    bug put 224 points between them on cream.
    """
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    win = QMainWindow()
    win.setStyleSheet(build_qss(pal))
    win.resize(900, 600)

    img = win.grab().toImage()

    page = _lum(QColor(pal.bg))
    # cx 0.9 / cy 0.05 — sample the glow's own centre and its surroundings
    worst, where = 0.0, None
    for x in range(0, 900, 25):
        for y in range(0, 600, 25):
            d = abs(_lum(QColor(img.pixel(x, y))) - page)
            if d > worst:
                worst, where = d, (x, y)
    assert worst <= 24.0, (
        f"{mode}/{accent}: pixel at {where} is {worst:.0f} luminance from the "
        f"page ({page:.0f}) — the corner glow is punching a hole, not glowing")
    win.deleteLater()


@pytest.mark.parametrize("mode,accent", CASES)
def test_the_combo_caret_is_a_triangle_not_a_block(qapp, mode, accent):
    """Rendered, because this is invisible to any test that reads the sheet.

    Qt Style Sheets cannot draw a CSS triangle. The idiom — zero width and
    height, transparent left/right borders, one solid border — parsed fine
    and drew a SOLID RECTANGLE in fg_dim, on every combo box and all 25+
    spin controls in Adaptability. A triangle narrows: count the caret's
    pixels per row and the rows must strictly shrink. A block does not.
    """
    from PySide6.QtWidgets import QComboBox

    pal = build_palette(accent=accent, **dict(MODES)[mode])
    box = QComboBox()
    box.addItems(["Light", "Dark"])
    box.setStyleSheet(build_qss(pal))
    box.resize(220, 34)
    img = box.grab().toImage()

    page = _lum(QColor(pal.bg_alt))
    x0, x1 = img.width() - 34, img.width() - 4
    span = x1 - x0
    # Widths of the caret, row by row, over the drop-down area. A row that
    # spans the WHOLE sample is the widget's own top/bottom border, not the
    # caret — the caret can never be that wide.
    widths = []
    for y in range(img.height()):
        n = sum(1 for x in range(x0, x1)
                if abs(_lum(QColor(img.pixel(x, y))) - page) > 18)
        if 0 < n < span:
            widths.append(n)
    # Drop 1-2px rows: those are the rounded border's corner fringe clipping
    # into the sample window, and the caret's own antialiased tip.
    body = [n for n in widths if n >= 3]
    assert len(body) >= 3, (
        f"{mode}/{accent}: no caret drawn at all (rows {widths})")
    assert all(a > b for a, b in zip(body, body[1:])), (
        f"{mode}/{accent}: caret rows {body} do not taper — that is a block")
    assert max(body) <= 14, f"{mode}/{accent}: caret is {max(body)}px wide"
    box.deleteLater()


@pytest.mark.parametrize("mode,accent", CASES)
def test_the_primary_button_acknowledges_a_click(qapp, mode, accent):
    """The Play button had NO pressed state, in every theme and accent.

    `QPushButton:pressed` is declared before `QPushButton[accent="true"]`,
    and an attribute selector and a pseudo-state carry equal CSS2
    specificity — so the later declaration won and the accent fill was never
    replaced. The app's primary call to action, which writes a playlist and
    launches Steam, was byte-identical up and down while every ordinary
    button beside it darkened.

    Rendered, and compared over the button's whole rect: hover cannot stand
    in for this, because on the dark themes accent -> accent_hover is only
    7-13 per channel.
    """
    from PySide6.QtWidgets import QPushButton

    pal = build_palette(accent=accent, **dict(MODES)[mode])
    btn = QPushButton("Play adaptive task")
    btn.setProperty("accent", True)
    btn.setStyleSheet(build_qss(pal))
    btn.resize(200, 34)

    up = btn.grab().toImage()
    btn.setDown(True)
    down = btn.grab().toImage()

    changed = sum(1 for x in range(btn.width()) for y in range(btn.height())
                  if up.pixel(x, y) != down.pixel(x, y))
    assert changed > 0.5 * btn.width() * btn.height(), (
        f"{mode}/{accent}: only {changed}px of "
        f"{btn.width() * btn.height()} changed when pressed")
    # and the change must be a real step, not a rounding wobble
    mid = (btn.width() // 2, btn.height() // 2)
    assert abs(_lum(QColor(up.pixel(*mid))) - _lum(QColor(down.pixel(*mid)))) > 8.0
    btn.deleteLater()


def test_the_glow_is_the_composite_the_alpha_asked_for():
    """What replaced the alpha reproduces Qt's own SourceOver, in sRGB.

    Pinned against an independent implementation rather than the helper's
    own arithmetic, and explicitly NOT against `color.mix`: mixing this in
    linear light quadrupled the glow (luminance 76 on a page of 23), which
    is the trap this test exists to keep shut.
    """
    from kovadapt.gui.theme import _GLOW_T, _glow

    for mode, kwargs in MODES:
        for accent in ACCENTS:
            pal = build_palette(accent=accent, **kwargs)
            t = _GLOW_T[pal.is_dark]
            want = "#{:02x}{:02x}{:02x}".format(*(
                round(int(pal.bg[i:i + 2], 16)
                      + (int(pal.accent[i:i + 2], 16)
                         - int(pal.bg[i:i + 2], 16)) * t)
                for i in (1, 3, 5)))
            assert _glow(pal) == want, f"{mode}/{accent}"
            # a whisper, not a wash: nearer the page than the accent is
            assert (color.contrast_ratio(_glow(pal), pal.bg)
                    < color.contrast_ratio(pal.accent, pal.bg)), f"{mode}/{accent}"
            assert _glow(pal) != color.mix(pal.bg, pal.accent, t), (
                f"{mode}/{accent}: linear-light mix crept back in")


def test_the_help_button_actually_renders_its_glyph(qapp):
    """The nav bar's "?" rendered COMPLETELY BLANK — zero inked pixels.

    The sheet gives every QPushButton `padding: 7px 18px` plus a 1px border,
    which is 38px of chrome inside a 30px fixed width, so
    SE_PushButtonContents comes back with a NEGATIVE width and Qt draws no
    text at all. The only ink left was the menu chevron, so the single entry
    point to the guide, the hints toggle and the data folder read as a third
    unlabelled dropdown next to "Auto theme" and "Indigo".
    """
    from PySide6.QtWidgets import QPushButton

    pal = build_palette(dark=False, accent="indigo")
    sheet = build_qss(pal)

    def ink_rows(extra: str) -> int:
        b = QPushButton("?")
        b.setStyleSheet(sheet + extra)
        b.setFixedWidth(30)
        b.resize(30, b.sizeHint().height())
        img = b.grab().toImage()
        fg = QColor(pal.fg)
        n = 0
        for y in range(img.height()):
            for x in range(2, img.width() - 2):
                c = QColor(img.pixel(x, y))
                if (abs(c.red() - fg.red()) + abs(c.green() - fg.green())
                        + abs(c.blue() - fg.blue())) < 200:
                    n += 1
                    break
        b.deleteLater()
        return n

    assert ink_rows("") == 0, (
        "the 30px button now draws something with the shipped padding — "
        "re-check whether app.py still needs its override")
    assert ink_rows("\nQPushButton { padding: 7px 0px; }") >= 5, (
        "the override no longer restores the glyph")


@pytest.fixture(scope="module")
def fusion(qapp):
    """Fusion, installed ONCE for the whole module.

    The chrome under test is Fusion's own PE_PanelMenu, and the app sets
    Fusion at startup — without this the test runs on whatever the platform
    default is, the defect does not reproduce, and the test passes with the
    fix removed. Which is exactly what it did.

    Once, not per case: setStyle replaces the proxy style the fix installs,
    so re-running the install per case stacks app-wide event filters.
    """
    previous_sheet = qapp.styleSheet()
    qapp.setStyle("Fusion")
    theme.install_popup_style(qapp)
    yield qapp
    qapp.setStyleSheet(previous_sheet)


@pytest.mark.parametrize("mode,accent", CASES)
def test_the_dropdown_popup_wears_no_native_chrome(qapp, fusion, mode, accent):
    """Rendered, because the offending pixels come from a widget no stylesheet
    selector can name.

    QSS styles `QComboBox QAbstractItemView` — the LIST. The list lives inside
    a `QComboBoxPrivateContainer`, and Fusion paints its own PE_PanelMenu
    under that container: measured on cream, a SQUARE #969288 outer line
    around a 6px-rounded list, plus a 5px #fffaeb band above and below. Both
    strays are `QColor(bg_raised).darker(160)` and `.lighter(108)` exactly,
    which is the fingerprint this looks for.

    The plate was wrong too. The list painted bg_raised while the box it drops
    out of is bg_alt — 14.0 luminance apart on light, 5.1 on dark — so the
    popup read as a menu arriving from elsewhere rather than as that control
    opening.
    """
    from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

    # Fusion explicitly: the chrome under test is Fusion's PE_PanelMenu, and
    # the app sets Fusion at startup (app.py). Without this the test runs on
    # whatever the platform default is, the defect does not reproduce, and the
    # test passes with the fix removed — which is exactly what it did.
    # App-wide, not per-widget: the fix is a QProxyStyle plus an event filter
    # installed by theme._apply, and neither exists if the sheet is only ever
    # pushed onto one widget the way the caret test above does it.
    pal = build_palette(accent=accent, **dict(MODES)[mode])
    theme._apply(qapp, pal)
    host = QWidget()
    lay = QVBoxLayout(host)
    combo = QComboBox()
    combo.addItems(["Auto theme", "Dark", "Light", "Midnight", "RGB"])
    lay.addWidget(combo)
    host.resize(300, 140)
    host.show()
    qapp.processEvents()
    combo.showPopup()
    qapp.processEvents()

    img = combo.view().window().grab().toImage().convertToFormat(QImage.Format_RGB32)
    # numpy, not a pixel loop: this runs over 20 theme x accent combos and a
    # nested Python loop over ~37k pixels apiece put four minutes on the suite.
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]
    packed = (arr[:, :, 2].astype(np.uint32) << 16 | arr[:, :, 1].astype(np.uint32) << 8
              | arr[:, :, 0].astype(np.uint32))
    total = packed.size
    values, counts = np.unique(packed, return_counts=True)

    def count(colour: QColor) -> int:
        hit = np.nonzero(values == (colour.rgb() & 0xFFFFFF))[0]
        return int(counts[hit[0]]) if hit.size else 0

    raised = QColor(pal.bg_raised)
    strays = {"Fusion panel outline": raised.darker(160),
              "Fusion margin band": raised.lighter(108),
              # The container's own fill, showing in the inset it holds the
              # list down by. bg_raised is the menu surface; this popup is a
              # control opening, so not one pixel of it belongs here.
              "container plate": raised}
    for what, colour in strays.items():
        n = count(colour)
        assert n <= total * 0.002, (
            f"{mode}/{accent}: {n}px of {what} ({colour.name()}) in the popup "
            f"— native container chrome is showing through")

    plate = int(values[int(np.argmax(counts))])
    assert plate == (QColor(pal.bg_alt).rgb() & 0xFFFFFF), (
        f"{mode}/{accent}: the popup's plate is #{plate:06x}, not the closed "
        f"box's own fill {pal.bg_alt}")
    combo.hidePopup()
    host.deleteLater()
