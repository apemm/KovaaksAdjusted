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

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMainWindow

from kovadapt.gui import color
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
