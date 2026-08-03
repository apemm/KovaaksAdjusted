"""Optimizer window layout (offscreen QPA).

The window is four stacked group boxes, and the one the window exists for —
System checkup, a twelve-row list of probes and Fix buttons — opened as a
79px slot. Two of the four boxes carry no stretch factor, so their full
sizeHint comes off the top before the stretch factors divide anything, and
110px of the watchdog's was a read-only log pane that had never logged
anything. The un-stretched box ended up TALLER than the stretch-3 one.

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no system font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QGroupBox,
    QScrollArea,
)

from kovadapt.config import Settings  # noqa: E402
from kovadapt.optimize.checkup import CheckResult  # noqa: E402
from kovadapt.optimize.hardware import HardwareInfo  # noqa: E402
from kovadapt.gui import theme  # noqa: E402
from kovadapt.gui.optimizer_window import OptimizerWindow  # noqa: E402

# Two sizes, because they answer different questions. FLOOR is the smallest
# window the app will ever open (a 800px-tall laptop panel); PANEL is what it
# opens at on the 1080p desktop most users are on. A layout claim that only
# holds on the dev machine is not a claim.
WIN_W = 860
FLOOR_H, PANEL_H = 720, 940


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.kovadapt or the real registry."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr("kovadapt.gui.optimizer_window.startup_registered",
                        lambda: False)
    # The constructor kicks off a real scan: detect_hardware shells out to
    # PowerShell and every checkup probe reads live registry/power state.
    # These tests are about layout, and drive _on_scan with fixed rows
    # instead, so the scan never starts and no thread outlives the test.
    monkeypatch.setattr(OptimizerWindow, "rescan", lambda self: None)


@pytest.fixture(scope="module")
def qapp():
    """A QApplication carrying a KNOWN stylesheet.

    Every measurement here is in pixels, and the app-wide QSS sets padding,
    borders and font size on every widget in the window. The QApplication is
    shared across the whole pytest session, so whichever module ran last left
    its theme on it: these tests passed alone and failed in the full suite,
    off nothing but load order. Pin the palette, and put back whatever was
    there so the next module inherits what it expected.
    """
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    previous = app.styleSheet()
    theme._apply(app, theme.build_palette(dark=True, accent="indigo"))
    yield app
    app.setStyleSheet(previous)


@pytest.fixture()
def settings(tmp_path):
    root = tmp_path / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    return Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof"),
                    telemetry_enabled=False, onboarding_done=True, motion="off")


def _rows(n: int = 12) -> list[CheckResult]:
    """A realistic finished scan: the live checkup returns twelve of these."""
    return [CheckResult(
        check_id=f"c{i}", title=f"Check {i}",
        detail="A sentence of detail explaining what this probe measured "
               "and why it matters for aim consistency.",
        status=["ok", "warn", "info", "bad"][i % 4],
        can_fix=(i % 3 == 0), safe=True) for i in range(n)]


def _built(settings, qapp, height: int = FLOOR_H) -> OptimizerWindow:
    w = OptimizerWindow(settings)
    w._on_scan(HardwareInfo(), _rows())
    w.resize(WIN_W, height)
    w.show()
    qapp.processEvents()
    return w


def _boxes(w) -> dict[str, QGroupBox]:
    return {b.title().split(" (")[0]: b for b in w.findChildren(QGroupBox)}


def _checkup_viewport(w) -> int:
    for sc in w.findChildren(QScrollArea):
        par = sc.parent()
        while par is not None and not isinstance(par, QGroupBox):
            par = par.parent()
        if par is not None and "checkup" in par.title().lower():
            return sc.viewport().height()
    raise AssertionError("no scroll area inside the System checkup box")


def test_the_checkup_gets_more_room_than_the_boxes_beside_it(qapp, settings):
    """It opened at 149px over 862px of rows while the watchdog box beside it,
    carrying NO stretch factor at all, took 135 — and 110 of that was an empty
    log pane. A QBoxLayout hands every un-stretched child its full sizeHint
    before the stretch factors divide what is left."""
    w = _built(settings, qapp)
    vp = _checkup_viewport(w)
    boxes = _boxes(w)

    assert vp >= 190, f"the checkup opens as a {vp}px slot at the floor size"
    assert vp > boxes["Watchdog"].height(), (
        f"checkup viewport {vp} is no bigger than the un-stretched watchdog "
        f"box {boxes['Watchdog'].height()}")
    assert boxes["System checkup"].height() > boxes["Recommended for your hardware"].height(), (
        "reference prose is given more room than the list this window is for")
    w.shutdown()


def test_on_a_1080p_panel_the_checkup_shows_a_usable_run_of_rows(qapp, settings):
    """The floor size is a floor. What most users get is this, and here the
    list has to be genuinely readable rather than merely better than 149."""
    w = _built(settings, qapp, PANEL_H)
    vp = _checkup_viewport(w)
    assert vp >= 250, f"only {vp}px of checkup on a 1080p panel"
    # ~76px per row: enough to read three and know there are more
    assert vp // 76 >= 3, f"{vp}px shows fewer than three rows"
    w.shutdown()


def test_the_watchdog_log_stays_out_of_the_way_until_it_has_a_line(qapp, settings):
    """An empty read-only pane was taking 110px off the top of the window
    before any stretch factor got a say. The "Evidence:" label above it
    already explains what will appear there, so the pane was not even
    carrying the explanation."""
    w = _built(settings, qapp, PANEL_H)
    assert not w.wd_log.isVisible(), "an empty log is holding layout space"
    before = _checkup_viewport(w)

    w._log("watchdog: KovaaK's launched, priority set to High")
    qapp.processEvents()

    assert w.wd_log.isVisible(), "the log never appears once it has something"
    assert "priority set to High" in w.wd_log.toPlainText()
    after = _checkup_viewport(w)
    # it costs the checkup room, which is the trade — but only once it is
    # actually carrying a record, and the checkup must not collapse for it
    assert after >= 240, f"the checkup collapsed to {after}px when the log opened"
    assert after <= before
    w.shutdown()


def test_opening_the_log_does_not_gut_the_checkup_at_the_floor_size(qapp, settings):
    """The narrow case the log cap was cut for: at 110px tall, one watchdog
    event dropped the checkup from 197px to 81 — smaller than a single row."""
    w = _built(settings, qapp)
    w._log("watchdog: KovaaK's launched, priority set to High")
    qapp.processEvents()
    vp = _checkup_viewport(w)
    assert vp >= 110, f"one log line cut the checkup to {vp}px"
    w.shutdown()


def test_the_window_does_not_open_taller_than_the_desktop(qapp, settings):
    """The height is taken from the screen now, so it has to stay inside it —
    a window that opens with its lower half off-screen is worse than a short
    one, and the checkup is the part that would be cut."""
    from PySide6.QtGui import QGuiApplication

    w = OptimizerWindow(settings)
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        assert w.height() <= screen.availableGeometry().height(), (
            "the window opens taller than the desktop it is on")
    assert w.height() >= 640, "never smaller than the authored minimum"
    w.shutdown()


def test_a_check_row_is_transparent_but_its_fix_button_is_not(qapp, settings):
    """_CheckRow is a QFrame, and theme.py's page-background rule reaches
    every QWidget subclass, so all twelve rows painted the PAGE colour over
    the checkup box's plate. It cannot go in the global exemption list the way
    QCheckBox did: `QFrame` as a type selector also matches QLabel,
    QScrollArea, QSplitter and QStackedWidget.

    So it carries its own sheet — SCOPED to its class, which PySide publishes
    to QSS under the Python name. Unscoped it cascades to the children and
    takes the Fix button's accent fill to #010102.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    from kovadapt.gui import theme
    from kovadapt.gui.optimizer_window import _CheckRow

    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout

    pal = theme.current()
    # Inside a group box, which is the whole point: the row has to be
    # transparent AGAINST a plate, and standalone there is no plate to
    # overpaint. Rendering it bare passes with the transparency removed.
    box = QGroupBox("System checkup")
    lay = QVBoxLayout(box)
    row = _CheckRow(CheckResult(check_id="c", title="Game process priority",
                                detail="KovaaK's runs at Normal.",
                                status="warn", can_fix=True, safe=True),
                    lambda *_: None)
    lay.addWidget(row)
    box.resize(620, 100)
    box.show()
    qapp.processEvents()

    pm = QPixmap(620, 100)
    pm.fill(QColor(pal.bg))
    box.render(pm)
    img = pm.toImage().convertToFormat(QImage.Format_RGB32)
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]

    # the row's own background, sampled between the detail text and the button
    rtl = row.mapTo(box, row.rect().topLeft())
    back = arr[rtl.y() + 4:rtl.y() + row.height() - 4,
               rtl.x() + row.width() - 140:rtl.x() + row.width() - 110]
    back = back.reshape(-1, 3)[:, ::-1].mean(axis=0)
    page = np.array([int(pal.bg[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    assert not np.allclose(back, page, atol=2), (
        f"the check row paints the page colour {pal.bg} over the checkup "
        f"box's plate")

    row_in_box = row
    btn = row_in_box.fix_btn
    tl = btn.mapTo(box, btn.rect().topLeft())
    # The most common colour in the button, not a mean over it: the label
    # sits in the middle in accent_fg and the corners are antialiased, so
    # every average lands somewhere between the fill and something else.
    # The fill is what most of the button IS.
    patch = arr[tl.y() + 2:tl.y() + btn.height() - 2,
                tl.x() + 2:tl.x() + btn.width() - 2][:, :, ::-1].reshape(-1, 3)
    packed = (patch[:, 0].astype(np.uint32) << 16 | patch[:, 1].astype(np.uint32) << 8
              | patch[:, 2].astype(np.uint32))
    values, counts = np.unique(packed, return_counts=True)
    fill = int(values[int(np.argmax(counts))])

    assert fill == int(pal.accent[1:], 16), (
        f"the Fix button's fill is #{fill:06x}, not the accent "
        f"{pal.accent} — the row's transparency has cascaded onto it")
    box.setParent(None)
    box.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("accent", ["indigo", "ocean", "mint", "rose", "ember"])
def test_an_info_row_is_not_painted_in_whatever_accent_you_picked(accent):
    """`info` was pal.accent, and on live probes info is the LARGEST bucket of
    the twelve checkup rows — nearly half the list wearing the user's colour
    preference as a status. Under rose every informational row reads as an
    error; under mint, as an all-clear. The ○ shape survives that, but colour
    is the channel a reader takes a verdict from.
    """
    from kovadapt.gui import theme
    from kovadapt.gui.optimizer_window import _status_color

    theme._apply(QApplication.instance() or QApplication([]),
                 theme.build_palette(dark=True, accent=accent))
    pal = theme.current()
    assert _status_color("info") != pal.accent, (
        f"{accent}: informational rows are painted in the accent")
    for status in ("ok", "warn", "bad"):
        assert _status_color(status) != pal.accent, (
            f"{accent}: {status} rows are painted in the accent")
    # ...and "here is a reading" must not look like "could not read this"
    assert _status_color("info") != _status_color("unknown")
