"""A NaN must never reach a paint handler, and never get into a profile.

`int(nan)` raises ValueError and `int(-inf)` raises OverflowError. Thrown from
inside `paintEvent` that leaks the QPainter, and Qt then aborts the process
("Cannot destroy paint device that is being painted", 0xC0000409 / 0xC0000005).
Catching the Python exception does NOT save it: the except block runs and the
process still dies a frame later, which is why this is tested by running a
subprocess and reading its exit code rather than by asserting in-process.

The path is real, not theoretical. `stats/parser.py` read accuracy as
`float(row[7])` straight out of KovaaK's CSV and `float("nan")` parses
happily; that lands in `profile.history`, `json.dumps` writes a bare `NaN`
token, `json.loads` reads it straight back, and `dashboard.py` hands the list
to AsciiTrend on launch. One malformed stats row made the app un-openable
until the profile was deleted by hand.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kovadapt.profile.player import PlayerProfile
from kovadapt.stats.parser import parse_stats_csv

REPO = Path(__file__).resolve().parent.parent

_CRASH_PROBE = textwrap.dedent(
    """
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", "C:" + chr(92) + "Windows" + chr(92) + "Fonts")
    sys.path.insert(0, {repo!r})
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    from kovadapt.gui import viz, theme
    theme._apply(app, theme.build_palette(dark=True, accent="indigo"))
    bad = float({value!r})
    if {which!r} == "bars":
        w = viz.AsciiBars(title="t"); w.set_data(["a", "b", "c"], [0.5, bad, 0.2])
    else:
        w = viz.AsciiTrend(); w.set_data([0.5, bad, 0.2])
    w.resize(600, 240)
    w.grab()
    print("SURVIVED")
    """
)


@pytest.mark.parametrize("which", ["bars", "trend"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_non_finite_value_does_not_take_the_process_down(which, value, tmp_path):
    """Subprocess, because the failure mode is a process abort that no
    in-process assertion can observe."""
    pytest.importorskip("PySide6")
    probe = tmp_path / "probe.py"
    probe.write_text(_CRASH_PROBE.format(repo=str(REPO), value=value, which=which),
                     encoding="utf-8")
    done = subprocess.run([sys.executable, str(probe)], capture_output=True,
                          text=True, timeout=180)
    assert done.returncode == 0, (
        f"{which} with {value} exited {done.returncode!r} — a non-finite value "
        f"is still killing the process.\nstderr: {done.stderr[-400:]}")
    assert "SURVIVED" in done.stdout


def test_the_charts_draw_their_empty_state_rather_than_partial_nonsense():
    """All-or-nothing: dropping the bad entries would silently renumber a run
    history, and a trend that quietly skips run 4 is worse than a panel saying
    it has nothing to draw."""
    pytest.importorskip("PySide6")
    from kovadapt.gui import viz

    assert viz.finite([0.5, 0.3]) == [0.5, 0.3]
    assert viz.finite([0.5, float("nan")]) == []
    assert viz.finite([float("inf")]) == []
    assert viz.finite([0.1, 0.2, float("-inf")]) == []


def test_a_malformed_stats_row_never_becomes_a_non_finite_measurement(tmp_path):
    """`float("nan")` does not raise, so the parser's existing row-skipping
    guard let it straight through."""
    csv = tmp_path / "Wall Task - Challenge - 2026.07.28-10.00.00 Stats.csv"
    csv.write_text(
        "Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,Damage Done,"
        "Damage Possible,Efficiency,Cheated,Overshots\n"
        "1,10:00:01.000,bot,gun,0.500s,4,3,0.75,100,100,1,0,0\n"
        "2,10:00:02.000,bot,gun,0.500s,4,3,nan,100,100,1,0,0\n"
        "3,10:00:03.000,bot,gun,0.500s,4,2,0.50,100,100,1,0,0\n",
        encoding="utf-8")

    run = parse_stats_csv(csv)
    accs = [k.accuracy for k in run.kills]
    assert all(math.isfinite(a) for a in accs), f"a non-finite accuracy survived: {accs}"
    assert accs == [0.75, 0.50], "the good rows either side were dropped too"


def test_a_profile_carrying_a_nan_still_opens_the_app(tmp_path, monkeypatch):
    """The last line of defence, and the one that matters for profiles ALREADY
    poisoned on disk: json writes a bare NaN token and reads it straight back,
    so the fix cannot live only at the parser."""
    pytest.importorskip("PySide6")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    prof = PlayerProfile(scenario="X [Adaptive]")
    prof.run_count = 3
    prof.history = [{"accuracy": 0.61, "kps": 1.0, "score": 400.0},
                    {"accuracy": float("nan"), "kps": 1.0, "score": 400.0}]
    prof.save(tmp_path)

    raw = next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")
    assert "NaN" in raw, "the fixture no longer reproduces a poisoned profile"
    with pytest.raises(ValueError):
        json.loads(raw, parse_constant=lambda _c: (_ for _ in ()).throw(ValueError))

    back = PlayerProfile.load("X [Adaptive]", tmp_path)
    accs = [float(h.get("accuracy", 0.0)) for h in back.history]
    assert any(not math.isfinite(a) for a in accs), "fixture no longer round-trips a NaN"

    from kovadapt.gui import viz
    assert viz.finite(accs) == [], (
        "a poisoned history reaches the chart as drawable data")
