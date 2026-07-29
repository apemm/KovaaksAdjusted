"""GameUserSettings.ini decoding: a healthy config is never offered for delete.

The config probe is the only check whose fix DESTROYS a user file, so a false
"corrupt" verdict is the worst outcome in the optimizer. Reading with the
process ANSI codepage produced exactly that on healthy Unreal-written files.

Everything here runs against a tmp LOCALAPPDATA; nothing reads or writes the
real game config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kovadapt.optimize.checkup import SystemCheckup, game_config_path, read_game_config
from kovadapt.optimize.hardware import HardwareInfo

# Enough of a real config to satisfy the section heuristic.
BODY = (
    "[/Script/Engine.GameUserSettings]\n"
    "bUseVSync=False\n"
    "ResolutionSizeX=2560\n"
    "[ScalabilityGroups]\n"
    "sg.ViewDistanceQuality=3\n"
)


@pytest.fixture
def config_probe(tmp_path, monkeypatch):
    """A checkup whose game_config_path() lands in tmp_path, with home
    redirected too so nothing can reach the developer's real state."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    ini = game_config_path()
    ini.parent.mkdir(parents=True)
    assert str(tmp_path) in str(ini)
    return ini, SystemCheckup("", HardwareInfo())


# Encodings Unreal actually writes. The accented/CJK cases are the regression:
# their UTF-8 bytes contain 0x81/0x8D/0x8F/0x90/0x9D, which are undefined slots
# in cp1252, so the old locale-codepage read raised UnicodeDecodeError and the
# healthy file was billed "unreadable/corrupt" with a delete button attached.
HEALTHY = {
    "ascii-utf8": BODY.encode("utf-8"),
    "utf8-bom": BODY.encode("utf-8-sig"),
    "utf8-acute": (BODY + "LastPlayerName=Ástrid\n").encode("utf-8"),   # C3 81
    "utf8-ydiaeresis": (BODY + "Crosshair=Ýggdrasil.png\n").encode("utf-8"),
    "utf8-hiragana": (BODY + "LastScenario=きもち\n").encode("utf-8"),
    "utf8-bom-acute": (BODY + "LastPlayerName=Íris\n").encode("utf-8-sig"),
    "utf16-le": BODY.encode("utf-16"),                          # AutoDetect save path
    # Unbommed UTF-16 is indistinguishable from garbage, so pin the BOM'd form.
    "utf16-be": b"\xfe\xff" + BODY.encode("utf-16-be"),
    "legacy-ansi": (BODY + "Note=café\n").encode("cp1252"),
}


@pytest.mark.parametrize("label", sorted(HEALTHY))
def test_healthy_config_is_never_offered_for_deletion(config_probe, label):
    ini, checkup = config_probe
    ini.write_bytes(HEALTHY[label])

    res = checkup._c_config()
    assert res.status == "ok", f"{label}: {res.detail}"
    assert not res.can_fix, f"{label}: healthy config offered a destructive fix"
    assert not res.fix_label


def test_decoder_round_trips_every_healthy_encoding(config_probe):
    """The decoded text must be the real ini content, not mojibake — the
    section/NUL heuristics downstream only work on real text."""
    ini, _ = config_probe
    for label, data in HEALTHY.items():
        ini.write_bytes(data)
        text = read_game_config(ini)
        assert text is not None, label
        assert "[ScalabilityGroups]" in text, label
        assert "\x00" not in text, label
        assert not text.startswith("﻿"), label   # BOM stripped, not kept


def test_genuinely_corrupt_config_is_still_caught(config_probe):
    """Tolerating encodings must not blunt the check it exists for."""
    ini, checkup = config_probe

    ini.write_bytes(b"\x00\x00garbage\x00")            # binary sludge
    res = checkup._c_config()
    assert res.status == "bad" and res.can_fix and not res.safe

    ini.write_bytes(b"\xff\xfe" + b"\x00\x01\x02")     # UTF-16 BOM, odd tail
    assert checkup._c_config().status == "bad"

    ini.write_bytes(b"nothing=here\n")                 # decodes, no sections
    res = checkup._c_config()
    assert res.status == "bad" and "missing expected sections" in res.detail


def test_missing_config_is_info_not_bad(config_probe):
    _ini, checkup = config_probe
    res = checkup._c_config()
    assert res.status == "info" and not res.can_fix
