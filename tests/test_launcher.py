"""Launcher: install detection, deep-link URLs, playlist writing.

Launch functions never fire for real here: _open_steam_url is monkeypatched
(on Windows os.startfile would genuinely start the game).
"""

from __future__ import annotations

import json

from kovadapt import launcher
from kovadapt.config import ADAPTIVE_SUFFIX, Settings


def make_settings(tmp_path):
    """Fake Steam-library tree: <lib>/steamapps/common/FPSAimTrainer/FPSAimTrainer."""
    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    (root / "Saved" / "SaveGames" / "Scenarios").mkdir(parents=True)
    return Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof")), root


# ------------------------------------------------------------------ deep link
def test_scenario_url_is_the_binary_template_form():
    url = launcher.scenario_url("1wall 6targets small [Adaptive]")
    assert url.startswith("steam://run/824270//?action=jump-to-scenario&name=")
    # urllib.parse.quote(safe="") — spaces AND brackets percent-encoded
    assert "1wall%206targets%20small%20%5BAdaptive%5D" in url
    assert url.endswith("&mode=challenge")


def test_scenario_url_mode_override():
    assert launcher.scenario_url("X", mode="freeplay").endswith("&mode=freeplay")


# ------------------------------------------------------------------- install
def test_check_install_finds_root_and_manifest(tmp_path):
    s, root = make_settings(tmp_path)
    st = launcher.check_install(s)
    assert st.root_found
    assert not st.manifest_found
    steamapps = root.parents[2]             # .../steamapps
    assert steamapps.name == "steamapps"
    (steamapps / f"appmanifest_{launcher.STEAM_APP_ID}.acf").write_text(
        '"appid" "824270"')
    assert launcher.check_install(s).manifest_found


def test_check_install_missing_root(tmp_path):
    s = Settings(kovaaks_root=str(tmp_path / "nowhere"),
                 profile_dir=str(tmp_path / "prof"))
    st = launcher.check_install(s)
    assert not st.root_found
    assert not st.manifest_found
    assert "not found" in st.describe()


# ----------------------------------------------------------------- playlists
def test_write_playlist_schema_and_encoding(tmp_path):
    s, _root = make_settings(tmp_path)
    p = launcher.write_playlist(s, [("X [Adaptive]", 3), ("Y", 0)])
    assert p.parent == s.playlists_dir
    raw = p.read_bytes()
    assert raw[:1] == b"{"                  # UTF-8 without BOM
    assert b"\t" in raw                     # tab indent like the game's files
    doc = json.loads(p.read_text(encoding="utf-8"))
    # key casing is load-bearing for the game's JSON converter
    assert doc["scenarioList"] == [
        {"scenario_name": "X [Adaptive]", "play_Count": 3},
        {"scenario_name": "Y", "play_Count": 1},   # clamped to >= 1
    ]
    assert doc["playlistName"] == launcher.PLAYLIST_NAME
    assert isinstance(doc["updated"], int)


def test_playlist_filename_sanitized(tmp_path):
    s, _root = make_settings(tmp_path)
    p = launcher.write_playlist(s, [("X", 1)], name='a<b>:c"|?*')
    assert p.name == "a_b_c_.json"


# -------------------------------------------------------------- play_adaptive
def test_play_adaptive_requires_variant(tmp_path):
    s, _root = make_settings(tmp_path)
    msg, ok = launcher.play_adaptive(s, "Foo")
    assert not ok
    assert "start adapting" in msg


def test_play_adaptive_writes_playlist_and_fires_deep_link(tmp_path, monkeypatch):
    s, _root = make_settings(tmp_path)
    adaptive = "Foo" + ADAPTIVE_SUFFIX
    (s.scenarios_dir / f"{adaptive}.sce").write_text("[Scenario]\n")
    fired: list[str] = []
    monkeypatch.setattr(launcher, "_open_steam_url",
                        lambda url: (fired.append(url), ("", True))[1])
    msg, ok = launcher.play_adaptive(s, "Foo")
    assert ok, msg
    doc = json.loads(launcher._playlist_path(s).read_text(encoding="utf-8"))
    assert doc["scenarioList"][0]["scenario_name"] == adaptive
    assert fired and "jump-to-scenario" in fired[0]
    assert "Foo%20%5BAdaptive%5D" in fired[0]
