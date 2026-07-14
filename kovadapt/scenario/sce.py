"""Lossless-ish reader/writer for KovaaK's .sce scenario files.

Format (verified against KovaaK's 3.9.x):
    - header: Key=Value lines
    - repeated INI-ish sections: [Aim Profile], [Bot Profile],
      [Character Profile], [Dodge Profile], [Weapon Profile], ...
    - trailing [Map Data]: an embedded reflex-format map with brushes and
      `entity` blocks (type PlayerSpawn / WorldSpawn / ...), tab-indented.

We keep every line verbatim and edit surgically, so untouched content
round-trips byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ENTITY_RE = re.compile(r"^\t?entity\s*$")
_POSITION_RE = re.compile(
    r"^(\s*Vector3 position )(-?[\d.]+) (-?[\d.]+) (-?[\d.]+)\s*$"
)


@dataclass
class SpawnPoint:
    """One PlayerSpawn entity in [Map Data]."""

    start: int          # line index of its `entity` line
    end: int            # line index one past the last line of the block
    x: float
    y: float
    z: float
    lines: list[str]    # verbatim block lines


class SceFile:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self._map_data_start = self._find_map_data()

    # ------------------------------------------------------------------ io
    @classmethod
    def read(cls, path: Path | str) -> "SceFile":
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return cls(text.split("\n"))

    def write(self, path: Path | str) -> None:
        Path(path).write_text("\n".join(self.lines), encoding="utf-8", newline="\n")

    def _find_map_data(self) -> int:
        for i, ln in enumerate(self.lines):
            if ln.strip() == "[Map Data]":
                return i
        return len(self.lines)

    # -------------------------------------------------------------- header
    def get_header(self, key: str) -> str | None:
        prefix = key + "="
        for ln in self.lines[: self._map_data_start]:
            if ln.startswith("["):
                break
            if ln.startswith(prefix):
                return ln[len(prefix):]
        return None

    def set_header(self, key: str, value: str) -> None:
        prefix = key + "="
        for i, ln in enumerate(self.lines[: self._map_data_start]):
            if ln.startswith("["):
                break
            if ln.startswith(prefix):
                self.lines[i] = f"{key}={value}"
                return
        self.lines.insert(0, f"{key}={value}")
        self._map_data_start += 1

    # ------------------------------------------------------------ sections
    def _section_spans(self, section: str) -> list[tuple[int, int]]:
        """[(start, end)) line spans for every `[section]` occurrence."""
        spans, start = [], None
        header = f"[{section}]"
        for i, ln in enumerate(self.lines):
            s = ln.strip()
            if s.startswith("[") and s.endswith("]"):
                if start is not None:
                    spans.append((start, i))
                    start = None
                if s == header:
                    start = i
        if start is not None:
            spans.append((start, len(self.lines)))
        return spans

    def find_section(self, section: str, name: str) -> tuple[int, int] | None:
        """Span of the `[section]` block whose Name= matches."""
        for start, end in self._section_spans(section):
            for ln in self.lines[start:end]:
                if ln.startswith("Name=") and ln[5:].strip() == name:
                    return start, end
        return None

    def get_in_section(self, section: str, name: str, key: str) -> str | None:
        span = self.find_section(section, name)
        if span is None:
            return None
        prefix = key + "="
        for ln in self.lines[span[0]: span[1]]:
            if ln.startswith(prefix):
                return ln[len(prefix):]
        return None

    def set_in_section(self, section: str, name: str, key: str, value) -> bool:
        span = self.find_section(section, name)
        if span is None:
            return False
        prefix = key + "="
        for i in range(span[0], span[1]):
            if self.lines[i].startswith(prefix):
                self.lines[i] = f"{key}={value}"
                return True
        return False

    # ---------------------------------------------------------- spawn points
    def spawn_points(self) -> list[SpawnPoint]:
        """Parse PlayerSpawn entity blocks inside [Map Data]."""
        pts: list[SpawnPoint] = []
        i, n = self._map_data_start, len(self.lines)
        while i < n:
            if _ENTITY_RE.match(self.lines[i]):
                start = i
                j = i + 1
                is_spawn, xyz = False, None
                while j < n and not _ENTITY_RE.match(self.lines[j]) and not self.lines[j].lstrip().startswith("brush"):
                    s = self.lines[j].strip()
                    if s == "type PlayerSpawn":
                        is_spawn = True
                    m = _POSITION_RE.match(self.lines[j])
                    if m:
                        xyz = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                    j += 1
                if is_spawn and xyz is not None:
                    pts.append(SpawnPoint(start, j, *xyz, lines=self.lines[start:j]))
                i = j
            else:
                i += 1
        return pts

    def replace_spawn_points(self, new_blocks: list[list[str]]) -> None:
        """Replace all PlayerSpawn entity blocks with the provided blocks
        (each a verbatim list of lines). Non-spawn content is untouched;
        new blocks are written contiguously at the position of the first."""
        pts = self.spawn_points()
        if not pts:
            raise ValueError("no PlayerSpawn entities found in [Map Data]")
        keep: list[str] = []
        cursor = 0
        first = pts[0].start
        for p in pts:
            keep.extend(self.lines[cursor: p.start])
            cursor = p.end
        tail = self.lines[cursor:]
        flat = [ln for block in new_blocks for ln in block]
        head = keep[:first]
        mid = keep[first:]
        self.lines = head + flat + mid + tail
        self._map_data_start = self._find_map_data()  # recompute index after splice
