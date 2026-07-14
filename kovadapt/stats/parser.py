"""Parser for KovaaK's per-run stats CSVs.

File layout (verified against KovaaK's 3.9.x output):
    1. per-kill rows:  Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,...
    2. blank line, weapon summary table
    3. blank line(s), "Key:,Value" summary pairs (Kills:, Score:, Scenario:, ...)

Filename: "<Scenario> - <Mode> - YYYY.MM.DD-HH.MM.SS Stats.csv"
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import KillEvent, Run

_FILENAME_RE = re.compile(
    r"^(?P<scenario>.+) - (?P<mode>.+) - "
    r"(?P<ts>\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}) Stats\.csv$"
)


def parse_stats_filename(name: str) -> tuple[str, str, datetime] | None:
    """-> (scenario, mode, started) or None if not a stats file."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    ts = datetime.strptime(m.group("ts"), "%Y.%m.%d-%H.%M.%S")
    return m.group("scenario"), m.group("mode"), ts


def _clock_seconds(hhmmss: str) -> float:
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_stats_csv(path: Path | str) -> Run:
    path = Path(path)
    meta = parse_stats_filename(path.name)
    scenario = meta[0] if meta else path.stem
    started = meta[2] if meta else datetime.fromtimestamp(path.stat().st_mtime)

    text = path.read_text(encoding="utf-8", errors="replace")
    run = Run(scenario=scenario, started=started, source_file=str(path))

    kill_section, in_kills, t0 = [], False, None
    for line in io.StringIO(text):
        line = line.rstrip("\n")
        if line.startswith("Kill #,"):
            in_kills = True
            continue
        if in_kills:
            if not line.strip():
                in_kills = False
                continue
            kill_section.append(line)
        elif "," in line:
            # summary "Key:,Value" pairs (also catches weapon table rows; those
            # keys simply never collide with the accessors we use)
            key, _, val = line.partition(",")
            if key.endswith(":"):
                run.summary[key] = val.strip()

    for row in csv.reader(kill_section):
        if len(row) < 13:
            continue
        try:
            clock = _clock_seconds(row[1])
            if t0 is None:
                t0 = clock
            run.kills.append(
                KillEvent(
                    index=int(row[0]),
                    timestamp=row[1],
                    t=clock - t0,
                    bot=row[2],
                    weapon=row[3],
                    ttk=float(row[4].rstrip("s")),
                    shots=int(row[5]),
                    hits=int(row[6]),
                    accuracy=float(row[7]),
                    cheated=row[11].strip() == "1",
                    overshots=int(row[12]),
                )
            )
        except (ValueError, IndexError):
            continue
    return run


def iter_runs(
    stats_dir: Path | str,
    scenario: str | None = None,
    since: datetime | None = None,
) -> Iterator[Run]:
    """Yield parsed runs, oldest first, optionally filtered by scenario name."""
    entries = []
    for p in Path(stats_dir).glob("*.csv"):
        meta = parse_stats_filename(p.name)
        if meta is None:
            continue
        s, _, ts = meta
        if scenario is not None and s != scenario:
            continue
        if since is not None and ts <= since:
            continue
        entries.append((ts, p))
    for _, p in sorted(entries):
        yield parse_stats_csv(p)
