"""Difficulty shadow policy — SCAFFOLD ONLY, nothing here is trained.

Intended role (future pass): run *alongside* ``AdaptationEngine.plan()``
and propose target-scale / movement adjustments learned from logged
``(profile state, emitted plan, next-run outcome)`` transitions. The
engine's deadband controller stays authoritative; the shadow policy only
ever logs what it *would* have done until its offline replay demonstrably
beats the controller. The watcher already appends one transition record
per processed run (``SessionWatcher._log_shadow``), so the training set
accumulates from day one; ``propose()`` is consulted there but, being
untrained, never influences the emitted plan.

Interface contract (stable, so the training pass can slot in):

- ``DifficultyShadowPolicy.propose(profile_state)`` takes the same profile
  snapshot the engine sees (see ``PROFILE_STATE_KEYS``) and returns a
  ``ShadowSuggestion`` or ``None``. The scaffold ALWAYS returns ``None``
  — there is no trained policy to consult.
- Transition logging is append-only JSONL at
  ``<profile_dir>/ml/shadow_log.jsonl``; one record per processed run,
  schema in ``SHADOW_LOG_SCHEMA``. The log is the future training set, so
  the schema is versioned from day one.

Torch-free on purpose: the scaffold must import on a core install.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Keys ``propose()`` expects in ``profile_state`` (a plain-dict snapshot of
#: ``PlayerProfile`` fields plus the fatigue stamp; all optional — the
#: policy must tolerate missing keys, matching the JSON round-trip contract).
PROFILE_STATE_KEYS = (
    "scenario",
    "archetype",
    "run_count",
    "ewma_accuracy",
    "ewma_ttk",
    "ewma_kps",
    "ewma_score",
    "ewma_bias",
    "target_scale",
    "movement",
    "fatigue",          # dict, session FatigueState snapshot (may be empty)
)

#: JSONL record layout for the shadow transition log. Field -> description.
#: ``schema`` is a version tag: bump it on any breaking change and keep the
#: reader tolerant of older records (same policy as the profile/report JSON).
SHADOW_LOG_SCHEMA = {
    "schema": "shadow-v3",
    "ts": "ISO timestamp of the processed run",
    "scenario": (
        "The adaptive scenario this transition belongs to. v2 specified a "
        "pairing rule that reads 'for the same scenario' and then recorded no "
        "scenario anywhere — not in the record, not in profile_state — so the "
        "rule could not be applied to its own data. With one scenario adapted "
        "it works by accident; the moment a second is interleaved, a forward "
        "pair joins one task's plan to another task's outcome, and run_count "
        "(which is per profile, i.e. per scenario) reads as gaps everywhere. "
        "Records without this field are unpairable in a multi-scenario log "
        "and `training_pairs` drops them rather than guessing."
    ),
    "profile_state": "dict of PROFILE_STATE_KEYS captured BEFORE observe()",
    "plan": "the emitted AdaptationPlan as a dict (target_scale, movement, ...)",
    "suggestion": "ShadowSuggestion as a dict, or null while untrained",
    "run_outcome": (
        "THIS run's measured result: {accuracy, score, kps}. The reward for "
        "record[i]'s plan is record[i+1]['run_outcome'] for the same "
        "scenario — the plan emitted after run i is what run i+1 was played "
        "on. Pair forward; do not read this as the outcome OF this record's "
        "plan. `kps` is NULL when the run had fewer than two kill rows: "
        "invincible-target scenarios report none, so a 0.0 there would be a "
        "structural fake in an append-only log, not a measurement. Treat "
        "null as missing, never as zero. Gaps are detectable without a "
        "sequence field: profile_state.run_count increments by one per "
        "processed run of a scenario, so a jump means a record was dropped "
        "and the neighbouring pair must not be used."
    ),
}

# v1 records carry an `outcome` key that is ALWAYS null. The field was
# specified as "next-run outcome when known" and nothing ever wrote it, so
# every v1 transition holds a state and an action with no reward attached —
# unusable for off-policy learning on its own. The reward is recoverable by
# joining v1 rows against the report library on (scenario, ts), and any
# training pass must do that for pre-v2 rows. It is fixed forward rather than
# back-filled because this log is append-only by design.
SHADOW_LOG_V1_OUTCOME_IS_ALWAYS_NULL = True

LOG_NAME = "shadow_log.jsonl"


@dataclass(frozen=True)
class ShadowSuggestion:
    """What the shadow policy would set, with its own confidence.

    ``target_scale``/``movement`` are absolute values in the same units the
    engine emits (multiplier on base size; 0..1 movement intensity), NOT
    deltas — matching the plan-is-absolute contract of the generator.
    """

    target_scale: float
    movement: float
    confidence: float   # 0..1, calibrated by the (future) training pass
    reason: str         # short human-readable provenance for the GUI/log


class DifficultyShadowPolicy:
    """NOT YET TRAINED — ``propose()`` always returns ``None``.

    The constructor takes ``profile_dir`` only to anchor the transition
    log; no model weights exist yet.
    """

    def __init__(self, profile_dir: Path | str | None = None) -> None:
        self.profile_dir = Path(profile_dir) if profile_dir is not None else None

    @property
    def log_path(self) -> Path | None:
        if self.profile_dir is None:
            return None
        return self.profile_dir / "ml" / LOG_NAME

    def propose(self, profile_state: dict) -> ShadowSuggestion | None:
        """Suggest difficulty for the state, or ``None`` when the policy has
        nothing trustworthy to say. The scaffold has no trained policy, so
        this is unconditionally ``None`` — callers must already handle that
        (it stays the answer for cold-start profiles even once trained)."""
        return None

    def log_transition(self, record: dict) -> Path | None:
        """Append one ``SHADOW_LOG_SCHEMA`` record to the JSONL log.

        Fills in the schema tag; returns the log path, or ``None`` when the
        policy was built without a ``profile_dir``. Append-only by design —
        the log is the future training set."""
        path = self.log_path
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"schema": SHADOW_LOG_SCHEMA["schema"], **record}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return path


def read_transitions(profile_dir: Path | str) -> tuple[list[dict], dict]:
    """(records, diagnostics) from the JSONL log, oldest first.

    Tolerant by policy: a malformed line is counted and skipped rather than
    raising, because this file is append-only and one bad write must not make
    the whole training set unreadable.
    """
    path = Path(profile_dir) / "ml" / LOG_NAME
    diag = {"lines": 0, "unreadable": 0, "by_schema": {}}
    records: list[dict] = []
    if not path.is_file():
        return records, diag
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        diag["lines"] += 1
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError("record is not an object")
        except (ValueError, TypeError):
            diag["unreadable"] += 1
            continue
        tag = str(rec.get("schema", "?"))
        diag["by_schema"][tag] = diag["by_schema"].get(tag, 0) + 1
        records.append(rec)
    return records, diag


def training_pairs(profile_dir: Path | str) -> tuple[list[dict], dict]:
    """([{state, plan, reward, scenario}], diagnostics) — the log's own
    documented pairing rule, executed.

    The rule: record[i]'s plan is rewarded by record[i+1]["run_outcome"] FOR
    THE SAME SCENARIO, because the plan emitted after run i is what run i+1
    was played on. Three things drop a pair, and each is counted rather than
    silently skipped:

    - **no scenario** (pre-v3): the rule cannot be applied to a record that
      does not say what it belongs to. See the schema note.
    - **a run_count gap**: `profile_state.run_count` increments by one per
      processed run of a scenario, so a jump means a record was dropped and
      the two neighbours are not consecutive runs.
    - **no reward**: v1 rows carry `outcome: null` and no `run_outcome` at
      all. Their reward is recoverable by joining the report library on
      (scenario, ts) — deliberately NOT done here, because that join belongs
      to a training pass that has the reports open, and inventing it here
      would put a guess where the log promises a measurement.

    Writing the reader is what found the missing field: the rule had been
    specified for a release without anything executing it.
    """
    records, diag = read_transitions(profile_dir)
    by_scenario: dict[str, list[dict]] = {}
    dropped = {"no_scenario": 0, "no_reward": 0, "run_count_gap": 0}
    for rec in records:
        name = rec.get("scenario")
        if not name:
            dropped["no_scenario"] += 1
            continue
        by_scenario.setdefault(str(name), []).append(rec)

    pairs: list[dict] = []
    for name, rows in by_scenario.items():
        rows.sort(key=lambda r: str(r.get("ts", "")))
        for cur, nxt in zip(rows, rows[1:]):
            reward = nxt.get("run_outcome")
            if not reward:
                dropped["no_reward"] += 1
                continue
            a = (cur.get("profile_state") or {}).get("run_count")
            b = (nxt.get("profile_state") or {}).get("run_count")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if int(b) - int(a) != 1:
                    dropped["run_count_gap"] += 1
                    continue
            pairs.append({
                "scenario": name,
                "state": cur.get("profile_state") or {},
                "plan": cur.get("plan") or {},
                "reward": reward,
            })
    diag["scenarios"] = sorted(by_scenario)
    diag["dropped"] = dropped
    diag["pairs"] = len(pairs)
    return pairs, diag
