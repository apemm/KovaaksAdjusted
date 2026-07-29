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
    "schema": "shadow-v1",
    "ts": "ISO timestamp of the processed run",
    "profile_state": "dict of PROFILE_STATE_KEYS captured BEFORE observe()",
    "plan": "the emitted AdaptationPlan as a dict (target_scale, movement, ...)",
    "suggestion": "ShadowSuggestion as a dict, or null while untrained",
    "outcome": "next-run outcome when known: {accuracy, score, kps} or null",
}

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
