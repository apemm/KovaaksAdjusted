"""Neural analysis over recorded mouse telemetry (optional ``[ml]`` extra).

The workstream so far:

- ``dataset``  — flick dataset built from the trace library under
  ``<profile_dir>/traces/`` (fixed-length normalized velocity curves).
- ``model``    — ``FlickEncoder``: 1D-CNN -> GRU -> 64-dim embedding with
  regression heads (overshoot, corrections, log-duration), ~1.4M params.
- ``train``    — seeded training loop; checkpoints to
  ``<profile_dir>/ml/flick_encoder.pt`` plus a metadata JSON.
- ``infer``    — checkpoint loading + per-flick scoring, all of it
  degrading to ``None`` when torch or the checkpoint is missing.
- ``shadow``   — scaffold for the future difficulty shadow policy: the
  watcher logs one transition per run to its JSONL schema, but no policy
  is trained and ``propose()`` never influences a plan.

Import contract (ARCHITECTURE.md): the core install stays numpy-only. This
package imports cleanly on any install — ``ML_AVAILABLE`` reports whether
torch is importable, and every entry point guards its torch usage so
callers probe and degrade instead of crashing. ``kovadapt/ml`` is the only
place allowed to import torch; ``model.py`` is the one submodule that
needs torch at import time and is therefore only ever imported lazily,
behind an ``ML_AVAILABLE``/try-except check.
"""

from __future__ import annotations

try:  # `except Exception`, not ImportError: a broken torch install can raise
    import torch  # noqa: F401  # OSError from DLL loading on Windows.

    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

__all__ = ["ML_AVAILABLE"]
