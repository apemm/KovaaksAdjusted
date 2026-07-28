"""Structural validation of the cited knowledge base (kovadapt/analysis/kb.py)."""

import ast

import pytest

from kovadapt.analysis import kb

CONFIDENCE_LEVELS = {"high", "medium", "low"}


def _primary_level(confidence: str) -> str:
    # Compound labels like "high (pattern), medium (sens attribution)" keep the researched
    # nuance verbatim; the leading token is the primary confidence level.
    return confidence.split()[0].strip("(),;").lower()


def test_sections_non_empty():
    assert isinstance(kb.PRINCIPLES, dict) and kb.PRINCIPLES
    assert isinstance(kb.DIAGNOSTICS, dict) and kb.DIAGNOSTICS
    assert isinstance(kb.BENCHMARKS, list) and kb.BENCHMARKS
    assert isinstance(kb.ROUTINES, list) and kb.ROUTINES
    assert isinstance(kb.GAPS, tuple) and kb.GAPS


def test_ids_unique_across_tables():
    overlap = set(kb.PRINCIPLES) & set(kb.DIAGNOSTICS)
    assert not overlap, f"ids in both tables: {overlap}"


def test_principles_shape():
    for pid, entry in kb.PRINCIPLES.items():
        assert entry["topic"].strip(), pid
        assert entry["text"].strip(), pid
        assert _primary_level(entry["confidence"]) in CONFIDENCE_LEVELS, pid
        assert isinstance(entry["sources"], tuple) and entry["sources"], pid
        assert all(isinstance(s, str) and s.strip() for s in entry["sources"]), pid


def test_diagnostics_shape():
    for did, entry in kb.DIAGNOSTICS.items():
        assert entry["signal"].strip(), did
        assert entry["condition"].strip(), did
        assert entry["interpretation"].strip(), did
        assert entry["prescription"].strip(), did
        assert _primary_level(entry["confidence"]) in CONFIDENCE_LEVELS, did
        assert isinstance(entry["sources"], tuple) and entry["sources"], did
        assert all(isinstance(s, str) and s.strip() for s in entry["sources"]), did


def test_benchmarks_shape():
    for entry in kb.BENCHMARKS:
        for key in ("system", "version", "categories", "scenarios", "thresholds",
                    "accuracy_caveat"):
            assert key in entry, (entry.get("system"), key)
        assert entry["system"].strip() and entry["version"].strip()
        assert entry["accuracy_caveat"].strip()


def test_routines_shape():
    ids = [r["id"] for r in kb.ROUTINES]
    assert len(ids) == len(set(ids))
    for r in kb.ROUTINES:
        assert r["name"].strip() and r["structure"].strip(), r["id"]


def test_gaps_shape():
    for gap in kb.GAPS:
        assert isinstance(gap, str) and gap.strip()
        # research-workflow tooling noise is excluded from the KB by policy
        assert "ORCHESTRATION FAULT" not in gap


def test_accessors():
    pid = next(iter(kb.PRINCIPLES))
    did = next(iter(kb.DIAGNOSTICS))
    assert kb.principle(pid) is kb.PRINCIPLES[pid]
    assert kb.diagnostic(did) is kb.DIAGNOSTICS[did]
    assert kb.sources_for(pid) == kb.PRINCIPLES[pid]["sources"]
    assert kb.sources_for(did) == kb.DIAGNOSTICS[did]["sources"]
    with pytest.raises(KeyError):
        kb.principle("no-such-id")
    with pytest.raises(KeyError):
        kb.diagnostic("no-such-id")
    with pytest.raises(KeyError):
        kb.sources_for("no-such-id")


def test_module_is_pure_data():
    # The analysis package is a leaf and the KB must not pull numpy or anything else:
    # the module source contains no import statements at all.
    with open(kb.__file__, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not imports
    # no UTF-8 mojibake survived the transcription
    for pattern in ("â€", "�"):
        assert pattern not in source
