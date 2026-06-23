"""Score-extraction tests — carries forward the fork's hard-won protections:
the SCORE= sentinel strictness and the #73 draft-result.json guard."""
import json

import pytest

from continual_auto_research.core import scoring


# ── SCORE= sentinel ──────────────────────────────────────────────────────────

def test_score_sentinel_last_wins():
    assert scoring.extract_score("SCORE=1\nnoise\nSCORE=2.5") == 2.5


def test_score_sentinel_is_strict():
    # The SCORE= sentinel regex itself is line-anchored + case-sensitive: neither
    # incidental `best_score=` nor an inline `... SCORE=999 ...` is the sentinel.
    assert scoring._SCORE_SENTINEL.findall("best_score=0.9") == []
    assert scoring._SCORE_SENTINEL.findall("... inline SCORE=999 ...") == []
    # A whole-line sentinel DOES match.
    assert scoring._SCORE_SENTINEL.findall("SCORE=3.5") == ["3.5"]


def test_extract_score_prose_fallback_is_lenient():
    # extract_score() is allowed to fall through to the lenient prose parser when
    # the strict sentinel doesn't fire — that's by design (it's the last resort).
    assert scoring.extract_score("the final score=0.9 today") == 0.9
    # but with no score-ish token at all, nothing is returned
    assert scoring.extract_score("best_metric was great") is None


def test_score_falls_back_to_json_and_prose():
    assert scoring.extract_score('```json\n{"score": 0.77}\n```') == 0.77
    assert scoring.extract_score("final score: 0.83") == 0.83
    assert scoring.extract_score("nothing numeric here") is None


# ── result.json draft guard (#73) ────────────────────────────────────────────

@pytest.mark.parametrize("status", ["draft", "proposal", "skipped", "error", "DRAFT"])
def test_draft_result_json_score_ignored(tmp_path, status):
    (tmp_path / "result.json").write_text(
        json.dumps({"score": 0.0, "direction": "min", "status": status}))
    out = scoring.read_result_json(str(tmp_path))
    assert out is None or "score" not in out, f"status={status} must not carry a score"


@pytest.mark.parametrize("status", ["done", "ok", "succeeded", "completed", "finished"])
def test_executed_status_keeps_score(tmp_path, status):
    (tmp_path / "result.json").write_text(
        json.dumps({"score": 1180.0, "direction": "min", "status": status}))
    out = scoring.read_result_json(str(tmp_path))
    assert out and out["score"] == 1180.0


def test_missing_status_keeps_score_legacy_shape(tmp_path):
    (tmp_path / "result.json").write_text('{"score": 203.7, "direction": "min"}')
    out = scoring.read_result_json(str(tmp_path))
    assert out and out["score"] == 203.7


def test_resolve_prefers_result_json_over_prose(tmp_path):
    (tmp_path / "result.json").write_text('{"score": 0.5, "direction": "min"}')
    # prose says something different; result.json must win
    assert scoring.resolve_score(str(tmp_path), "SCORE=999") == 0.5


def test_resolve_draft_falls_through_to_none(tmp_path):
    # the exact iter-1 poisoning repro: a draft 0.0 must NOT be the score
    (tmp_path / "result.json").write_text('{"score": 0.0, "status": "draft"}')
    assert scoring.resolve_score(str(tmp_path), "DRAFT — not executed") is None


def test_resolve_falls_back_to_prose_when_no_file(tmp_path):
    assert scoring.resolve_score(str(tmp_path), "SCORE=42") == 42.0
