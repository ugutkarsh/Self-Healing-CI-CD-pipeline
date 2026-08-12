"""Unit tests for token-optimized log/diff extraction."""

from __future__ import annotations

import pytest

from src.diff_parser import (
    build_filtered_input,
    extract_failures,
    filter_diff_by_files,
    parse_changed_files,
    select_primary_failure,
)

SAMPLE_PYTEST_LOG = """
============================= test session starts ==============================
collected 2 items

tests/test_sample_app.py::test_addition_passes PASSED
tests/test_sample_app.py::test_division_breaks_intentionally FAILED

=================================== FAILURES ===================================
____________________ test_division_breaks_intentionally ______________________

    def test_division_breaks_intentionally() -> None:
>       assert divide(10, 2) == 4
E       assert 5.0 == 4

tests/test_sample_app.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_sample_app.py::test_division_breaks_intentionally
"""

SAMPLE_DIFF = """\
diff --git a/tests/sample_app.py b/tests/sample_app.py
index abc123..def456 100644
--- a/tests/sample_app.py
+++ b/tests/sample_app.py
@@ -8,4 +8,4 @@ def add(a: int, b: int) -> int:
 
 def divide(numerator: int, denominator: int) -> float:
-    return numerator // denominator
+    return numerator / denominator
diff --git a/README.md b/README.md
index 111..222 100644
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""


def test_extract_failures_pytest() -> None:
    failures = extract_failures(SAMPLE_PYTEST_LOG)
    assert len(failures) >= 1
    primary = select_primary_failure(failures)
    assert "AssertionError" in primary.stack_trace or "assert" in primary.stack_trace


def test_parse_changed_files() -> None:
    files = parse_changed_files(SAMPLE_DIFF)
    assert "tests/sample_app.py" in files
    assert "README.md" in files


def test_filter_diff_by_files() -> None:
    filtered = filter_diff_by_files(SAMPLE_DIFF, ["tests/sample_app.py"])
    assert "sample_app.py" in filtered
    assert "README.md" not in filtered


def test_build_filtered_input_respects_budget() -> None:
    payload = build_filtered_input(
        raw_log=SAMPLE_PYTEST_LOG,
        unified_diff=SAMPLE_DIFF,
        commit_sha="abc1234",
        max_chars=5000,
        pr_author="octocat",
    )
    assert payload.commit_sha == "abc1234"
    assert "assert" in payload.failing_excerpt.lower()
    assert "sample_app.py" in payload.relevant_diff
    assert "octocat" in payload.pr_metadata


def test_build_filtered_input_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        build_filtered_input(raw_log="", unified_diff="", commit_sha="x")


def test_final_clip_truncated_flag_respects_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: final safety clip must use `_truncate`'s bool, not `True or extra`."""
    calls: list[tuple[str, int]] = []

    def _recording_truncate(text: str, limit: int) -> tuple[str, bool]:
        calls.append((text, limit))
        # First two calls: initial excerpt/diff bounds — no truncation.
        if len(calls) <= 2:
            return text, False
        # Final safety clip — still no truncation.
        return text, False

    monkeypatch.setattr("src.diff_parser._truncate", _recording_truncate)

    payload = build_filtered_input(
        raw_log="failure",
        unified_diff="diff-content",
        commit_sha="c",
        max_chars=20,
    )

    assert len(calls) >= 3
    assert payload.truncated is False
