"""
Token-optimized log and diff extraction for Auto-Heal CI.

CI pipelines often emit megabytes of stdout. Passing that verbatim to an LLM
is expensive and noisy. This module extracts:

  1. The most relevant failing stack trace / assertion block.
  2. Unified diff hunks that touch files referenced in the failure.
  3. A compact metadata envelope for the AI analyzer.

All public functions are pure (no I/O) so they are easy to unit test offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedFailure:
    """A single extracted failure block from raw CI logs."""

    framework: str
    summary_line: str
    stack_trace: str
    referenced_files: tuple[str, ...] = ()


@dataclass
class FilteredDiagnosticInput:
    """
    Cost-optimized payload handed to the AI analyzer.

    Attributes:
        commit_sha: Git commit under test.
        failing_excerpt: Trimmed log excerpt containing the primary failure.
        relevant_diff: Unified diff limited to files implicated in the failure.
        changed_files: All paths touched by the merge (for blast-radius hints).
        pr_metadata: Human-readable PR context lines (author, title, URL).
        truncated: True when content was clipped to respect max_input_chars.
    """

    commit_sha: str
    failing_excerpt: str
    relevant_diff: str
    changed_files: list[str] = field(default_factory=list)
    pr_metadata: str = ""
    truncated: bool = False


# ---------------------------------------------------------------------------
# Log parsing heuristics
# ---------------------------------------------------------------------------

# Common pytest / unittest failure markers.
_PYTEST_FAILURE_START = re.compile(
    r"^={3,}\s*(FAILURES|ERRORS)\s*={3,}\s*$", re.MULTILINE
)
_PYTEST_FAILURE_BLOCK = re.compile(
    r"^_{3,}\s*.+?\s*_{3,}\s*\n(.*?)(?=^_{3,}|^={3,}|^FAILED|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SHORT_TRACEBACK = re.compile(
    r"(Traceback \(most recent call last\):.*?)(?:\n\S|\Z)",
    re.DOTALL,
)
_ASSERTION_ERROR = re.compile(
    r"(AssertionError:.*?)(?:\n\S|\Z)", re.DOTALL
)
_NPM_TEST_FAIL = re.compile(
    r"(FAIL\s+.*?(?:\n\s+at\s+.+)+)", re.MULTILINE | re.DOTALL
)
_FILE_REF = re.compile(
    r"(?:File \"([^\"]+)\"|([\w./\\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java)):\d+)"
)


def _unique_paths(paths: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in paths:
        normalized = raw.replace("\\", "/").lstrip("./")
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def extract_file_references(text: str) -> tuple[str, ...]:
    """Pull file paths mentioned in a log excerpt."""
    refs: list[str] = []
    for match in _FILE_REF.finditer(text):
        path = match.group(1) or match.group(2)
        if path:
            refs.append(path)
    return _unique_paths(refs)


def extract_failures(raw_log: str) -> list[ExtractedFailure]:
    """
    Parse raw CI stdout and return structured failure blocks.

    Supports pytest-style output, Python tracebacks, and common npm/jest
    failure patterns. Returns an empty list when no failure signature matches.
    """
    if not raw_log.strip():
        return []

    failures: list[ExtractedFailure] = []

    # pytest FAILURES section
    if _PYTEST_FAILURE_START.search(raw_log):
        for block_match in _PYTEST_FAILURE_BLOCK.finditer(raw_log):
            body = block_match.group(1).strip()
            header = block_match.group(0).splitlines()[0]
            refs = extract_file_references(body)
            failures.append(
                ExtractedFailure(
                    framework="pytest",
                    summary_line=header,
                    stack_trace=body,
                    referenced_files=refs,
                )
            )

    # Generic Python traceback (fallback)
    if not failures:
        for tb_match in _SHORT_TRACEBACK.finditer(raw_log):
            body = tb_match.group(1).strip()
            failures.append(
                ExtractedFailure(
                    framework="python",
                    summary_line=body.splitlines()[-1][:200],
                    stack_trace=body,
                    referenced_files=extract_file_references(body),
                )
            )

    # AssertionError without full traceback
    if not failures:
        for assert_match in _ASSERTION_ERROR.finditer(raw_log):
            body = assert_match.group(1).strip()
            failures.append(
                ExtractedFailure(
                    framework="assertion",
                    summary_line=body.splitlines()[0][:200],
                    stack_trace=body,
                    referenced_files=extract_file_references(body),
                )
            )

    # npm / jest style
    if not failures:
        for npm_match in _NPM_TEST_FAIL.finditer(raw_log):
            body = npm_match.group(1).strip()
            failures.append(
                ExtractedFailure(
                    framework="npm",
                    summary_line=body.splitlines()[0][:200],
                    stack_trace=body,
                    referenced_files=extract_file_references(body),
                )
            )

    # Last resort: grab tail of log (often contains the error)
    if not failures:
        tail = raw_log.strip()[-4000:]
        failures.append(
            ExtractedFailure(
                framework="unknown",
                summary_line=tail.splitlines()[-1][:200] if tail else "Unknown failure",
                stack_trace=tail,
                referenced_files=extract_file_references(tail),
            )
        )

    return failures


def select_primary_failure(failures: list[ExtractedFailure]) -> ExtractedFailure:
    """Choose the most informative failure block (first pytest, else first)."""
    if not failures:
        raise ValueError("No failures available to select.")
    for failure in failures:
        if failure.framework == "pytest":
            return failure
    return failures[0]


# ---------------------------------------------------------------------------
# Diff filtering
# ---------------------------------------------------------------------------

_DIFF_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@ .+? @@", re.MULTILINE)


def parse_changed_files(unified_diff: str) -> list[str]:
    """Return all file paths present in a unified diff."""
    files: list[str] = []
    for match in _DIFF_FILE_HEADER.finditer(unified_diff):
        # Prefer the "b/" side (post-change path).
        path = match.group(2).strip()
        if path != "/dev/null":
            files.append(path)
    return files


def filter_diff_by_files(
    unified_diff: str,
    target_files: Iterable[str],
    *,
    context_lines: int = 3,
    max_hunks_per_file: int = 8,
) -> str:
    """
    Return a subset of `unified_diff` containing only hunks for `target_files`.

    When `target_files` is empty, returns the first N hunks across the diff
    (still bounded) so the model receives *some* change context.

    Args:
        unified_diff: Full git diff text.
        target_files: Paths implicated by the failing stack trace.
        context_lines: Unused placeholder for future context expansion.
        max_hunks_per_file: Cap hunks per file to control token usage.
    """
    _ = context_lines  # reserved for future unified-diff context expansion

    if not unified_diff.strip():
        return ""

    targets = {f.replace("\\", "/").lstrip("./") for f in target_files}
    # Also match basename-only hits (logs sometimes omit directory prefix).
    target_basenames = {t.split("/")[-1] for t in targets}

    chunks: list[str] = []
    current_file: Optional[str] = None
    current_chunk_lines: list[str] = []
    collecting = False
    hunks_for_file = 0

    def _file_in_scope(path: str) -> bool:
        normalized = path.replace("\\", "/")
        basename = normalized.split("/")[-1]
        return not targets or normalized in targets or basename in target_basenames

    def _flush() -> None:
        nonlocal hunks_for_file, collecting
        if collecting and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
        current_chunk_lines.clear()
        collecting = False
        hunks_for_file = 0

    for line in unified_diff.splitlines():
        file_match = _DIFF_FILE_HEADER.match(line)
        if file_match:
            _flush()
            current_file = file_match.group(2).strip()
            collecting = _file_in_scope(current_file)
            if collecting:
                current_chunk_lines = [line]
            continue

        if not collecting or current_file is None:
            continue

        if _HUNK_HEADER.match(line):
            hunks_for_file += 1
            if hunks_for_file > max_hunks_per_file:
                continue

        current_chunk_lines.append(line)

    _flush()
    return "\n".join(chunks)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.7)
    tail = limit - head - 40
    return (
        text[:head]
        + "\n\n... [truncated for token budget] ...\n\n"
        + text[-tail:],
        True,
    )


def build_filtered_input(
    *,
    raw_log: str,
    unified_diff: str,
    commit_sha: str,
    max_chars: int = 24_000,
    pr_author: Optional[str] = None,
    pr_title: Optional[str] = None,
    pr_url: Optional[str] = None,
) -> FilteredDiagnosticInput:
    """
    Orchestrate log + diff extraction into a single bounded payload.

    This is the primary entry point used by the recovery engine before calling
    the AI analyzer.

    Raises:
        ValueError: When both log and diff are empty (nothing to analyze).
    """
    if not raw_log.strip() and not unified_diff.strip():
        raise ValueError("Both raw_log and unified_diff are empty; cannot diagnose.")

    failures = extract_failures(raw_log)
    primary = select_primary_failure(failures) if failures else None

    referenced = primary.referenced_files if primary else ()
    relevant_diff = filter_diff_by_files(unified_diff, referenced)
    if not relevant_diff.strip() and unified_diff.strip():
        # No overlap — include a small prefix of the full diff instead.
        relevant_diff, _ = _truncate(unified_diff, max_chars // 3)

    changed_files = parse_changed_files(unified_diff)

    excerpt = primary.stack_trace if primary else raw_log.strip()[-4000:]
    excerpt, excerpt_truncated = _truncate(excerpt, max_chars // 2)
    relevant_diff, diff_truncated = _truncate(relevant_diff, max_chars // 2)

    meta_lines = [
        f"commit: {commit_sha}",
    ]
    if pr_author:
        meta_lines.append(f"pr_author: {pr_author}")
    if pr_title:
        meta_lines.append(f"pr_title: {pr_title}")
    if pr_url:
        meta_lines.append(f"pr_url: {pr_url}")
    if primary:
        meta_lines.append(f"failure_framework: {primary.framework}")
        meta_lines.append(f"failure_summary: {primary.summary_line}")

    combined_truncated = excerpt_truncated or diff_truncated
    total = "\n".join(meta_lines) + excerpt + relevant_diff
    if len(total) > max_chars:
        # Final safety clip on the excerpt side.
        budget = max_chars - len(relevant_diff) - len("\n".join(meta_lines)) - 100
        excerpt, extra = _truncate(excerpt, max(500, budget))
        combined_truncated = combined_truncated or extra

    return FilteredDiagnosticInput(
        commit_sha=commit_sha,
        failing_excerpt=excerpt,
        relevant_diff=relevant_diff,
        changed_files=changed_files,
        pr_metadata="\n".join(meta_lines),
        truncated=combined_truncated,
    )
