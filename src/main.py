#!/usr/bin/env python3
"""
Auto-Heal CI recovery entrypoint.

Invoked by the GitHub Actions workflow when the post-merge test suite fails.
Orchestrates diff filtering, AI diagnosis, rollback PR creation, and Issue reporting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ai_analyzer import (
    AIAnalyzerError,
    DiagnosticEngine,
    build_mock_report,
    report_to_json,
)
from src.config import load_config
from src.diff_parser import build_filtered_input
from src.github_service import GitHubService, GitHubServiceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("auto_heal")


def _read_file(path: Path) -> str:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def run_recovery(
    *,
    log_path: Path,
    diff_path: Path,
    dry_run: bool = False,
    mock_ai: bool = False,
) -> int:
    """
    Execute the full Auto-Heal recovery pipeline.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        config = load_config(
            require_ai_key=not mock_ai,
            require_github=not (dry_run and mock_ai),
        )
    except (EnvironmentError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    raw_log = _read_file(log_path)
    unified_diff = _read_file(diff_path)

    try:
        filtered = build_filtered_input(
            raw_log=raw_log,
            unified_diff=unified_diff,
            commit_sha=config.run.commit_sha,
            max_chars=config.ai.max_input_chars,
            pr_author=config.run.pr_author,
            pr_title=config.run.pr_title,
            pr_url=config.run.pr_url,
        )
    except ValueError as exc:
        logger.error("Input filtering failed: %s", exc)
        return 1

    failure_summary = filtered.failing_excerpt.splitlines()[0][:200]

    if mock_ai:
        logger.info("Using offline mock AI report (--mock-ai).")
        report = build_mock_report(filtered)
    else:
        try:
            engine = DiagnosticEngine.from_config(config.ai)
            report = engine.analyze(filtered)
        except AIAnalyzerError as exc:
            logger.error("AI diagnosis failed: %s", exc)
            return 1

    logger.info("Diagnostic report:\n%s", report_to_json(report))

    if dry_run:
        logger.info("Dry run — skipping GitHub rollback/issue creation.")
        return 0

    gh = GitHubService(config.github)
    try:
        rollback = gh.create_revert_branch_and_pr(
            merge_commit_sha=config.run.commit_sha,
            pr_number=config.run.pr_number,
            pr_title=config.run.pr_title,
            failure_summary=failure_summary,
        )
        logger.info(
            "Opened rollback PR #%s: %s",
            rollback.pull_request.number,
            rollback.pull_request.html_url,
        )

        issue = gh.create_diagnostic_issue(
            report=report,
            run=config.run,
            rollback_pr=rollback.pull_request,
        )
        logger.info("Created diagnostic issue: %s", issue.issue_url)
    except GitHubServiceError as exc:
        logger.error("GitHub operation failed: %s", exc)
        return 1
    finally:
        gh.close()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Heal CI recovery engine")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("test-output.log"),
        help="Path to captured CI stdout/stderr",
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        default=Path("merge.diff"),
        help="Path to unified diff of the merge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run AI analysis only; skip GitHub API mutations",
    )
    parser.add_argument(
        "--mock-ai",
        action="store_true",
        help="Skip LLM API calls; emit a local mock diagnostic report",
    )
    args = parser.parse_args()
    sys.exit(
        run_recovery(
            log_path=args.log_file,
            diff_path=args.diff_file,
            dry_run=args.dry_run,
            mock_ai=args.mock_ai,
        )
    )


if __name__ == "__main__":
    main()
