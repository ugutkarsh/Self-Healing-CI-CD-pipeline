"""
GitHub API integration for Auto-Heal CI.

Handles revert branches, Rollback PRs, diagnostic Issues, and cross-linking
between artifacts using PyGithub with explicit error handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from github import Auth, Github, GithubException, InputGitTreeElement
from github.GithubException import UnknownObjectException
from github.PullRequest import PullRequest
from github.Repository import Repository

from src.ai_analyzer import DiagnosticReport, report_to_markdown
from src.config import GitHubConfig, RunContext

logger = logging.getLogger(__name__)


class GitHubServiceError(Exception):
    """Raised when a GitHub API operation fails."""


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of creating an automated rollback PR."""

    branch_name: str
    pull_request: PullRequest
    revert_commit_sha: str


@dataclass(frozen=True)
class DiagnosticIssueResult:
    """Outcome of posting the AI diagnostic report."""

    issue_number: int
    issue_url: str


class GitHubService:
    """Thin wrapper around PyGithub for Auto-Heal CI workflows."""

    def __init__(self, config: GitHubConfig) -> None:
        self._config = config
        self._gh = Github(
            auth=Auth.Token(config.token),
            base_url=config.api_url,
            per_page=50,
        )
        self._repo: Repository = self._gh.get_repo(config.repository)

    @property
    def repository(self) -> Repository:
        return self._repo

    def _upsert_branch_ref(self, branch_name: str, sha: str) -> None:
        """
        Create or update a branch ref pointing at `sha`.

        PyGithub's ``create_git_ref`` does not accept ``force``; if the branch
        already exists from a prior Auto-Heal run, update it instead.
        """
        ref_path = f"refs/heads/{branch_name}"
        try:
            ref = self._repo.create_git_ref(ref=ref_path, sha=sha)
            logger.info("Created revert branch %s @ %s", branch_name, ref.object.sha)
        except GithubException as exc:
            if exc.status != 422:
                raise
            existing = self._repo.get_git_ref(f"heads/{branch_name}")
            existing.edit(sha=sha, force=True)
            logger.info("Updated existing revert branch %s @ %s", branch_name, sha)

    def create_revert_branch_and_pr(
        self,
        *,
        merge_commit_sha: str,
        pr_number: Optional[int],
        pr_title: Optional[str],
        failure_summary: str,
    ) -> RollbackResult:
        """
        Create an isolated revert branch and open a Rollback PR targeting main.

        The branch name follows `revert/pr-<PR_NUMBER>` when a PR number is known,
        otherwise `revert/commit-<short_sha>`.
        """
        short_sha = merge_commit_sha[:7]
        if pr_number is not None:
            branch_name = f"revert/pr-{pr_number}"
            title = f"[Auto-Heal] Rollback PR #{pr_number}: {pr_title or short_sha}"
        else:
            branch_name = f"revert/commit-{short_sha}"
            title = f"[Auto-Heal] Rollback merge {short_sha}"

        try:
            merge_commit = self._repo.get_git_commit(merge_commit_sha)
        except UnknownObjectException as exc:
            raise GitHubServiceError(
                f"Merge commit {merge_commit_sha} not found."
            ) from exc
        except GithubException as exc:
            raise GitHubServiceError(f"Failed to fetch merge commit: {exc}") from exc

        if merge_commit.parents and len(merge_commit.parents) >= 1:
            parent_sha = merge_commit.parents[0].sha
        else:
            raise GitHubServiceError(
                f"Commit {merge_commit_sha} has no parent; cannot revert."
            )

        try:
            parent_commit = self._repo.get_git_commit(parent_sha)
            base_tree = self._repo.get_git_tree(parent_sha, recursive=True)
        except GithubException as exc:
            raise GitHubServiceError(f"Failed to load parent tree: {exc}") from exc

        # Build a revert tree by restoring blobs from the parent at each changed path.
        merge_tree = self._repo.get_git_tree(merge_commit.tree.sha, recursive=True)
        parent_paths = {item.path: item for item in base_tree.tree if item.path}
        merge_paths = {item.path: item for item in merge_tree.tree if item.path}

        changed_paths = set(merge_paths) ^ set(parent_paths)
        changed_paths.update(
            p
            for p in merge_paths
            if p in parent_paths
            and merge_paths[p].sha != parent_paths[p].sha
        )

        elements: list[InputGitTreeElement] = []
        for path in sorted(changed_paths):
            if path in parent_paths:
                blob = parent_paths[path]
                elements.append(
                    InputGitTreeElement(
                        path=path,
                        mode=blob.mode,
                        type=blob.type,
                        sha=blob.sha,
                    )
                )
            else:
                # File added in merge — delete by setting sha to None.
                elements.append(
                    InputGitTreeElement(path=path, mode="100644", type="blob", sha=None)
                )

        try:
            new_tree = self._repo.create_git_tree(elements, base_tree)
            revert_commit = self._repo.create_git_commit(
                message=f"Revert merge {merge_commit_sha}\n\nAuto-Heal CI rollback.",
                tree=new_tree,
                parents=[parent_commit],
            )

            self._upsert_branch_ref(branch_name, revert_commit.sha)
        except GithubException as exc:
            raise GitHubServiceError(f"Failed to create revert branch: {exc}") from exc

        body = (
            "## Automated Rollback\n\n"
            "Post-merge CI failed on `main`. Auto-Heal CI opened this PR to safely "
            "isolate the breaking change by reverting the merge commit.\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Merge commit | `{merge_commit_sha}` |\n"
            f"| Original PR | #{pr_number if pr_number else 'n/a'} |\n"
            f"| Failure summary | {failure_summary[:500]} |\n\n"
            "### Next steps\n\n"
            "1. Review the linked diagnostic Issue for root-cause analysis.\n"
            "2. Merge this PR to restore green CI on `main`.\n"
            "3. Fix forward in a follow-up PR using the suggested patch.\n"
        )

        try:
            pr = self._repo.create_pull(
                title=title,
                body=body,
                head=branch_name,
                base=self._config.base_branch,
            )
        except GithubException as exc:
            raise GitHubServiceError(f"Failed to open rollback PR: {exc}") from exc

        return RollbackResult(
            branch_name=branch_name,
            pull_request=pr,
            revert_commit_sha=revert_commit.sha,
        )

    def create_diagnostic_issue(
        self,
        *,
        report: DiagnosticReport,
        run: RunContext,
        rollback_pr: PullRequest,
        assignee: Optional[str] = None,
    ) -> DiagnosticIssueResult:
        """
        Post structured AI diagnostics as a GitHub Issue and cross-link the Rollback PR.
        """
        title = (
            f"[Auto-Heal] CI failure after merge "
            f"(PR #{run.pr_number})" if run.pr_number else "[Auto-Heal] CI failure on main"
        )

        labels = ["auto-heal", "ci-failure", "needs-triage"]
        body_parts = [
            "## Auto-Heal CI Diagnostic Report",
            "",
            report_to_markdown(report),
            "",
            "---",
            "",
            "### Run context",
            "",
            f"- **Commit:** `{run.commit_sha}`",
            f"- **Workflow run:** {run.workflow_run_id or 'n/a'}",
            f"- **Rollback PR:** #{rollback_pr.number} — {rollback_pr.html_url}",
        ]
        if run.pr_url:
            body_parts.append(f"- **Original PR:** {run.pr_url}")
        if run.pr_author:
            body_parts.append(f"- **Original author:** @{run.pr_author}")

        body = "\n".join(body_parts)

        try:
            issue = self._repo.create_issue(
                title=title,
                body=body,
                labels=labels,
                assignee=assignee or run.pr_author,
            )
        except GithubException as exc:
            raise GitHubServiceError(f"Failed to create diagnostic issue: {exc}") from exc

        # Cross-link: comment on Rollback PR pointing to the Issue.
        try:
            rollback_pr.create_issue_comment(
                f"📋 **AI diagnostic report:** {issue.html_url}\n\n"
                f"cc @{run.pr_author}" if run.pr_author else ""
            )
        except GithubException as exc:
            logger.warning("Could not comment on rollback PR: %s", exc)

        return DiagnosticIssueResult(
            issue_number=issue.number,
            issue_url=issue.html_url,
        )

    def close(self) -> None:
        """Release underlying HTTP connections."""
        self._gh.close()
