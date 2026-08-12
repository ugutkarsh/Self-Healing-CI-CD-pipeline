"""
Configuration loader for Auto-Heal CI.

Reads environment variables and GitHub Actions context. Secrets are never
logged or persisted — only referenced at runtime for API calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AIProvider(str, Enum):
    """Supported LLM providers for diagnostic analysis."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class GitHubConfig:
    """GitHub repository and authentication settings."""

    token: str
    repository: str  # "owner/repo"
    base_branch: str = "main"
    api_url: str = field(
        default_factory=lambda: os.getenv("GITHUB_API_URL", "https://api.github.com")
    )


@dataclass(frozen=True)
class AIConfig:
    """LLM provider settings with cost/latency guardrails."""

    provider: AIProvider
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    max_retries: int = 2
    # Hard cap on tokens sent to the model (pre-filtered upstream).
    max_input_chars: int = 24_000


@dataclass(frozen=True)
class RunContext:
    """Metadata captured from the failing CI run."""

    commit_sha: str
    pr_number: Optional[int]
    pr_author: Optional[str]
    pr_title: Optional[str]
    pr_url: Optional[str]
    workflow_run_id: Optional[str]
    actor: Optional[str]


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    github: GitHubConfig
    ai: AIConfig
    run: RunContext
    revert_branch_prefix: str = "revert/pr-"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is missing or empty."
        )
    return value


def _optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, "").strip()
    return value or default


# Substrings that indicate a README placeholder rather than a real secret.
_PLACEHOLDER_MARKERS = (
    "your_key",
    "your_token",
    "your-api-key",
    "changeme",
    "placeholder",
    "replace_me",
    "xxx",
)


def _validate_secret(name: str, value: str, *, allow_mock: bool = False) -> None:
    """
    Reject obvious placeholder values before calling external APIs.

    Raises:
        ValueError: When the value looks like documentation filler text.
    """
    if allow_mock:
        return

    normalized = value.strip().lower()
    if any(marker in normalized for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(
            f"{name} looks like a README placeholder ({value!r}). "
            "Export a real secret from your provider dashboard, or run with "
            "`--mock-ai` to exercise the pipeline offline."
        )


def load_config(
    *,
    require_ai_key: bool = True,
    require_github: bool = True,
) -> AppConfig:
    """
    Build configuration from the current process environment.

    Raises:
        EnvironmentError: When mandatory variables are absent.
        ValueError: When enum values or numeric settings are invalid.
    """
    provider_raw = _optional_env("AUTO_HEAL_AI_PROVIDER", "openai") or "openai"
    try:
        provider = AIProvider(provider_raw.lower())
    except ValueError as exc:
        raise ValueError(
            f"Invalid AUTO_HEAL_AI_PROVIDER '{provider_raw}'. "
            f"Choose one of: {[p.value for p in AIProvider]}"
        ) from exc

    default_model = "gpt-4o-mini" if provider == AIProvider.OPENAI else "claude-3-5-haiku-20241022"

    ai_key_var = (
        "OPENAI_API_KEY"
        if provider == AIProvider.OPENAI
        else "ANTHROPIC_API_KEY"
    )

    timeout_raw = _optional_env("AUTO_HEAL_AI_TIMEOUT_SECONDS", "60") or "60"
    retries_raw = _optional_env("AUTO_HEAL_AI_MAX_RETRIES", "2") or "2"
    max_chars_raw = _optional_env("AUTO_HEAL_MAX_INPUT_CHARS", "24000") or "24000"

    pr_number_raw = _optional_env("AUTO_HEAL_PR_NUMBER") or _optional_env(
        "GITHUB_EVENT_PULL_REQUEST_NUMBER"
    )
    pr_number: Optional[int] = int(pr_number_raw) if pr_number_raw else None

    github_token = (
        _require_env("GITHUB_TOKEN")
        if require_github
        else (_optional_env("GITHUB_TOKEN", "mock-token-local") or "mock-token-local")
    )
    github_repo = (
        _require_env("GITHUB_REPOSITORY")
        if require_github
        else (_optional_env("GITHUB_REPOSITORY", "local/dev") or "local/dev")
    )
    if require_github:
        _validate_secret("GITHUB_TOKEN", github_token)

    github = GitHubConfig(
        token=github_token,
        repository=github_repo,
        base_branch=_optional_env("AUTO_HEAL_BASE_BRANCH", "main") or "main",
    )

    if require_ai_key:
        api_key = _require_env(ai_key_var)
        _validate_secret(ai_key_var, api_key)
    else:
        api_key = "mock-local-key"

    ai = AIConfig(
        provider=provider,
        api_key=api_key,
        model=_optional_env("AUTO_HEAL_AI_MODEL", default_model) or default_model,
        timeout_seconds=float(timeout_raw),
        max_retries=int(retries_raw),
        max_input_chars=int(max_chars_raw),
    )

    run = RunContext(
        commit_sha=_optional_env("GITHUB_SHA", "unknown") or "unknown",
        pr_number=pr_number,
        pr_author=_optional_env("AUTO_HEAL_PR_AUTHOR"),
        pr_title=_optional_env("AUTO_HEAL_PR_TITLE"),
        pr_url=_optional_env("AUTO_HEAL_PR_URL"),
        workflow_run_id=_optional_env("GITHUB_RUN_ID"),
        actor=_optional_env("GITHUB_ACTOR"),
    )

    return AppConfig(github=github, ai=ai, run=run)
