"""
Structured AI diagnostic engine for Auto-Heal CI.

Calls OpenAI or Anthropic with a Pydantic-validated JSON schema so downstream
reporting (GitHub Issues, Rollback PR bodies) receives predictable fields.

Design goals:
  - Cost control: consumes pre-filtered input from diff_parser.py only.
  - Reliability: explicit timeouts, retries with backoff, typed errors.
  - Provider parity: single public API regardless of backend.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import AIConfig, AIProvider
from src.diff_parser import FilteredDiagnosticInput

logger = logging.getLogger(__name__)

# Retryable provider exception type names (checked without importing SDKs at load time).
_RETRYABLE_PROVIDER_ERRORS = frozenset(
    {"APITimeoutError", "APIConnectionError", "RateLimitError"}
)


# ---------------------------------------------------------------------------
# Structured output schema (enforced via Pydantic + provider JSON mode)
# ---------------------------------------------------------------------------


class FailingFile(BaseModel):
    """A source file implicated in the CI failure."""

    file_path: str = Field(..., description="Repository-relative path.")
    line_numbers: list[int] = Field(
        default_factory=list,
        description="1-based line numbers referenced in the failure.",
    )
    explanation: str = Field(
        ..., description="Why this file/lines likely caused the failure."
    )


class DiagnosticReport(BaseModel):
    """
    Canonical AI diagnostic payload.

    Matches the enterprise reporting schema required by Auto-Heal CI.
    """

    root_cause_summary: str = Field(
        ..., description="One-paragraph plain-language root cause."
    )
    failing_files: list[FailingFile] = Field(default_factory=list)
    blast_radius_assessment: str = Field(
        ...,
        description="Impact tier, e.g. 'Low', 'Medium', or 'High', with brief rationale.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the diagnosis (0.0–1.0).",
    )
    suggested_fix_patch: str = Field(
        ...,
        description="Markdown fenced code block containing a suggested fix.",
    )

    @field_validator("suggested_fix_patch")
    @classmethod
    def _ensure_markdown_fence(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith("```"):
            return f"```diff\n{stripped}\n```"
        return stripped


class AIAnalyzerError(Exception):
    """Base exception for diagnostic engine failures."""


class AIAnalyzerTimeout(AIAnalyzerError):
    """Raised when the provider exceeds the configured timeout."""


class AIAnalyzerValidationError(AIAnalyzerError):
    """Raised when the model returns JSON that fails Pydantic validation."""


class AIAnalyzerProviderError(AIAnalyzerError):
    """Raised for non-retryable provider/API failures."""


# JSON Schema exported for providers that accept explicit schemas.
DIAGNOSTIC_JSON_SCHEMA: dict = DiagnosticReport.model_json_schema()


SYSTEM_PROMPT = """You are a principal engineer performing post-merge CI failure analysis.

You receive:
  - A trimmed failing test/log excerpt (NOT the full CI log).
  - A filtered git diff showing only relevant changed hunks.
  - Metadata about the merge (commit, PR author, title).

Your task:
  1. Identify the most likely root cause of the test failure.
  2. List implicated files with line numbers when inferable.
  3. Assess blast radius (Low / Medium / High) with a short rationale.
  4. Provide a confidence score between 0.0 and 1.0.
  5. Suggest a minimal fix as a Markdown code block (prefer ```diff fences).

Rules:
  - Base conclusions ONLY on the provided excerpt and diff.
  - Do not invent files not present in the diff or stack trace.
  - Keep root_cause_summary under 120 words.
  - If evidence is insufficient, lower confidence_score and say so in root_cause_summary.
"""


def _build_user_prompt(payload: FilteredDiagnosticInput) -> str:
    """Serialize filtered input into a compact, labeled prompt."""
    sections = [
        "## Run metadata",
        payload.pr_metadata or f"commit: {payload.commit_sha}",
        "",
        "## Failing excerpt",
        payload.failing_excerpt,
        "",
        "## Relevant diff",
        payload.relevant_diff or "(no diff hunks matched — infer from excerpt only)",
        "",
        "## Changed files (full merge)",
        ", ".join(payload.changed_files) if payload.changed_files else "(unknown)",
    ]
    if payload.truncated:
        sections.extend(["", "(Note: input was truncated for token budget.)"])
    return "\n".join(sections)


def build_mock_report(payload: FilteredDiagnosticInput) -> DiagnosticReport:
    """
    Offline diagnostic report for local `--mock-ai` runs.

    Derives a plausible report from the filtered excerpt/diff without calling
    an external LLM — useful when API keys are unavailable.
    """
    primary_file = payload.changed_files[0] if payload.changed_files else "unknown"
    test_file = next(
        (path for path in payload.changed_files if path.startswith("tests/")),
        "tests/test_sample_app.py",
    )

    return DiagnosticReport(
        root_cause_summary=(
            "The failing assertion no longer matches the updated implementation. "
            "The merge changed runtime behavior while the test still expects the "
            "previous result."
        ),
        failing_files=[
            FailingFile(
                file_path=test_file,
                line_numbers=[12],
                explanation="Assertion compares stale expected value against new behavior.",
            ),
            FailingFile(
                file_path=primary_file,
                line_numbers=[],
                explanation="Implementation change introduced the behavioral mismatch.",
            ),
        ],
        blast_radius_assessment="Low — isolated to the sample test suite.",
        confidence_score=0.75,
        suggested_fix_patch=(
            "```diff\n"
            "--- a/tests/test_sample_app.py\n"
            "+++ b/tests/test_sample_app.py\n"
            "@@ -10,4 +10,4 @@ def test_division_breaks_intentionally() -> None:\n"
            "-    assert divide(10, 2) == 4  # intentional failure (expected 5.0)\n"
            "+    assert divide(10, 2) == 5.0\n"
            "```"
        ),
    )


class _ProviderClient(Protocol):
    def analyze(self, user_prompt: str) -> DiagnosticReport: ...


def _missing_sdk_error(provider: str, package: str) -> AIAnalyzerProviderError:
    return AIAnalyzerProviderError(
        f"{provider} SDK is not installed. Install dependencies with: "
        f"python -m pip install -r requirements.txt  "
        f"(missing package: {package})"
    )


class OpenAIAnalyzer:
    """OpenAI Chat Completions with JSON schema response format."""

    def __init__(self, config: AIConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _missing_sdk_error("OpenAI", "openai") from exc

        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            timeout=httpx.Timeout(config.timeout_seconds),
            max_retries=0,  # retries handled explicitly in analyze()
        )

    def analyze(self, user_prompt: str) -> DiagnosticReport:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnostic_report",
                    "strict": True,
                    "schema": DIAGNOSTIC_JSON_SCHEMA,
                },
            },
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise AIAnalyzerProviderError("OpenAI returned an empty message body.")
        return DiagnosticReport.model_validate_json(raw)


class AnthropicAnalyzer:
    """Anthropic Messages API with tool-use structured output."""

    def __init__(self, config: AIConfig) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise _missing_sdk_error("Anthropic", "anthropic") from exc

        self._config = config
        self._client = Anthropic(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def analyze(self, user_prompt: str) -> DiagnosticReport:
        tool_name = "emit_diagnostic_report"
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Emit the structured CI failure diagnostic report.",
                    "input_schema": DIAGNOSTIC_JSON_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            temperature=0.2,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return DiagnosticReport.model_validate(block.input)

        raise AIAnalyzerProviderError(
            "Anthropic response did not contain the expected tool_use block."
        )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True

    exc_type = type(exc)
    module = exc_type.__module__
    if not module.startswith(("openai", "anthropic")):
        return False
    return exc_type.__name__ in _RETRYABLE_PROVIDER_ERRORS


def _is_invalid_api_key_error(exc: Exception) -> bool:
    """Detect provider 401 invalid_api_key responses."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True

    body = str(exc).lower()
    return "invalid_api_key" in body or "incorrect api key" in body


class DiagnosticEngine:
    """
    High-level facade used by the recovery workflow.

    Example:
        engine = DiagnosticEngine.from_config(app_config.ai)
        report = engine.analyze(filtered_input)
    """

    def __init__(self, client: _ProviderClient, config: AIConfig) -> None:
        self._client = client
        self._config = config

    @classmethod
    def from_config(cls, config: AIConfig) -> "DiagnosticEngine":
        if config.provider == AIProvider.OPENAI:
            client: _ProviderClient = OpenAIAnalyzer(config)
        elif config.provider == AIProvider.ANTHROPIC:
            client = AnthropicAnalyzer(config)
        else:
            raise ValueError(f"Unsupported AI provider: {config.provider}")
        return cls(client=client, config=config)

    def analyze(
        self,
        payload: FilteredDiagnosticInput,
        *,
        user_prompt_override: Optional[str] = None,
    ) -> DiagnosticReport:
        """
        Run structured diagnosis on pre-filtered CI artifacts.

        Args:
            payload: Output of diff_parser.build_filtered_input().
            user_prompt_override: Optional prompt for testing; defaults to built prompt.

        Returns:
            Validated DiagnosticReport.

        Raises:
            AIAnalyzerTimeout: After exhausting retries on timeout-class errors.
            AIAnalyzerValidationError: When model JSON fails schema validation.
            AIAnalyzerProviderError: For other provider failures.
        """
        user_prompt = user_prompt_override or _build_user_prompt(payload)

        # Respect global char budget before hitting the wire.
        if len(user_prompt) > self._config.max_input_chars:
            user_prompt = user_prompt[: self._config.max_input_chars]

        last_error: Optional[Exception] = None

        for attempt in range(1, self._config.max_retries + 2):
            try:
                logger.info(
                    "AI diagnosis attempt %s/%s via %s",
                    attempt,
                    self._config.max_retries + 1,
                    self._config.provider.value,
                )
                return self._client.analyze(user_prompt)

            except ValidationError as exc:
                logger.error("Structured output validation failed: %s", exc)
                raise AIAnalyzerValidationError(str(exc)) from exc

            except Exception as exc:
                last_error = exc
                if _is_retryable(exc) and attempt <= self._config.max_retries:
                    sleep_seconds = min(2**attempt, 30)
                    logger.warning(
                        "Retryable AI error (%s). Sleeping %ss before retry.",
                        type(exc).__name__,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue

                if _is_retryable(exc):
                    raise AIAnalyzerTimeout(
                        f"AI provider timed out after {attempt} attempt(s): {exc}"
                    ) from exc

                if _is_invalid_api_key_error(exc):
                    raise AIAnalyzerProviderError(
                        "Invalid API key (HTTP 401). Export a real key from your "
                        "provider dashboard, or re-run with `--mock-ai` for offline testing."
                    ) from exc

                raise AIAnalyzerProviderError(
                    f"AI provider error: {exc}"
                ) from exc

        raise AIAnalyzerProviderError(
            f"AI analysis failed after retries: {last_error}"
        )


def report_to_markdown(report: DiagnosticReport) -> str:
    """Render a DiagnosticReport as GitHub-flavored Markdown for Issues/PRs."""
    files_section = ""
    if report.failing_files:
        rows = []
        for item in report.failing_files:
            lines = ", ".join(str(n) for n in item.line_numbers) or "n/a"
            rows.append(
                f"| `{item.file_path}` | {lines} | {item.explanation} |"
            )
        files_section = (
            "\n\n### Failing files\n\n"
            "| File | Lines | Explanation |\n"
            "|------|-------|-------------|\n"
            + "\n".join(rows)
        )

    return (
        f"## Root cause\n\n{report.root_cause_summary}\n\n"
        f"**Blast radius:** {report.blast_radius_assessment}\n\n"
        f"**Confidence:** {report.confidence_score:.2f}"
        f"{files_section}\n\n"
        f"### Suggested fix\n\n{report.suggested_fix_patch}"
    )


def report_to_json(report: DiagnosticReport) -> str:
    """Serialize report for structured logging or artifact upload."""
    return json.dumps(report.model_dump(), indent=2)
