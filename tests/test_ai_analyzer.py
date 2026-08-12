"""Unit tests for structured AI analyzer (mocked providers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.ai_analyzer import (
    AIAnalyzerValidationError,
    DiagnosticEngine,
    DiagnosticReport,
    FailingFile,
    report_to_markdown,
)
from src.config import AIConfig, AIProvider
from src.diff_parser import FilteredDiagnosticInput


@pytest.fixture
def sample_payload() -> FilteredDiagnosticInput:
    return FilteredDiagnosticInput(
        commit_sha="deadbeef",
        failing_excerpt="AssertionError: assert 5.0 == 4",
        relevant_diff="diff --git a/tests/sample_app.py b/tests/sample_app.py\n+return a / b",
        changed_files=["tests/sample_app.py"],
        pr_metadata="commit: deadbeef\npr_author: octocat",
    )


@pytest.fixture
def sample_report() -> DiagnosticReport:
    return DiagnosticReport(
        root_cause_summary="Division returns float but test expects int.",
        failing_files=[
            FailingFile(
                file_path="tests/test_sample_app.py",
                line_numbers=[12],
                explanation="Assertion compares float to int literal.",
            )
        ],
        blast_radius_assessment="Low — isolated test expectation mismatch.",
        confidence_score=0.91,
        suggested_fix_patch="```diff\n- assert divide(10, 2) == 4\n+ assert divide(10, 2) == 5.0\n```",
    )


def test_report_schema_validation() -> None:
    with pytest.raises(ValidationError):
        DiagnosticReport(
            root_cause_summary="x",
            blast_radius_assessment="Low",
            confidence_score=1.5,  # out of range
            suggested_fix_patch="fix",
        )


def test_report_to_markdown(sample_report: DiagnosticReport) -> None:
    md = report_to_markdown(sample_report)
    assert "Root cause" in md
    assert "sample_app" in md
    assert "0.91" in md


def test_diagnostic_engine_delegates_to_client(sample_payload: FilteredDiagnosticInput, sample_report: DiagnosticReport) -> None:
    mock_client = MagicMock()
    mock_client.analyze.return_value = sample_report

    config = AIConfig(
        provider=AIProvider.OPENAI,
        api_key="test-key",
        model="gpt-4o-mini",
        max_retries=0,
    )
    engine = DiagnosticEngine(client=mock_client, config=config)
    result = engine.analyze(sample_payload)

    assert result.root_cause_summary == sample_report.root_cause_summary
    mock_client.analyze.assert_called_once()


def test_validation_error_wrapped(sample_payload: FilteredDiagnosticInput) -> None:
    mock_client = MagicMock()
    mock_client.analyze.side_effect = ValidationError.from_exception_data(
        "DiagnosticReport",
        [{"type": "missing", "loc": ("root_cause_summary",), "input": {}}],
    )

    config = AIConfig(provider=AIProvider.OPENAI, api_key="k", model="m", max_retries=0)
    engine = DiagnosticEngine(client=mock_client, config=config)

    with pytest.raises(AIAnalyzerValidationError):
        engine.analyze(sample_payload)


def test_openai_schema_has_additional_properties_false() -> None:
    from src.ai_analyzer import DIAGNOSTIC_JSON_SCHEMA

    assert DIAGNOSTIC_JSON_SCHEMA["additionalProperties"] is False
    assert "failing_files" in DIAGNOSTIC_JSON_SCHEMA["required"]
    failing_file = DIAGNOSTIC_JSON_SCHEMA["$defs"]["FailingFile"]
    assert failing_file["additionalProperties"] is False
    assert set(failing_file["required"]) == set(failing_file["properties"].keys())
