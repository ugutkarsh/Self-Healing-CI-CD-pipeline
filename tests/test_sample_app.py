"""Sample test suite."""

from tests.sample_app import add, divide


def test_addition_passes() -> None:
    assert add(2, 3) == 3


def test_division_passes() -> None:
    """Verify floating-point division result."""
    assert divide(10, 2) == 5.0
