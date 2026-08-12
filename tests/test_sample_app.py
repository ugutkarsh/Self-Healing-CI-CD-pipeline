"""Sample test suite — one passing, one intentionally breaking."""

from tests.sample_app import add, divide


def test_addition_passes() -> None:
    assert add(2, 3) == 5


def test_division_breaks_intentionally() -> None:
    """This test fails to demonstrate Auto-Heal CI rollback flow."""
    # Bug: integer division expected but float returned.
    assert divide(10, 2) == 4  # intentional failure (expected 5.0)
