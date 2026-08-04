"""Tests for grian.dispatch battery optimisation."""

from grian.dispatch import capture_ratio, schedule


def test_schedule_importable():
    """The schedule function is importable."""
    assert callable(schedule)


def test_schedule_solves_simple():
    """Schedule solves a simple price profile and returns expected keys."""
    result = schedule(prices=[50, 100, 50, 100])
    assert result["status"] == "optimal"
    assert "charge" in result
    assert "discharge" in result
    assert "soc" in result
    assert "revenue" in result
    assert result["revenue"] >= 0


def test_capture_ratio_perfect():
    """Perfect foresight gives a capture ratio of 1.0."""
    assert capture_ratio(1000.0, 1000.0) == 1.0


def test_capture_ratio_half():
    """Half the perfect revenue gives 0.5."""
    assert capture_ratio(500.0, 1000.0) == 0.5


def test_capture_ratio_zero_denominator():
    """Zero perfect revenue returns 0.0, not a division error."""
    assert capture_ratio(0.0, 0.0) == 0.0
