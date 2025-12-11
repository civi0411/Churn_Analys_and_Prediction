"""
tests/ops/dataops/test_validator.py

Tests cho `DataValidator` (validate_quality basic smoke test).
"""

from src.ops.dataops.validator import DataValidator


def test_validator_basic():
    """Kiểm tra `DataValidator.validate_quality` trả về dict có key 'quality_score' với DataFrame rỗng."""
    validator = DataValidator()
    # Test with empty DataFrame
    import pandas as pd

    df = pd.DataFrame()
    result = validator.validate_quality(df)
    assert isinstance(result, dict)
    assert 'quality_score' in result
