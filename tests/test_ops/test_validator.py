from src.ops.dataops.validator import DataValidator

def test_validator_basic():
    validator = DataValidator()
    # Test with empty DataFrame
    import pandas as pd
    df = pd.DataFrame()
    result = validator.validate_quality(df)
    assert isinstance(result, dict)
    assert 'quality_score' in result
