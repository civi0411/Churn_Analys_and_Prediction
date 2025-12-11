"""
tests/test_data/test_transformer.py

Các unit test cho `DataTransformer`.
Mục tiêu: kiểm tra `fit_transform`, `transform` và factory resampler.

Important keywords: Args, Returns, Notes
"""
import pytest
import pandas as pd

from src.data.transformer import DataTransformer, IMBLEARN_AVAILABLE


def make_config():
    return {
        'data': {'target_col': 'Churn'},
        'create_features': True,
        'missing_strategy': {'numerical': 'median', 'categorical': 'mode'},
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'scaler_type': 'standard',
        'categorical_encoding': 'label',
        'feature_selection': False,
        'preprocessing': {'use_smote': False}
    }


def test_fit_transform_and_transform(sample_raw_df, mock_logger):
    """Fit trên training và transform dữ liệu test; kiểm tra learned params và kích thước output.

    Args:
        sample_raw_df: fixture chứa DataFrame mẫu
        mock_logger: fixture logger giả
    """
    cfg = make_config()
    transformer = DataTransformer(cfg, mock_logger)

    # Fit-transform on sample_raw_df
    X_train, y_train = transformer.fit_transform(sample_raw_df)

    assert X_train is not None
    assert y_train is not None
    assert len(X_train) == len(y_train)

    # Learned params should have imputers / scaler possibly set (depending on data types)
    assert isinstance(transformer._learned_params, dict)

    # Now transform a new copy and ensure no exceptions and same columns (or selected_features applied)
    X_test, y_test = transformer.transform(sample_raw_df.copy())

    assert X_test is not None
    # If selected_features was set, ensure returned columns match
    if transformer._learned_params.get('selected_features'):
        assert list(X_test.columns) == list(transformer._learned_params['selected_features'])
    else:
        # Otherwise ensure X_test has at least one column
        assert X_test.shape[1] > 0


def test_get_resampler(sample_raw_df, mock_logger):
    """Kiểm tra `get_resampler` trả về resampler khi `imblearn` có sẵn, ngược lại trả None."""
    cfg = make_config()
    cfg['preprocessing']['use_smote'] = True
    transformer = DataTransformer(cfg, mock_logger)

    resampler = transformer.get_resampler()

    if IMBLEARN_AVAILABLE:
        # Should return a resampler instance with fit_resample method
        assert resampler is not None
        assert hasattr(resampler, 'fit_resample')
    else:
        # imblearn not installed -> None
        assert resampler is None
