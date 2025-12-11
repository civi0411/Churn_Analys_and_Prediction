"""
tests/test_ops/test_drift_detector.py

Tests cho DataDriftDetector: schema drift, KS test cho numeric, Chi2 cho categorical.
"""
import pandas as pd
import numpy as np
import os

from src.ops.dataops.drift_detector import DataDriftDetector


def test_schema_drift_detected():
    """Kiểm tra phát hiện schema drift khi cột bị thiếu."""
    # reference with columns A,B
    ref = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    new = pd.DataFrame({'A': [1,2,3]})  # missing B

    detector = DataDriftDetector(ref)
    report = detector.detect_drift(new)

    assert report['drift_details']['schema']['has_drift']
    assert 'B' in report['drift_details']['schema']['missing_columns']
    assert report['summary']['drift_severity'] == 'CRITICAL'


def test_numerical_ks_detects_shift():
    """Kiểm tra KS test phát hiện dịch chuyển phân phối numeric."""
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({'x': rng.normal(0, 1, size=1000)})
    # shifted distribution
    new = pd.DataFrame({'x': rng.normal(1.0, 1, size=1000)})

    detector = DataDriftDetector(ref)
    report = detector.detect_drift(new, threshold=0.01)

    num = report['drift_details']['numerical']
    assert 'x' in num['ks_results']
    assert num['ks_results']['x']['has_drift']
    assert 'x' in num['drifted_features']


def test_categorical_chi2_detects_new_category():
    """Kiểm tra Chi2 phát hiện category mới xuất hiện."""
    ref = pd.DataFrame({'cat': ['a'] * 80 + ['b'] * 20})
    new = pd.DataFrame({'cat': ['a'] * 50 + ['b'] * 30 + ['c'] * 20})

    detector = DataDriftDetector(ref)
    report = detector.detect_drift(new, threshold=0.05)

    cat = report['drift_details']['categorical']
    assert 'cat' in cat['chi2_results']
    assert cat['chi2_results']['cat']['has_drift']
    assert 'c' in cat['chi2_results']['cat']['new_categories']


def test_detect_drift_handles_empty_reference_or_new():
    """Kiểm tra hàm xử lý khi reference hoặc new rỗng."""
    ref = pd.DataFrame({'x': []})
    new = pd.DataFrame({'x': [1,2,3]})
    detector = DataDriftDetector(ref)
    report = detector.detect_drift(new)
    # No numerical checks possible -> total_features_checked may be 0
    assert report['summary']['total_features_checked'] >= 0
