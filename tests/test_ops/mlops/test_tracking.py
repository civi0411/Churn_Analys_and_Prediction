"""
tests/test_ops/test_tracking.py

Tests cho ExperimentTracker (khởi tạo cơ bản).
"""
import pytest
from src.ops.mlops.tracking import ExperimentTracker


def test_experiment_tracker_init():
    """Kiểm tra khởi tạo `ExperimentTracker` có thuộc tính `base_dir`."""
    tracker = ExperimentTracker('artifacts/experiments')
    assert hasattr(tracker, 'base_dir')
