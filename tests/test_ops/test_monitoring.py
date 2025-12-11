"""
tests/test_ops/test_monitoring.py

Tests cho ModelMonitor (khởi tạo cơ bản).
"""
import pytest
from src.ops.mlops.monitoring import ModelMonitor


def test_model_monitor_init():
    """Kiểm tra khởi tạo `ModelMonitor` có thuộc tính `base_dir`."""
    monitor = ModelMonitor('artifacts/monitoring')
    assert hasattr(monitor, 'base_dir')
