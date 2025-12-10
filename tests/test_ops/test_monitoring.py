import pytest
from src.ops.mlops.monitoring import ModelMonitor

def test_model_monitor_init():
    monitor = ModelMonitor('artifacts/monitoring')
    assert hasattr(monitor, 'base_dir')

