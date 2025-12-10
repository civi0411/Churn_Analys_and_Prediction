import pytest
from src.ops.mlops.tracking import ExperimentTracker

def test_experiment_tracker_init():
    tracker = ExperimentTracker('artifacts/experiments')
    assert hasattr(tracker, 'base_dir')

