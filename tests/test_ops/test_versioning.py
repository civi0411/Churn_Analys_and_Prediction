import pytest
from src.ops.dataops.versioning import DataVersioning

def test_data_versioning_init():
    dv = DataVersioning('artifacts/versions')
    assert hasattr(dv, 'versions_dir')
