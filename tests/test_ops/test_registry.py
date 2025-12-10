import pytest
from src.ops.mlops.registry import ModelRegistry

def test_model_registry_init():
    reg = ModelRegistry('artifacts/registry')
    assert hasattr(reg, 'registry_dir')

