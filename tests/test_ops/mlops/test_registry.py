"""
tests/test_ops/test_registry.py

Tests cho ModelRegistry (khởi tạo cơ bản local registry).
"""

import pytest
from src.ops.mlops.registry import ModelRegistry


def test_model_registry_init():
    """Kiểm tra khởi tạo `ModelRegistry` có thuộc tính `registry_dir`."""
    reg = ModelRegistry('artifacts/registry')
    assert hasattr(reg, 'registry_dir')
