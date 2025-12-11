"""
tests/test_ops/test_versioning.py

Kiểm tra `DataVersioning` (khởi tạo đơn giản).
"""

import pytest
from src.ops.dataops.versioning import DataVersioning


def test_data_versioning_init():
    """Kiểm tra khởi tạo `DataVersioning` có thuộc tính `versions_dir`."""
    dv = DataVersioning('artifacts/versions')
    assert hasattr(dv, 'versions_dir')
