"""
tests/test_ops/reporting.py

Smoke test import for reporting module.
"""

import pytest
import importlib


def test_import_reporting():
    """Kiểm tra module `src.ops.reporting` import được thành công."""
    reporting = importlib.import_module('src.ops.reporting')
    # Basic smoke: module loaded
    assert reporting is not None
