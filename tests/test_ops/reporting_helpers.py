"""
tests/test_ops/reporting_helpers.py

Smoke test import for reporting_helpers module.
"""

import pytest
import importlib


def test_import_reporting_helpers():
    """Kiểm tra module `src.ops.reporting_helpers` import được thành công."""
    mod = importlib.import_module('src.ops.reporting_helpers')
    assert mod is not None
