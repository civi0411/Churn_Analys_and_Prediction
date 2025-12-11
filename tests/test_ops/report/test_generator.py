"""
tests/test_ops/report/test_generator.py

Tests đơn giản cho `ReportGenerator` khởi tạo.
"""

from src.ops.report.generator import ReportGenerator


def test_report_generator_init():
    """Kiểm tra khởi tạo ReportGenerator có thuộc tính experiments_base_dir."""
    gen = ReportGenerator(experiments_base_dir='artifacts/experiments', logger=None)
    assert hasattr(gen, 'experiments_base_dir')
