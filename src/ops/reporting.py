"""Legacy compatibility shim for reporting.
This file now re-exports the ReportGenerator implemented under src.ops.report
The heavy implementation was moved to src/ops/report/generator.py to organize report-related code.
"""
from .report import ReportGenerator

__all__ = ["ReportGenerator"]
