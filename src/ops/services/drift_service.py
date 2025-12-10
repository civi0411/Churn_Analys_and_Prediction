"""
Deprecated shim: DriftService functionality moved into src.ops.dataops.drift_detector.DataDriftDetector.save_report
This module provides a thin wrapper for backwards compatibility.
"""
import warnings
from typing import Dict, Any, Optional, Tuple
from ..dataops.drift_detector import DataDriftDetector


class DriftService:
    def __init__(self, reference_data=None, logger=None):
        warnings.warn("DriftService has been moved to src.ops.dataops.drift_detector. This shim will be removed in future.", DeprecationWarning)
        self.logger = logger
        self.reference_data = reference_data

    def run_drift(self, new_data, reference_data=None, reports_dir: Optional[str] = None,
                  threshold: float = 0.05, sample_frac: float = 1.0, max_rows: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        ref = reference_data if reference_data is not None else self.reference_data
        if ref is None:
            raise ValueError("Reference data must be provided to run drift")
        detector = DataDriftDetector(ref, self.logger)
        report = detector.detect_drift(new_data, threshold=threshold, sample_frac=sample_frac, max_rows=max_rows)
        json_path, md_path = detector.save_report(report, reports_dir)
        return report, json_path
