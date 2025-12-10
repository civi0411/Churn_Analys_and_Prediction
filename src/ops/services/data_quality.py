"""
Deprecated shim: DataQuality service was moved into src.ops.dataops.validator.
This wrapper forwards calls to preserve backward compatibility.
"""
import warnings
from typing import Dict, Any, Optional
from ..dataops.validator import DataValidator


class DataQualityService:
    def __init__(self, logger=None):
        warnings.warn("DataQualityService has been moved to src.ops.dataops.validator. This shim will be removed in future.", DeprecationWarning)
        self.validator = DataValidator(logger)
        self.logger = logger

    def run_business_validation(self, df, run_dir: Optional[str] = None, persist: bool = True,
                                mask_pii: bool = True, n_examples: int = 5) -> Dict[str, Any]:
        """
        Backwards-compatible wrapper. Calls DataValidator.validate_business_rules and DataValidator.persist_business_report.
        """
        report = self.validator.validate_business_rules(df, n_examples=n_examples)
        if persist and run_dir:
            try:
                self.validator.persist_business_report(report, run_dir)
            except Exception:
                if self.logger:
                    self.logger.warning("Failed to persist business rules report (shim)")
        return report
