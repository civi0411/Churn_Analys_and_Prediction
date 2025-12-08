"""
src/ops/dataops/validator.py

Data Validator - Check data quality.
"""
import pandas as pd
from typing import Dict, Any


class DataValidator:
    """Data Validator - Check data quality (null, duplicate)."""

    def __init__(self, logger=None):
        self.logger = logger

    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check basic data quality."""
        n_rows = len(df)
        if n_rows == 0:
            return {"null_ratio": 0, "duplicate_ratio": 0, "rows": 0}

        null_ratio = df.isnull().sum().sum() / (n_rows * len(df.columns))
        dup_ratio = df.duplicated().sum() / n_rows

        if self.logger:
            if null_ratio > 0.2:
                self.logger.warning(f"  [WARN] High Null Ratio: {null_ratio:.2%}")
            if dup_ratio > 0.05:
                self.logger.warning(f"  [WARN] High Duplicate Ratio: {dup_ratio:.2%}")

        return {
            "null_ratio": null_ratio,
            "duplicate_ratio": dup_ratio,
            "rows": n_rows
        }
