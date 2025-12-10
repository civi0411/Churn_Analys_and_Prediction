"""
src/ops/dataops/validator.py

Data Validator - Check data quality.
"""
import pandas as pd
import re
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

    def validate_business_rules(self, df: pd.DataFrame, n_examples: int = 5) -> Dict[str, Any]:
        """
        Validate domain/business rules and return a structured report.

        Rules implemented:
        - Range checks:
            * Tenure >= 0
            * CashbackAmount >= 0
        - Logical checks:
            * If CashbackAmount > 0 then OrderCount > 0
        - Format checks:
            * Email: basic email regex for values that are non-null
            * Phone: basic phone number regex for values that are non-null
        - Referential checks (best-effort):
            * If `CustomerID` column exists, ensure no nulls (presence) and optionally uniqueness

        Returns a dict with keys:
            - passed (bool): True if no violations
            - range_violations, logic_violations, format_violations, referential_violations
              each is a mapping from rule -> {count, examples}
        """
        report = {
            'passed': True,
            'range_violations': {},
            'logic_violations': {},
            'format_violations': {},
            'referential_violations': {}
        }

        if df is None or len(df) == 0:
            return report

        # Helper to collect examples
        def _examples(idx_list):
            return list(idx_list[:n_examples])

        # ===== Range checks =====
        if 'Tenure' in df.columns:
            mask = df['Tenure'].notna() & (df['Tenure'] < 0)
            cnt = int(mask.sum())
            if cnt > 0:
                report['passed'] = False
                report['range_violations']['Tenure'] = {
                    'count': cnt,
                    'examples': _examples(df.index[mask].tolist())
                }
                if self.logger:
                    self.logger.warning(f"BusinessRule[TENURE_RANGE]: {cnt} rows with Tenure < 0")

        if 'CashbackAmount' in df.columns:
            mask = df['CashbackAmount'].notna() & (df['CashbackAmount'] < 0)
            cnt = int(mask.sum())
            if cnt > 0:
                report['passed'] = False
                report['range_violations']['CashbackAmount'] = {
                    'count': cnt,
                    'examples': _examples(df.index[mask].tolist())
                }
                if self.logger:
                    self.logger.warning(f"BusinessRule[CASHBACK_RANGE]: {cnt} rows with CashbackAmount < 0")

        # ===== Logical checks =====
        if 'CashbackAmount' in df.columns and 'OrderCount' in df.columns:
            mask = df['CashbackAmount'].notna() & (df['CashbackAmount'] > 0) & (
                df['OrderCount'].isna() | (df['OrderCount'] <= 0)
            )
            cnt = int(mask.sum())
            if cnt > 0:
                report['passed'] = False
                report['logic_violations']['OrderCount_for_Cashback'] = {
                    'count': cnt,
                    'examples': _examples(df.index[mask].tolist())
                }
                if self.logger:
                    self.logger.warning(f"BusinessRule[ORDER_LOGIC]: {cnt} rows with Cashback>0 but OrderCount<=0 or missing")

        # ===== Format checks =====
        # Email
        if 'Email' in df.columns:
            email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
            mask = df['Email'].notna() & ~df['Email'].astype(str).str.match(email_re)
            cnt = int(mask.sum())
            if cnt > 0:
                report['passed'] = False
                report['format_violations']['Email'] = {
                    'count': cnt,
                    'examples': _examples(df.index[mask].tolist())
                }
                if self.logger:
                    self.logger.warning(f"BusinessRule[EMAIL_FMT]: {cnt} rows with invalid Email format")

        # Phone
        if 'Phone' in df.columns:
            # Basic phone: starts with optional +, contains digits, spaces, dashes, min length 7 digits
            phone_re = re.compile(r"^\+?[0-9\-\s()]{7,}$")
            mask = df['Phone'].notna() & ~df['Phone'].astype(str).str.match(phone_re)
            cnt = int(mask.sum())
            if cnt > 0:
                report['passed'] = False
                report['format_violations']['Phone'] = {
                    'count': cnt,
                    'examples': _examples(df.index[mask].tolist())
                }
                if self.logger:
                    self.logger.warning(f"BusinessRule[PHONE_FMT]: {cnt} rows with invalid Phone format")

        # ===== Referential checks (best-effort) =====
        if 'CustomerID' in df.columns:
            mask_null = df['CustomerID'].isna()
            cnt_null = int(mask_null.sum())
            if cnt_null > 0:
                report['passed'] = False
                report['referential_violations']['CustomerID_null'] = {
                    'count': cnt_null,
                    'examples': _examples(df.index[mask_null].tolist())
                }
                if self.logger:
                    self.logger.warning(f"BusinessRule[REF_CUSTOMER]: {cnt_null} rows with null CustomerID")

            # Optional: check uniqueness as potential referential problem (duplicate ids)
            dup_mask = df['CustomerID'].duplicated(keep=False)
            cnt_dup = int(dup_mask.sum())
            if cnt_dup > 0:
                # duplicate customer ids may be valid, but flag for review
                report['referential_violations']['CustomerID_duplicates'] = {
                    'count': cnt_dup,
                    'examples': _examples(df.index[dup_mask].tolist())
                }
                if self.logger:
                    self.logger.info(f"BusinessRule[REF_CUSTOMER]: {cnt_dup} rows with duplicated CustomerID (flagged)")

        return report

