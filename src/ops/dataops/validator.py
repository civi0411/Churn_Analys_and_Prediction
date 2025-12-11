"""
Module `ops.dataops.validator` - kiểm tra chất lượng dữ liệu và business rules.

Important keywords: Args, Returns, Methods, Notes
"""
import pandas as pd
import re
from typing import Dict, Any


class DataValidator:
    """
    Kiểm tra data quality và business rules.

    Methods:
        validate_quality(df) -> basic quality metrics
        validate_business_rules(df) -> structured report of violations
    """

    def __init__(self, logger=None):
        """_
        Khởi tạo DataValidator.

        Args:
            logger (logging.Logger, optional): Đối tượng logger để ghi lại các cảnh báo vi phạm. Defaults to None.
        """
        self.logger = logger

    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Kiểm tra các chỉ số chất lượng dữ liệu cơ bản như tỷ lệ giá trị thiếu và tỷ lệ trùng lặp.

        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra.

        Returns:
            Dict[str, Any]: Dictionary chứa kết quả:
                - 'null_ratio': Tỷ lệ giá trị thiếu trên toàn bộ bảng.
                - 'duplicate_ratio': Tỷ lệ hàng bị trùng lặp.
                - 'rows': Tổng số dòng dữ liệu.
                - 'quality_score': Điểm chất lượng tổng hợp (thang 0.0 - 1.0).
        """
        n_rows = len(df)
        if n_rows == 0:
            # For empty DataFrame return sensible defaults including a quality_score key
            return {"null_ratio": 0, "duplicate_ratio": 0, "rows": 0, "quality_score": 1.0}

        null_ratio = df.isnull().sum().sum() / (n_rows * len(df.columns))
        dup_ratio = df.duplicated().sum() / n_rows

        if self.logger:
            if null_ratio > 0.2:
                self.logger.warning(f"  [WARN] High Null Ratio: {null_ratio:.2%}")
            if dup_ratio > 0.05:
                self.logger.warning(f"  [WARN] High Duplicate Ratio: {dup_ratio:.2%}")

        # Simple composite quality score in range [0, 1]. Higher is better.
        # We average the complement of null and duplicate ratios. This is simple and stable.
        try:
            quality_score = float(max(0.0, 1.0 - ((null_ratio + dup_ratio) / 2.0)))
        except Exception:
            quality_score = 0.0

        return {
            "null_ratio": null_ratio,
            "duplicate_ratio": dup_ratio,
            "rows": n_rows,
            "quality_score": quality_score,
        }

    def validate_business_rules(self, df: pd.DataFrame, n_examples: int = 5) -> Dict[str, Any]:
        """
        Kiểm tra dữ liệu dựa trên các quy tắc nghiệp vụ (Domain/Business Rules) đã định nghĩa.

        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra.
            n_examples (int, optional): Số lượng ví dụ vi phạm tối đa cần lưu lại cho mỗi loại lỗi. Defaults to 5.

        Returns:
            Dict[str, Any]: Báo cáo cấu trúc về các vi phạm, bao gồm:
                - 'passed': (bool) True nếu dữ liệu vượt qua tất cả các quy tắc.
                - 'range_violations': Vi phạm về phạm vi giá trị (vd: số âm không hợp lệ).
                - 'logic_violations': Vi phạm về logic chéo giữa các cột (vd: có Cashback nhưng không có Order).
                - 'format_violations': Vi phạm về định dạng chuỗi (Email, Phone).
                - 'referential_violations': Vi phạm về ràng buộc tham chiếu (Null ID, Duplicate ID).
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
