"""
src/ops/dataops.py
Data Operations: Validation & Versioning
"""
import os
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ..utils import IOHandler, ensure_dir, compute_file_hash

# ==================== DATA VALIDATOR ====================

class DataValidator:
    """Kiểm tra chất lượng dữ liệu cơ bản"""

    def __init__(self, logger=None):
        self.logger = logger

    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
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


# ==================== DATA VERSIONING ====================

class DataVersioning:
    """Quản lý phiên bản dữ liệu dựa trên mã Hash (MD5)"""

    def __init__(self, versions_dir: str):
        self.versions_dir = versions_dir
        ensure_dir(self.versions_dir)
        self.history_file = os.path.join(self.versions_dir, "versions.json")
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            self.history = IOHandler.load_json(self.history_file)
        else:
            self.history = {}

    def create_version(self, file_path: str, dataset_name: str = "raw_data") -> str:
        """Tạo version mới cho dataset"""
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Cannot version missing file: {file_path}")

        file_hash = compute_file_hash(file_path)

        # Check idempotency (Nếu hash cũ trùng hash mới -> trả về version cũ)
        if dataset_name in self.history:
            if self.history[dataset_name] and self.history[dataset_name][-1]['hash'] == file_hash:
                return self.history[dataset_name][-1]['version_id']

        ver_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        info = {
            "version_id": ver_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "hash": file_hash,
            "path": file_path
        }

        if dataset_name not in self.history:
            self.history[dataset_name] = []
        self.history[dataset_name].append(info)
        IOHandler.save_json(self.history, self.history_file)

        return ver_id