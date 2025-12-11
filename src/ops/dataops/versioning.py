"""
Module `ops.dataops.versioning` - quản lý version dữ liệu bằng hash-based versioning.

Important keywords: Args, Returns, Methods
"""
import os
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ...utils import IOHandler, ensure_dir, compute_file_hash


class DataVersioning:
    """
    Quản lý phiên bản dữ liệu (hash-based).

    Methods:
        create_version(file_path, dataset_name, df, ...)
        get_version_info, list_versions, compare_versions, add_lineage
    """

    def __init__(self, versions_dir: str, logger=None):
        self.versions_dir = versions_dir
        self.logger = logger
        ensure_dir(self.versions_dir)
        self.history_file = os.path.join(self.versions_dir, "versions.json")
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            self.history = IOHandler.load_json(self.history_file)
        else:
            self.history = {}

    def _save_history(self):
        IOHandler.save_json(self.history, self.history_file)

    def create_version(
        self,
        file_path: str,
        dataset_name: str = "raw_data",
        df: pd.DataFrame = None,
        description: str = None,
        tags: list = None
    ) -> str:
        """Tạo phiên bản mới cho tập dữ liệu."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không thể version file không tồn tại: {file_path}")

        file_hash = compute_file_hash(file_path)

        # Kiểm tra idempotency
        if dataset_name in self.history:
            if self.history[dataset_name] and self.history[dataset_name][-1]["hash"] == file_hash:
                return self.history[dataset_name][-1]["version_id"]

        ver_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        info = {
            "version_id": ver_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hash": file_hash,
            "path": file_path,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
        }

        if description:
            info["description"] = description
        if tags:
            info["tags"] = tags

        if df is not None:
            info["metadata"] = self._extract_metadata(df)
            info["profile"] = self._extract_profile(df)

        if dataset_name not in self.history:
            self.history[dataset_name] = []
        self.history[dataset_name].append(info)
        self._save_history()

        if self.logger:
            self.logger.info(f"Data Version  | {ver_id}")

        return ver_id

    def _extract_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }

    def _extract_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        profile = {
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum())
        }

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().to_dict()
            profile["numeric_stats"] = {
                col: {
                    "min": round(stats[col].get("min", 0), 2),
                    "max": round(stats[col].get("max", 0), 2),
                    "mean": round(stats[col].get("mean", 0), 2),
                    "std": round(stats[col].get("std", 0), 2)
                }
                for col in numeric_cols[:10]
            }

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            profile["categorical_stats"] = {
                col: {
                    "unique": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }
                for col in cat_cols[:5]
            }

        return profile

    def add_lineage(self, dataset_name: str, parent_name: str, transformation: str = None):
        """Thêm thông tin nguồn gốc."""
        if dataset_name not in self.history or not self.history[dataset_name]:
            return

        latest = self.history[dataset_name][-1]

        if "lineage" not in latest:
            latest["lineage"] = {}

        latest["lineage"]["parent"] = parent_name
        if transformation:
            latest["lineage"]["transformation"] = transformation
        latest["lineage"]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._save_history()

    def get_version_info(self, dataset_name: str, version_id: str = None) -> Dict:
        """Lấy thông tin phiên bản."""
        if dataset_name not in self.history or not self.history[dataset_name]:
            return {}

        if version_id is None:
            return self.history[dataset_name][-1]

        for version in self.history[dataset_name]:
            if version.get("version_id") == version_id:
                return version

        return {}

    def get_lineage_tree(self, dataset_name: str) -> Dict:
        """Lấy cây nguồn gốc."""
        lineage = {"dataset": dataset_name, "parents": []}

        info = self.get_version_info(dataset_name)
        while info and "lineage" in info:
            parent = info["lineage"].get("parent")
            if parent:
                lineage["parents"].append({
                    "name": parent,
                    "transformation": info["lineage"].get("transformation", "unknown")
                })
                info = self.get_version_info(parent)
            else:
                break

        return lineage

    def list_versions(self, dataset_name: str = None) -> Dict:
        """Liệt kê tất cả các phiên bản."""
        if dataset_name:
            return {dataset_name: self.history.get(dataset_name, [])}
        return self.history

    def compare_versions(self, dataset_name: str, v1_id: str, v2_id: str) -> Dict:
        """So sánh hai phiên bản."""
        v1 = self.get_version_info(dataset_name, v1_id)
        v2 = self.get_version_info(dataset_name, v2_id)

        if not v1 or not v2:
            return {"error": "Không tìm thấy phiên bản"}

        comparison = {
            "v1": v1_id,
            "v2": v2_id,
            "hash_changed": v1.get("hash") != v2.get("hash"),
            "changes": {}
        }

        if "metadata" in v1 and "metadata" in v2:
            m1, m2 = v1["metadata"], v2["metadata"]
            comparison["changes"]["rows"] = {
                "v1": m1.get("rows"),
                "v2": m2.get("rows"),
                "diff": m2.get("rows", 0) - m1.get("rows", 0)
            }
            comparison["changes"]["columns"] = {
                "v1": m1.get("columns"),
                "v2": m2.get("columns"),
                "diff": m2.get("columns", 0) - m1.get("columns", 0)
            }

        return comparison
