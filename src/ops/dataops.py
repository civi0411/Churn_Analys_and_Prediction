"""
src/ops/dataops.py

Data Operations Module - Quản lý chất lượng và phiên bản dữ liệu

Module này cung cấp các công cụ DataOps:
    - DataValidator: Kiểm tra chất lượng dữ liệu (null, duplicate)
    - DataVersioning: Quản lý phiên bản dữ liệu với hash-based versioning

Tính năng chính:
    - Validation: Kiểm tra null ratio, duplicate ratio
    - Versioning: Hash-based versioning với MD5
    - Metadata: Lưu trữ thông tin rows, columns, dtypes
    - Profiling: Thống kê cơ bản (min, max, mean, unique values)
    - Lineage: Theo dõi nguồn gốc dữ liệu (raw → processed → train/test)

Example:
    >>> validator = DataValidator(logger)
    >>> quality = validator.validate_quality(df)
    >>> print(f"Null ratio: {quality['null_ratio']:.2%}")

    >>> versioning = DataVersioning('artifacts/versions', logger)
    >>> version_id = versioning.create_version('data.csv', 'raw_data', df=df)

Author: Churn Prediction Team
"""
import os
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from ..utils import IOHandler, ensure_dir, compute_file_hash, get_files_in_folder

# ==================== DATA VALIDATOR ====================

class DataValidator:
    """
    Data Validator - Kiểm tra chất lượng dữ liệu.

    Kiểm tra các vấn đề chất lượng dữ liệu phổ biến:
        - Missing values (null ratio)
        - Duplicate rows (duplicate ratio)

    Attributes:
        logger: Logger instance để ghi log warnings

    Example:
        >>> validator = DataValidator(logger)
        >>> quality = validator.validate_quality(df)
        >>> print(quality)
        {'null_ratio': 0.05, 'duplicate_ratio': 0.01, 'rows': 5630}
    """

    def __init__(self, logger=None):
        """
        Khởi tạo DataValidator.

        Args:
            logger: Logger instance (optional). Nếu có, sẽ log warnings
                   khi phát hiện vấn đề chất lượng nghiêm trọng.
        """
        self.logger = logger

    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Kiểm tra chất lượng dữ liệu cơ bản.

        Kiểm tra:
            - Null ratio: Tỷ lệ ô null trên tổng số ô
            - Duplicate ratio: Tỷ lệ dòng trùng lặp

        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra

        Returns:
            Dict[str, Any]: Dictionary chứa:
                - null_ratio (float): Tỷ lệ null (0.0 - 1.0)
                - duplicate_ratio (float): Tỷ lệ duplicate (0.0 - 1.0)
                - rows (int): Số dòng trong DataFrame

        Warnings:
            - Log WARNING nếu null_ratio > 20%
            - Log WARNING nếu duplicate_ratio > 5%

        Example:
            >>> quality = validator.validate_quality(df)
            >>> if quality['null_ratio'] > 0.1:
            ...     print("High missing rate!")
        """
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
    """
    Quản lý phiên bản dữ liệu với:
    - Hash-based versioning
    - Metadata tracking (rows, columns, dtypes)
    - Data profiling (statistics)
    - Data lineage (raw → processed → train/test)
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
        """Lưu history ra file"""
        IOHandler.save_json(self.history, self.history_file)

    def create_version(
        self,
        file_path: str,
        dataset_name: str = "raw_data",
        df: pd.DataFrame = None,
        description: str = None,
        tags: list = None
    ) -> str:
        """
        Tạo version mới cho dataset với đầy đủ metadata

        Args:
            file_path: Đường dẫn file
            dataset_name: Tên dataset (raw_data, train, test, etc.)
            df: DataFrame để extract metadata (optional)
            description: Mô tả dataset
            tags: Tags để phân loại

        Returns:
            version_id
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot version missing file: {file_path}")

        file_hash = compute_file_hash(file_path)

        # Check idempotency (Nếu hash cũ trùng hash mới -> trả về version cũ)
        if dataset_name in self.history:
            if self.history[dataset_name] and self.history[dataset_name][-1]['hash'] == file_hash:
                return self.history[dataset_name][-1]['version_id']

        ver_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Base info
        info = {
            "version_id": ver_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "hash": file_hash,
            "path": file_path,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
        }

        # Optional metadata
        if description:
            info["description"] = description
        if tags:
            info["tags"] = tags

        # Extract metadata from DataFrame if provided
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
        """Extract metadata từ DataFrame"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }

    def _extract_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic statistics profile từ DataFrame"""
        profile = {
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum())
        }

        # Numerical stats
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().to_dict()
            profile["numeric_stats"] = {
                col: {
                    "min": round(stats[col].get('min', 0), 2),
                    "max": round(stats[col].get('max', 0), 2),
                    "mean": round(stats[col].get('mean', 0), 2),
                    "std": round(stats[col].get('std', 0), 2)
                }
                for col in numeric_cols[:10]  # Top 10 columns
            }

        # Categorical stats
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            profile["categorical_stats"] = {
                col: {
                    "unique": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }
                for col in cat_cols[:5]  # Top 5 columns
            }

        return profile

    def add_lineage(self, dataset_name: str, parent_name: str, transformation: str = None):
        """
        Thêm thông tin lineage (nguồn gốc data)

        Args:
            dataset_name: Dataset con (e.g., 'train')
            parent_name: Dataset cha (e.g., 'raw_data')
            transformation: Mô tả transformation (e.g., 'split 80/20')
        """
        if dataset_name not in self.history or not self.history[dataset_name]:
            return

        latest = self.history[dataset_name][-1]

        if "lineage" not in latest:
            latest["lineage"] = {}

        latest["lineage"]["parent"] = parent_name
        if transformation:
            latest["lineage"]["transformation"] = transformation
        latest["lineage"]["created_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self._save_history()

    def get_version_info(self, dataset_name: str, version_id: str = None) -> Dict:
        """
        Lấy thông tin của một version

        Args:
            dataset_name: Tên dataset
            version_id: Version cụ thể (None = latest)
        """
        if dataset_name not in self.history or not self.history[dataset_name]:
            return {}

        if version_id is None:
            return self.history[dataset_name][-1]

        for version in self.history[dataset_name]:
            if version.get('version_id') == version_id:
                return version

        return {}

    def get_lineage_tree(self, dataset_name: str) -> Dict:
        """
        Lấy cây lineage của dataset

        Returns:
            Dict với parent chain
        """
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
        """Liệt kê tất cả versions"""
        if dataset_name:
            return {dataset_name: self.history.get(dataset_name, [])}
        return self.history

    def compare_versions(self, dataset_name: str, v1_id: str, v2_id: str) -> Dict:
        """
        So sánh 2 versions của cùng dataset

        Returns:
            Dict với các thay đổi
        """
        v1 = self.get_version_info(dataset_name, v1_id)
        v2 = self.get_version_info(dataset_name, v2_id)

        if not v1 or not v2:
            return {"error": "Version not found"}

        comparison = {
            "v1": v1_id,
            "v2": v2_id,
            "hash_changed": v1.get("hash") != v2.get("hash"),
            "changes": {}
        }

        # Compare metadata if available
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


# ==================== BATCH DATA LOADER ====================

class BatchDataLoader:
    """
    Batch Data Loader - Gom nhiều file raw data thành 1 DataFrame.

    Use case: Mỗi tháng/quý có thêm raw data mới, cần gom lại để train lại model.

    Hỗ trợ:
        - Load tất cả files trong folder
        - Load các files mới (chưa được load trước đó)
        - Tracking files đã load
        - Deduplication sau khi merge

    Architecture:
        - Sử dụng IOHandler.read_data() (src/utils.py) để đọc từng file
        - Thêm các features: tracking, dedup, logging, _source_file column

    Example:
        >>> loader = BatchDataLoader('data/raw/monthly', logger)
        >>> df = loader.load_all()  # Load tất cả
        >>> df_new = loader.load_new_only()  # Chỉ load files mới
    """

    def __init__(self, batch_folder: str, logger=None, tracking_file: str = None):
        """
        Khởi tạo BatchDataLoader.

        Args:
            batch_folder: Thư mục chứa các files raw data
            logger: Logger instance
            tracking_file: File JSON để track files đã load (optional)
        """
        self.batch_folder = batch_folder
        self.logger = logger
        self.tracking_file = tracking_file or os.path.join(batch_folder, ".batch_tracking.json")
        self._load_tracking()

    def _load_tracking(self):
        """Load tracking history."""
        if os.path.exists(self.tracking_file):
            self.tracking = IOHandler.load_json(self.tracking_file)
        else:
            self.tracking = {
                "loaded_files": [],
                "last_load": None,
                "total_rows_loaded": 0
            }

    def _save_tracking(self):
        """Save tracking history."""
        IOHandler.save_json(self.tracking, self.tracking_file)

    def _get_all_files(self) -> List[str]:
        """Lấy danh sách tất cả files trong batch folder (delegate to utils)."""
        return get_files_in_folder(self.batch_folder)

    def _get_new_files(self) -> List[str]:
        """Lấy danh sách files mới (chưa được load)."""
        all_files = self._get_all_files()
        loaded_files = set(self.tracking.get("loaded_files", []))

        new_files = [f for f in all_files if f not in loaded_files]
        return new_files

    def load_all(self, deduplicate: bool = True, reset_tracking: bool = False,
                 file_list: List[str] = None) -> pd.DataFrame:
        """
        Load và merge TẤT CẢ files trong batch folder.

        Args:
            deduplicate: Có remove duplicate rows sau khi merge không
            reset_tracking: Reset tracking (coi như chưa load file nào)
            file_list: Danh sách files cụ thể (từ CLI filter). Nếu None, load tất cả.

        Returns:
            pd.DataFrame: DataFrame đã merge
        """
        if reset_tracking:
            self.tracking = {"loaded_files": [], "last_load": None, "total_rows_loaded": 0}

        # Sử dụng file_list nếu có, nếu không thì lấy tất cả files
        all_files = file_list if file_list else self._get_all_files()

        if not all_files:
            if self.logger:
                self.logger.warning(f"No files found in {self.batch_folder}")
            return pd.DataFrame()

        if self.logger:
            self.logger.info(f"Batch Load    | Folder: {self.batch_folder}")
            self.logger.info(f"Batch Load    | Files: {len(all_files)}")

        df_list = []
        for file_path in all_files:
            try:
                df = IOHandler.read_data(file_path)
                df['_source_file'] = os.path.basename(file_path)  # Track nguồn
                df_list.append(df)

                if self.logger:
                    self.logger.info(f"  Loaded      | {os.path.basename(file_path)} ({len(df):,} rows)")

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"  Skip        | {os.path.basename(file_path)}: {e}")

        if not df_list:
            return pd.DataFrame()

        # Merge
        merged_df = pd.concat(df_list, ignore_index=True)

        # Deduplicate
        if deduplicate:
            before = len(merged_df)
            # Drop duplicates (trừ cột _source_file)
            cols_to_check = [c for c in merged_df.columns if c != '_source_file']
            merged_df = merged_df.drop_duplicates(subset=cols_to_check, keep='last')
            after = len(merged_df)

            if self.logger and before != after:
                self.logger.info(f"Deduplicated  | Removed {before - after:,} duplicates")

        # Update tracking
        self.tracking["loaded_files"] = all_files
        self.tracking["last_load"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.tracking["total_rows_loaded"] = len(merged_df)
        self._save_tracking()

        if self.logger:
            self.logger.info(f"Batch Result  | Total: {len(merged_df):,} rows")

        return merged_df

    def load_new_only(self, deduplicate: bool = True) -> pd.DataFrame:
        """
        Chỉ load các files MỚI (chưa được load trước đó).

        Hữu ích khi muốn xem data mới mà không cần load lại tất cả.

        Args:
            deduplicate: Có remove duplicate rows không

        Returns:
            pd.DataFrame: DataFrame chỉ chứa data mới
        """
        new_files = self._get_new_files()

        if not new_files:
            if self.logger:
                self.logger.info("Batch Load    | No new files found")
            return pd.DataFrame()

        if self.logger:
            self.logger.info(f"Batch Load    | New files: {len(new_files)}")

        df_list = []
        for file_path in new_files:
            try:
                df = IOHandler.read_data(file_path)
                df['_source_file'] = os.path.basename(file_path)
                df_list.append(df)

                if self.logger:
                    self.logger.info(f"  Loaded      | {os.path.basename(file_path)} ({len(df):,} rows)")

                # Add to tracking
                self.tracking["loaded_files"].append(file_path)

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"  Skip        | {os.path.basename(file_path)}: {e}")

        if not df_list:
            return pd.DataFrame()

        merged_df = pd.concat(df_list, ignore_index=True)

        if deduplicate:
            cols_to_check = [c for c in merged_df.columns if c != '_source_file']
            merged_df = merged_df.drop_duplicates(subset=cols_to_check, keep='last')

        # Update tracking
        self.tracking["last_load"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.tracking["total_rows_loaded"] += len(merged_df)
        self._save_tracking()

        return merged_df

    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của batch loader.

        Returns:
            Dict chứa thông tin files đã load, files mới, etc.
        """
        all_files = self._get_all_files()
        new_files = self._get_new_files()

        return {
            "batch_folder": self.batch_folder,
            "total_files": len(all_files),
            "loaded_files": len(self.tracking.get("loaded_files", [])),
            "new_files": len(new_files),
            "new_file_names": [os.path.basename(f) for f in new_files],
            "last_load": self.tracking.get("last_load"),
            "total_rows_loaded": self.tracking.get("total_rows_loaded", 0)
        }

    def reset(self):
        """Reset tracking - coi như chưa load file nào."""
        self.tracking = {
            "loaded_files": [],
            "last_load": None,
            "total_rows_loaded": 0
        }
        self._save_tracking()

        if self.logger:
            self.logger.info("Batch Tracking Reset")

