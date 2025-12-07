"""
src/utils.py
Utility functions: Logging, IO, Hashing, Config, Filename parsing
"""
import os, sys, yaml, json, logging, joblib, pickle, random, glob
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd


# ===================== Helper functions =====================

def ensure_dir(dir_path: str) -> None:
    """Create directory if not exists. Ignore empty path."""
    if not dir_path:
        return
    os.makedirs(dir_path, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_timestamp() -> str:
    """Return current timestamp as YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_base_filename(file_path: str) -> str:
    """
    Extract base filename without extension.

    Example:
        'data/raw/E Commerce Dataset.xlsx' -> 'E Commerce Dataset'
        'data/raw/customers.csv' -> 'customers'
    """
    return Path(file_path).stem


def build_processed_filename(raw_path: str, processed_dir: str) -> str:
    """
    Build processed filename: [raw_filename]_processed.csv

    Example:
        raw_path: 'data/raw/E Commerce Dataset.xlsx'
        -> 'data/processed/E Commerce Dataset_processed.csv'
    """
    base_name = extract_base_filename(raw_path)
    return os.path.join(processed_dir, f"{base_name}_processed.csv")


def build_split_filenames(raw_path: str, train_test_dir: str) -> Tuple[str, str]:
    """
    Build train/test filenames with timestamp.

    Returns:
        (train_path, test_path)

    Example:
        raw_path: 'data/raw/E Commerce Dataset.xlsx'
        -> ('data/train_test/E Commerce Dataset_train_20240520_143022.parquet',
            'data/train_test/E Commerce Dataset_test_20240520_143022.parquet')
    """
    base_name = extract_base_filename(raw_path)
    train_path = os.path.join(train_test_dir, f"{base_name}_train.parquet")
    test_path = os.path.join(train_test_dir, f"{base_name}_test.parquet")

    return train_path, test_path


def get_latest_train_test(train_test_dir: str, raw_filename: str = None) -> Tuple[str, str]:
    """
    Get the latest train/test files from train_test directory.

    Args:
        train_test_dir: Directory containing train/test files
        raw_filename: Optional base filename to filter (e.g., 'E Commerce Dataset')
                     If None, returns the latest pair regardless of source

    Returns:
        (train_path, test_path) or raises FileNotFoundError
    """
    if not os.path.exists(train_test_dir):
        raise FileNotFoundError(f"Train/test directory not found: {train_test_dir}")

    # Build search patterns
    if raw_filename:
        base = extract_base_filename(raw_filename) if '.' in raw_filename else raw_filename
        train_pattern = f"{base}_train.parquet"
        test_pattern = f"{base}_test.parquet"
    else:
        train_pattern = "*_train.parquet"
        test_pattern = "*_test.parquet"

    # Find all matching files
    train_files = sorted(glob.glob(os.path.join(train_test_dir, train_pattern)))
    test_files = sorted(glob.glob(os.path.join(train_test_dir, test_pattern)))

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"No train/test files found in {train_test_dir} "
            f"{'for ' + raw_filename if raw_filename else ''}"
        )

    # Return latest (sorted by name which includes timestamp)
    return train_files[-1], test_files[-1]


def compute_file_hash(path: str) -> str:
    """
    Compute MD5 hash.
    - If path is a file: Hash the file.
    - If path is a directory: Hash all files inside and combine them.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found for hashing: {path}")

    hash_md5 = hashlib.md5()

    if os.path.isfile(path):
        # Hash single file
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    elif os.path.isdir(path):
        # Hash directory
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                hash_md5.update(file.encode('utf-8'))
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_files_in_folder(folder: str, extensions: list = None) -> list:
    """
    Lấy danh sách tất cả files trong folder theo extensions.

    Args:
        folder (str): Đường dẫn thư mục
        extensions (list): Danh sách extensions (default: csv, xlsx, xls, parquet)

    Returns:
        list: Danh sách đường dẫn files (sorted)

    Example:
        >>> files = get_files_in_folder('data/raw')
        >>> files = get_files_in_folder('data/raw', ['.csv', '.xlsx'])
    """
    if extensions is None:
        extensions = ['.csv', '.xlsx', '.xls', '.parquet']

    all_files = []
    for ext in extensions:
        pattern = f"*{ext}"
        all_files.extend(glob.glob(os.path.join(folder, pattern)))
        all_files.extend(glob.glob(os.path.join(folder, pattern.upper())))

    # Filter temp files và deduplicate
    all_files = [f for f in all_files if not os.path.basename(f).startswith(('~', '.'))]
    return sorted(list(set(all_files)))


def filter_files_by_date(folder: str, from_date: str = None, to_date: str = None) -> list:
    """
    Lọc files trong folder theo pattern ngày tháng trong tên file.

    Stateless utility - chỉ parse và filter, không tracking.

    Hỗ trợ patterns:
        - data_2024_01.csv (YYYY_MM)
        - data_2024-01.csv (YYYY-MM)
        - data_202401.csv (YYYYMM)
        - data_2024_Q1.csv (YYYY_Q#)

    Args:
        folder (str): Đường dẫn thư mục
        from_date (str): Ngày bắt đầu (format: YYYY-MM hoặc YYYY-Q#)
        to_date (str): Ngày kết thúc (format: YYYY-MM hoặc YYYY-Q#)

    Returns:
        list: Danh sách files đã filter (sorted)

    Example:
        >>> files = filter_files_by_date('data/raw/monthly', '2024-01', '2024-06')
        >>> files = filter_files_by_date('data/raw/quarterly', '2024-Q1', '2024-Q2')
    """
    import re
    from datetime import datetime as dt

    all_files = get_files_in_folder(folder)

    if not from_date and not to_date:
        return all_files

    def parse_filter_date(date_str: str, is_end: bool = False) -> dt:
        """Parse filter date string thành datetime."""
        if not date_str:
            return dt.min if not is_end else dt.max

        # Quarter format: 2024-Q1
        q_match = re.match(r'(\d{4})[-_]?Q(\d)', date_str, re.IGNORECASE)
        if q_match:
            year, quarter = int(q_match.group(1)), int(q_match.group(2))
            month = quarter * 3 if is_end else (quarter - 1) * 3 + 1
            return dt(year, month, 1)

        # Month format: 2024-01
        m_match = re.match(r'(\d{4})[-_]?(\d{2})', date_str)
        if m_match:
            return dt(int(m_match.group(1)), int(m_match.group(2)), 1)

        return dt.min if not is_end else dt.max

    def extract_date_from_filename(filename: str) -> dt:
        """Extract date từ tên file."""
        basename = os.path.basename(filename)

        # Quarter pattern: 2024_Q1, 2024-Q2
        q_match = re.search(r'(\d{4})[-_]?Q(\d)', basename, re.IGNORECASE)
        if q_match:
            year, quarter = int(q_match.group(1)), int(q_match.group(2))
            return dt(year, (quarter - 1) * 3 + 1, 1)

        # Month pattern: 2024_01, 2024-01, 202401
        m_match = re.search(r'(\d{4})[-_]?(\d{2})', basename)
        if m_match:
            year, month = int(m_match.group(1)), int(m_match.group(2))
            if 1 <= month <= 12:
                return dt(year, month, 1)

        return dt.min

    from_dt = parse_filter_date(from_date, is_end=False)
    to_dt = parse_filter_date(to_date, is_end=True)

    filtered = []
    for f in all_files:
        file_date = extract_date_from_filename(f)
        # Include if: no date pattern found OR date is within range
        if file_date == dt.min or from_dt <= file_date <= to_dt:
            filtered.append(f)

    return sorted(filtered)


# ===================== Config Loader =====================

class ConfigLoader:
    """Load YAML config with basic validation."""

    @staticmethod
    def load_config(config_path: str = "config/config.yaml") -> Dict:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If YAML is empty or invalid.
            yaml.YAMLError: If YAML syntax is invalid.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}") from e

        if config is None:
            raise ValueError(f"Config file is empty or invalid: {config_path}")

        if not isinstance(config, dict):
            raise ValueError(f"Config root must be a mapping (dict): {config_path}")

        return config


# ===================== Logger =====================

# ===================== Logger với Filtering =====================

class SystemLogFilter(logging.Filter):
    """
    Filter cho System Log (Dành cho Dev):
    - WARNING, ERROR, CRITICAL (luôn ghi)
    - INFO: CHỈ ghi các dòng QUAN TRỌNG (không ghi dòng = dư thừa)
    """

    def filter(self, record):
        # Luôn ghi WARNING/ERROR/CRITICAL
        if record.levelno >= logging.WARNING:
            return True

        # Với INFO: chỉ ghi milestone thực sự quan trọng
        if record.levelno == logging.INFO:
            msg = record.getMessage()
            # Chỉ cho phép dòng = nếu có keyword quan trọng kèm theo
            if msg.strip().startswith('='):
                # Bỏ qua dòng chỉ toàn dấu =
                if msg.strip().strip('=').strip() == '':
                    return False

            # ✅ CHỈ GHI CÁC MILESTONE QUAN TRỌNG
            important_keywords = [
                'PIPELINE STARTED',
                'PIPELINE FINISHED',
                'Pipeline Initialized',
                'STAGE:',
                'Mode:',
                'Run ID:',
                '[SYSTEM]',
                '[SUCCESS]',
                '[FAILURE]',
                'TRAINING RESULTS',
                'Best Model:',
                'START PIPELINE',  # Từ main.py
            ]

            return any(keyword in msg for keyword in important_keywords)

        return False


class RunLogFilter(logging.Filter):
    """
    Filter cho Run Log (Dành cho User):
    - CHỈ ghi INFO (full chi tiết)
    - KHÔNG ghi WARNING/DEBUG/ERROR (clean)
    """

    def filter(self, record):
        # Chỉ cho phép INFO
        return record.levelno == logging.INFO


class Logger:
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str, log_dir: str = "artifacts/logs",
                   level: str = "INFO", run_log_path: str = None) -> logging.Logger:
        """
        Tạo logger với 3 handlers:
        1. Console: INFO+ (hiển thị mọi thứ)
        2. System Log: INFO (filtered) + WARNING + ERROR + CRITICAL
        3. Run Log: CHỈ INFO (clean cho user)

        Args:
            name: Logger name
            log_dir: Global log directory
            level: Log level
            run_log_path: Path to run-specific log file (optional)
        """

        # Singleton check
        if name in cls._loggers:
            logger = cls._loggers[name]

            # ✅ NẾU CÓ run_log_path MỚI → THÊM HANDLER
            if run_log_path:
                cls._add_run_handler(logger, run_log_path)

            return logger

        ensure_dir(log_dir)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Logger nhận tất cả

        if logger.hasHandlers():
            logger.handlers.clear()

        # Formatter cho Console và System Log
        full_formatter = logging.Formatter(
            "[%(asctime)s] | %(levelname)-7s | %(name)-10s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Formatter cho Run Log (bỏ %(name)s - gọn hơn)
        run_formatter = logging.Formatter(
            "[%(asctime)s] | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ===== HANDLER 1: CONSOLE (INFO+) =====
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(full_formatter)
        logger.addHandler(console_handler)

        # ===== HANDLER 2: SYSTEM LOG (Filtered INFO + WARNING+) =====
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        global_log_file = os.path.join(log_dir, f"MAIN_{timestamp}.log")
        global_file_handler = logging.FileHandler(global_log_file, encoding="utf-8")
        global_file_handler.setLevel(logging.INFO)
        global_file_handler.setFormatter(full_formatter)

        # ✅ THÊM FILTER: Chỉ ghi milestone INFO + WARNING/ERROR
        global_file_handler.addFilter(SystemLogFilter())
        logger.addHandler(global_file_handler)

        # ===== HANDLER 3: RUN LOG (CHỈ INFO, KHÔNG WARNING) =====
        if run_log_path:
            cls._add_run_handler(logger, run_log_path)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def _add_run_handler(cls, logger: logging.Logger, run_log_path: str):
        """Thêm handler cho run-specific log (CHỈ INFO)"""
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == os.path.abspath(run_log_path):
                    return

        ensure_dir(os.path.dirname(run_log_path))

        run_formatter = logging.Formatter(
            "[%(asctime)s] | %(levelname)-7s | %(message)s",  # ← ĐÚNG RỒI
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        run_file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
        run_file_handler.setLevel(logging.INFO)
        run_file_handler.setFormatter(run_formatter)
        run_file_handler.addFilter(RunLogFilter())
        logger.addHandler(run_file_handler)

# ===================== IO Handler =====================

class IOHandler:
    """
    Handle IO operations:
    - Read / write tabular data (csv, excel, json, parquet)
    - Save / load models (joblib, pickle)
    - Save / load JSON files
    """

    _SUPPORTED_EXT = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    @staticmethod
    def read_data(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read data from file into DataFrame.

        Raises:
            ValueError: If file extension is not supported.
            IOError: For IO or parsing errors.
        """
        ext = Path(file_path).suffix.lower()

        if ext not in IOHandler._SUPPORTED_EXT:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(sorted(IOHandler._SUPPORTED_EXT))}"
            )

        try:
            if ext == ".csv":
                return pd.read_csv(file_path, **kwargs)
            elif ext in {".xlsx", ".xls"}:
                return pd.read_excel(file_path, **kwargs)
            elif ext == ".json":
                try:
                    return pd.read_json(file_path, orient='records', **kwargs)
                except ValueError:
                    return pd.read_json(file_path, **kwargs)
            elif ext == ".parquet":
                return pd.read_parquet(file_path, **kwargs)

        except (ValueError, TypeError) as e:
            raise IOError(f"Error reading file {file_path}: {e}") from e

        raise RuntimeError(f"Unhandled file format or error with extension: {ext}")

    # NOTE: Để đọc nhiều files với tracking, deduplication, logging
    # → Dùng BatchDataLoader (src/ops/dataops.py)

    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Save DataFrame to file.

        Raises:
            ValueError: If file extension is not supported.
            IOError: For other IO errors.
        """
        ensure_dir(os.path.dirname(file_path))
        ext = Path(file_path).suffix.lower()

        if ext not in IOHandler._SUPPORTED_EXT:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(sorted(IOHandler._SUPPORTED_EXT))}"
            )

        try:
            if ext == ".csv":
                df.to_csv(file_path, index=False, **kwargs)
            elif ext in {".xlsx", ".xls"}:
                df.to_excel(file_path, index=False, **kwargs)
            elif ext == ".json":
                df.to_json(file_path, **kwargs)
            elif ext == ".parquet":
                df.to_parquet(file_path, index=False, **kwargs)
        except Exception as e:
            raise IOError(f"Error saving file {file_path}: {e}") from e

    # ---------- Model IO ----------

    @staticmethod
    def save_model(model: Any, file_path: str, method: str = "joblib") -> None:
        """
        Save ML model to file.

        Args:
            model: Model object to save.
            file_path: Output path.
            method: "joblib" or "pickle".

        Raises:
            ValueError: If method is not supported.
            IOError: If save fails.
        """
        ensure_dir(os.path.dirname(file_path))

        if method not in {"joblib", "pickle"}:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Only 'joblib' and 'pickle' are supported."
            )

        try:
            if method == "joblib":
                joblib.dump(model, file_path)
            else:  # "pickle"
                with open(file_path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            raise IOError(f"Error saving model to {file_path}: {e}") from e

    @staticmethod
    def load_model(file_path: str, method: str = "joblib") -> Any:
        """
        Load ML model from file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If method is not supported.
            IOError: If load fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        if method not in {"joblib", "pickle"}:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Only 'joblib' and 'pickle' are supported."
            )

        try:
            if method == "joblib":
                return joblib.load(file_path)
            else:  # "pickle"
                with open(file_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            raise IOError(f"Error loading model from {file_path}: {e}") from e

    # ---------- JSON IO ----------

    @staticmethod
    def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
        """Save dict to JSON file."""
        ensure_dir(os.path.dirname(file_path))
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Error saving JSON to {file_path}: {e}") from e

    @staticmethod
    def load_json(file_path: str) -> Dict:
        """
        Load JSON file as dict.

        Raises:
            FileNotFoundError: If file does not exist.
            IOError: For read / decode errors.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Error loading JSON from {file_path}: {e}") from e

    @staticmethod
    def save_yaml(data: Dict, file_path: str, indent: int = 4) -> None:
        """Save dict to YAML file."""
        ensure_dir(os.path.dirname(file_path))
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, indent=indent, allow_unicode=True)
        except Exception as e:
            raise IOError(f"Error saving YAML to {file_path}: {e}") from e


# ===================== Report Generator =====================

class ReportGenerator:
    """
    Universal Report Generator - Tạo báo cáo cho nhiều mục đích.

    Hỗ trợ tạo report cho:
        - EDA: Thống kê dữ liệu, missing values, distributions
        - Data: Data versioning, quality, lineage
        - Training: Model metrics, comparison, feature importance
        - Comparison: So sánh nhiều experiments

    Formats hỗ trợ:
        - Markdown (.md)
        - JSON (.json)

    Attributes:
        output_dir (str): Thư mục lưu reports
        logger: Logger instance

    Example:
        >>> reporter = ReportGenerator('artifacts/reports')
        >>> reporter.generate_eda_report(df, 'eda_report')
        >>> reporter.generate_training_report(run_id, best_model, metrics)
    """

    def __init__(self, output_dir: str = "artifacts/reports", logger=None):
        """
        Khởi tạo ReportGenerator.

        Args:
            output_dir (str): Thư mục lưu reports
            logger: Logger instance (optional)
        """
        self.output_dir = output_dir
        self.logger = logger
        ensure_dir(output_dir)

    def _save_markdown(self, lines: list, filename: str) -> str:
        """Helper để lưu markdown report."""
        report_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if self.logger:
            self.logger.info(f"Report Generated | {filename}.md")
        return report_path

    def _save_json(self, data: Dict, filename: str) -> str:
        """Helper để lưu JSON report."""
        report_path = os.path.join(self.output_dir, f"{filename}.json")
        IOHandler.save_json(data, report_path)
        if self.logger:
            self.logger.info(f"Report Generated | {filename}.json")
        return report_path

    # ==================== EDA REPORT ====================

    def generate_eda_report(
        self,
        df: pd.DataFrame,
        filename: str = "eda_report",
        target_col: str = None,
        format: str = "markdown"
    ) -> str:
        """
        Tạo báo cáo Exploratory Data Analysis.

        Args:
            df (pd.DataFrame): DataFrame cần phân tích
            filename (str): Tên file output (không có extension)
            target_col (str): Tên cột target (optional)
            format (str): 'markdown' hoặc 'json'

        Returns:
            str: Đường dẫn file report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Thống kê cơ bản
        n_rows, n_cols = df.shape
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / (n_rows * n_cols)) * 100
        duplicates = df.duplicated().sum()

        # Phân loại columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Missing by column
        missing_by_col = df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

        if format == "json":
            report = {
                'timestamp': timestamp,
                'shape': {'rows': n_rows, 'columns': n_cols},
                'missing': {
                    'total': int(missing_total),
                    'percentage': round(missing_pct, 2),
                    'by_column': missing_by_col.to_dict()
                },
                'duplicates': int(duplicates),
                'column_types': {
                    'numeric': numeric_cols,
                    'categorical': categorical_cols
                },
                'numeric_stats': df[numeric_cols].describe().to_dict() if numeric_cols else {},
                'target_distribution': df[target_col].value_counts().to_dict() if target_col and target_col in df.columns else {}
            }
            return self._save_json(report, filename)

        # Markdown format
        lines = [
            f"# 📊 EDA Report",
            f"",
            f"**Generated:** {timestamp}",
            f"",
            f"---",
            f"",
            f"## 📋 Dataset Overview",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Rows | {n_rows:,} |",
            f"| Columns | {n_cols} |",
            f"| Missing Values | {missing_total:,} ({missing_pct:.2f}%) |",
            f"| Duplicate Rows | {duplicates:,} |",
            f"| Numeric Columns | {len(numeric_cols)} |",
            f"| Categorical Columns | {len(categorical_cols)} |",
            f""
        ]

        # Missing values by column
        if len(missing_by_col) > 0:
            lines.extend([
                f"## ❌ Missing Values by Column",
                f"",
                f"| Column | Missing Count | Missing % |",
                f"|--------|---------------|-----------|"
            ])
            for col, count in missing_by_col.head(10).items():
                pct = (count / n_rows) * 100
                lines.append(f"| {col} | {count:,} | {pct:.1f}% |")
            lines.append("")

        # Target distribution
        if target_col and target_col in df.columns:
            target_dist = df[target_col].value_counts()
            lines.extend([
                f"## 🎯 Target Distribution: `{target_col}`",
                f"",
                f"| Class | Count | Percentage |",
                f"|-------|-------|------------|"
            ])
            for cls, count in target_dist.items():
                pct = (count / n_rows) * 100
                lines.append(f"| {cls} | {count:,} | {pct:.1f}% |")
            lines.append("")

        # Numeric stats summary
        if numeric_cols:
            lines.extend([
                f"## 📈 Numeric Columns Summary",
                f"",
                f"| Column | Min | Max | Mean | Std |",
                f"|--------|-----|-----|------|-----|"
            ])
            for col in numeric_cols[:10]:
                stats = df[col].describe()
                lines.append(f"| {col} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['mean']:.2f} | {stats['std']:.2f} |")
            lines.append("")

        lines.extend([
            f"---",
            f"",
            f"*Report generated by ReportGenerator*"
        ])

        return self._save_markdown(lines, filename)

    # ==================== DATA REPORT ====================

    def generate_data_report(
        self,
        version_info: Dict,
        quality_info: Dict = None,
        lineage_info: Dict = None,
        filename: str = "data_report",
        format: str = "markdown"
    ) -> str:
        """
        Tạo báo cáo Data Versioning và Quality.

        Args:
            version_info (Dict): Thông tin version từ DataVersioning
            quality_info (Dict): Thông tin quality từ DataValidator
            lineage_info (Dict): Thông tin lineage (optional)
            filename (str): Tên file output
            format (str): 'markdown' hoặc 'json'

        Returns:
            str: Đường dẫn file report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if format == "json":
            report = {
                'timestamp': timestamp,
                'version': version_info,
                'quality': quality_info,
                'lineage': lineage_info
            }
            return self._save_json(report, filename)

        lines = [
            f"# 📦 Data Report",
            f"",
            f"**Generated:** {timestamp}",
            f"",
            f"---",
            f"",
            f"## 🔖 Version Information",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Version ID | `{version_info.get('version_id', 'N/A')}` |",
            f"| Timestamp | {version_info.get('timestamp', 'N/A')} |",
            f"| Hash | `{version_info.get('hash', 'N/A')[:16]}...` |",
            f"| File Size | {version_info.get('file_size_mb', 'N/A')} MB |",
            f""
        ]

        # Metadata
        if 'metadata' in version_info:
            meta = version_info['metadata']
            lines.extend([
                f"## 📊 Metadata",
                f"",
                f"| Property | Value |",
                f"|----------|-------|",
                f"| Rows | {meta.get('rows', 'N/A'):,} |",
                f"| Columns | {meta.get('columns', 'N/A')} |",
                f"| Memory | {meta.get('memory_mb', 'N/A')} MB |",
                f""
            ])

        # Quality
        if quality_info:
            lines.extend([
                f"## ✅ Data Quality",
                f"",
                f"| Metric | Value | Status |",
                f"|--------|-------|--------|",
                f"| Null Ratio | {quality_info.get('null_ratio', 0):.2%} | {'⚠️' if quality_info.get('null_ratio', 0) > 0.1 else '✅'} |",
                f"| Duplicate Ratio | {quality_info.get('duplicate_ratio', 0):.2%} | {'⚠️' if quality_info.get('duplicate_ratio', 0) > 0.05 else '✅'} |",
                f""
            ])

        # Lineage
        if lineage_info and lineage_info.get('parents'):
            lines.extend([
                f"## 🔗 Data Lineage",
                f"",
                f"```",
                f"{lineage_info.get('dataset', 'current')}"
            ])
            for parent in lineage_info.get('parents', []):
                lines.append(f"  ← {parent.get('name')} ({parent.get('transformation', 'unknown')})")
            lines.extend([f"```", f""])

        lines.extend([
            f"---",
            f"",
            f"*Report generated by ReportGenerator*"
        ])

        return self._save_markdown(lines, filename)

    # ==================== TRAINING REPORT ====================

    def generate_training_report(
        self,
        run_id: str,
        best_model_name: str,
        all_metrics: Dict[str, Dict[str, float]],
        feature_importance: pd.DataFrame = None,
        config: Dict = None,
        format: str = "markdown"
    ) -> str:
        """
        Tạo báo cáo Training.

        Args:
            run_id: ID của run
            best_model_name: Tên model tốt nhất
            all_metrics: Dict metrics của tất cả models
            feature_importance: DataFrame feature importance
            config: Config dict
            format: 'markdown' hoặc 'json'

        Returns:
            str: Đường dẫn file report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = f"report_{run_id}"

        if format == "json":
            report = {
                'run_id': run_id,
                'timestamp': timestamp,
                'best_model': {
                    'name': best_model_name,
                    'metrics': all_metrics.get(best_model_name, {})
                },
                'all_models': all_metrics,
                'feature_importance': feature_importance.to_dict('records') if feature_importance is not None and len(feature_importance) > 0 else [],
                'config_summary': {
                    'test_size': config.get('data', {}).get('test_size') if config else None,
                    'cv_folds': config.get('tuning', {}).get('cv_folds') if config else None,
                    'scoring': config.get('tuning', {}).get('scoring') if config else None,
                    'use_smote': config.get('preprocessing', {}).get('use_smote') if config else None
                }
            }
            return self._save_json(report, filename)

        lines = [
            f"# 🏆 Training Report",
            f"",
            f"**Run ID:** `{run_id}`  ",
            f"**Generated:** {timestamp}",
            f"",
            f"---",
            f"",
            f"## 🥇 Best Model",
            f"",
            f"**Model:** `{best_model_name.upper()}`",
            f""
        ]

        # Best model metrics
        if best_model_name in all_metrics:
            best_metrics = all_metrics[best_model_name]
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in best_metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {metric.upper()} | {value:.4f} |")
            lines.append("")

        # All models comparison
        lines.extend([
            f"## 📈 Model Comparison",
            f"",
            f"| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
            f"|-------|----------|-----------|--------|-----|---------|"
        ])

        for model_name, metrics in all_metrics.items():
            marker = " 🏆" if model_name == best_model_name else ""
            lines.append(
                f"| {model_name}{marker} | "
                f"{metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('precision', 0):.4f} | "
                f"{metrics.get('recall', 0):.4f} | "
                f"{metrics.get('f1', 0):.4f} | "
                f"{metrics.get('roc_auc', 0):.4f} |"
            )
        lines.append("")

        # Feature importance
        if feature_importance is not None and len(feature_importance) > 0:
            lines.extend([
                f"## 🎯 Top 10 Feature Importance",
                f"",
                f"| Rank | Feature | Importance |",
                f"|------|---------|------------|"
            ])
            for idx, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                lines.append(f"| {idx+1} | {row['feature']} | {row['importance']:.4f} |")
            lines.append("")

        # Config summary
        if config:
            lines.extend([
                f"## ⚙️ Configuration",
                f"",
                f"| Parameter | Value |",
                f"|-----------|-------|",
                f"| Test Size | {config.get('data', {}).get('test_size', 'N/A')} |",
                f"| Random State | {config.get('data', {}).get('random_state', 'N/A')} |",
                f"| CV Folds | {config.get('tuning', {}).get('cv_folds', 'N/A')} |",
                f"| Scoring | {config.get('tuning', {}).get('scoring', 'N/A')} |",
                f"| SMOTE | {'✅' if config.get('preprocessing', {}).get('use_smote') else '❌'} |",
                f""
            ])

        lines.extend([
            f"---",
            f"",
            f"*Report generated by ReportGenerator*"
        ])

        return self._save_markdown(lines, filename)

    # ==================== COMPARISON REPORT ====================

    def generate_comparison_report(
        self,
        run_ids: List[str],
        experiments_dir: str,
        metric: str = 'f1',
        filename: str = None
    ) -> str:
        """
        Tạo báo cáo so sánh nhiều runs.

        Args:
            run_ids: Danh sách run IDs
            experiments_dir: Thư mục experiments
            metric: Metric để so sánh
            filename: Tên file output

        Returns:
            str: Đường dẫn file report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if filename is None:
            filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        lines = [
            f"# 📊 Experiments Comparison",
            f"",
            f"**Generated:** {timestamp}",
            f"",
            f"## Runs Compared: {len(run_ids)}",
            f"",
            f"| Run ID | Best Model | {metric.upper()} |",
            f"|--------|------------|-------|"
        ]

        best_run = None
        best_score = -1

        for run_id in run_ids:
            metrics_file = os.path.join(experiments_dir, run_id, "metrics.json")
            if os.path.exists(metrics_file):
                metrics = IOHandler.load_json(metrics_file)

                run_best_model = None
                run_best_score = -1

                for model_name, model_metrics in metrics.items():
                    if isinstance(model_metrics, dict) and metric in model_metrics:
                        if model_metrics[metric] > run_best_score:
                            run_best_score = model_metrics[metric]
                            run_best_model = model_name

                if run_best_model:
                    if run_best_score > best_score:
                        best_score = run_best_score
                        best_run = run_id
                    lines.append(f"| {run_id} | {run_best_model} | {run_best_score:.4f} |")

        lines.extend([
            f"",
            f"## 🏆 Best Overall",
            f"",
            f"**Run:** `{best_run}`  ",
            f"**{metric.upper()}:** `{best_score:.4f}`",
            f"",
            f"---",
            f"",
            f"*Report generated by ReportGenerator*"
        ])

        return self._save_markdown(lines, filename)
