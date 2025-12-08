"""
src/utils.py
Utility functions: Logging, IO, Hashing, Config, Filename parsing
"""
import os, sys, yaml, json, logging, joblib, pickle, random, glob
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
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

    Notes:
        Example usages:
            - get_files_in_folder('data/raw')
            - get_files_in_folder('data/raw', ['.csv', '.xlsx'])
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

    Notes:
        Example usages:
            - filter_files_by_date('data/raw/monthly', '2024-01', '2024-06')
            - filter_files_by_date('data/raw/quarterly', '2024-Q1', '2024-Q2')
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

class SystemLogFilter:
    """Filter cho System Log - giữ các milestone INFO + WARNING/ERROR"""

    def filter(self, record):
        msg = record.getMessage()

        # Keep all warnings and errors
        if record.levelno >= logging.WARNING:
            return True

        # Allow DEBUG records in system log (for diagnostics)
        if record.levelno == logging.DEBUG:
            return True

        # For INFO level, only allow a small set of milestone messages
        if record.levelno == logging.INFO:
            important_keywords = [
                'Report written',
                'Report Generated',
                'Model Registry',
                'SHAP Plot',
                'Saved Plot',
                'Saved:',
                'TRAINING RESULTS',
                'BEST MODEL',
            ]
            try:
                return any(kw in msg for kw in important_keywords)
            except Exception:
                return False

        return False


class RunLogFilter(logging.Filter):
    """
    Filter cho Run Log (Dành cho User):
    - CHỈ ghi INFO (full chi tiết)
    - KHÔNG ghi WARNING/DEBUG/ERROR (clean)
    """

    def filter(self, record):
        # Per-run log should capture INFO and above so operators see full run progress
        return record.levelno >= logging.INFO


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
        # Console should show normal INFO logs for operators; don't filter console.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(full_formatter)
        logger.addHandler(console_handler)

        # ===== HANDLER 2: SYSTEM LOG (Filtered INFO + WARNING+) =====
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Name system log file with the logger name (MAIN/TEST) so files are identifiable
        global_log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        global_file_handler = logging.FileHandler(global_log_file, encoding="utf-8")
        # Capture DEBUG + higher; filter will strip undesired INFO
        global_file_handler.setLevel(logging.DEBUG)
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

    # NOTE: Multiple-file ingestion (multi-file / folder-based loaders) removed; use IOHandler.read_data for single-file ingestion

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


# Lazy re-exports to avoid circular imports
def __getattr__(name: str):
    """
    Lazily import and expose symbols that live in other modules but are expected
    to be importable from `src.utils` (backwards-compatibility / test convenience).

    Currently supports:
      - ReportGenerator (from src.ops.reporting)
    """
    if name == 'ReportGenerator':
        # Local import to avoid circular import at module import time
        from .ops.reporting import ReportGenerator as _RG
        globals()['ReportGenerator'] = _RG
        return _RG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
