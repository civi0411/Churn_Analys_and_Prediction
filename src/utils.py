"""
Tiện ích chung (utils) cho dự án: IO, logging, config, helper.

Tóm tắt:
- Chứa các hàm phục vụ file I/O, quản lý thời gian, seed, và ConfigLoader/Logger.

Important keywords: Args, Returns, Raises, Notes, Class
"""

import os, sys, yaml, json, logging, joblib, pickle, random, glob
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, cast
import numpy as np
import pandas as pd
import re

# Simple state to record last system message (global) to avoid duplicate separators across modules
_last_system_message: Dict[str, str] = {}
_last_system_message_global: str = ""

# Module-level logging format constants (used by get_logger and _add_run_handler)
LOG_FILE_TIME_FMT = "%Y-%m-%d %H:%M:%S"
LOG_CONSOLE_TIME_FMT = "%H:%M:%S"
LOG_SYSTEM_FMT = "[%(asctime)s] | %(levelname)-7s | %(name)-15s | %(message)s"
LOG_RUN_FMT = "[%(asctime)s] | %(levelname)-7s | %(message)s"
LOG_CONSOLE_FMT = "[%(asctime)s] | %(levelname)-7s | %(message)s"


# ===================== Helper functions =====================

def ensure_dir(dir_path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại.

    Args:
        dir_path (str): Đường dẫn thư mục (bỏ qua nếu rỗng).
    """
    if not dir_path:
        return
    os.makedirs(dir_path, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    """Đặt seed cho random và numpy để tái tạo kết quả.

    Args:
        seed (int, optional): Giá trị seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_timestamp() -> str:
    """
    Trả về timestamp theo format YYYYMMDD_HHMMSS.

    Returns:
        str: timestamp hiện tại.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_base_filename(file_path: str) -> str:
    """
    Lấy tên file không phần mở rộng (basename).

    Args:
        file_path (str): Đường dẫn file.

    Returns:
        str: Base filename (without extension).
    """
    return Path(file_path).stem


def build_processed_filename(raw_path: str, processed_dir: str) -> str:
    """
    Tạo đường dẫn file output cho bước xử lý dữ liệu dựa trên tên file gốc.

    Args:
        raw_path (str): Đường dẫn file dữ liệu thô.
        processed_dir (str): Thư mục chứa dữ liệu đã xử lý.

    Returns:
        str: Đường dẫn file output (VD: 'processed_dir/file_processed.csv').
    """
    base_name = extract_base_filename(raw_path)
    return os.path.join(processed_dir, f"{base_name}_processed.csv")


def build_split_filenames(raw_path: str, train_test_dir: str) -> Tuple[str, str]:
    """
    Tạo cặp đường dẫn file Train và Test dựa trên tên file gốc.

    Args:
        raw_path (str): Đường dẫn file gốc.
        train_test_dir (str): Thư mục chứa file Train/Test.

    Returns:
        Tuple[str, str]: Cặp đường dẫn (train_path, test_path).
    """
    base_name = extract_base_filename(raw_path)
    train_path = os.path.join(train_test_dir, f"{base_name}_train.parquet")
    test_path = os.path.join(train_test_dir, f"{base_name}_test.parquet")

    return train_path, test_path


def get_latest_train_test(train_test_dir: str, raw_filename: str = None) -> Tuple[str, str]:
    """
   Tự động tìm cặp file Train/Test mới nhất trong thư mục dựa trên timestamp trong tên file.

    Args:
        train_test_dir (str): Thư mục chứa các file Train/Test.
        raw_filename (str, optional): Tên file gốc để lọc (nếu có nhiều nguồn dữ liệu). Defaults to None.

    Returns:
        Tuple[str, str]: Đường dẫn tới file Train và Test mới nhất.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file nào phù hợp.
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
    Tính mã băm MD5 cho một file hoặc toàn bộ thư mục (để kiểm tra sự thay đổi dữ liệu).

    Args:
        path (str): Đường dẫn tới file hoặc thư mục.

    Returns:
        str: Mã hash MD5 (hex string).
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
    """
    Hỗ trợ tải và kiểm tra tính hợp lệ của file cấu hình YAML.

    Methods:
        load_config: Tải file YAML và trả về dictionary.
    """

    @staticmethod
    def load_config(config_path: str = "config/config.yaml") -> Dict:
        """
        Đọc file cấu hình YAML.

        Args:
            config_path (str, optional): Đường dẫn tới file config. Defaults to "config/config.yaml".

        Returns:
            Dict: Nội dung cấu hình.

        Raises:
            FileNotFoundError: Nếu file không tồn tại.
            ValueError: Nếu file rỗng hoặc không đúng định dạng.
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
    """
    Filter cho SYSTEM LOG (artifacts/logs/):
    - Ghi tất cả DEBUG (chi tiết kỹ thuật)
    - Ghi tất cả WARNING và ERROR
    - Chỉ ghi INFO cho các milestone quan trọng (tiêu đề các stage)

    Mục đích: Log kỹ thuật chi tiết cho dev/debug
    """

    def filter(self, record):
        msg = record.getMessage()

        # Keep all DEBUG, WARNING, ERROR, CRITICAL
        if record.levelno in (logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL):
            return True

        # Allow DEBUG records in system log (for diagnostics)
        if record.levelno == logging.DEBUG:
            return True

        # For INFO: chỉ giữ các milestone/stage headers
        if record.levelno == logging.INFO:
            milestone_keywords = [
                'STAGE:',  # Các giai đoạn pipeline
                'Pipeline',  # Khởi tạo/kết thúc pipeline
                'Initialized',  # Component initialization
                'Completed',  # Stage completion
                '=' * 60,  # Separator lines (headers)
                '=' * 70,
            ]
            return any(kw in msg for kw in milestone_keywords)

        return False


class SystemMsgCleaner(logging.Filter):
    """Bộ làm sạch thông điệp log: Loại bỏ dòng trống thừa, chuẩn hóa các dòng phân cách và khử trùng lặp liên tiếp."""

    def filter(self, record):
        try:
            msg = str(record.getMessage())
            # Collapse multiple newlines and trim
            msg = re.sub(r"\n\s*\n+", "\n", msg)
            msg = msg.strip()

            # Standardize separator-only messages to a canonical length
            if re.fullmatch(r"=+", msg):
                # choose 70-character separator
                msg = '=' * 70

            # If message empty after cleaning, skip
            if not msg:
                return False

            # Deduplicate consecutive same messages (global) to avoid repeated separators
            global _last_system_message_global
            if _last_system_message_global == msg:
                return False
            _last_system_message_global = msg

            # Replace the record message with cleaned version (keeps exc_info elsewhere)
            record.msg = msg
            record.args = ()
            return True
        except Exception:
            return True


class RunLogFilter(logging.Filter):
    """
    Filter cho RUN LOG (artifacts/experiments/<run_id>/run.log):
    - Ghi tất cả INFO (báo cáo tiến trình chi tiết)
    - Ghi WARNING với thông điệp rút gọn
    - Ghi ERROR với message chuyển hướng đến system log

    Mục đích: Báo cáo dễ đọc cho người dùng
    """

    IMPORTANT_INFO_KEYWORDS = [
        # === STAGE HEADERS ===
        '=' * 60,
        '=' * 70,
        'STAGE:',
        'PIPELINE',

        # === DATA OPERATIONS ===
        'Loaded data:',
        'Data Version',
        'Split:',
        'Saved:',
        'Data Quality',

        # === MODEL OPERATIONS ===
        'TRAINING RUN',
        'BEST MODEL:',
        '[EVALUATION]',
        'Model Registry',
        'Model Health:',

        # === VISUALIZATION ===
        'EDA Complete',
        'Visualization Completed',

        # === RESULTS ===
        'TRAINING RESULTS',
        'Best Model',
        'F1 Score',
        'ROC-AUC',
        'Accuracy',
        '[SUCCESS]',

        # === PREDICTIONS ===
        'Prediction saved:',
        'CRITICAL DRIFT',

        # === REPORTS ===
        'Report Generated',
    ]

    def filter(self, record):
        msg = record.getMessage()
        # === INFO: Chỉ ghi các actions quan trọng ===
        if record.levelno == logging.INFO:
            # Skip only clearly technical/irrelevant INFO
            skip_patterns = [
                'run_specific_dir not set',  # Internal warnings
                'Transformer State',  # Technical artifact saving
            ]

            if any(pattern in msg for pattern in skip_patterns):
                return False

            # Allow most INFO messages for run log (user-facing), keep it comprehensive
            return True

        # === WARNING: Rút gọn message ===
        if record.levelno == logging.WARNING:
            # Skip technical warnings
            if 'TreeExplainer' in msg or 'KernelExplainer' in msg:
                return False
            # Keep business warnings
            return True

        # For ERROR/CRITICAL: thay message thành 1 dòng redirect message that points to latest system log
        if record.levelno >= logging.ERROR:
            try:
                sys_path = getattr(Logger, '_latest_system_log', None) or 'artifacts/logs/'
                sys_path = os.path.abspath(sys_path)
                # Single-line redirect message
                record.msg = f"ERROR | Hệ thống gặp lỗi và không thể chạy tiếp. Xem: {sys_path}"
                record.args = ()  # Clear args to use modified msg
                # Remove exception traceback from run log (we redirect users to system log)
                if hasattr(record, 'exc_info'):
                    record.exc_info = None
                if hasattr(record, 'exc_text'):
                    record.exc_text = None
            except Exception:
                record.msg = "ERROR | Hệ thống gặp lỗi và không thể chạy tiếp. Xem system log."
                record.args = ()
            return True

        return False


class Logger:
    """
        2-Tier Logging System:

        Tier 1 - SYSTEM LOG (artifacts/logs/MAIN_<timestamp>.log):
            - Mục đích: Chi tiết kỹ thuật đầy đủ cho debugging
            - Nội dung:
              + Tất cả DEBUG messages
              + Tất cả WARNING/ERROR/CRITICAL
              + INFO chỉ cho milestone (tiêu đề stages)
            - Format: [timestamp] | LEVEL | MODULE | message
            - Audience: Developers, Technical debugging

        Tier 2 - RUN LOG (artifacts/experiments/<run_id>/run.log):
            - Mục đích: Báo cáo tiến trình dễ đọc cho người dùng
            - Nội dung:
              + Tất cả INFO messages (chi tiết hoạt động)
              + WARNING messages
              + ERROR chỉ hiện message chuyển hướng
            - Format: [timestamp] | LEVEL | message
            - Audience: End users, Data scientists

        """
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str, log_dir: str = None,
                   level: str = "INFO", run_log_path: str = None) -> logging.Logger:
        """
        Tạo logger với 2-tier architecture.

        Args:
            name: Tên logger (thường là 'MAIN')
            log_dir: Thư mục system logs (default: 'artifacts/logs')
            level: Log level (default: 'INFO')
            run_log_path: Đường dẫn run log (optional, set sau khi start run)

        Returns:
            logging.Logger: Configured logger instance
        """

        # Default log_dir
        if log_dir is None:
            log_dir = "artifacts/logs"

        # Singleton check
        if name in cls._loggers:
            logger = cls._loggers[name]
            if run_log_path:
                cls._add_run_handler(logger, run_log_path)  # Nếu có run_log_path mới, add handler
            return logger

        ensure_dir(log_dir)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        # System log: Full technical format (include logger name/module)
        system_formatter = logging.Formatter(LOG_SYSTEM_FMT, datefmt=LOG_FILE_TIME_FMT)

        # Run log: User-friendly format (no module name)
        run_formatter = logging.Formatter(LOG_RUN_FMT, datefmt=LOG_FILE_TIME_FMT)

        # Console: short time, aligned level
        console_formatter = logging.Formatter(LOG_CONSOLE_FMT, datefmt=LOG_CONSOLE_TIME_FMT)

        # === HANDLER 1: CONSOLE ===
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        # Normalize console output similarly to system logs
        console_handler.addFilter(SystemMsgCleaner())
        logger.addHandler(console_handler)

        # === HANDLER 2: SYSTEM LOG ===
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        system_log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        # Use utf-8-sig so PowerShell/Windows tools display unicode characters correctly
        system_file_handler = logging.FileHandler(system_log_file, encoding="utf-8-sig")
        system_file_handler.setLevel(logging.DEBUG)  # Capture all
        system_file_handler.setFormatter(system_formatter)
        # Attach cleaner first so messages are normalized before milestone filtering
        system_file_handler.addFilter(SystemMsgCleaner())  # Attach message cleaner
        system_file_handler.addFilter(SystemLogFilter())
        logger.addHandler(system_file_handler)

        # Record latest system log path for run-log redirect messages
        try:
            cls._latest_system_log = system_log_file
        except Exception:
            pass

        # === HANDLER 3: RUN LOG ===
        if run_log_path:
            cls._add_run_handler(logger, run_log_path)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def _add_run_handler(cls, logger: logging.Logger, run_log_path: str):
        """
        Thêm handler cho run-specific log.

        Args:
            logger: Logger instance
            run_log_path: Full path to run log file
        """
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == os.path.abspath(run_log_path):
                    return

        ensure_dir(os.path.dirname(run_log_path))

        # Create run formatter locally using module-level constants to ensure consistency
        run_formatter_local = logging.Formatter(LOG_RUN_FMT, datefmt=LOG_FILE_TIME_FMT)

        # Use utf-8-sig for run logs as well
        run_file_handler = logging.FileHandler(run_log_path, encoding="utf-8-sig")
        run_file_handler.setLevel(logging.INFO)
        run_file_handler.setFormatter(run_formatter_local)
        run_file_handler.addFilter(RunLogFilter())
        logger.addHandler(run_file_handler)

# ===================== IO Handler =====================

class IOHandler:
    """
    Lớp tiện ích xử lý các thao tác Nhập/Xuất (I/O) tập trung:
    - Đọc/Ghi dữ liệu bảng (CSV, Excel, Parquet).
    - Lưu/Tải mô hình (Joblib, Pickle).
    - Đọc/Ghi file cấu hình (JSON, YAML).

    Methods:
        read_data: Đọc dữ liệu từ file vào DataFrame.
        save_data: Lưu DataFrame xuống file.
        save_model: Lưu đối tượng mô hình.
        load_model: Tải đối tượng mô hình.
        save_json / load_json: Xử lý file JSON.
        save_yaml: Xử lý file YAML.
    """

    _SUPPORTED_EXT = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    @staticmethod
    def read_data(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ nhiều định dạng file khác nhau (.csv, .xlsx, .parquet).

        Args:
            file_path (str): Đường dẫn file dữ liệu.
            **kwargs: Các tham số phụ trợ cho hàm đọc của pandas (vd: sheet_name).

        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu.
        """
        import pandas as pd
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ['.xlsx', '.xls']:
            sheet_name = kwargs.get('sheet_name', None)
            try:
                if sheet_name is not None:
                    try:
                        return pd.read_excel(file_path, sheet_name=sheet_name)
                    except Exception:
                        # Nếu sheet_name không tồn tại, đọc sheet đầu tiên
                        return pd.read_excel(file_path, sheet_name=0)
                else:
                    return pd.read_excel(file_path, sheet_name=0)
            except Exception as e:
                raise IOError(f"Error reading file {file_path}: {e}") from e
        elif ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # NOTE: Multiple-file ingestion (multi-file / folder-based loaders) removed; use IOHandler.read_data for single-file ingestion

    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Lưu DataFrame xuống file với định dạng tương ứng đuôi file.

        Args:
            df (pd.DataFrame): Dữ liệu cần lưu.
            file_path (str): Đường dẫn đích.
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
        Tuần tự hóa (Serialize) và lưu mô hình máy học xuống đĩa.

        Args:
            model (Any): Đối tượng mô hình.
            file_path (str): Đường dẫn lưu file.
            method (str, optional): Phương thức lưu ('joblib' hoặc 'pickle'). Defaults to "joblib".
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
                    pickle.dump(model, cast(Any, f))
        except Exception as e:
            raise IOError(f"Error saving model to {file_path}: {e}") from e

    @staticmethod
    def load_model(file_path: str, method: str = "joblib") -> Any:
        """
        Tải mô hình đã lưu từ đĩa lên bộ nhớ.

        Args:
            file_path (str): Đường dẫn file mô hình.
            method (str, optional): Phương thức tải ('joblib' hoặc 'pickle'). Defaults to "joblib".

        Returns:
            Any: Đối tượng mô hình đã tải.
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
                    return pickle.load(cast(Any, f))
        except Exception as e:
            raise IOError(f"Error loading model from {file_path}: {e}") from e

    # ---------- JSON IO ----------

    @staticmethod
    def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
        """
        Ghi dữ liệu dictionary xuống file JSON với định dạng thụt lề (indent) dễ đọc.

        Args:
            data (Dict): Dữ liệu cần lưu.
            file_path (str): Đường dẫn file đích.
            indent (int, optional): Số khoảng trắng thụt đầu dòng
        """
        ensure_dir(os.path.dirname(file_path))
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Error saving JSON to {file_path}: {e}") from e

    @staticmethod
    def load_json(file_path: str) -> Dict:
        """
        Đọc nội dung từ file JSON và chuyển đổi thành Dictionary.

        Args:
            file_path (str): Đường dẫn file JSON cần đọc.

        Returns:
            Dict: Dữ liệu đã được tải.

        Raises:
            FileNotFoundError: Nếu file không tồn tại.
            IOError: Nếu xảy ra lỗi khi đọc hoặc giải mã JSON.
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
        """
        Ghi dữ liệu dictionary xuống file YAML (hỗ trợ Unicode).

        Args:
            data (Dict): Dữ liệu cần lưu.
            file_path (str): Đường dẫn file đích.
            indent (int, optional): Số khoảng trắng thụt đầu dòng. Defaults to 4.
        """
        ensure_dir(os.path.dirname(file_path))
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, indent=indent, allow_unicode=True)
        except Exception as e:
            raise IOError(f"Error saving YAML to {file_path}: {e}") from e


# Public API for `src/utils.py`
__all__ = [
    "ensure_dir",
    "set_random_seed",
    "get_timestamp",
    "extract_base_filename",
    "build_processed_filename",
    "build_split_filenames",
    "get_latest_train_test",
    "compute_file_hash",
    "get_files_in_folder",
    "filter_files_by_date",
    "ConfigLoader",
    "SystemLogFilter",
    "RunLogFilter",
    "Logger",
    "IOHandler",
]

