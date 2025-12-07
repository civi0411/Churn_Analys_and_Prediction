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


    @staticmethod
    def read_batch_folder(folder_path: str, **kwargs) -> pd.DataFrame:
        """
        Read and merge all files in folder (Batch Processing).
        Auto-detect csv, xlsx, json, parquet.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Batch folder not found: {folder_path}")

        patterns = ["*.xlsx", "*.xls", "*.csv", "*.json", "*.parquet"]
        all_files = []
        for p in patterns:
            all_files.extend(glob.glob(os.path.join(folder_path, p)))
            all_files.extend(glob.glob(os.path.join(folder_path, p.upper())))

        all_files = sorted(list(set(all_files)))

        if not all_files:
            raise ValueError(f"Folder {folder_path} is empty or has no valid data files.")

        df_list = []
        print(f"   > Detected {len(all_files)} files in batch folder.")

        for f in all_files:
            if os.path.basename(f).startswith('~'):
                continue
            try:
                df = IOHandler.read_data(f, **kwargs)
                df_list.append(df)
            except Exception as e:
                print(f"   > [WARN] Skipping {os.path.basename(f)}: {e}")

        if not df_list:
            raise ValueError("Failed to read any files in batch.")

        return pd.concat(df_list, ignore_index=True)

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

