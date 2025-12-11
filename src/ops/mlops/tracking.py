"""
Module `ops.mlops.tracking` - ExperimentTracker (đơn giản, file-based experiment tracking).

Important keywords: Args, Methods, Returns, Notes
"""
import os
import sys
import platform
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ...utils import IOHandler, ensure_dir


class ExperimentTracker:
    """
    Hệ thống theo dõi thí nghiệm (Experiment Tracking) thay thế đơn giản cho MLflow, sử dụng file CSV và JSON để lưu trữ log.

    Methods:
        start_run: Bắt đầu một phiên chạy (run) mới.
        end_run: Kết thúc phiên chạy hiện tại.
        log_params: Ghi lại các tham số (parameters) của thí nghiệm.
        log_metrics: Ghi lại các chỉ số (metrics) của thí nghiệm.
        log_artifact: Lưu trữ các file kết quả (model, plot, data).
        get_run_info: Lấy thông tin chi tiết về một phiên chạy.
        log_metadata: Ghi lại thông tin môi trường và hệ thống.
    """

    def __init__(self, base_dir: str = "artifacts/experiments"):
        """
        Khởi tạo ExperimentTracker, thiết lập thư mục gốc và file log tổng hợp.

        Args:
            base_dir (str, optional): Đường dẫn thư mục gốc để lưu trữ các thí nghiệm. Defaults to "artifacts/experiments".
        """
        self.base_dir = base_dir
        self.experiments_file = os.path.join(base_dir, "experiments.csv")
        ensure_dir(base_dir)

        if not os.path.exists(self.experiments_file):
            # Create empty dataframe with object dtype for string columns to avoid dtype coercion later
            cols = ["run_id", "run_name", "start_time", "end_time", "status", "duration_seconds"]
            empty = {c: pd.Series(dtype='object') for c in cols}
            pd.DataFrame(empty).to_csv(self.experiments_file, index=False)

        self.current_run_id = None
        self.current_run_dir = None
        self.run_start_time = None

    def start_run(self, run_name: str = None) -> str:
        """
        Bắt đầu một phiên chạy (run) mới, tạo thư mục riêng cho run và ghi nhận vào lịch sử.

        Args:
            run_name (str, optional): Tên định danh cho run. Nếu None, tự động tạo theo timestamp. Defaults to None.

        Returns:
            str: ID của run vừa khởi tạo.
        """
        self.run_start_time = datetime.now()
        if run_name is None:
            run_name = self.run_start_time.strftime("%Y%m%d_%H%M%S")
        self.current_run_id = run_name

        self.current_run_dir = os.path.join(self.base_dir, self.current_run_id)
        ensure_dir(self.current_run_dir)

        new_row = pd.DataFrame([{
            "run_id": self.current_run_id,
            "run_name": run_name,
            "start_time": self.run_start_time.isoformat(),
            "end_time": None,
            "status": "RUNNING",
            "duration_seconds": None
        }])

        try:
            experiments_df = pd.read_csv(self.experiments_file, dtype=str)
            # Align new_row columns with experiments_df to avoid dtype issues
            new_row = new_row.reindex(columns=experiments_df.columns)
            # Append by direct assignment (avoid pd.concat warnings)
            row_dict = new_row.iloc[0].to_dict()
            experiments_df.loc[len(experiments_df)] = row_dict
            experiments_df.to_csv(self.experiments_file, index=False)
        except Exception as e:
            print(f"Error logging run start: {e}")

        return self.current_run_id

    def end_run(self, status: str = "FINISHED"):
        """
        Kết thúc phiên chạy hiện tại, tính toán thời gian chạy và cập nhật trạng thái.

        Args:
            status (str, optional): Trạng thái kết thúc (vd: 'FINISHED', 'FAILED'). Defaults to "FINISHED".
        """
        if self.current_run_id is None:
            return

        end_time = datetime.now()
        duration = (end_time - self.run_start_time).total_seconds()

        try:
            experiments_df = pd.read_csv(self.experiments_file, dtype=str)
            # Ensure time columns are treated as object (strings) to avoid setting incompatible dtypes
            for tcol in ["start_time", "end_time"]:
                if tcol in experiments_df.columns:
                    experiments_df[tcol] = experiments_df[tcol].astype(object)
            mask = experiments_df["run_id"] == self.current_run_id
            experiments_df.loc[mask, "end_time"] = end_time.isoformat()
            experiments_df.loc[mask, "status"] = status
            experiments_df.loc[mask, "duration_seconds"] = duration
            experiments_df.to_csv(self.experiments_file, index=False)
        except Exception as e:
            print(f"Error logging run end: {e}")

        self.current_run_id = None
        self.current_run_dir = None
        self.run_start_time = None

    def log_params(self, params: Dict[str, Any]):
        """
        Lưu các tham số cấu hình (hyperparameters, config) vào file JSON trong thư mục run.

        Args:
            params (Dict[str, Any]): Dictionary chứa các tham số cần log.
        """
        if self.current_run_dir is None:
            return

        params_file = os.path.join(self.current_run_dir, "params.json")
        IOHandler.save_json(params, params_file)

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Lưu các chỉ số đánh giá (metrics) vào file JSON. Nếu file đã tồn tại, sẽ cập nhật thêm.

        Args:
            metrics (Dict[str, float]): Dictionary chứa các metrics (vd: accuracy, loss).
        """
        if self.current_run_dir is None:
            return

        metrics_file = os.path.join(self.current_run_dir, "metrics.json")

        if os.path.exists(metrics_file):
            existing = IOHandler.load_json(metrics_file)
            existing.update(metrics)
            metrics = existing

        IOHandler.save_json(metrics, metrics_file)

    def get_run_info(self, run_id: str = None) -> Dict:
        """
        Truy xuất toàn bộ thông tin đã log của một run (params, metrics).

        Args:
            run_id (str, optional): ID của run cần lấy thông tin. Nếu None, lấy run hiện tại. Defaults to None.

        Returns:
            Dict: Dictionary chứa thông tin chi tiết của run.
        """
        if run_id is None:
            run_id = self.current_run_id

        if run_id is None:
            return {}

        run_dir = os.path.join(self.base_dir, run_id)

        info = {
            "run_id": run_id,
            "run_dir": run_dir
        }

        params_file = os.path.join(run_dir, "params.json")
        if os.path.exists(params_file):
            info["params"] = IOHandler.load_json(params_file)

        metrics_file = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_file):
            info["metrics"] = IOHandler.load_json(metrics_file)

        return info

    def log_metadata(self, metadata: Dict[str, Any]):
        """
        Ghi lại thông tin môi trường hệ thống (Python version, OS, Timestamp) và các metadata tùy chỉnh khác.

        Args:
            metadata (Dict[str, Any]): Thông tin bổ sung cần log.
        """
        if self.current_run_dir is None:
            return

        auto_metadata = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
        }

        full_metadata = {**auto_metadata, **metadata}

        metadata_file = os.path.join(self.current_run_dir, "metadata.json")
        IOHandler.save_json(full_metadata, metadata_file)
