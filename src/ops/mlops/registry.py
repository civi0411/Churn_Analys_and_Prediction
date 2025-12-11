"""
Module `ops.mlops.registry` - quản lý lưu trữ và phiên bản mô hình (local registry).

Important keywords: Args, Methods, Returns
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime
from ...utils import IOHandler, ensure_dir


class ModelRegistry:
    """
    Hệ thống quản lý kho lưu trữ mô hình (Model Registry) cục bộ, hỗ trợ lưu trữ, đánh phiên bản và truy xuất mô hình.

    Methods:
        register_model: Đăng ký và lưu một mô hình mới vào kho.
        get_latest_model: Lấy mô hình mới nhất của một loại.
        get_model_by_version: Lấy mô hình theo phiên bản cụ thể.
        list_models: Liệt kê danh sách các mô hình đã đăng ký.
        get_best_model: Lấy mô hình tốt nhất dựa trên một chỉ số đánh giá (metric).
    """

    def __init__(self, registry_dir: str):
        """
        Khởi tạo ModelRegistry.

        Args:
            registry_dir (str): Đường dẫn thư mục gốc để lưu trữ các file mô hình và metadata registry.
        """
        self.registry_dir = registry_dir
        ensure_dir(self.registry_dir)
        self.registry_file = os.path.join(self.registry_dir, "registry.json")
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_file):
            self.registry = IOHandler.load_json(self.registry_file)
        else:
            self.registry = {}

    def register_model(self, model_name: str, model: Any, metrics: Dict[str, float],
                       run_id: str = None) -> str:
        """
        Lưu mô hình xuống đĩa và ghi nhận thông tin vào registry (metadata).

        Args:
            model_name (str): Tên định danh của loại mô hình (vd: 'RandomForest', 'ChurnModel').
            model (Any): Đối tượng mô hình đã huấn luyện (sklearn model, pipeline...).
            metrics (Dict[str, float]): Dictionary chứa các chỉ số đánh giá hiệu năng (vd: {'f1': 0.85}).
            run_id (str, optional): ID của lần chạy huấn luyện (Experiment Run ID) để truy vết nguồn gốc. Defaults to None.

        Returns:
            str: Đường dẫn tuyệt đối tới file mô hình đã được lưu.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = len(self.registry.get(model_name, [])) + 1

        filename = f"{model_name}_v{version}_{timestamp}.joblib"
        save_path = os.path.join(self.registry_dir, filename)
        IOHandler.save_model(model, save_path)

        info = {
            "version": version,
            "timestamp": timestamp,
            "path": save_path,
            "metrics": metrics,
            "run_id": run_id
        }

        if model_name not in self.registry:
            self.registry[model_name] = []
        self.registry[model_name].append(info)
        IOHandler.save_json(self.registry, self.registry_file)

        return save_path

    def get_latest_model(self, model_name: str) -> Optional[Any]:
        """
        Tải và trả về phiên bản mới nhất của một loại mô hình.

        Args:
            model_name (str): Tên loại mô hình cần lấy.

        Returns:
            Optional[Any]: Đối tượng mô hình mới nhất, hoặc None nếu chưa có mô hình nào được đăng ký.
        """
        if model_name not in self.registry or not self.registry[model_name]:
            return None

        latest = self.registry[model_name][-1]
        return IOHandler.load_model(latest["path"])

    def get_model_by_version(self, model_name: str, version: int) -> Optional[Any]:
        """
        Tải và trả về một mô hình dựa trên số phiên bản cụ thể.

        Args:
            model_name (str): Tên loại mô hình.
            version (int): Số thứ tự phiên bản (vd: 1, 2...).

        Returns:
            Optional[Any]: Đối tượng mô hình tương ứng, hoặc None nếu không tìm thấy.
        """
        if model_name not in self.registry:
            return None

        for info in self.registry[model_name]:
            if info["version"] == version:
                return IOHandler.load_model(info["path"])

        return None

    def list_models(self, model_name: str = None) -> Dict:
        """
        Liệt kê lịch sử và thông tin các phiên bản của mô hình.

        Args:
            model_name (str, optional): Tên loại mô hình cụ thể. Nếu None, trả về danh sách tất cả các loại. Defaults to None.

        Returns:
            Dict: Dictionary chứa thông tin metadata của các mô hình đã đăng ký.
        """
        if model_name:
            return {model_name: self.registry.get(model_name, [])}
        return self.registry

    def get_best_model(self, model_name: str, metric: str = "f1") -> Optional[Any]:
        """
        Tìm và tải mô hình có hiệu năng tốt nhất dựa trên một metric chỉ định.

        Args:
            model_name (str): Tên loại mô hình.
            metric (str, optional): Tên chỉ số dùng để so sánh (vd: 'f1', 'accuracy'). Defaults to "f1".

        Returns:
            Optional[Any]: Đối tượng mô hình tốt nhất, hoặc None nếu không tìm thấy.
        """
        if model_name not in self.registry or not self.registry[model_name]:
            return None

        best_info = max(
            self.registry[model_name],
            key=lambda x: x["metrics"].get(metric, 0)
        )

        return IOHandler.load_model(best_info["path"])
