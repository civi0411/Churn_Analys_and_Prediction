"""
src/ops/mlops/registry.py

Model Registry - Manage model versions.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime
from ...utils import IOHandler, ensure_dir


class ModelRegistry:
    """Manage model storage and versioning (Local)."""

    def __init__(self, registry_dir: str):
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
        """Register new model."""
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
        """Get latest model."""
        if model_name not in self.registry or not self.registry[model_name]:
            return None

        latest = self.registry[model_name][-1]
        return IOHandler.load_model(latest["path"])

    def get_model_by_version(self, model_name: str, version: int) -> Optional[Any]:
        """Get model by specific version."""
        if model_name not in self.registry:
            return None

        for info in self.registry[model_name]:
            if info["version"] == version:
                return IOHandler.load_model(info["path"])

        return None

    def list_models(self, model_name: str = None) -> Dict:
        """List all models."""
        if model_name:
            return {model_name: self.registry.get(model_name, [])}
        return self.registry

    def get_best_model(self, model_name: str, metric: str = "f1") -> Optional[Any]:
        """Get model with best metric."""
        if model_name not in self.registry or not self.registry[model_name]:
            return None

        best_info = max(
            self.registry[model_name],
            key=lambda x: x["metrics"].get(metric, 0)
        )

        return IOHandler.load_model(best_info["path"])
