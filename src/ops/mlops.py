"""
src/ops/mlops.py
ML Operations: Experiment Tracking, Model Registry, Monitoring, Explainability
"""
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..utils import IOHandler, ensure_dir

# ==================== EXPERIMENT TRACKER (Custom) ====================

class ExperimentTracker:
    """
    Custom Experiment Tracker - Thay thế MLflow
    Lưu trữ experiments và runs vào CSV/JSON
    """

    def __init__(self, base_dir: str = "artifacts/experiments"):
        self.base_dir = base_dir
        self.experiments_file = os.path.join(base_dir, "experiments.csv")
        ensure_dir(base_dir)

        # Initialize experiments file if not exists
        if not os.path.exists(self.experiments_file):
            pd.DataFrame(columns=[
                'run_id', 'run_name', 'start_time', 'end_time',
                'status', 'duration_seconds'
            ]).to_csv(self.experiments_file, index=False)

        self.current_run_id = None
        self.current_run_dir = None
        self.run_start_time = None

    def start_run(self, run_name: str = None) -> str:
        """Bắt đầu một run mới"""
        self.run_start_time = datetime.now()
        if run_name is None:
           run_name = self.run_start_time.strftime('%Y%m%d_%H%M%S')
        self.current_run_id = run_name

        # Create run directory
        self.current_run_dir = os.path.join(self.base_dir, self.current_run_id)
        ensure_dir(self.current_run_dir)

        # Log to experiments file
        new_row = pd.DataFrame([{
            'run_id': self.current_run_id,
            'run_name': run_name,
            'start_time': self.run_start_time.isoformat(),
            'end_time': None,
            'status': 'RUNNING',
            'duration_seconds': None
        }])

        try:
            experiments_df = pd.read_csv(self.experiments_file)
            experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)
            experiments_df.to_csv(self.experiments_file, index=False)
        except Exception as e:
            print(f"Error logging run start: {e}")

        return self.current_run_id

    def end_run(self, status: str = "FINISHED"):
        """Kết thúc run hiện tại"""
        if self.current_run_id is None:
            return

        end_time = datetime.now()
        duration = (end_time - self.run_start_time).total_seconds()

        try:
            # Update experiments file
            experiments_df = pd.read_csv(self.experiments_file)
            mask = experiments_df['run_id'] == self.current_run_id
            experiments_df.loc[mask, 'end_time'] = end_time.isoformat()
            experiments_df.loc[mask, 'status'] = status
            experiments_df.loc[mask, 'duration_seconds'] = duration
            experiments_df.to_csv(self.experiments_file, index=False)
        except Exception as e:
            print(f"Error logging run end: {e}")

        self.current_run_id = None
        self.current_run_dir = None
        self.run_start_time = None

    def log_params(self, params: Dict[str, Any]):
        """Lưu parameters"""
        if self.current_run_dir is None:
            return

        params_file = os.path.join(self.current_run_dir, "params.json")
        IOHandler.save_json(params, params_file)

    def log_metrics(self, metrics: Dict[str, float]):
        """Lưu metrics"""
        if self.current_run_dir is None:
            return

        metrics_file = os.path.join(self.current_run_dir, "metrics.json")

        # Load existing metrics if any
        if os.path.exists(metrics_file):
            existing = IOHandler.load_json(metrics_file)
            existing.update(metrics)
            metrics = existing

        IOHandler.save_json(metrics, metrics_file)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Copy artifact vào run directory"""
        if self.current_run_dir is None or not os.path.exists(local_path):
            return

        import shutil
        artifacts_dir = os.path.join(self.current_run_dir, "artifacts")
        ensure_dir(artifacts_dir) # Ensure artifacts subdir exists

        try:
            if os.path.isfile(local_path):
                dest = os.path.join(artifacts_dir, os.path.basename(local_path))
                shutil.copy2(local_path, dest)
            elif os.path.isdir(local_path):
                dest = os.path.join(artifacts_dir, os.path.basename(local_path))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(local_path, dest)
        except Exception as e:
            print(f"Error logging artifact {local_path}: {e}")

    def get_run_info(self, run_id: str = None) -> Dict:
        """Lấy thông tin của một run"""
        if run_id is None:
            run_id = self.current_run_id

        if run_id is None:
            return {}

        run_dir = os.path.join(self.base_dir, run_id)

        info = {
            'run_id': run_id,
            'run_dir': run_dir
        }

        # Load params
        params_file = os.path.join(run_dir, "params.json")
        if os.path.exists(params_file):
            info['params'] = IOHandler.load_json(params_file)

        # Load metrics
        metrics_file = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_file):
            info['metrics'] = IOHandler.load_json(metrics_file)

        return info

    def log_metadata(self, metadata: Dict[str, Any]):
        """Log thông tin môi trường (git commit, Python version, packages...)"""
        if self.current_run_dir is None:
            return

        # Tự động thu thập metadata cơ bản
        import sys
        import platform

        auto_metadata = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'timestamp': datetime.now().isoformat(),
        }

        # Merge với user metadata
        full_metadata = {**auto_metadata, **metadata}

        metadata_file = os.path.join(self.current_run_dir, "metadata.json")
        IOHandler.save_json(full_metadata, metadata_file)


# ==================== MODEL REGISTRY ====================

class ModelRegistry:
    """Quản lý lưu trữ và phiên bản Model (Local)"""

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
        """Đăng ký model mới"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = len(self.registry.get(model_name, [])) + 1

        # Save physical model
        filename = f"{model_name}_v{version}_{timestamp}.joblib"
        save_path = os.path.join(self.registry_dir, filename)
        IOHandler.save_model(model, save_path)

        # Update registry metadata
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
        """Lấy model mới nhất"""
        if model_name not in self.registry or not self.registry[model_name]:
            return None

        latest = self.registry[model_name][-1]
        return IOHandler.load_model(latest['path'])


# ==================== MODEL MONITORING ====================

class ModelMonitor:
    """
    Monitor model performance trong production
    Track metrics theo thời gian
    """

    def __init__(self, base_dir: str = "artifacts/monitoring"):
        self.base_dir = base_dir
        ensure_dir(base_dir)
        self.performance_log = os.path.join(base_dir, "performance_log.csv")

        # Initialize log file
        if not os.path.exists(self.performance_log):
            pd.DataFrame(columns=[
                'timestamp', 'model_name', 'model_version',
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                'n_samples', 'notes'
            ]).to_csv(self.performance_log, index=False)

    def log_performance(self, model_name: str, metrics: Dict[str, float],
                       model_version: str = None, n_samples: int = None,
                       notes: str = None):
        """Ghi lại performance của model"""
        new_row = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_version': model_version or 'unknown',
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1': metrics.get('f1'),
            'roc_auc': metrics.get('roc_auc'),
            'n_samples': n_samples,
            'notes': notes
        }])

        try:
            df = pd.read_csv(self.performance_log)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.performance_log, index=False)
        except Exception as e:
            print(f"Error logging performance: {e}")

    def get_performance_history(self, model_name: str = None) -> pd.DataFrame:
        """Lấy lịch sử performance"""
        if not os.path.exists(self.performance_log):
            return pd.DataFrame()

        df = pd.read_csv(self.performance_log)

        if model_name:
            df = df[df['model_name'] == model_name]

        return df

    def detect_drift(self, model_name: str, metric: str = 'f1',
                    threshold: float = 0.05) -> Dict:
        """
        Phát hiện performance drift
        So sánh metric hiện tại với baseline (giá trị đầu tiên)
        """
        history = self.get_performance_history(model_name)

        if len(history) < 2:
            return {'drift_detected': False, 'message': 'Not enough data'}

        baseline = history.iloc[0][metric]
        current = history.iloc[-1][metric]

        if pd.isna(baseline) or pd.isna(current):
            return {'drift_detected': False, 'message': 'Missing metric values'}

        drift = abs(current - baseline)
        drift_detected = drift > threshold

        return {
            'drift_detected': drift_detected,
            'baseline': float(baseline),
            'current': float(current),
            'drift': float(drift),
            'threshold': threshold,
            'message': f"Drift: {drift:.4f} ({'ALERT' if drift_detected else 'OK'})"
        }

    def create_alert(self, model_name: str, alert_type: str,
                     message: str, severity: str = "WARNING") -> None:
        """
        Tạo cảnh báo khi phát hiện vấn đề

        Args:
            alert_type: 'drift', 'performance_drop', 'data_quality'
            severity: 'INFO', 'WARNING', 'CRITICAL'
        """
        alerts_dir = os.path.join(self.base_dir, "alerts")
        ensure_dir(alerts_dir)

        alert = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'alert_type': alert_type,
            'severity': severity,
            'message': message
        }

        # Lưu vào file riêng
        alert_file = os.path.join(
            alerts_dir,
            f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        IOHandler.save_json(alert, alert_file)

        # Log vào file tổng hợp
        alerts_log = os.path.join(self.base_dir, "alerts_log.csv")
        new_row = pd.DataFrame([alert])

        try:
            if os.path.exists(alerts_log):
                df = pd.read_csv(alerts_log)
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
            df.to_csv(alerts_log, index=False)
        except Exception as e:
            print(f"Error logging alert: {e}")

    def check_health(self, model_name: str, current_metrics: Dict[str, float],
                     baseline_metrics: Dict[str, float] = None,
                     thresholds: Dict[str, float] = None) -> Dict:
        """
        Kiểm tra sức khỏe model

        Returns:
            {
                'status': 'HEALTHY' | 'WARNING' | 'CRITICAL',
                'issues': [...],
                'recommendations': [...]
            }
        """
        if thresholds is None:
            thresholds = {
                'f1_min': 0.70,  # F1 tối thiểu
                'drift_max': 0.10,  # Drift tối đa 10%
                'accuracy_min': 0.75
            }

        issues = []
        recommendations = []
        status = 'HEALTHY'

        # Check 1: Metrics dưới ngưỡng
        for metric, threshold in thresholds.items():
            if metric.endswith('_min'):
                metric_name = metric.replace('_min', '')
                if metric_name in current_metrics:
                    if current_metrics[metric_name] < threshold:
                        issues.append(
                            f"{metric_name.upper()} below threshold: {current_metrics[metric_name]:.3f} < {threshold}")
                        status = 'WARNING'
                        recommendations.append(f"Consider retraining with more data or feature engineering")

        # Check 2: Drift detection
        if baseline_metrics:
            for metric in ['f1', 'accuracy', 'roc_auc']:
                if metric in current_metrics and metric in baseline_metrics:
                    drift = abs(current_metrics[metric] - baseline_metrics[metric])
                    if drift > thresholds.get('drift_max', 0.10):
                        issues.append(f"{metric.upper()} drift detected: {drift:.3f}")
                        status = 'CRITICAL' if drift > 0.15 else 'WARNING'
                        recommendations.append(f"Model drift detected. Retrain recommended.")

                        # Tạo alert
                        self.create_alert(
                            model_name=model_name,
                            alert_type='drift',
                            message=f"{metric.upper()} drift: {drift:.3f}",
                            severity='CRITICAL' if drift > 0.15 else 'WARNING'
                        )

        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }


# ==================== MODEL EXPLAINABILITY ====================

class ModelExplainer:
    """
    Model Explainability - SHAP & Feature Importance
    """

    def __init__(self, model, X_train, feature_names: List[str]):
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            self.model = model.named_steps['model']
        else:
            self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Lấy feature importance từ model"""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    def explain_with_shap(self, X_sample, save_path: str = None):
        """
        Generate SHAP explanations
        Requires: pip install shap
        """
        try:
            import shap
            import matplotlib.pyplot as plt

            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

            return shap_values

        except ImportError:
            print("[WARN] SHAP not installed. Run: pip install shap")
            return None
        except Exception as e:
            print(f"[ERROR] SHAP explanation failed: {e}")
            return None