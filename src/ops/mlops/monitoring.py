"""
Module `ops.mlops.monitoring` - ghi nhận hiệu năng mô hình và phát hiện drift hiệu năng.

Important keywords: Args, Methods, Returns
"""
import os
import hashlib
import pandas as pd
from typing import Dict
from datetime import datetime
from ...utils import ensure_dir


class ModelMonitor:
    """
    Monitor model performance in production.

    Methods:
        log_performance, get_performance_history, detect_drift, create_alert, check_health
    """

    def __init__(self, base_dir: str = "artifacts/monitoring"):
        self.base_dir = base_dir
        ensure_dir(base_dir)
        self.performance_log = os.path.join(base_dir, "performance_log.csv")

        if not os.path.exists(self.performance_log):
            pd.DataFrame(columns=[
                "timestamp", "model_name", "model_version",
                "accuracy", "precision", "recall", "f1", "roc_auc",
                "n_samples", "notes"
            ]).to_csv(self.performance_log, index=False)

    def log_performance(self, model_name: str, metrics: Dict[str, float],
                        model_version: str = None, n_samples: int = None,
                        notes: str = None):
        """Ghi nhận hiệu năng mô hình."""
        new_row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "model_version": model_version or "unknown",
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "n_samples": n_samples,
            "notes": notes
        }])

        try:
            if os.path.exists(self.performance_log):
                df = pd.read_csv(self.performance_log)
                if df.empty:
                    df = new_row
                else:
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
            df.to_csv(self.performance_log, index=False)
        except Exception as e:
            print(f"Error logging performance: {e}")

    def get_performance_history(self, model_name: str = None) -> pd.DataFrame:
        """Lấy lịch sử hiệu năng."""
        if not os.path.exists(self.performance_log):
            return pd.DataFrame()

        df = pd.read_csv(self.performance_log)

        if model_name:
            df = df[df["model_name"] == model_name]

        return df

    def detect_drift(self, model_name: str, metric: str = "f1",
                     threshold: float = 0.05) -> Dict:
        """Phát hiện drift hiệu năng."""
        history = self.get_performance_history(model_name)

        if len(history) < 2:
            return {"drift_detected": False, "message": "Not enough data"}

        baseline = history.iloc[0][metric]
        current = history.iloc[-1][metric]

        if pd.isna(baseline) or pd.isna(current):
            return {"drift_detected": False, "message": "Missing metric values"}

        drift = abs(current - baseline)
        drift_detected = drift > threshold

        return {
            "drift_detected": drift_detected,
            "baseline": float(baseline),
            "current": float(current),
            "drift": float(drift),
            "threshold": threshold,
            "message": f"Drift: {drift:.4f} ({'ALERT' if drift_detected else 'OK'})"
        }

    def create_alert(self, model_name: str, alert_type: str,
                     message: str, severity: str = "WARNING") -> bool:
        """Tạo cảnh báo khi phát hiện sự cố.

        Behaviors:
        - Append an alert row into `base_dir/alerts_log.csv`.
        - Avoid duplicate alerts: compute `alert_id` = md5(model|type|severity|message) and skip if exists.
        - Return True if alert is written, False if skipped or on failure.
        """
        ensure_dir(self.base_dir)

        timestamp = datetime.now().isoformat()
        alert = {
            "timestamp": timestamp,
            "model_name": model_name,
            "alert_type": alert_type,
            "severity": severity,
            "message": message
        }

        # stable id (exclude timestamp) so identical alerts deduplicate
        raw_id = f"{model_name}|{alert_type}|{severity}|{message}"
        alert_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
        alert["alert_id"] = alert_id

        alerts_log = os.path.join(self.base_dir, "alerts_log.csv")
        new_row = pd.DataFrame([alert])

        try:
            # Lightweight dedupe: if file exists and contains alert_id, skip
            if os.path.exists(alerts_log):
                try:
                    df_existing = pd.read_csv(alerts_log)
                except Exception:
                    df_existing = pd.DataFrame()

                if not df_existing.empty and 'alert_id' in df_existing.columns:
                    if (df_existing['alert_id'] == alert_id).any():
                        return False

            # Append row atomically-ish using to_csv mode='a'
            new_row.to_csv(alerts_log, mode='a', header=not os.path.exists(alerts_log), index=False)
            return True
        except Exception as e:
            print(f"Error logging alert: {e}")
            return False

    def check_health(self, model_name: str, current_metrics: Dict[str, float],
                     baseline_metrics: Dict[str, float] = None,
                     thresholds: Dict[str, float] = None) -> Dict:
        """Kiểm tra sức khỏe mô hình."""
        if thresholds is None:
            thresholds = {
                "f1_min": 0.70,
                "drift_max": 0.10,
                "accuracy_min": 0.75
            }

        issues = []
        recommendations = []
        status = "HEALTHY"

        for metric, threshold in thresholds.items():
            if metric.endswith("_min"):
                metric_name = metric.replace("_min", "")
                if metric_name in current_metrics:
                    if current_metrics[metric_name] < threshold:
                        issues.append(
                            f"{metric_name.upper()} dưới ngưỡng tối thiểu: "
                            f"{current_metrics[metric_name]:.3f} < {threshold}"
                        )
                        status = "WARNING"
                        recommendations.append(
                            "Xem xét việc huấn luyện lại với nhiều dữ liệu hơn hoặc kỹ thuật tạo đặc trưng"
                        )

        if baseline_metrics:
            for metric in ["f1", "accuracy", "roc_auc"]:
                if metric in current_metrics and metric in baseline_metrics:
                    drift = abs(current_metrics[metric] - baseline_metrics[metric])
                    if drift > thresholds.get("drift_max", 0.10):
                        issues.append(f"Phát hiện drift {metric.upper()}: {drift:.3f}")
                        status = "CRITICAL" if drift > 0.15 else "WARNING"
                        recommendations.append("Phát hiện drift mô hình. Khuyến nghị huấn luyện lại.")

                        self.create_alert(
                            model_name=model_name,
                            alert_type="drift",
                            message=f"Drift {metric.upper()}: {drift:.3f}",
                            severity="CRITICAL" if drift > 0.15 else "WARNING"
                        )

        return {
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
