"""
Module `models.evaluator` - đánh giá hiệu năng mô hình phân loại.

Important keywords: Args, Returns, Methods
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score
)


class ModelEvaluator:
    """
    Class: ModelEvaluator
    Chịu trách nhiệm tính toán các chỉ số đánh giá (metrics) cho mô hình phân loại.

    Methods:
        evaluate(model, X_test, y_test, model_name): Thực hiện đánh giá toàn diện và trả về kết quả.
    """

    def __init__(self, logger=None):
        """
        Constructor: __init__
        Khởi tạo ModelEvaluator.

        Args:
            logger (logging.Logger, optional): Đối tượng logger để ghi lại kết quả đánh giá. Defaults to None.
        """
        self.logger = logger

    def evaluate(self, model: Any, X_test, y_test, model_name: str = 'model') -> Dict[str, Any]:
        """
        Method: evaluate
        Tính toán các metrics cơ bản và trả về dictionary chứa kết quả chi tiết.

        Args:
            model (Any): Mô hình đã được huấn luyện (cần có method .predict, và tùy chọn .predict_proba).
            X_test (pd.DataFrame or np.ndarray): Dữ liệu đặc trưng (features) tập kiểm thử.
            y_test (pd.Series or np.ndarray): Nhãn thực tế (true labels) tập kiểm thử.
            model_name (str, optional): Tên mô hình để hiển thị trong log. Defaults to 'model'.

        Returns:
            Dict[str, Any]: Dictionary chứa kết quả đánh giá, bao gồm các keys:
                - 'metrics': Dict các chỉ số (accuracy, precision, recall, f1, roc_auc).
                - 'y_pred': Mảng dự đoán nhãn.
                - 'y_pred_proba': Mảng dự đoán xác suất (nếu có).
                - 'classification_report': Báo cáo chi tiết dạng text.
                - 'confusion_matrix': Ma trận nhầm lẫn.
                - 'roc_curve_data': Tuple (fpr, tpr, auc) để vẽ biểu đồ ROC.
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        roc_curve_data = None
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_curve_data = (fpr, tpr, metrics['roc_auc'])

        # Tổng hợp kết quả
        result = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve_data': roc_curve_data
        }

        if self.logger:
            self.logger.info(f"[EVALUATION] {model_name.upper()}")
            log_msg = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"  {log_msg}")

        return result