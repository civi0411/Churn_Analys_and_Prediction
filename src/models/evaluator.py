"""
src/models/evaluator.py
Nhiệm vụ: Tính toán metrics, confusion matrix, ROC curve
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score
)


class ModelEvaluator:
    """Class chuyên trách đánh giá hiệu năng model"""

    def __init__(self, logger=None):
        self.logger = logger

    def evaluate(self, model: Any, X_test, y_test, model_name: str = 'model') -> Dict[str, Any]:
        """
        Tính toán toàn bộ metrics cho một model

        Returns:
            Dictionary chứa metrics, confusion matrix, report, roc_data
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