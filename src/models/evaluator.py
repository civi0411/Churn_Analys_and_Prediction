"""
src/models/evaluator.py

Model Evaluation Module

Module này chịu trách nhiệm đánh giá hiệu năng của ML models:
    - Classification metrics: Accuracy, Precision, Recall, F1-Score
    - ROC-AUC score và ROC curve data
    - Confusion matrix
    - Classification report

Metrics được tính toán:
    - Accuracy: Tỷ lệ dự đoán đúng
    - Precision: TP / (TP + FP) - Độ chính xác khi dự đoán positive
    - Recall: TP / (TP + FN) - Khả năng tìm đúng positive
    - F1-Score: Harmonic mean của Precision và Recall
    - ROC-AUC: Area Under ROC Curve - Khả năng phân biệt classes

Example:
    >>> evaluator = ModelEvaluator(logger)
    >>> result = evaluator.evaluate(model, X_test, y_test, 'xgboost')
    >>> print(result['metrics'])
    {'accuracy': 0.978, 'precision': 0.927, 'recall': 0.942, 'f1': 0.935, 'roc_auc': 0.995}

Author: Churn Prediction Team
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score
)


class ModelEvaluator:
    """
    Model Evaluator - Đánh giá hiệu năng classification models.

    Tính toán comprehensive metrics cho binary classification:
        - Basic metrics: Accuracy, Precision, Recall, F1
        - Probability-based: ROC-AUC, ROC curve
        - Detailed: Confusion matrix, Classification report

    Attributes:
        logger: Logger instance để log kết quả

    Example:
        >>> evaluator = ModelEvaluator(logger)
        >>> result = evaluator.evaluate(trained_model, X_test, y_test, 'xgboost')
        >>> print(f"F1: {result['metrics']['f1']:.4f}")
    """

    def __init__(self, logger=None):
        """
        Khởi tạo ModelEvaluator.

        Args:
            logger: Logger instance (optional). Nếu có, sẽ log
                   kết quả evaluation với format đẹp.
        """
        self.logger = logger

    def evaluate(self, model: Any, X_test, y_test, model_name: str = 'model') -> Dict[str, Any]:
        """
        Tính toán toàn bộ evaluation metrics cho một model.

        Args:
            model: Trained sklearn estimator (phải có predict() method)
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target (ground truth)
            model_name (str): Tên model để logging

        Returns:
            Dict[str, Any]: Dictionary chứa:
                - metrics (Dict): {accuracy, precision, recall, f1, roc_auc}
                - y_pred (np.array): Predictions
                - y_pred_proba (np.array): Probability predictions (nếu có)
                - classification_report (str): Sklearn classification report
                - confusion_matrix (np.array): Confusion matrix 2x2
                - roc_curve_data (Tuple): (fpr, tpr, auc) cho plotting

        Note:
            - ROC-AUC chỉ được tính nếu model có predict_proba()
            - Với models không có predict_proba (như SVM linear),
              roc_auc và roc_curve_data sẽ là None

        Example:
            >>> result = evaluator.evaluate(model, X_test, y_test, 'xgboost')
            >>> print(f"Accuracy: {result['metrics']['accuracy']:.2%}")
            >>> print(f"Confusion Matrix:\\n{result['confusion_matrix']}")
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