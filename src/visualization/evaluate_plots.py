"""
src/visualization/evaluate_plots.py
Model Evaluation & Performance Visualization Module

Module này tạo các biểu đồ đánh giá model performance:
    - Confusion Matrix: Ma trận nhầm lẫn với số lượng và tỷ lệ %
    - ROC Curve: So sánh ROC curves của nhiều models
    - Feature Importance: Bar chart top N features quan trọng
    - Model Comparison: Grouped bar chart so sánh metrics

Features:
    - Tự động lưu plots vào thư mục artifacts
    - Style nhất quán với seaborn
    - Hỗ trợ so sánh nhiều models trên cùng 1 plot

Output:
    Các file PNG được lưu vào: artifacts/figures/evaluation/
    - confusion_matrix_{model}.png
    - roc_curve.png
    - feature_importance_top_N.png
    - model_comparison.png

Example:
    >>> visualizer = EvaluateVisualizer(config, logger)
    >>> visualizer.plot_confusion_matrix(y_true, y_pred, 'xgboost')
    >>> visualizer.plot_roc_curve(roc_data_dict)
    >>> visualizer.plot_model_comparison(metrics_dict)

Author: Churn Prediction Team
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc
from ..utils import ensure_dir


class EvaluateVisualizer:
    """
    Evaluate Visualizer - Tạo các biểu đồ đánh giá model.

    Tất cả plots được tự động lưu vào thư mục artifacts/figures/evaluation/
    với DPI cao (300) cho quality tốt.

    Attributes:
        config (dict): Configuration dictionary
        logger: Logger instance
        save_dir (str): Thư mục lưu plots
        eval_dir (str): Thư mục evaluation cụ thể

    Example:
        >>> viz = EvaluateVisualizer(config, logger)
        >>> viz.plot_confusion_matrix(y_true, y_pred, 'xgboost')
        >>> viz.plot_roc_curve({'xgboost': (fpr, tpr, auc)})
    """

    def __init__(self, config: dict, logger=None, run_specific_dir=None):
        """
        Khởi tạo EvaluateVisualizer.

        Args:
            config (dict): Configuration dictionary chứa:
                - artifacts.figures_dir: Thư mục lưu figures
            logger: Logger instance (optional)
            run_specific_dir (str, optional): Thư mục cho run cụ thể
        """
        self.config = config
        self.logger = logger
        self.save_dir = run_specific_dir or config.get('artifacts', {}).get('figures_dir', 'artifacts/figures')
        self.eval_dir = os.path.join(self.save_dir, 'evaluation')
        ensure_dir(self.eval_dir)

        sns.set_style("whitegrid")

    def _save_plot(self, fig, filename: str):
        """
        Helper để lưu figure và log.

        Args:
            fig: Matplotlib figure
            filename (str): Tên file (sẽ lưu vào eval_dir)
        """
        path = os.path.join(self.eval_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Saved Plot      | EVALUATION | {filename}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """
        Vẽ Confusion Matrix với số lượng và tỷ lệ %.

        Hiển thị cả số lượng tuyệt đối và tỷ lệ phần trăm
        trong mỗi ô của confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name (str): Tên model (để đặt tên file)

        Output:
            Saves: confusion_matrix_{model_name}.png
        """
        cm = confusion_matrix(y_true, y_pred)

        # Calculate percentages
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100

        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax, cbar=False)

        ax.set_title(f"Confusion Matrix: {model_name}")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        self._save_plot(fig, f"confusion_matrix_{model_name}.png")

    def plot_roc_curve(self, roc_data_dict: Dict[str, tuple]):
        """
        Vẽ nhiều ROC Curves trên cùng 1 biểu đồ để so sánh.

        ROC (Receiver Operating Characteristic) curve thể hiện
        trade-off giữa True Positive Rate và False Positive Rate.

        Args:
            roc_data_dict (Dict[str, tuple]): Dictionary chứa ROC data
                Format: {'ModelName': (fpr, tpr, auc_score)}

        Output:
            Saves: roc_curve.png

        Example:
            >>> roc_data = {
            ...     'XGBoost': (fpr1, tpr1, 0.995),
            ...     'Random Forest': (fpr2, tpr2, 0.991)
            ... }
            >>> viz.plot_roc_curve(roc_data)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, (fpr, tpr, auc_score) in roc_data_dict.items():
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right")

        self._save_plot(fig, "roc_curve.png")

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n=20):
        """
        Vẽ Feature Importance dạng Horizontal Bar Chart.

        Hiển thị top N features quan trọng nhất với importance score.

        Args:
            importance_df (pd.DataFrame): DataFrame với columns ['feature', 'importance']
            top_n (int): Số features cần hiển thị (default: 20)

        Output:
            Saves: feature_importance_top_{top_n}.png
        """
        if importance_df is None or importance_df.empty: return

        df_plot = importance_df.head(top_n).sort_values('importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(df_plot['feature'], df_plot['importance'], color='teal')

        ax.set_title(f"Top {top_n} Feature Importance")
        ax.set_xlabel("Importance Score")

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)

        self._save_plot(fig, f"feature_importance_top_{top_n}.png")

    def plot_model_comparison(self, metrics_dict: Dict[str, Dict], metrics_to_plot=['accuracy', 'f1', 'recall']):
        """
        Vẽ Grouped Bar Chart so sánh metrics giữa các models.

        Mỗi model sẽ có một nhóm bars thể hiện các metrics khác nhau.

        Args:
            metrics_dict (Dict[str, Dict]): Dictionary chứa metrics của các models
                Format: {'ModelName': {'accuracy': 0.95, 'f1': 0.92, ...}}
            metrics_to_plot (List[str]): Danh sách metrics cần so sánh

        Output:
            Saves: model_comparison.png

        Example:
            >>> metrics = {
            ...     'XGBoost': {'accuracy': 0.978, 'f1': 0.935, 'recall': 0.942},
            ...     'Random Forest': {'accuracy': 0.961, 'f1': 0.888, 'recall': 0.915}
            ... }
            >>> viz.plot_model_comparison(metrics)
        """
        data = []
        for model_name, metrics in metrics_dict.items():
            for m in metrics_to_plot:
                if m in metrics:
                    data.append({
                        'Model': model_name,
                        'Metric': m.upper(),
                        'Score': metrics[m]
                    })

        if not data: return

        df_comp = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_comp, x='Model', y='Score', hue='Metric', palette='viridis', ax=ax)

        ax.set_title("Model Performance Comparison")
        ax.set_ylim(0, 1.1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add values on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

        self._save_plot(fig, "model_comparison.png")