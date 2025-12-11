"""
Module `visualization.evaluate_plots` - biểu đồ đánh giá mô hình (confusion, ROC, feature importance, comparison).

Important keywords: Args, Returns, Methods
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
    Hỗ trợ vẽ các biểu đồ đánh giá mô hình và lưu ảnh.

    Methods:
        plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_model_comparison
    """

    def __init__(self, config: dict, logger=None, run_specific_dir=None):
        """Khởi tạo EvaluateVisualizer.

        Args:
            config (dict): cấu hình
            logger: logger instance
            run_specific_dir (str, optional): thư mục lưu cho run
        """
        self.config = config
        self.logger = logger
        if run_specific_dir is None:
            self.save_dir = None
            self.eval_dir = None
            if self.logger:
                self.logger.warning("EvaluateVisualizer: run_specific_dir not set, no figures will be saved.")
        else:
            self.save_dir = run_specific_dir
            self.eval_dir = os.path.join(self.save_dir, 'evaluation')
            ensure_dir(self.eval_dir)
        sns.set_style("whitegrid")

    def _save_plot(self, fig, filename: str):
        """Lưu figure vào thư mục evaluation.

        Args:
            fig: matplotlib figure
            filename: tên file để lưu
        """
        if not self.eval_dir:
            if self.logger:
                self.logger.warning(f"EvaluateVisualizer: eval_dir not set, cannot save {filename}")
            return
        path = os.path.join(self.eval_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Saved Plot      | EVAL       | {filename}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Vẽ confusion matrix và lưu ảnh.

        Args:
            y_true: nhãn thật
            y_pred: nhãn dự đoán
            model_name: tên model (dùng để đặt tên file)
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
        """Vẽ ROC curve so sánh nhiều model.

        Args:
            roc_data_dict: dict chứa (fpr, tpr, auc) cho mỗi model
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
        """Vẽ biểu đồ feature importance top N.

        Args:
            importance_df: DataFrame có columns ['feature','importance']
            top_n: số features hiển thị
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
        """So sánh metrics giữa các model và lưu ảnh.

        Args:
            metrics_dict: dict format {'Model': {'accuracy':..., 'f1':...}}
            metrics_to_plot: danh sách các metrics cần vẽ
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