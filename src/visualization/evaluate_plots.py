"""
src/visualization/evaluate_plots.py
Model Evaluation & Performance Visualization
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
    def __init__(self, config: dict, logger=None, run_specific_dir=None):
        self.config = config
        self.logger = logger
        self.save_dir = run_specific_dir or config.get('artifacts', {}).get('figures_dir', 'artifacts/figures')
        self.eval_dir = os.path.join(self.save_dir, 'evaluation')
        ensure_dir(self.eval_dir)

        sns.set_style("whitegrid")

    def _save_plot(self, fig, filename: str):
        path = os.path.join(self.eval_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Saved eval plot: {path}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Confusion Matrix đẹp với số lượng và %"""
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
        Vẽ nhiều ROC Curve trên cùng 1 biểu đồ để so sánh
        roc_data_dict: { 'ModelName': (fpr, tpr, auc_score) }
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

        self._save_plot(fig, "roc_curve_comparison.png")

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n=20):
        """Feature Importance - Horizontal Bar Chart"""
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

        self._save_plot(fig, "feature_importance.png")

    def plot_model_comparison(self, metrics_dict: Dict[str, Dict], metrics_to_plot=['accuracy', 'f1', 'recall']):
        """
        Grouped Bar Chart so sánh các metrics giữa các models
        metrics_dict: {'Logistic': {'accuracy': 0.8, ...}, 'RF': {...}}
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