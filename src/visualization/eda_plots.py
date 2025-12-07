"""
src/visualization/eda_plots.py
Advanced Exploratory Data Analysis Visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Optional
from ..utils import ensure_dir


class EDAVisualizer:
    def __init__(self, config: dict, logger=None, run_specific_dir=None):
        self.config = config
        self.logger = logger
        # Nếu có run_dir thì lưu vào đó, nếu không thì lưu vào thư mục chung
        self.save_dir = run_specific_dir or config.get('artifacts', {}).get('figures_dir', 'artifacts/figures')
        self.eda_dir = os.path.join(self.save_dir, 'eda')
        ensure_dir(self.eda_dir)

        # Setup Style
        sns.set_palette("muted")
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10})

    def _save_plot(self, fig, filename: str):
        path = os.path.join(self.eda_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Saved plot: {path}")

    def plot_target_distribution(self, y: pd.Series):
        """Vẽ biểu đồ Donut Chart cho Target"""
        fig, ax = plt.subplots(figsize=(6, 6))

        counts = y.value_counts()
        labels = counts.index
        sizes = counts.values

        # Donut Chart
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85, explode=[0.05] * len(labels),
            colors=sns.color_palette("pastel")
        )

        # Draw circle in the middle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        ax.set_title(f"Target Distribution: {y.name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_plot(fig, "target_distribution_donut.png")

    def plot_missing_values(self, df: pd.DataFrame):
        """Biểu đồ cột thể hiện % Missing Data"""
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            return

        missing_pct = (missing / len(df)) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="viridis", ax=ax)

        ax.set_title("Percentage of Missing Values by Feature", fontsize=14)
        ax.set_xlabel("Missing Percentage (%)")

        # Add text labels
        for i, v in enumerate(missing_pct.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center')

        self._save_plot(fig, "missing_values.png")

    def plot_numerical_distributions(self, df: pd.DataFrame, numerical_cols: List[str]):
        """Vẽ Hist + KDE cho các cột số"""
        if not numerical_cols: return

        # Vẽ từng file riêng hoặc grid lớn (ở đây chọn grid 3 cột)
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Dist: {col}")

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._save_plot(fig, "numerical_distributions.png")

    def plot_categorical_vs_target(self, df: pd.DataFrame, cat_cols: List[str], target_col: str):
        """Vẽ Stacked Bar Chart (100%) so sánh tỷ lệ Churn"""
        if not cat_cols or target_col not in df.columns: return

        for col in cat_cols:
            if df[col].nunique() > 10: continue  # Skip if too many categories

            ct = pd.crosstab(df[col], df[target_col], normalize='index')

            fig, ax = plt.subplots(figsize=(8, 5))
            ct.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn_r', alpha=0.8)

            ax.set_title(f"Churn Rate by {col}")
            ax.set_ylabel("Proportion")
            plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            self._save_plot(fig, f"cat_vs_target_{col}.png")

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Heatmap correlation (Chỉ hiện tam giác dưới)"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2: return

        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False
        )

        ax.set_title("Feature Correlation Matrix", fontsize=16)
        plt.tight_layout()
        self._save_plot(fig, "correlation_matrix.png")

    def plot_outliers_boxplot(self, df: pd.DataFrame, num_cols: List[str]):
        """Boxplot để soi outliers"""
        if not num_cols: return

        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            sns.boxplot(y=df[col], ax=axes[i], color='orange')
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._save_plot(fig, "outliers_boxplot.png")