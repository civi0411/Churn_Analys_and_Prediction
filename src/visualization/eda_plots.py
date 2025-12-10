"""
src/visualization/eda_plots.py

Exploratory Data Analysis Visualization Module

Tạo các biểu đồ EDA:
    1. missing_values.png - Missing values bar chart
    2. target_distribution.png - Target pie chart
    3. numerical_distributions.png - Histogram + KDE (grid)
    4. boxplots.png - Boxplots cho outliers (grid)
    5. correlation_matrix.png - Heatmap correlation
    6. correlation_with_target.png - Top features với target
    7. churn_by_category.png - Churn rate theo categories (grid)

Example:
    >>> visualizer = EDAVisualizer(config, logger)
    >>> visualizer.run_full_eda(df, target_col='Churn')

Author: Churn Prediction Team
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List
from ..utils import ensure_dir


class EDAVisualizer:
    """
    EDA Visualizer - Tạo các biểu đồ phân tích dữ liệu.

    Outputs:
        - missing_values.png: Bar chart % missing
        - target_distribution.png: Pie chart target
        - numerical_distributions.png: Histograms
        - boxplots.png: Outlier detection
        - correlation_matrix.png: Heatmap
        - correlation_with_target.png: Top features
        - churn_by_category.png: Churn rate by category

    Attributes:
        config (dict): Configuration
        logger: Logger instance
        eda_dir (str): Output directory

    Example:
        >>> viz = EDAVisualizer(config, logger)
        >>> viz.run_full_eda(df, 'Churn')
    """

    def __init__(self, config: dict, logger=None, run_specific_dir=None):
        """
        Khởi tạo EDAVisualizer.

        Args:
            config: Config dict (artifacts.figures_dir)
            logger: Logger instance
            run_specific_dir: Thư mục run cụ thể (optional)
        """
        self.config = config
        self.logger = logger
        if run_specific_dir is None:
            self.save_dir = None
            self.eda_dir = None
            if self.logger:
                self.logger.warning("EDAVisualizer: run_specific_dir not set, no figures will be saved.")
        else:
            self.save_dir = run_specific_dir
            self.eda_dir = os.path.join(self.save_dir, 'eda')
            ensure_dir(self.eda_dir)

        sns.set_palette("muted")
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10})

    def _save_plot(self, fig, filename: str):
        """Lưu figure và log."""
        if not self.eda_dir:
            if self.logger:
                self.logger.warning(f"EDAVisualizer: eda_dir not set, cannot save {filename}")
            return
        path = os.path.join(self.eda_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Saved Plot      | EDA        | {filename}")

    # ==================== 1. MISSING VALUES ====================

    def plot_missing_values(self, df: pd.DataFrame):
        """
        Bar chart hiển thị % missing values.

        Output: missing_values.png
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if len(missing) == 0:
            return

        missing_pct = (missing / len(df)) * 100

        fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))

        bars = ax.barh(missing_pct.index, missing_pct.values, color='coral')
        ax.set_xlabel('Missing %')
        ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, missing_pct.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        self._save_plot(fig, "missing_values.png")

    # ==================== 2. TARGET DISTRIBUTION ====================

    def plot_target_distribution(self, y: pd.Series):
        """
        Pie/Donut chart cho target distribution.

        Output: target_distribution.png
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        counts = y.value_counts()
        colors = sns.color_palette("pastel")[:len(counts)]

        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors, explode=[0.03] * len(counts),
            startangle=90, pctdistance=0.75
        )

        # Donut style
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        ax.add_artist(centre_circle)

        ax.set_title(f'Target Distribution: {y.name}', fontsize=14, fontweight='bold')

        # Add count labels
        total = counts.sum()
        legend_labels = [f'{idx}: {cnt:,} ({cnt/total*100:.1f}%)' for idx, cnt in counts.items()]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        self._save_plot(fig, "target_distribution.png")

    # ==================== 3. NUMERICAL DISTRIBUTIONS ====================

    def plot_numerical_distributions(self, df: pd.DataFrame, num_cols: List[str]):
        """
        Grid of Histogram + KDE cho numerical features.

        Output: numerical_distributions.png
        """
        if not num_cols:
            return

        n = len(num_cols)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(num_cols):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='steelblue')
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_xlabel('')

        # Hide empty subplots
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Numerical Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_plot(fig, "numerical_distributions.png")

    # ==================== 4. BOXPLOTS ====================

    def plot_boxplots(self, df: pd.DataFrame, num_cols: List[str]):
        """
        Grid of Boxplots để detect outliers.

        Output: boxplots.png
        """
        if not num_cols:
            return

        n = len(num_cols)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(num_cols):
            sns.boxplot(y=df[col], ax=axes[i], color='coral')
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_ylabel('')

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Boxplots (Outlier Detection)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_plot(fig, "boxplots.png")

    # ==================== 5. CORRELATION MATRIX ====================

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """
        Heatmap correlation matrix (lower triangle).

        Output: correlation_matrix.png
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return

        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.5}, annot=False, ax=ax)

        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_plot(fig, "correlation_matrix.png")

    # ==================== 6. CORRELATION WITH TARGET ====================

    def plot_correlation_with_target(self, df: pd.DataFrame, target_col: str, top_n: int = 15):
        """
        Bar chart Top N features correlated với target.

        Output: correlation_with_target.png
        """
        if target_col not in df.columns:
            return

        numeric_df = df.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns or numeric_df.shape[1] < 2:
            return

        # Tính correlation với target
        correlations = numeric_df.corr()[target_col].drop(target_col)
        top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
        original = correlations[top_corr.index].sort_values()

        fig, ax = plt.subplots(figsize=(10, max(6, len(original) * 0.4)))

        colors = ['crimson' if v < 0 else 'steelblue' for v in original.values]
        bars = ax.barh(original.index, original.values, color=colors)

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title(f'Top {top_n} Features Correlated with {target_col}',
                    fontsize=14, fontweight='bold')

        for bar, val in zip(bars, original.values):
            x_pos = val + 0.02 if val >= 0 else val - 0.06
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        self._save_plot(fig, "correlation_with_target.png")

    # ==================== 7. CHURN BY CATEGORY ====================

    def plot_churn_by_category(self, df: pd.DataFrame, target_col: str, max_features: int = 6):
        """
        Grid of bar charts: Churn rate theo từng categorical feature.

        Output: churn_by_category.png
        """
        if target_col not in df.columns:
            return

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        valid_cols = [c for c in cat_cols if df[c].nunique() <= 8][:max_features]

        if not valid_cols:
            return

        n = len(valid_cols)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        avg_churn = df[target_col].mean() * 100

        for i, col in enumerate(valid_cols):
            # Tính churn rate
            grouped = df.groupby(col)[target_col].mean() * 100
            grouped = grouped.sort_values()

            colors = ['crimson' if v > avg_churn else 'steelblue' for v in grouped.values]
            bars = axes[i].barh(grouped.index.astype(str), grouped.values, color=colors)

            axes[i].axvline(x=avg_churn, color='orangered', linestyle='--',
                           linewidth=2, label=f'Avg: {avg_churn:.1f}%')
            axes[i].set_xlabel('Churn Rate (%)')
            axes[i].set_title(col, fontweight='bold')
            axes[i].legend(loc='lower right', fontsize=8)

            for bar, val in zip(bars, grouped.values):
                axes[i].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}%', va='center', fontsize=8)

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Churn Rate by Category', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_plot(fig, "churn_by_category.png")

    # ==================== RUN ALL ====================

    def run_full_eda(self, df: pd.DataFrame, target_col: str = None):
        """
        Chạy Full EDA - Tạo tất cả plots.

        Output:
            - missing_values.png
            - target_distribution.png (nếu có target)
            - numerical_distributions.png
            - boxplots.png
            - correlation_matrix.png
            - correlation_with_target.png (nếu có target)
            - churn_by_category.png (nếu có target + categorical)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        # 1. Missing values
        self.plot_missing_values(df)

        # 2. Target distribution
        if target_col and target_col in df.columns:
            self.plot_target_distribution(df[target_col])

        # 3. Numerical distributions
        if numeric_cols:
            self.plot_numerical_distributions(df, numeric_cols)

        # 4. Boxplots
        if numeric_cols:
            self.plot_boxplots(df, numeric_cols)

        # 5. Correlation matrix
        self.plot_correlation_matrix(df)

        # 6. Correlation with target
        if target_col:
            self.plot_correlation_with_target(df, target_col)

        # 7. Churn by category
        if target_col:
            self.plot_churn_by_category(df, target_col)

        if self.logger:
            self.logger.info(f"EDA Complete   | {self.eda_dir}")
