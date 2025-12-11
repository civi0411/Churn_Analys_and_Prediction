"""
Module `visualization.eda_plots` - Chứa các công cụ vẽ biểu đồ EDA và lưu figure.

Important keywords: Args, Returns, Notes, Methods, Class
"""

import matplotlib
# Chuyển backend sang 'Agg' để vẽ hình ổn định (không cần màn hình hiển thị GUI)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List
from ..utils import ensure_dir


class EDAVisualizer:
    """
    Class: EDAVisualizer
    Lớp hỗ trợ tạo và lưu các biểu đồ Phân tích Dữ liệu Khám phá (EDA) theo phong cách Dashboard.

    Methods:
        run_full_eda: Chạy toàn bộ quy trình vẽ biểu đồ EDA.
        plot_missing_values: Vẽ biểu đồ giá trị thiếu.
        plot_target_distribution: Vẽ biểu đồ phân phối biến mục tiêu.
        plot_numerical_distributions: Vẽ biểu đồ phân phối các biến số.
        plot_boxplots: Vẽ biểu đồ hộp để phát hiện ngoại lai.
        plot_correlation_matrix: Vẽ ma trận tương quan.
        plot_correlation_with_target: Vẽ biểu đồ tương quan với biến mục tiêu.
        plot_churn_by_category: Vẽ tỷ lệ Churn theo nhóm phân loại.
    """

    def __init__(self, config: dict, logger=None, run_specific_dir=None):
        """
        Constructor: __init__
        Khởi tạo đối tượng EDAVisualizer.

        Args:
            config (dict): Dictionary chứa cấu hình dự án.
            logger (logging.Logger, optional): Đối tượng logger để ghi log. Defaults to None.
            run_specific_dir (str, optional): Đường dẫn thư mục của run hiện tại để lưu các figure. Defaults to None.

        Notes:
            Thiết lập bảng màu (palette) và style mặc định cho biểu đồ tại đây.
        """
        self.config = config
        self.logger = logger
        if run_specific_dir is None:
            self.save_dir = None
            self.eda_dir = None
            if self.logger:
                self.logger.warning("EDAVisualizer: run_specific_dir chưa được thiết lập, sẽ không lưu figures.")
        else:
            self.save_dir = run_specific_dir
            self.eda_dir = os.path.join(self.save_dir, 'eda')
            ensure_dir(self.eda_dir)

        # --- DASHBOARD COLOR PALETTE (Màu sắc đậm đà, tương phản cao) ---
        self.colors = {
            'primary': '#1F77B4',     # Steel Blue (Màu chính cho các cột, dữ liệu thường)
            'secondary': '#3498DB',   # Blue (Màu xanh đậm hơn một chút cho boxplot)
            'danger': '#D62728',      # Brick Red (Màu đỏ cho Cảnh báo/Churn/Negative)
            'success': '#2CA02C',     # Green (Màu xanh lá cho trạng thái tốt)
            'accent': '#FF7F0E',      # Safety Orange (Màu cam cho đường nhấn/KDE)
            'neutral': '#7F7F7F',     # Grey (Màu trung tính)
            'text': '#333333',        # Dark Grey (Màu chữ)
            'line': '#333333'         # Dark Line (Màu đường kẻ trục/lưới)
        }

        # Cấu hình Global cho Matplotlib/Seaborn
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold',
            'figure.facecolor': 'white',
            'axes.facecolor': '#FAFAFA', # Nền xám cực nhạt giúp nổi bật dữ liệu
            'axes.grid': True,
            'grid.color': '#E0E0E0',
            'grid.linestyle': '-',
            'grid.alpha': 0.6,
            'text.color': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text']
        })

    def _save_plot(self, fig, filename: str):
        """
        Method: _save_plot
        Lưu figure (biểu đồ) vào thư mục EDA đã được cấu hình.

        Args:
            fig (matplotlib.figure.Figure): Đối tượng figure cần lưu.
            filename (str): Tên file ảnh (ví dụ: 'plot.png').

        Returns:
            None
        """
        if not self.eda_dir:
            return

        try:
            path = os.path.join(self.eda_dir, filename)
            fig.savefig(path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            if self.logger:
                self.logger.info(f"Saved Plot      | EDA        | {filename}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save plot {filename}: {e}")
            plt.close(fig)

    # ==================== 1. MISSING VALUES ====================

    def plot_missing_values(self, df: pd.DataFrame):
        """
        Method: plot_missing_values
        Vẽ biểu đồ thanh (bar chart) hiển thị phần trăm giá trị thiếu của từng cột.

        Args:
            df (pd.DataFrame): DataFrame dữ liệu đầu vào.

        Returns:
            None (Lưu file 'missing_values.png').
        """
        try:
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=True)

            if len(missing) == 0:
                return

            missing_pct = (missing / len(df)) * 100

            fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))

            # Sử dụng màu cam (Accent) để cảnh báo về dữ liệu thiếu
            bars = ax.barh(missing_pct.index, missing_pct.values,
                           color=self.colors['accent'], edgecolor='white', linewidth=1)

            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Values Overview')

            for bar, val in zip(bars, missing_pct.values):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

            plt.tight_layout()
            self._save_plot(fig, "missing_values.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting missing values: {e}")

    # ==================== 2. TARGET DISTRIBUTION ====================

    def plot_target_distribution(self, y: pd.Series):
        """
        Method: plot_target_distribution
        Vẽ biểu đồ Donut Chart hiển thị phân phối của biến mục tiêu (Target).

        Args:
            y (pd.Series): Series chứa dữ liệu của cột Target.

        Returns:
            None (Lưu file 'target_distribution.png').
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = y.value_counts()

            # Logic màu: Class 0 (Thường là tốt) -> Xanh, Class 1 (Churn) -> Đỏ
            if set(counts.index).issubset({0, 1}):
                colors_map = {0: self.colors['primary'], 1: self.colors['danger']}
                colors = [colors_map.get(i, self.colors['neutral']) for i in counts.index]
            else:
                colors = sns.color_palette("Paired")

            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, explode=[0.02] * len(counts),
                startangle=90, pctdistance=0.75,
                wedgeprops={"edgecolor": "white", 'linewidth': 3},
                textprops={'fontsize': 11, 'weight': 'bold'}
            )

            # Tạo lỗ tròn ở giữa để thành biểu đồ Donut
            centre_circle = plt.Circle((0, 0), 0.60, fc='white')
            ax.add_artist(centre_circle)

            ax.set_title(f'Target Distribution: {y.name}', color=self.colors['line'])

            total = counts.sum()
            legend_labels = [f'{idx}: {cnt:,} ({cnt/total*100:.1f}%)' for idx, cnt in counts.items()]
            ax.legend(wedges, legend_labels, title="Legend", loc='center left', bbox_to_anchor=(1, 0.5))

            plt.tight_layout()
            self._save_plot(fig, "target_distribution.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting target distribution: {e}")

    # ==================== 3. NUMERICAL DISTRIBUTIONS ====================

    def plot_numerical_distributions(self, df: pd.DataFrame, num_cols: List[str]):
        """
        Method: plot_numerical_distributions
        Vẽ biểu đồ Histogram kết hợp đường mật độ (KDE) cho các biến số.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu.
            num_cols (List[str]): Danh sách tên các cột dữ liệu số.

        Returns:
            None (Lưu file 'numerical_distributions.png').
        """
        try:
            if not num_cols:
                return

            n = len(num_cols)
            n_cols = min(3, n)
            n_rows = (n + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = np.array(axes).flatten() if n > 1 else [axes]

            for i, col in enumerate(num_cols):
                # Histogram: Màu xanh Primary đậm đà (alpha=0.8)
                sns.histplot(
                    data=df, x=col, kde=True, ax=axes[i],
                    color=self.colors['primary'],
                    alpha=0.8,
                    edgecolor=None
                )
                axes[i].set_title(col)
                axes[i].set_xlabel('')

                # KDE Line: Màu cam Accent để nổi bật trên nền xanh
                if axes[i].lines:
                    plt.setp(axes[i].lines, linewidth=2.5, color=self.colors['accent'])

            for j in range(n, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle('Numerical Distributions', fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            self._save_plot(fig, "numerical_distributions.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting numerical distributions: {e}")

    # ==================== 4. BOXPLOTS ====================

    def plot_boxplots(self, df: pd.DataFrame, num_cols: List[str]):
        """
        Method: plot_boxplots
        Vẽ biểu đồ hộp (Boxplot) để phát hiện giá trị ngoại lai (outliers).

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu.
            num_cols (List[str]): Danh sách tên các cột dữ liệu số.

        Returns:
            None (Lưu file 'boxplots.png').
        """
        try:
            if not num_cols:
                return

            n = len(num_cols)
            n_cols = min(4, n)
            n_rows = (n + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.array(axes).flatten() if n > 1 else [axes]

            for i, col in enumerate(num_cols):
                # Box: Tô màu xanh (Secondary), viền xám đậm (Text color)
                sns.boxplot(
                    y=df[col], ax=axes[i],
                    color=self.colors['secondary'],
                    linecolor=self.colors['text'],
                    linewidth=1.2,
                    fliersize=3,
                    flierprops={'marker': 'o', 'markerfacecolor': self.colors['danger'], 'markeredgecolor': 'none', 'alpha': 0.7}
                )
                axes[i].set_title(col)
                axes[i].set_ylabel('')

            for j in range(n, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle('Outlier Detection (Boxplots)', fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            self._save_plot(fig, "boxplots.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting boxplots: {e}")

    # ==================== 5. CORRELATION MATRIX ====================

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """
        Method: plot_correlation_matrix
        Vẽ biểu đồ nhiệt (Heatmap) thể hiện ma trận tương quan giữa các biến số.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu.

        Returns:
            None (Lưu file 'correlation_matrix.png').
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return

            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))

            fig, ax = plt.subplots(figsize=(12, 10))

            # Heatmap 'coolwarm': Xanh (Âm) -> Đỏ (Dương), dễ nhìn cho dashboard
            sns.heatmap(
                corr, mask=mask, cmap='coolwarm', center=0,
                square=True,
                linewidths=0.5, linecolor='white',
                annot=False, cbar_kws={"shrink": 0.6}, ax=ax
            )

            ax.set_title('Correlation Matrix', fontsize=15)
            plt.tight_layout()
            self._save_plot(fig, "correlation_matrix.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting correlation matrix: {e}")

    # ==================== 6. CORRELATION WITH TARGET ====================

    def plot_correlation_with_target(self, df: pd.DataFrame, target_col: str, top_n: int = 15):
        """
        Method: plot_correlation_with_target
        Vẽ biểu đồ thanh ngang thể hiện Top N đặc trưng có tương quan cao nhất với biến Target.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu.
            target_col (str): Tên cột biến mục tiêu.
            top_n (int, optional): Số lượng đặc trưng hiển thị. Defaults to 15.

        Returns:
            None (Lưu file 'correlation_with_target.png').
        """
        try:
            if target_col not in df.columns:
                return

            numeric_df = df.select_dtypes(include=[np.number])
            if target_col not in numeric_df.columns or numeric_df.shape[1] < 2:
                return

            correlations = numeric_df.corr()[target_col].drop(target_col)
            top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
            original = correlations[top_corr.index].sort_values()

            fig, ax = plt.subplots(figsize=(10, max(6, len(original) * 0.4)))

            colors = [self.colors['danger'] if v < 0 else self.colors['primary'] for v in original.values]

            bars = ax.barh(original.index, original.values, color=colors, height=0.7)

            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title(f'Top {top_n} Features Correlated with {target_col}')

            for bar, val in zip(bars, original.values):
                x_pos = val + 0.02 if val >= 0 else val - 0.08
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=9, fontweight='bold',
                       color=self.colors['danger'] if val < 0 else self.colors['primary'])

            plt.tight_layout()
            self._save_plot(fig, "correlation_with_target.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting correlation with target: {e}")

    # ==================== 7. CHURN BY CATEGORY ====================

    def plot_churn_by_category(self, df: pd.DataFrame, target_col: str, max_features: int = 6):
        """
        Method: plot_churn_by_category
        Vẽ biểu đồ tỷ lệ Churn (trung bình của Target) theo từng nhóm của biến phân loại.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu.
            target_col (str): Tên cột biến mục tiêu (thường là 0/1).
            max_features (int, optional): Số lượng biến phân loại tối đa để vẽ (để tránh quá tải). Defaults to 6.

        Returns:
            None (Lưu file 'churn_by_category.png').
        """
        try:
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
                grouped = df.groupby(col)[target_col].mean() * 100
                grouped = grouped.sort_values()

                # Logic màu: Cao hơn trung bình -> Đỏ (Nguy hiểm), Thấp hơn -> Xanh (Tốt)
                colors = [self.colors['danger'] if v > avg_churn else self.colors['success'] for v in grouped.values]

                bars = axes[i].barh(grouped.index.astype(str), grouped.values, color=colors, alpha=0.9)

                # Đường trung bình
                axes[i].axvline(x=avg_churn, color=self.colors['text'], linestyle='--',
                               linewidth=2.0, label=f'Avg: {avg_churn:.1f}%')

                axes[i].set_xlabel('Churn Rate (%)')
                axes[i].set_title(col)
                axes[i].legend(loc='lower right', fontsize=8)

                for bar, val in zip(bars, grouped.values):
                    offset = -5 if val > 10 else 1
                    txt_color = 'white' if val > 10 else 'black'
                    axes[i].text(val + offset, bar.get_y() + bar.get_height()/2,
                               f'{val:.1f}%', va='center', fontsize=9, fontweight='bold', color=txt_color)

            for j in range(n, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle('Churn Rate by Category', fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            self._save_plot(fig, "churn_by_category.png")
        except Exception as e:
            if self.logger: self.logger.error(f"Error plotting churn by category: {e}")

    # ==================== RUN ALL ====================

    def run_full_eda(self, df: pd.DataFrame, target_col: str = None):
        """
        Method: run_full_eda
        Phương thức điều phối chính: Chạy toàn bộ các hàm vẽ biểu đồ EDA phù hợp với dữ liệu.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu thô hoặc đã qua xử lý sơ bộ.
            target_col (str, optional): Tên cột biến mục tiêu. Defaults to None.

        Returns:
            None: Kết quả là các file ảnh được lưu trong thư mục EDA.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        # 1. Vẽ biểu đồ giá trị thiếu
        self.plot_missing_values(df)

        # 2. Vẽ phân phối Target (nếu có)
        if target_col and target_col in df.columns:
            self.plot_target_distribution(df[target_col])

        # 3. Vẽ phân phối biến số và Boxplot
        if numeric_cols:
            self.plot_numerical_distributions(df, numeric_cols)
            self.plot_boxplots(df, numeric_cols)

        # 4. Vẽ ma trận tương quan
        self.plot_correlation_matrix(df)

        # 5. Vẽ tương quan với Target và Churn theo nhóm (nếu có Target)
        if target_col:
            self.plot_correlation_with_target(df, target_col)
            self.plot_churn_by_category(df, target_col)

        if self.logger:
            self.logger.info(f"EDA Complete   | {self.eda_dir}")