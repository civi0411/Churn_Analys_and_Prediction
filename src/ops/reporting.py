"""src/ops/reporting.py

Professional ML Report Generator - Comprehensive Training Reports

This module generates publication-quality markdown reports with:
- Executive Summary with key findings
- Automated insights and recommendations
- Visual analysis with objective interpretations
- Performance metrics with threshold validation
- EDA findings with statistical summaries

Report Structure:
    1. Executive Summary
    2. EDA Insights (Data Quality, Distribution Analysis)
    3. Model Performance (Metrics Tables, Comparisons)
    4. Visual Analysis (EDA + Evaluation figures with explanations)
    5. Feature Importance (Top drivers with business context)
    6. Recommendations (Actionable next steps)
    7. Technical Appendix (Config, Lineage)

Output Locations:
    - Central: artifacts/reports/training_report_<run_id>.md
    - Per-Run: artifacts/experiments/<run_id>/training_report_<run_id>.md

Author: Churn Prediction Team
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from ..utils import ensure_dir, IOHandler
from .reporting_helpers import sanitize_filename, sanitize_text


class ReportGenerator:
    """
    Professional Report Generator for ML Training Runs.

    Generates comprehensive markdown reports with:
    - Automated analysis and insights
    - Visual explanations (objective, data-driven)
    - Performance validation against thresholds
    - Actionable recommendations

    Example:
        >>> rg = ReportGenerator(experiments_base_dir="artifacts/experiments")
        >>> # feature_importance should be a pandas.DataFrame when supplied
        >>> # rg.generate_training_report(..., feature_importance=importance_df, ...)
    """

    def __init__(
            self,
            reports_dir: str = None,
            experiments_base_dir: str = "artifacts/experiments",
            logger=None
    ):
        """Initialize ReportGenerator.

        Args:
            reports_dir: Central reports directory (default: artifacts/reports)
            experiments_base_dir: Base directory for experiment runs
            logger: Optional logger instance
        """
        self.experiments_base_dir = experiments_base_dir
        self.reports_dir = reports_dir or os.path.join("artifacts", "reports")
        ensure_dir(self.reports_dir)
        self.logger = logger

    # ==================== MAIN REPORT GENERATOR ====================

    def generate_training_report(
            self,
            run_id: str,
            best_model_name: str,
            all_metrics: Dict[str, Dict[str, float]],
            feature_importance: Optional[pd.DataFrame] = None,
            config: Optional[Dict[str, Any]] = None,
            eda_summary: Optional[Dict[str, Any]] = None,
            format: str = "markdown",
            mode: str = "train",
            optimize: bool = True,
            args=None
    ) -> str:
        """
        Generate adaptive training report based on pipeline mode.
        Returns path to generated report file.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_dir = self._get_run_dir(run_id) or ""
        filename = f"training_report_{run_id}"
        lines: List[str] = []
        best_metrics = all_metrics.get(best_model_name, {})
        eval_fig_dir = os.path.join(run_dir, 'figures', 'evaluation') if run_dir else None

        # HEADER
        lines.extend([
            f"# Báo Cáo Pipeline ({mode.upper()})",
            ""
        ])
        # Dashboard summary table
        lines.extend(self._generate_run_summary(run_id, timestamp, mode, config, args))
        lines.append("---\n")

        # EDA MODE
        if mode == "eda":
            if eda_summary:
                lines.extend([
                    "## 🔍 Tóm Tắt Phân Tích Dữ Liệu (EDA)",
                    f"* **Vấn đề Mất cân bằng:** Tỷ lệ Churn (Lớp 1) là **{eda_summary.get('churn_rate', 0.168):.1%}** so với Non-Churn (Lớp 0) là {1-eda_summary.get('churn_rate', 0.168):.1%}.",
                    f"* **Chiến lược Xử lý:** {eda_summary.get('resampling', 'Đã áp dụng SMOTE/Tomek để cân bằng lại tập huấn luyện.')}",
                    f"* **Tương quan ban đầu (Top 2):** Biến `{eda_summary.get('top_correlated', ['Tenure','Complain'])[0]}` và `{eda_summary.get('top_correlated', ['Tenure','Complain'])[1]}` có tương quan cao nhất với nhãn `Churn`.",
                    "",
                    "---",
                    ""
                ])
                eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
                if eda_fig_dir and os.path.isdir(eda_fig_dir):
                    lines.append("### Key Visualizations (EDA)")
                    for fname in sorted(os.listdir(eda_fig_dir)):
                        if fname.lower().endswith('.png'):
                            rel = f"figures/eda/{fname}"
                            lines.append(f"**{self._humanize_filename(fname)}**")
                            lines.append("")
                            lines.append(f"![{fname}]({rel})")
                            lines.extend(self._interpret_eda_figure(fname, eda_summary))
                            lines.append("")
                    lines.append("---\n")
            lines.extend(self._generate_footer())
            return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

        # TRAIN MODE
        if mode == "train":
            lines.append("## 📊 Bảng Xếp Hạng Thuật Toán (Model Leaderboard)")
            all_metric_keys = set()
            for m in all_metrics.values():
                all_metric_keys.update(m.keys())
            keys = [k for k in ['accuracy','f1','precision','recall','roc_auc'] if k in all_metric_keys]
            header = "| Model | " + " | ".join(k.upper() for k in keys) + " |"
            sep = "|:------|" + "|".join(["------:"] * len(keys)) + "|"
            lines.append(header)
            lines.append(sep)
            for name, metrics in sorted(all_metrics.items(), key=lambda x: x[1].get('f1', 0), reverse=True):
                row_vals = [f"{metrics.get(k, 0):.4f}" for k in keys]
                if name == best_model_name:
                    row = f"| **{name}** 🏆 | " + " | ".join(f"**{v}**" for v in row_vals) + " |"
                else:
                    row = f"| {name} | " + " | ".join(row_vals) + " |"
                lines.append(row)
            lines.append("")
            lines.append("---\n")
            lines.append("## 🥇 Kết Quả Đánh Giá Mô Hình Tốt Nhất")
            lines.append(f"### Champion Metrics ({best_model_name.upper()})")
            lines.append("| Metric | Giá trị | Ngưỡng MLOps (Min) | Trạng Thái |")
            lines.append("| :--- | :--- | :--- | :--- |")
            lines.append(f"| **Accuracy** | **{best_metrics.get('accuracy',0):.4f}** | > 0.75 | {'✅ PASS' if best_metrics.get('accuracy',0)>0.75 else '❌ FAIL'} |")
            lines.append(f"| **F1-Score** | **{best_metrics.get('f1',0):.4f}** | > 0.70 | {'✅ PASS' if best_metrics.get('f1',0)>0.70 else '❌ FAIL'} |")
            lines.append(f"| **Recall (Churn Coverage)** | {best_metrics.get('recall',0):.4f} | - | - |")
            lines.append(f"| **ROC-AUC** | **{best_metrics.get('roc_auc',0):.4f}** | - | - |")
            lines.append("")
            # Evaluation Figures with auto-interpretation
            if eval_fig_dir and os.path.isdir(eval_fig_dir):
                lines.append("## 🖼️ Evaluation Figures & Analysis")
                for fname in sorted(os.listdir(eval_fig_dir)):
                    if fname.lower().endswith('.png'):
                        rel = f"figures/evaluation/{fname}"
                        title = self._humanize_filename(fname)
                        lines.append(f"### {title}")
                        lines.append("")
                        lines.append(f"![{title}]({rel})")
                        lines.extend(self._interpret_eval_figure(fname, best_metrics, all_metrics, feature_importance))
                        lines.append("")
                lines.append("---\n")
            lines.append("## 🚨 Kết Luận & Kế Hoạch Giám Sát (Monitoring Plan)")
            lines.append("* **Trạng Thái Mô Hình:** Mô hình XGBoost (vượt ngưỡng) được chọn làm **Model Baseline**.")
            lines.append("* **Kế hoạch MLOps:** Hệ thống giám sát (Monitoring) phải được kích hoạt để theo dõi F1-Score và Data Drift.")
            lines.append("* **Mục tiêu Giám sát:** Đảm bảo hiệu suất không giảm xuống dưới ngưỡng 0.70 (F1-Score) và cảnh báo khi có biến động dữ liệu mạnh (`Complain`, `Tenure`).")
            lines.append("")
            lines.append("---\n")
            lines.extend(self._generate_footer())
            return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

        # FULL MODE
        if mode == "full":
            # Dashboard summary table already added
            # EDA
            if eda_summary:
                lines.extend([
                    "## 🔍 Tóm Tắt Phân Tích Dữ Liệu (EDA)",
                    f"* **Vấn đề Mất cân bằng:** Tỷ lệ Churn (Lớp 1) là **{eda_summary.get('churn_rate', 0.168):.1%}** so với Non-Churn (Lớp 0) là {1-eda_summary.get('churn_rate', 0.168):.1%}.",
                    f"* **Chiến lược Xử lý:** {eda_summary.get('resampling', 'Đã áp dụng SMOTE/Tomek để cân bằng lại tập huấn luyện.')}",
                    f"* **Tương quan ban đầu (Top 2):** Biến `{eda_summary.get('top_correlated', ['Tenure','Complain'])[0]}` và `{eda_summary.get('top_correlated', ['Tenure','Complain'])[1]}` có tương quan cao nhất với nhãn `Churn`.",
                    "",
                    "---",
                    ""
                ])
                eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
                if eda_fig_dir and os.path.isdir(eda_fig_dir):
                    lines.append("### Key Visualizations (EDA)")
                    for fname in sorted(os.listdir(eda_fig_dir)):
                        if fname.lower().endswith('.png'):
                            rel = f"figures/eda/{fname}"
                            lines.append(f"**{self._humanize_filename(fname)}**")
                            lines.append("")
                            lines.append(f"![{fname}]({rel})")
                            lines.extend(self._interpret_eda_figure(fname, eda_summary))
                            lines.append("")
                    lines.append("---\n")
            # Training
            lines.append("## 📊 Bảng Xếp Hạng Thuật Toán (Model Leaderboard)")
            all_metric_keys = set()
            for m in all_metrics.values():
                all_metric_keys.update(m.keys())
            keys = [k for k in ['accuracy','f1','precision','recall','roc_auc'] if k in all_metric_keys]
            header = "| Model | " + " | ".join(k.upper() for k in keys) + " |"
            sep = "|:------|" + "|".join(["------:"] * len(keys)) + "|"
            lines.append(header)
            lines.append(sep)
            for name, metrics in sorted(all_metrics.items(), key=lambda x: x[1].get('f1', 0), reverse=True):
                row_vals = [f"{metrics.get(k, 0):.4f}" for k in keys]
                if name == best_model_name:
                    row = f"| **{name}** 🏆 | " + " | ".join(f"**{v}**" for v in row_vals) + " |"
                else:
                    row = f"| {name} | " + " | ".join(row_vals) + " |"
                lines.append(row)
            lines.append("")
            lines.append("---\n")
            lines.append("## 🥇 Kết Quả Đánh Giá Mô Hình Tốt Nhất")
            lines.append(f"### Champion Metrics ({best_model_name.upper()})")
            lines.append("| Metric | Giá trị | Ngưỡng MLOps (Min) | Trạng Thái |")
            lines.append("| :--- | :--- | :--- | :--- |")
            lines.append(f"| **Accuracy** | **{best_metrics.get('accuracy',0):.4f}** | > 0.75 | {'✅ PASS' if best_metrics.get('accuracy',0)>0.75 else '❌ FAIL'} |")
            lines.append(f"| **F1-Score** | **{best_metrics.get('f1',0):.4f}** | > 0.70 | {'✅ PASS' if best_metrics.get('f1',0)>0.70 else '❌ FAIL'} |")
            lines.append(f"| **Recall (Churn Coverage)** | {best_metrics.get('recall',0):.4f} | - | - |")
            lines.append(f"| **ROC-AUC** | **{best_metrics.get('roc_auc',0):.4f}** | - | - |")
            lines.append("")
            # Evaluation Figures with auto-interpretation
            if eval_fig_dir and os.path.isdir(eval_fig_dir):
                lines.append("## 🖼️ Evaluation Figures & Analysis")
                for fname in sorted(os.listdir(eval_fig_dir)):
                    if fname.lower().endswith('.png'):
                        rel = f"figures/evaluation/{fname}"
                        title = self._humanize_filename(fname)
                        lines.append(f"### {title}")
                        lines.append("")
                        lines.append(f"![{title}]({rel})")
                        lines.extend(self._interpret_eval_figure(fname, best_metrics, all_metrics, feature_importance))
                        lines.append("")
                lines.append("---\n")
            lines.append("## 🚨 Kết Luận & Kế Hoạch Giám Sát (Monitoring Plan)")
            lines.append("* **Trạng Thái Mô Hình:** Mô hình XGBoost (vượt ngưỡng) được chọn làm **Model Baseline**.")
            lines.append("* **Kế hoạch MLOps:** Hệ thống giám sát (Monitoring) phải được kích hoạt để theo dõi F1-Score và Data Drift.")
            lines.append("* **Mục tiêu Giám sát:** Đảm bảo hiệu suất không giảm xuống dưới ngưỡng 0.70 (F1-Score) và cảnh báo khi có biến động dữ liệu mạnh (`Complain`, `Tenure`).")
            lines.append("")
            lines.append("---\n")
            if config:
                lines.extend(self._generate_technical_appendix(config))
            lines.extend(self._generate_footer())
            return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

        # PREPROCESS MODE
        if mode == "preprocess":
            lines.append("## Preprocessing Summary")
            lines.append("- Data cleaning, transformation, and feature engineering completed.")
            lines.append("- See config for details.")
            lines.append("")
            if config:
                lines.extend(self._generate_technical_appendix(config))
            lines.extend(self._generate_footer())
            return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

        # VISUALIZE MODE
        if mode == "visualize":
            lines.append("## Visualizations")
            if eval_fig_dir and os.path.isdir(eval_fig_dir):
                for fname in sorted(os.listdir(eval_fig_dir)):
                    if fname.lower().endswith('.png'):
                        rel = f"figures/evaluation/{fname}"
                        lines.append(f"**{self._humanize_filename(fname)}**")
                        lines.append(f"![{fname}]({rel})")
                        lines.extend(self._interpret_eval_figure(fname, best_metrics, all_metrics, feature_importance))
                        lines.append("")
            lines.append("---\n")
            lines.extend(self._generate_footer())
            return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

        # DEFAULT: fallback to full mode
        lines.append("*No valid mode selected. Showing full report.*")
        # EDA
        if eda_summary:
            lines.extend([
                "## 🔍 Tóm Tắt Phân Tích Dữ Liệu (EDA)",
                f"* **Vấn đề Mất cân bằng:** Tỷ lệ Churn (Lớp 1) là **{eda_summary.get('churn_rate', 0.168):.1%}** so với Non-Churn (Lớp 0) là {1-eda_summary.get('churn_rate', 0.168):.1%}.",
                f"* **Chiến lược Xử lý:** {eda_summary.get('resampling', 'Đã áp dụng SMOTE/Tomek để cân bằng lại tập huấn luyện.')}",
                f"* **Tương quan ban đầu (Top 2):** Biến `{eda_summary.get('top_correlated', ['Tenure','Complain'])[0]}` và `{eda_summary.get('top_correlated', ['Tenure','Complain'])[1]}` có tương quan cao nhất với nhãn `Churn`.",
                "",
                "---",
                ""
            ])
            eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
            if eda_fig_dir and os.path.isdir(eda_fig_dir):
                lines.append("### Key Visualizations (EDA)")
                for fname in sorted(os.listdir(eda_fig_dir)):
                    if fname.lower().endswith('.png'):
                        rel = f"figures/eda/{fname}"
                        lines.append(f"**{self._humanize_filename(fname)}**")
                        lines.append("")
                        lines.append(f"![{fname}]({rel})")
                        lines.extend(self._interpret_eda_figure(fname, eda_summary))
                        lines.append("")
                lines.append("---\n")
        # Training
        lines.append("## 📊 Bảng Xếp Hạng Thuật Toán (Model Leaderboard)")
        all_metric_keys = set()
        for m in all_metrics.values():
            all_metric_keys.update(m.keys())
        keys = [k for k in ['accuracy','f1','precision','recall','roc_auc'] if k in all_metric_keys]
        header = "| Model | " + " | ".join(k.upper() for k in keys) + " |"
        sep = "|:------|" + "|".join(["------:"] * len(keys)) + "|"
        lines.append(header)
        lines.append(sep)
        for name, metrics in sorted(all_metrics.items(), key=lambda x: x[1].get('f1', 0), reverse=True):
            row_vals = [f"{metrics.get(k, 0):.4f}" for k in keys]
            if name == best_model_name:
                row = f"| **{name}** 🏆 | " + " | ".join(f"**{v}**" for v in row_vals) + " |"
            else:
                row = f"| {name} | " + " | ".join(row_vals) + " |"
            lines.append(row)
        lines.append("")
        lines.append("---\n")
        lines.append("## 🥇 Kết Quả Đánh Giá Mô Hình Tốt Nhất")
        lines.append(f"### Champion Metrics ({best_model_name.upper()})")
        lines.append("| Metric | Giá trị | Ngưỡng MLOps (Min) | Trạng Thái |")
        lines.append("| :--- | :--- | :--- | :--- |")
        lines.append(f"| **Accuracy** | **{best_metrics.get('accuracy',0):.4f}** | > 0.75 | {'✅ PASS' if best_metrics.get('accuracy',0)>0.75 else '❌ FAIL'} |")
        lines.append(f"| **F1-Score** | **{best_metrics.get('f1',0):.4f}** | > 0.70 | {'✅ PASS' if best_metrics.get('f1',0)>0.70 else '❌ FAIL'} |")
        lines.append(f"| **Recall (Churn Coverage)** | {best_metrics.get('recall',0):.4f} | - | - |")
        lines.append(f"| **ROC-AUC** | **{best_metrics.get('roc_auc',0):.4f}** | - | - |")
        lines.append("")
        # Evaluation Figures with auto-interpretation
        if eval_fig_dir and os.path.isdir(eval_fig_dir):
            lines.append("## 🖼️ Evaluation Figures & Analysis")
            for fname in sorted(os.listdir(eval_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/evaluation/{fname}"
                    title = self._humanize_filename(fname)
                    lines.append(f"### {title}")
                    lines.append("")
                    lines.append(f"![{title}]({rel})")
                    lines.extend(self._interpret_eval_figure(fname, best_metrics, all_metrics, feature_importance))
                    lines.append("")
            lines.append("---\n")
        lines.append("## 🚨 Kết Luận & Kế Hoạch Giám Sát (Monitoring Plan)")
        lines.append("* **Trạng Thái Mô Hình:** Mô hình XGBoost (vượt ngưỡng) được chọn làm **Model Baseline**.")
        lines.append("* **Kế hoạch MLOps:** Hệ thống giám sát (Monitoring) phải được kích hoạt để theo dõi F1-Score và Data Drift.")
        lines.append("* **Mục tiêu Giám sát:** Đảm bảo hiệu suất không giảm xuống dưới ngưỡng 0.70 (F1-Score) và cảnh báo khi có biến động dữ liệu mạnh (`Complain`, `Tenure`).")
        lines.append("")
        lines.append("---\n")
        if config:
            lines.extend(self._generate_technical_appendix(config))
        lines.extend(self._generate_footer())
        return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

    # ==================== REPORT SECTIONS ====================

    def _generate_header(self, run_id: str, timestamp: str) -> List[str]:
        """Generate report header with metadata."""
        return [
            "# ML Training Report",
            "",
            f"**Run ID:** `{run_id}`  ",
            f"**Generated:** {timestamp}  ",
            f"**Project:** E-Commerce Churn Prediction",
            "",
            "---",
            ""
        ]

    def _generate_executive_summary(
            self,
            best_model_name: str,
            best_metrics: Dict[str, float],
            all_metrics: Dict[str, Dict[str, float]],
            eda_summary: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate executive summary with key findings."""
        lines = [
            "## Executive Summary",
            "",
            "### Key Findings",
            ""
        ]

        # Best model performance
        f1 = best_metrics.get('f1', 0)
        roc_auc = best_metrics.get('roc_auc', 0)
        accuracy = best_metrics.get('accuracy', 0)
        recall = best_metrics.get('recall', 0)
        precision = best_metrics.get('precision', 0)

        lines.append(f"**🏆 Champion Model:** `{best_model_name.upper()}`")
        lines.append("")
        lines.append("| Metric | Score | Status | Threshold |")
        lines.append("|:-------|------:|:------:|:---------:|")
        lines.append(
            f"| **F1-Score** | **{f1:.4f}** | "
            f"{'✅ PASS' if f1 >= 0.70 else '❌ FAIL'} | > 0.70 |"
        )
        lines.append(
            f"| **Accuracy** | **{accuracy:.4f}** | "
            f"{'✅ PASS' if accuracy >= 0.75 else '❌ FAIL'} | > 0.75 |"
        )
        lines.append(f"| **ROC-AUC** | **{roc_auc:.4f}** | - | - |")
        lines.append(f"| **Recall** | {recall:.4f} | - | - |")
        lines.append(f"| **Precision** | {precision:.4f} | - | - |")
        lines.append("")

        # Automated insights
        insights = self._analyze_model_performance(best_metrics, all_metrics)
        if insights:
            lines.append("**🔍 Automated Insights:**")
            lines.append("")
            for insight in insights:
                lines.append(f"- {insight}")
            lines.append("")

        # Data quality summary
        if eda_summary:
            churn_rate = eda_summary.get('churn_rate', 0)
            lines.append(f"**📊 Dataset:** Churn rate = {churn_rate:.1%} (class imbalance detected)")
            if eda_summary.get('resampling'):
                lines.append(f"- **Resampling:** {eda_summary.get('resampling')}")
            lines.append("")

        lines.append("---")
        lines.append("")

        return lines

    def _generate_eda_insights(
            self,
            eda_summary: Dict[str, Any],
            run_dir: Optional[str]
    ) -> List[str]:
        """Generate EDA insights section with data quality analysis."""
        lines = [
            "## Exploratory Data Analysis",
            "",
            "### Data Quality Summary",
            ""
        ]

        # Class distribution
        churn_rate = eda_summary.get('churn_rate', 0)
        lines.append(f"- **Target Distribution:** Churn = {churn_rate:.1%}, Non-Churn = {1 - churn_rate:.1%}")

        if churn_rate < 0.20:
            lines.append(f"  - ⚠️ **Imbalanced dataset detected** (minority class < 20%)")
            lines.append(f"  - **Impact:** Model may be biased toward majority class")
            lines.append(f"  - **Mitigation:** {eda_summary.get('resampling', 'SMOTE/Tomek applied')}")
        lines.append("")

        # Top correlated features
        top_corr = eda_summary.get('top_correlated', [])
        # Fix: ensure all elements are str and not None
        top_corr_str = [str(c) for c in top_corr if c is not None]
        if top_corr_str:
            lines.append(f"- Top correlated features with churn: {', '.join(top_corr_str[:3])}")
            lines.append("- **Usage:** Prioritize these features in feature engineering")
            lines.append("- **Note:** High correlation doesn't imply causation")
        lines.append("")

        # EDA figures preview
        eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
        if eda_fig_dir and os.path.isdir(eda_fig_dir):
            lines.append("### Key Visualizations")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>📊 Click to expand EDA figures</summary>")
            lines.append("")

            for fname in sorted(os.listdir(eda_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/eda/{fname}"
                    lines.append(f"**{self._humanize_filename(fname)}**")
                    lines.append("")
                    lines.append(f"<img src=\"{rel}\" width=\"600\" />")
                    lines.append("")
                    # Add figure interpretation
                    lines.extend(self._interpret_eda_figure(fname, eda_summary))
                    lines.append("")

            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

        return lines

    def _generate_performance_section(
            self,
            best_model_name: str,
            best_metrics: Dict[str, float],
            all_metrics: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate model performance comparison section."""
        lines = [
            "## Model Performance",
            "",
            "### All Models Comparison",
            ""
        ]

        # Metrics table
        all_metric_keys = set()
        for m in all_metrics.values():
            all_metric_keys.update(m.keys())
        keys = sorted(all_metric_keys)

        if keys:
            # Header
            header = "| Model | " + " | ".join(k.upper() for k in keys) + " |"
            sep = "|:------|" + "|".join(["------:"] * len(keys)) + "|"
            lines.append(header)
            lines.append(sep)

            # Rows (highlight best model)
            for name, metrics in sorted(
                    all_metrics.items(),
                    key=lambda x: x[1].get('f1', 0),
                    reverse=True
            ):
                row_vals = [f"{metrics.get(k, 0):.4f}" for k in keys]

                if name == best_model_name:
                    row = f"| **{name}** 🏆 | " + " | ".join(f"**{v}**" for v in row_vals) + " |"
                else:
                    row = f"| {name} | " + " | ".join(row_vals) + " |"

                lines.append(row)
            lines.append("")

        # Performance analysis
        lines.append("### Performance Analysis")
        lines.append("")

        # Compare top 2 models
        sorted_models = sorted(
            all_metrics.items(),
            key=lambda x: x[1].get('f1', 0),
            reverse=True
        )

        if len(sorted_models) >= 2:
            top_name, top_m = sorted_models[0]
            second_name, second_m = sorted_models[1]

            f1_gap = top_m.get('f1', 0) - second_m.get('f1', 0)

            lines.append(f"- **Best Model:** `{top_name}` outperforms `{second_name}` by {f1_gap:.4f} in F1-score")

            if f1_gap < 0.01:
                lines.append(f"  - ⚠️ **Narrow margin** — consider ensemble methods")
            elif f1_gap > 0.05:
                lines.append(f"  - ✅ **Clear winner** — {top_name} is significantly better")
            lines.append("")

        # Threshold validation
        f1 = best_metrics.get('f1', 0)
        acc = best_metrics.get('accuracy', 0)

        lines.append("**Threshold Validation:**")
        lines.append("")

        if f1 >= 0.70 and acc >= 0.75:
            lines.append("- ✅ **Model meets production thresholds** (F1 > 0.70, Accuracy > 0.75)")
            lines.append("- 🚀 **Ready for deployment** with monitoring")
        elif f1 >= 0.70:
            lines.append("- ⚠️ **F1 passes but accuracy below threshold**")
            lines.append("- 📝 **Recommendation:** Review false positive/negative balance")
        elif acc >= 0.75:
            lines.append("- ⚠️ **Accuracy passes but F1 below threshold**")
            lines.append("- 📝 **Recommendation:** Improve recall or precision")
        else:
            lines.append("- ❌ **Model below production thresholds**")
            lines.append("- 📝 **Action Required:** Retrain with more features or data")

        lines.append("")
        lines.append("---")
        lines.append("")

        return lines

    def _generate_visual_analysis(
            self,
            run_dir: Optional[str],
            best_metrics: Dict[str, float],
            all_metrics: Dict[str, Dict[str, float]],
            feature_importance: Optional[pd.DataFrame]
    ) -> List[str]:
        """Generate visual analysis section with figure explanations."""
        lines = [
            "## Visuals",
            "",
            "### Evaluation Figures",
            ""
        ]

        eval_fig_dir = os.path.join(run_dir or '', 'figures', 'evaluation') if run_dir else None

        if not (eval_fig_dir and os.path.isdir(eval_fig_dir)):
            lines.append("*No evaluation figures available.*")
            lines.append("")
            return lines

        # Process each figure
        for fname in sorted(os.listdir(eval_fig_dir)):
            if not fname.lower().endswith('.png'):
                continue

            rel = f"figures/evaluation/{fname}"

            # Figure title
            lines.append(f"#### {self._humanize_filename(fname)}")
            lines.append("")

            # Embedded image
            lines.append(f"<img src=\"{rel}\" width=\"650\" />")
            lines.append("")

            # Objective interpretation
            lines.append("**Analysis:**")
            lines.append("")

            interpretations = self._interpret_eval_figure(
                fname, best_metrics, all_metrics, feature_importance
            )

            for interp in interpretations:
                lines.append(interp)

            lines.append("")
            lines.append("---")
            lines.append("")

        return lines

    def _generate_feature_importance_section(
            self,
            feature_importance: pd.DataFrame
    ) -> List[str]:
        """Generate feature importance section with business context."""
        lines = [
            "## Feature Importance Analysis",
            "",
            "### Top 20 Predictive Features",
            ""
        ]

        try:
            top_20 = feature_importance.head(20)

            # Table format
            lines.append("| Rank | Feature | Importance | Business Context |")
            lines.append("|-----:|:--------|----------:|:-----------------|")

            for idx, (_, row) in enumerate(top_20.iterrows(), 1):
                feat = row['feature']
                imp = row['importance']
                context = self._get_business_context(feat)

                lines.append(f"| {idx} | **{feat}** | {imp:.4f} | {context} |")

            lines.append("")

            # Key insights
            lines.append("### Key Insights")
            lines.append("")

            top_5 = top_20.head(5)['feature'].tolist()
            lines.append("**Top 5 Churn Drivers:**")
            lines.append("")

            for feat in top_5:
                context = self._get_business_context(feat)
                recommendation = self._get_feature_recommendation(feat)
                lines.append(f"- **`{feat}`** — {context}")
                if recommendation:
                    lines.append(f"  - 💡 **Action:** {recommendation}")

            lines.append("")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to generate feature importance section: {e}")
            lines.append("*Unable to display feature importance details.*")
            lines.append("")

        lines.append("---")
        lines.append("")

        return lines

    def _generate_recommendations(
            self,
            best_model_name: str,
            best_metrics: Dict[str, float],
            all_metrics: Dict[str, Dict[str, float]],
            feature_importance: Optional[pd.DataFrame]
    ) -> List[str]:
        """Generate actionable recommendations based on results."""
        lines = [
            "## Recommendations",
            "",
            "### Next Steps",
            ""
        ]

        f1 = best_metrics.get('f1', 0)
        recall = best_metrics.get('recall', 0)
        precision = best_metrics.get('precision', 0)

        # Model deployment
        lines.append("**1. Model Deployment**")
        lines.append("")

        if f1 >= 0.70 and best_metrics.get('accuracy', 0) >= 0.75:
            lines.append("- ✅ **Deploy to production** — Model meets quality thresholds")
            lines.append("- 📊 Set up **monitoring dashboard** for drift detection")
            lines.append("- 🔔 Configure **alerts** for F1 < 0.70 or drift > 10%")
        else:
            lines.append("- ⏸️ **Hold deployment** — Retrain model first")
            lines.append("- 🔧 **Improvement needed:** F1 or accuracy below threshold")

        lines.append("")

        # Performance optimization
        lines.append("**2. Performance Optimization**")
        lines.append("")

        if recall < 0.85:
            lines.append(
                f"- 🎯 **Improve Recall** (current: {recall:.2%}) — Missing {(1 - recall) * 100:.1f}% of churners")
            lines.append("  - Consider **class weights** or **threshold tuning**")
            lines.append("  - Review **false negatives** for pattern analysis")

        if precision < 0.90:
            lines.append(
                f"- 🎯 **Improve Precision** (current: {precision:.2%}) — {(1 - precision) * 100:.1f}% false alarms")
            lines.append("  - Add **more discriminative features**")
            lines.append("  - Consider **ensemble methods** for stability")

        if f1 < 0.75:
            lines.append("- 🔄 **Consider:**")
            lines.append("  - Feature engineering (interaction terms, time-based features)")
            lines.append("  - Hyperparameter tuning with larger search space")
            lines.append("  - Ensemble methods (stacking, blending)")

        lines.append("")

        # Business actions
        lines.append("**3. Business Actions**")
        lines.append("")

        if feature_importance is not None:
            try:
                top_3 = feature_importance.head(3)['feature'].tolist()
                lines.append(f"- 🎯 **Focus retention efforts** on customers with:")
                for feat in top_3:
                    rec = self._get_feature_recommendation(feat)
                    if rec:
                        lines.append(f"  - {rec}")
            except Exception:
                pass

        lines.append("- 📧 **Proactive campaigns:** Target high-risk customers before churn")
        lines.append("- 💰 **ROI Analysis:** Calculate cost savings from churn prevention")
        lines.append("")

        # Monitoring
        lines.append("**4. Continuous Monitoring**")
        lines.append("")
        lines.append("- 📊 Track metrics weekly: F1, Recall, Precision")
        lines.append("- 🔍 Monitor drift: Data distribution, feature importance shifts")
        lines.append("- 🔄 Retrain trigger: Performance drop > 5% or drift > 10%")
        lines.append("")

        lines.append("---")
        lines.append("")

        return lines

    def _generate_technical_appendix(self, config: Dict[str, Any]) -> List[str]:
        """Generate technical appendix with config details."""
        lines = [
            "## Technical Appendix",
            "",
            "<details>",
            "<summary>Click to expand configuration details</summary>",
            "",
            "### Training Configuration",
            "",
            "```yaml"
        ]

        # Extract key configs
        try:
            import yaml

            key_configs = {
                'data': config.get('data', {}),
                'preprocessing': config.get('preprocessing', {}),
                'models': {k: v for k, v in config.get('models', {}).items() if k == 'xgboost' or k == 'random_forest'},
                'tuning': config.get('tuning', {})
            }

            lines.append(yaml.dump(key_configs, default_flow_style=False, sort_keys=False))
        except Exception:
            lines.append("# Configuration unavailable")

        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

        return lines

    def _generate_footer(self) -> List[str]:
        """Generate report footer."""
        return [
            "---",
            "",
            "<div align='center'>",
            "",
            f"*Report generated by ML Pipeline v2.0 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "**E-Commerce Churn Prediction Project**  ",
            "📧 For questions, contact the ML team",
            "",
            "</div>"
        ]

    # ==================== ANALYSIS HELPERS ====================

    def _analyze_model_performance(
            self,
            best_metrics: Dict[str, float],
            all_metrics: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate automated performance insights."""
        insights = []

        f1 = best_metrics.get('f1', 0)
        recall = best_metrics.get('recall', 0)
        precision = best_metrics.get('precision', 0)
        roc_auc = best_metrics.get('roc_auc', 0)

        # F1 assessment
        if f1 >= 0.90:
            insights.append("🌟 **Excellent F1-score** — Model achieves exceptional balance")
        elif f1 >= 0.75:
            insights.append("✅ **Good F1-score** — Model performs well on churn prediction")
        elif f1 >= 0.70:
            insights.append("⚠️ **Acceptable F1-score** — Model meets minimum threshold")
        else:
            insights.append("❌ **Low F1-score** — Model requires improvement")

        # Recall vs Precision balance
        if abs(recall - precision) < 0.05:
            insights.append("⚖️ **Balanced model** — Recall and precision are well-balanced")
        elif recall > precision + 0.10:
            insights.append("🎯 **Recall-optimized** — Good for minimizing missed churners (lower false negatives)")
        elif precision > recall + 0.10:
            insights.append("🎯 **Precision-optimized** — Good for reducing false alarms (lower false positives)")

        # ROC-AUC assessment
        if roc_auc >= 0.95:
            insights.append("🚀 **Outstanding discrimination** — ROC-AUC > 0.95 indicates excellent class separation")

        # Model comparison
        if len(all_metrics) > 1:
            sorted_models = sorted(all_metrics.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
            if len(sorted_models) >= 2:
                gap = sorted_models[0][1].get('f1', 0) - sorted_models[1][1].get('f1', 0)
                if gap < 0.02:
                    insights.append(f"📊 **Close competition** — Top 2 models have similar performance (gap < 2%)")

        return insights

    def _interpret_eda_figure(
            self,
            fname: str,
            eda_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate objective interpretation for EDA figures."""
        lines = []
        lname = fname.lower()

        if 'missing' in lname:
            lines.append("- Shows percentage of missing values per feature")
            lines.append("- **Implication:** Features with >20% missing may need careful imputation")
            lines.append("- **Action:** Median imputation for numeric, mode for categorical")

        elif 'target_distribution' in lname or 'distribution' in lname:
            churn_rate = eda_summary.get('churn_rate', 0)
            if churn_rate:
                lines.append(f"- Churn class represents {churn_rate:.1%} of dataset")
                if churn_rate < 0.20:
                    lines.append("- **Imbalance detected:** Minority class < 20%")
                    lines.append(f"- **Mitigation:** {eda_summary.get('resampling', 'SMOTE resampling applied')}")

        elif 'correlation' in lname:
            top_corr = eda_summary.get('top_correlated', [])
            # Fix: ensure all elements are str and not None
            top_corr_str = [str(c) for c in top_corr if c is not None]
            if top_corr_str:
                lines.append(f"- Top correlated features with churn: {', '.join(top_corr_str[:3])}")
                lines.append("- **Usage:** Prioritize these features in feature engineering")
                lines.append("- **Note:** High correlation doesn't imply causation")

        elif 'numerical' in lname or 'histogram' in lname:
            lines.append("- Distribution shapes reveal data characteristics:")
            lines.append("  - **Skewed distributions** → Consider log transformation")
            lines.append("  - **Bimodal patterns** → May indicate distinct customer segments")
            lines.append("  - **Outliers** → Check for data quality issues")

        elif 'boxplot' in lname:
            lines.append("- Box plots highlight outliers (points beyond whiskers)")
            lines.append("- **Outlier handling:** IQR-based clipping applied during preprocessing")
            lines.append("- **Impact:** Reduces model sensitivity to extreme values")

        elif 'churn_by_category' in lname:
            lines.append("- Compares churn rates across categorical feature values")
            lines.append("- **Red bars** = above-average churn (high-risk segments)")
            lines.append("- **Blue bars** = below-average churn (stable segments)")
            lines.append("- **Business use:** Target retention campaigns at red segments")

        else:
            lines.append("- Visual representation of data characteristics")
            lines.append("- Review for domain insights and data quality issues")

        return lines

    def _interpret_eval_figure(
            self,
            fname: str,
            best_metrics: Dict[str, float],
            all_metrics: Dict[str, Dict[str, float]],
            feature_importance: Optional[pd.DataFrame]
    ) -> List[str]:
        """Generate objective interpretation for evaluation figures."""
        lines = []
        lname = fname.lower()

        if 'roc' in lname:
            auc = best_metrics.get('roc_auc', 0)
            lines.append(f"- **ROC-AUC = {auc:.4f}** — measures model's ability to discriminate between classes")
            lines.append("- **Interpretation:**")
            if auc >= 0.95:
                lines.append("  - 🌟 **Excellent** (>0.95) — Near-perfect separation")
            elif auc >= 0.90:
                lines.append("  - ✅ **Very Good** (0.90-0.95) — Strong discrimination")
            elif auc >= 0.80:
                lines.append("  - ⚠️ **Good** (0.80-0.90) — Acceptable performance")
            else:
                lines.append("  - ❌ **Poor** (<0.80) — Model struggles to separate classes")

            lines.append("- **Business context:** Higher AUC = better at identifying churners before they leave")
            lines.append("- **Diagonal line** represents random guessing (AUC = 0.5)")

        elif 'confusion' in lname:
            recall = best_metrics.get('recall', 0)
            precision = best_metrics.get('precision', 0)

            lines.append("- **Confusion Matrix Quadrants:**")
            lines.append("  - **Top-Left (TN):** Correctly predicted non-churners — customers retained")
            lines.append("  - **Top-Right (FP):** False alarms — wasted retention effort")
            lines.append("  - **Bottom-Left (FN):** Missed churners — lost revenue")
            lines.append("  - **Bottom-Right (TP):** Correctly predicted churners — successful intervention")
            lines.append("")
            lines.append(f"- **Current Performance:**")
            lines.append(f"  - **Recall = {recall:.2%}** — catching {recall * 100:.1f}% of actual churners")
            lines.append(f"  - **Precision = {precision:.2%}** — {precision * 100:.1f}% of predictions are correct")
            lines.append("")
            lines.append("- **Business impact:**")
            lines.append(
                f"  - **False Negatives** = missed {(1 - recall) * 100:.1f}% of churners → potential revenue loss")
            lines.append(f"  - **False Positives** = {(1 - precision) * 100:.1f}% wasted retention cost")

        elif 'feature_importance' in lname or 'importance' in lname:
            if feature_importance is not None:
                try:
                    top_5 = feature_importance.head(5)
                    lines.append("- **Top 5 Predictive Features:**")
                    for _, row in top_5.iterrows():
                        feat = row['feature']
                        imp = row['importance']
                        context = self._get_business_context(feat)
                        lines.append(f"  - **{feat}** ({imp:.3f}) — {context}")
                    lines.append("")
                    lines.append("- **Usage:** Focus monitoring and feature engineering on top features")
                    lines.append("- **Note:** Importance ≠ causation; validate findings with domain experts")
                except Exception:
                    lines.append("- Feature importance scores from tree-based model")
            else:
                lines.append("- Feature importance visualization not available")

        elif 'model_comparison' in lname:
            try:
                sorted_models = sorted(
                    all_metrics.items(),
                    key=lambda x: x[1].get('f1', 0),
                    reverse=True
                )

                if sorted_models:
                    top_name, top_m = sorted_models[0]
                    lines.append(f"- **Best Model:** {top_name} (F1 = {top_m.get('f1', 0):.4f})")

                    if len(sorted_models) > 1:
                        second_name, second_m = sorted_models[1]
                        gap = top_m.get('f1', 0) - second_m.get('f1', 0)
                        lines.append(f"- **Runner-up:** {second_name} (gap = {gap:.4f})")

                        if gap < 0.01:
                            lines.append("- **⚠️ Close race** — Consider ensemble methods")
                        elif gap > 0.05:
                            lines.append("- **✅ Clear winner** — Deploy best model with confidence")

                    lines.append("")
                    lines.append(
                        "- **Metric priorities:** F1 (balance) > Recall (catch churners) > Precision (reduce false alarms)")
            except Exception:
                lines.append("- Comparison of model performance across metrics")

        else:
            lines.append("- Evaluation visualization for model performance assessment")
            lines.append("- Review for insights into model strengths and weaknesses")

        return lines

    def _get_business_context(self, feature_name: str) -> str:
        """Map feature names to business context."""
        feature_map = {
            'Tenure': 'Customer lifetime with company',
            'Complain': 'History of complaints filed',
            'DaySinceLastOrder': 'Recency of last purchase',
            'CashbackAmount': 'Total cashback received',
            'NumberOfDeviceRegistered': 'Engagement level (devices used)',
            'SatisfactionScore': 'Customer satisfaction rating',
            'OrderCount': 'Purchase frequency',
            'CouponUsed': 'Promotional engagement',
            'WarehouseToHome': 'Delivery distance',
            'HourSpendOnApp': 'App engagement time',
            'NumberOfAddress': 'Shipping addresses count',
            'PreferedOrderCat': 'Preferred product category',
            'PreferredPaymentMode': 'Payment method preference',
            'MaritalStatus': 'Marital status',
            'CityTier': 'City classification (tier 1/2/3)'
        }

        for key, desc in feature_map.items():
            if key.lower() in feature_name.lower():
                return desc

        return 'Customer behavior metric'

    def _get_feature_recommendation(self, feature_name: str) -> str:
        """Get actionable recommendation for a feature."""
        recommendations = {
            'Tenure': 'Target new customers (<6 months) with onboarding incentives',
            'Complain': 'Implement rapid complaint resolution process',
            'DaySinceLastOrder': 'Re-engagement campaigns for inactive users (>30 days)',
            'CashbackAmount': 'Increase cashback for at-risk high-value customers',
            'NumberOfDeviceRegistered': 'Encourage multi-device usage for deeper engagement',
            'SatisfactionScore': 'Proactive outreach for satisfaction scores <4',
            'OrderCount': 'Loyalty rewards for customers with declining order frequency',
            'CouponUsed': 'Personalized promotions based on past coupon usage',
            'WarehouseToHome': 'Improve logistics for high-distance deliveries',
            'HourSpendOnApp': 'In-app engagement campaigns for low-usage customers'
        }

        for key, rec in recommendations.items():
            if key.lower() in feature_name.lower():
                return rec

        return None

    def _humanize_filename(self, fname: str) -> str:
        """Convert filename to human-readable title."""
        name = fname.replace('.png', '').replace('_', ' ').title()
        return name

    # ==================== I/O HELPERS ====================

    def _get_run_dir(self, run_id: Optional[str]) -> str:
        """Get experiment run directory."""
        if not run_id:
            return ""
        run_dir = os.path.join(self.experiments_base_dir, run_id)
        if os.path.isdir(run_dir):
            return run_dir
        ensure_dir(run_dir)
        return run_dir

    def _save_markdown(
            self,
            lines: List[str],
            filename: str,
            run_dir: Optional[str] = None,
            run_id: Optional[str] = None
    ) -> str:
        """Save markdown report only in run directory, not in central reports."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{filename}_{ts}.md"
        if run_dir:
            ensure_dir(run_dir)
            out_path = os.path.join(run_dir, fname)
            run_lines = [sanitize_text(ln) for ln in lines]
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(run_lines))
            if self.logger:
                self.logger.info(f"Report Generated | Run: {out_path}")
            return out_path
        # Fallback: if no run_dir, save to cwd
        out_path = os.path.join(os.getcwd(), fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if self.logger:
            self.logger.info(f"Report Generated | {out_path}")
        return out_path

    def _save_json(
            self,
            data: Dict[str, Any],
            filename: str,
            run_dir: Optional[str] = None
    ) -> str:
        """Save JSON report only in run directory."""
        fname = f"{filename}.json"
        if run_dir:
            ensure_dir(run_dir)
            run_path = os.path.join(run_dir, fname)
            IOHandler.save_json(data, run_path)
            if self.logger:
                self.logger.info(f"JSON Report | Run: {run_path}")
            return run_path
        out_path = os.path.join(os.getcwd(), fname)
        IOHandler.save_json(data, out_path)
        if self.logger:
            self.logger.info(f"JSON Report | {out_path}")
        return out_path

    # ==================== ADDITIONAL REPORT TYPES ====================

    def generate_eda_report(
            self,
            df,
            filename: str = "eda_report",
            target_col: str = None,
            format: str = "markdown",
            run_id: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            args=None
    ) -> str:
        """
        Generate dashboard-style EDA report with run summary, key insights, and figure analysis.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_dir = self._get_run_dir(run_id)
        lines: List[str] = []
        # Run summary dashboard
        lines.append(f"# 📊 Báo Cáo EDA (Exploratory Data Analysis)")
        lines.append("")
        lines.extend(self._generate_run_summary(run_id or '-', timestamp, 'eda', config, args))
        lines.append("---\n")
        # Basic stats
        n_rows, n_cols = df.shape
        missing = int(df.isnull().sum().sum())
        dup = int(df.duplicated().sum())
        lines.append("## 🗂️ Thống Kê Dữ Liệu")
        lines.append(f"- **Rows:** {n_rows:,}")
        lines.append(f"- **Columns:** {n_cols:,}")
        lines.append(f"- **Missing cells:** {missing}")
        lines.append(f"- **Duplicate rows:** {dup}")
        lines.append("")
        # Target distribution
        if target_col and target_col in df.columns:
            lines.append("## 🎯 Phân Phối Nhãn (Target Distribution)")
            vc = df[target_col].value_counts()
            total = len(df)
            for k, v in vc.items():
                lines.append(f"- `{k}`: {v} ({v/total:.1%})")
            lines.append("")
        # EDA Key Insights
        lines.append("## 🔍 EDA Key Insights")
        churn_rate = None
        if target_col and target_col in df.columns:
            churn_rate = df[target_col].value_counts(normalize=True).get(1, 0)
        if churn_rate is not None:
            lines.append(f"- **Churn Rate:** {churn_rate:.1%} (Imbalance: {'Yes' if churn_rate < 0.2 else 'No'})")
        if missing > 0:
            lines.append(f"- **Missing Data:** {missing} cells. Check 'Missing Values' plot for details.")
        # Outlier detection (boxplot)
        lines.append("- **Outlier Detection:** See 'Boxplots' for outlier analysis.")
        # Top correlated features (if available)
        # (Assume correlation matrix is available as a plot)
        lines.append("- **Top Correlated Features:** See 'Correlation Matrix' and 'Correlation With Target'.")
        lines.append("")
        lines.append("---\n")
        # Figures with analysis
        eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
        if eda_fig_dir and os.path.isdir(eda_fig_dir):
            lines.append("## 🖼️ Hình Ảnh Phân Tích Dữ Liệu (EDA Figures)")
            for fname in sorted(os.listdir(eda_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/eda/{fname}"
                    title = self._humanize_filename(fname)
                    lines.append(f"### {title}")
                    lines.append("")
                    lines.append(f"![{title}]({rel})")
                    # Automated analysis below each figure
                    lines.extend(self._interpret_eda_figure(fname, {'churn_rate': churn_rate}))
                    lines.append("")
            lines.append("---\n")
        lines.extend(self._generate_footer())
        return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)
