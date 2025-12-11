"""
Module `ops.report.generator` - tạo báo cáo Markdown cho mỗi run (figures + summaries).

Important keywords: Args, Returns, Methods
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, cast
import pandas as pd
from ...utils import ensure_dir


def sanitize_text(text):
    """Clean text for markdown output."""
    if not isinstance(text, str):
        return str(text)
    return text.replace('|', '').replace('`', '').replace('\n', ' ').strip()


def figure_insights(fname, best_metrics, all_metrics, feature_importance):
    """Return short insights for given figure filename (placeholder)."""
    insights = []
    # Example: Add custom logic for each figure type
    if 'missing' in fname:
        insights.append('Biểu đồ này cho thấy tỷ lệ thiếu dữ liệu trên từng trường.')
    elif 'correlation' in fname:
        insights.append('Biểu đồ này thể hiện mức độ tương quan giữa các biến và nhãn Churn.')
    elif 'churn_by_category' in fname:
        insights.append('Phân tích churn theo từng phân khúc ngành hàng.')
    elif 'roc_curve' in fname:
        insights.append('Đường cong ROC cho thấy khả năng phân biệt giữa các lớp của mô hình.')
    elif 'confusion_matrix' in fname:
        insights.append('Ma trận nhầm lẫn giúp đánh giá chi tiết các loại lỗi của mô hình.')
    elif 'feature_importance' in fname:
        insights.append('Biểu đồ này cho thấy các đặc trưng quan trọng nhất ảnh hưởng đến dự đoán churn.')
    # Add more cases as needed
    return insights


class ReportGenerator:
    """
    Tạo báo cáo markdown cho từng run với các ảnh nhúng.

    Methods:
        generate_report(...)
    """

    def __init__(self, experiments_base_dir: str = "artifacts/experiments", logger=None):
        self.experiments_base_dir = experiments_base_dir
        self.logger = logger
        # Ensure base directory exists on init for compatibility with tests
        ensure_dir(self.experiments_base_dir)

    def _get_run_dir(self, run_id: str) -> Optional[str]:
        if not run_id:
            return None
        return os.path.join(self.experiments_base_dir, run_id)

    def generate_report(
            self,
            run_id: str,
            mode: str,
            optimize: bool = False,
            best_model_name: str = None,
            all_metrics: Dict[str, Dict[str, float]] = None,
            feature_importance: Optional[pd.DataFrame] = None,
            eda_summary: Optional[Dict[str, Any]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate comprehensive markdown report based on mode.

        Args:
            run_id: Run identifier
            mode: 'eda', 'train', 'full', 'preprocess', 'visualize'
            optimize: Whether hyperparameter tuning was used
            best_model_name: Name of best performing model
            all_metrics: Dictionary of all model metrics
            feature_importance: DataFrame with feature importance
            eda_summary: EDA statistics summary
            config: Configuration dict

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_dir = self._get_run_dir(run_id)

        if not run_dir:
            if self.logger:
                self.logger.warning("No run_id provided, skipping report generation")
            return None

        ensure_dir(run_dir)
        report_path = os.path.join(run_dir, "report.md")

        lines = []

        # === HEADER ===
        lines.append(f"# ML Pipeline Report - Run {sanitize_text(run_id)}")
        lines.append("")
        lines.append(f"**Generated:** {timestamp}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # === RUN INFORMATION ===
        lines.append("## Run Information")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|:----------|:------|")
        lines.append(f"| **Run ID** | `{sanitize_text(run_id)}` |")
        lines.append(f"| **Mode** | `{mode.upper()}` |")
        lines.append(f"| **Optimize** | `{optimize}` |")
        if config:
            lines.append(f"| **Data Source** | `{config.get('data', {}).get('raw_path', 'N/A')}` |")
            lines.append(f"| **Test Size** | `{config.get('data', {}).get('test_size', 0.2)}` |")
        if best_model_name:
            lines.append(f"| **Best Model** | `{best_model_name}` |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # === EDA SECTION ===
        if mode in ['eda', 'full'] and eda_summary:
            lines.extend(self._generate_eda_section(run_dir, eda_summary))

        # === TRAINING SECTION ===
        if mode in ['train', 'full'] and all_metrics:
            lines.extend(self._generate_training_section(
                run_dir, best_model_name, all_metrics, feature_importance
            ))

        # === PREPROCESSING SECTION ===
        if mode == 'preprocess':
            lines.append("## Data Preprocessing")
            lines.append("")
            lines.append("Data preprocessing completed successfully.")
            lines.append("")
            lines.append("**Steps performed:**")
            lines.append("1. Data loading and validation")
            lines.append("2. Data cleaning (duplicates, standardization)")
            lines.append("3. Train/test split with stratification")
            lines.append("4. Feature transformation and scaling")
            lines.append("")
            lines.append("**Output files:**")
            lines.append(f"- `{run_dir}/data/` - Processed datasets")
            lines.append("")
            lines.append("---")
            lines.append("")

        # === FOOTER ===
        lines.append("---")
        lines.append("")
        lines.append("_Report auto-generated by ML Pipeline_")
        lines.append("")

        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        if self.logger:
            self.logger.info(f"Report Generated | {report_path}")

        return report_path

    def _generate_eda_section(self, run_dir: str, eda_summary: Dict) -> list:
        """Generate EDA section with visualizations."""
        lines = []

        lines.append("## Exploratory Data Analysis")
        lines.append("")

        # Statistics
        lines.append("### Dataset Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|:-------|:------|")
        lines.append(f"| **Rows** | {eda_summary.get('n_rows', 'N/A'):,} |")
        lines.append(f"| **Columns** | {eda_summary.get('n_cols', 'N/A')} |")
        lines.append(f"| **Missing Values** | {eda_summary.get('missing', 0)} |")
        lines.append(f"| **Duplicates** | {eda_summary.get('duplicate', 0)} |")

        if 'churn_rate' in eda_summary:
            churn_rate = eda_summary['churn_rate']
            try:
                churn_pct = float(churn_rate) * 100
                lines.append(f"| **Churn Rate** | {churn_pct:.2f}% |")
            except:
                lines.append(f"| **Churn Rate** | {churn_rate} |")

        lines.append("")

        # Key Insights
        if 'top_correlated' in eda_summary and eda_summary['top_correlated']:
            lines.append("### Key Insights")
            lines.append("")
            top_features = eda_summary['top_correlated'][:3]
            lines.append(f"**Top correlated features with target:** {', '.join(top_features)}")
            lines.append("")

        # Visualizations
        eda_fig_dir = os.path.join(run_dir, 'figures', 'eda')
        if os.path.isdir(eda_fig_dir):
            lines.append("### Visualizations")
            lines.append("")

            for fname in sorted(os.listdir(eda_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel_path = f"figures/eda/{fname}"
                    lines.append(f"#### {fname.replace('_', ' ').replace('.png', '').title()}")
                    lines.append("")
                    lines.append(f"![{fname}]({rel_path})")
                    lines.append("")

                    # Add insights
                    insights = figure_insights(fname, {}, {}, None)
                    if insights:
                        for insight in insights:
                            lines.append(insight)
                        lines.append("")

        lines.append("---")
        lines.append("")
        return lines

    def _generate_training_section(self, run_dir: str, best_model_name: str, all_metrics: Dict[str, Dict[str, float]], feature_importance: Optional[pd.DataFrame] = None) -> list:
        """Generate training section for markdown report."""
        lines = []
        lines.append("## Model Training & Evaluation")
        lines.append("")
        if best_model_name:
            lines.append(f"**Best Model:** {best_model_name.upper()}")
            lines.append("")
        lines.append("### Metrics")
        lines.append("")
        lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
        lines.append("|:------|:--------:|:--------:|:------:|:--:|:-------:|")
        for mname, metrics in (all_metrics or {}).items():
            acc = metrics.get('accuracy', '')
            prec = metrics.get('precision', '')
            rec = metrics.get('recall', '')
            f1 = metrics.get('f1', '')
            roc = metrics.get('roc_auc', '')
            f1s = f"{float(f1):.4f}" if f1 != '' and f1 is not None else ''
            lines.append(f"| {mname.upper()} | {acc} | {prec} | {rec} | {f1s} | {roc} |")
        lines.append("")
        if feature_importance is not None:
            lines.append("### Feature Importance")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|:--------|:----------:|")
            for _, row in feature_importance.iterrows():
                lines.append(f"| {row['feature']} | {row['importance']} |")
            lines.append("")
        # Add evaluation figures if available
        eval_fig_dir = os.path.join(run_dir, 'figures', 'evaluation')
        if os.path.isdir(eval_fig_dir):
            lines.append("### Evaluation Visualizations")
            lines.append("")
            for fname in sorted(os.listdir(eval_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel_path = f"figures/evaluation/{fname}"
                    lines.append(f"#### {fname.replace('_', ' ').replace('.png', '').title()}")
                    lines.append("")
                    lines.append(f"![{fname}]({rel_path})")
                    lines.append("")
                    insights = figure_insights(fname, {}, all_metrics, feature_importance)
                    if insights:
                        for insight in insights:
                            lines.append(insight)
                        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    def generate_training_report(
            self,
            run_id: str,
            best_model_name: str,
            all_metrics: Dict[str, Dict[str, float]],
            feature_importance: Optional[pd.DataFrame] = None,
            config: Optional[Dict[str, Any]] = None,
            format: str = 'markdown'
    ) -> Optional[str]:
        """
        Compatibility wrapper used by tests. Produces either a markdown (.md) or JSON (.json)
        training report under the run directory.
        """
        run_dir = self._get_run_dir(run_id)
        if not run_dir:
            if self.logger:
                self.logger.warning("No run_id provided, skipping report generation")
            return None

        ensure_dir(run_dir)

        fmt = format.lower()
        if fmt not in ('markdown', 'json'):
            fmt = 'markdown'

        if fmt == 'markdown':
            report_path = os.path.join(run_dir, 'training_report.md')
            lines = []
            lines.append(f"# Training Report - {sanitize_text(run_id)}")
            lines.append("")
            lines.append(f"**Best Model**: {best_model_name.upper()}")
            lines.append("")
            lines.append("## Metrics")
            lines.append("")
            # Table header
            lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
            lines.append("|:------|:--------:|:--------:|:------:|:--:|:-------:|")

            for mname, metrics in (all_metrics or {}).items():
                acc = metrics.get('accuracy', '')
                prec = metrics.get('precision', '')
                rec = metrics.get('recall', '')
                f1 = metrics.get('f1', '')
                roc = metrics.get('roc_auc', '')
                # Format f1 to 4 decimals if present
                f1s = f"{float(f1):.4f}" if f1 != '' and f1 is not None else ''
                lines.append(f"| {mname.upper()} | {acc} | {prec} | {rec} | {f1s} | {roc} |")

            if feature_importance is not None:
                lines.append("")
                lines.append("## Feature Importance")
                lines.append("")
                # Simple table
                lines.append("| Feature | Importance |")
                lines.append("|:--------|:----------:|")
                for _, row in feature_importance.iterrows():
                    lines.append(f"| {row['feature']} | {row['importance']} |")

            # Include configuration summary if provided
            if config:
                lines.append("")
                lines.append("## Configuration")
                lines.append("")
                try:
                    # Briefly summarize top-level keys and values
                    for k, v in (config or {}).items():
                        lines.append(f"- **{k}**: {sanitize_text(str(v))}")
                except Exception:
                    lines.append("- Configuration summary not available")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            if self.logger:
                self.logger.info(f"Report Generated | {report_path}")

            return report_path

        else:  # json
            report_path = os.path.join(run_dir, 'training_report.json')
            payload = {
                'run_id': run_id,
                'best_model': best_model_name,
                'all_metrics': all_metrics,
            }
            if feature_importance is not None:
                # Convert DataFrame to serializable list-of-dicts and cast for type-checkers
                payload['feature_importance'] = cast(Any, feature_importance.to_dict(orient='records'))

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)

            if self.logger:
                self.logger.info(f"Report Generated | {report_path}")

            return report_path
