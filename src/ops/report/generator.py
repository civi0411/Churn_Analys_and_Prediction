import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import pandas as pd

from ..reporting_helpers import sanitize_text, figure_insights
from ...utils import ensure_dir


class ReportGenerator:
    """Simplified, focused ReportGenerator placed under src.ops.report

    Responsibilities:
    - Produce per-run markdown report under the run directory
    - Support modes: eda, train, full, visualize, preprocess
    - Embed figures using relative links to run-specific figures
    - Generate concise, dashboard-like markdown
    """

    def __init__(self, reports_dir: str = None, experiments_base_dir: str = "artifacts/experiments", logger=None):
        self.experiments_base_dir = experiments_base_dir
        self.reports_dir = reports_dir or os.path.join("artifacts", "reports")
        ensure_dir(self.reports_dir)
        self.logger = logger

    def _get_run_dir(self, run_id: str) -> Optional[str]:
        if not run_id:
            return None
        run_dir = os.path.join(self.experiments_base_dir, run_id)
        return run_dir

    def generate_training_report(self, run_id: str, best_model_name: str, all_metrics: Dict[str, Dict[str, float]],
                                 feature_importance: Optional[pd.DataFrame] = None, config: Optional[Dict[str, Any]] = None,
                                 eda_summary: Optional[Dict[str, Any]] = None, format: str = "markdown", mode: str = "train",
                                 optimize: bool = True, args=None) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_dir = self._get_run_dir(run_id)
        if run_dir:
            ensure_dir(run_dir)
        filename_md = f"training_report_{run_id}.md"
        filename_json = f"training_report_{run_id}.json"

        lines: List[str] = []
        best_metrics = all_metrics.get(best_model_name, {})
        eval_fig_dir = os.path.join(run_dir, 'figures', 'evaluation') if run_dir else None

        # Header / Dashboard
        lines.append(f"# 🏆 Báo Cáo Huấn Luyện Tự Động (Training Report) — {sanitize_text(run_id)}")
        lines.append("")
        lines.append("## 🎯 Tổng Quan Dữ Liệu & Quy Trình")
        lines.append("")
        lines.append("| Chi tiết | Giá trị |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Run ID** | `{sanitize_text(run_id)}` |")
        lines.append(f"| **Thời gian Thực hiện** | {timestamp} |")
        lines.append(f"| **Mode** | {mode} |")
        lines.append(f"| **Optimize** | {optimize} |")
        lines.append(f"| **Best Model** | {best_model_name} |")
        lines.append("")
        lines.append("---\n")

        if mode == 'eda' and eda_summary:
            churn_rate = eda_summary.get('churn_rate', None)
            lines.append("## 🔍 Tóm Tắt Phân Tích Dữ Liệu (EDA)")
            if churn_rate is not None:
                # explicit guard for static type checkers
                assert churn_rate is not None
                # safe formatting - coerce to float when possible
                try:
                    churn_rate_f = float(churn_rate)
                    lines.append(f"* **Vấn đề Mất cân bằng:** Tỷ lệ Churn (Lớp 1) là **{churn_rate_f:.2%}** — Non-Churn **{(1-churn_rate_f):.2%}**.")
                except Exception:
                    lines.append(f"* **Vấn đề Mất cân bằng:** Tỷ lệ Churn (Lớp 1): {churn_rate}")
            if 'top_correlated' in eda_summary:
                top = eda_summary.get('top_correlated', [])[:3]
                if top:
                    lines.append(f"* **Tương quan ban đầu (Top):** {', '.join(top)}")
            lines.append("")
            eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
            if eda_fig_dir and os.path.isdir(eda_fig_dir):
                lines.append("### Key Visualizations (EDA)")
                for fname in sorted(os.listdir(eda_fig_dir)):
                    if fname.lower().endswith('.png'):
                        rel = f"figures/eda/{fname}"
                        lines.append(f"**{fname}**")
                        lines.append("")
                        lines.append(f"![{fname}]({rel})")
                        insights = figure_insights(fname, best_metrics or {}, all_metrics or {}, feature_importance)
                        for l in insights:
                            lines.append(l)
                        lines.append("")
            lines.append(self._footer())
            path = os.path.join(run_dir or self.reports_dir, filename_md)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return path

        # TRAIN or FULL mode
        if mode in ('train', 'full'):
            # Leaderboard
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

            # Champion metrics
            lines.append("---\n")
            lines.append(f"## 🥇 Kết Quả Đánh Giá Mô Hình Tốt Nhất — {best_model_name.upper()}")
            lines.append("")
            lines.append("| Metric | Giá trị | Ngưỡng MLOps (Min) | Trạng Thái |")
            lines.append("| :--- | :--- | :--- | :--- |")
            lines.append(f"| **Accuracy** | **{best_metrics.get('accuracy',0):.4f}** | > 0.75 | {'✅ PASS' if best_metrics.get('accuracy',0)>0.75 else '❌ FAIL'} |")
            lines.append(f"| **F1-Score** | **{best_metrics.get('f1',0):.4f}** | > 0.70 | {'✅ PASS' if best_metrics.get('f1',0)>0.70 else '❌ FAIL'} |")
            lines.append(f"| **Recall (Churn Coverage)** | {best_metrics.get('recall',0):.4f} | - | - |")
            lines.append(f"| **ROC-AUC** | **{best_metrics.get('roc_auc',0):.4f}** | - | - |")
            lines.append("")

            # Evaluation figures
            if eval_fig_dir and os.path.isdir(eval_fig_dir):
                lines.append("## 🖼️ Evaluation Figures & Analysis")
                for fname in sorted(os.listdir(eval_fig_dir)):
                    if fname.lower().endswith('.png'):
                        rel = f"figures/evaluation/{fname}"
                        lines.append(f"### {fname}")
                        lines.append("")
                        lines.append(f"![{fname}]({rel})")
                        insights = figure_insights(fname, best_metrics, all_metrics, feature_importance)
                        for l in insights:
                            lines.append(l)
                        lines.append("")
                lines.append("---\n")

            # Feature importance
            if feature_importance is not None:
                lines.append("## 📈 Giải Thích Mô Hình (Feature Importance)")
                try:
                    top = feature_importance.head(10)
                    lines.append("| Feature | Importance |")
                    lines.append("| :--- | ---: |")
                    for _, r in top.iterrows():
                        lines.append(f"| {r['feature']} | {r['importance']:.4f} |")
                    lines.append("")
                except Exception:
                    pass

            lines.append("## 🚨 Kết Luận & Kế Hoạch Giám Sát")
            lines.append(f"- Mô hình được chọn làm baseline: {best_model_name}")
            lines.append("- Kích hoạt monitoring cho F1 và Data Drift.")
            lines.append("")
            # Compact experiment signature (short, reproducible summary)
            if config:
                try:
                    sig = {
                        'run_id': run_id,
                        'timestamp': timestamp,
                        'mode': mode,
                        'optimize': bool(optimize),
                        'data_path': config.get('data', {}).get('raw_path'),
                        'random_state': config.get('data', {}).get('random_state'),
                        'models': list(config.get('models', {}).keys())[:10]
                    }
                    import yaml
                    lines.append('---')
                    lines.append('## Experiment Signature')
                    lines.append('```yaml')
                    lines.extend(yaml.dump(sig, default_flow_style=False, allow_unicode=True).splitlines())
                    lines.append('```')
                except Exception:
                    # fallback to minimal signature
                    lines.append('---')
                    lines.append('## Experiment Signature')
                    lines.append(f"- run_id: {run_id}")
                    lines.append(f"- mode: {mode}")
                    lines.append(f"- optimize: {optimize}")
                    lines.append(f"- data_path: {config.get('data', {}).get('raw_path')}")
                    lines.append("")

            lines.append(self._footer())

            path_md = os.path.join(run_dir or self.reports_dir, filename_md)
            md_text = '\n'.join(lines)
            with open(path_md, 'w', encoding='utf-8') as f:
                f.write(md_text)

            # Optionally also emit JSON summary when requested
            if format == 'json':
                summary = {
                    'run_id': run_id,
                    'timestamp': timestamp,
                    'mode': mode,
                    'optimize': optimize,
                    'best_model': best_model_name,
                    'best_metrics': best_metrics,
                    'all_metrics': all_metrics
                }
                path_json = os.path.join(run_dir or self.reports_dir, filename_json)
                with open(path_json, 'w', encoding='utf-8') as fj:
                    json.dump(summary, fj, indent=2, ensure_ascii=False)
                return path_json

            # Markdown-first: always return Markdown path unless caller explicitly requested JSON
            return path_md

        # fallback
        path = os.path.join(run_dir or self.reports_dir, filename_md)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(["No content for selected mode."]))
        return path

    def generate_eda_report(self, df: pd.DataFrame, filename: str, target_col: Optional[str] = None, format: str = 'markdown', run_id: Optional[str] = None) -> str:
        """Generate a focused EDA markdown report and save to the run directory if provided.

        Embeds figures found under <run_dir>/figures/eda and auto-generates insights.
        """
        run_dir = self._get_run_dir(run_id) if run_id else None
        if run_dir:
            ensure_dir(run_dir)
        filename_md = f"{filename}.md" if filename.endswith('.md') is False else filename
        lines: List[str] = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines.append(f"# EDA Report — {sanitize_text(filename)}")
        lines.append("")
        lines.append(f"**Generated:** {timestamp}")
        lines.append("")
        lines.append("## Dataset Summary")
        lines.append("")
        lines.append(f"- Rows: {df.shape[0]}")
        lines.append(f"- Columns: {df.shape[1]}")
        if target_col and target_col in df.columns:
            try:
                churn_rate = df[target_col].value_counts(normalize=True).get(1, 0)
                lines.append(f"- Churn Rate: {churn_rate:.2%}")
            except Exception:
                lines.append(f"- Churn Rate: unknown")
        lines.append("")

        eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
        if eda_fig_dir and os.path.isdir(eda_fig_dir):
            lines.append("## Key Visualizations")
            for fname in sorted(os.listdir(eda_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/eda/{fname}"
                    lines.append(f"### {fname}")
                    lines.append("")
                    lines.append(f"![{fname}]({rel})")
                    insights = figure_insights(fname, {}, {}, None)
                    for l in insights:
                        lines.append(l)
                    lines.append("")

        lines.append(self._footer())

        path_md = os.path.join(run_dir or self.reports_dir, filename_md)
        md_text = '\n'.join(lines)
        with open(path_md, 'w', encoding='utf-8') as f:
            f.write(md_text)

        return path_md

    def _footer(self) -> str:
        return "---\n\n_Report auto-generated by ML Pipeline_"
