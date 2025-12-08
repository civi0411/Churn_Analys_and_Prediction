"""src/ops/reporting.py

ReportGenerator lives here (DataOps / MLOps concern).
Writes reports into experiments run directory and central artifacts/reports.
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..utils import ensure_dir, IOHandler


class ReportGenerator:
    """
    Report generator for EDA / Data / Training / Comparison reports.

    This class writes reports both to a central `reports_dir` folder
    and, when run_id is provided, into the experiment run directory
    `experiments/<run_id>/report_<run_id>.md` so every run contains its own
    report bundle.
    """

    def __init__(self, reports_dir: str = None, experiments_base_dir: str = "artifacts/experiments", logger=None):
        """Initialize ReportGenerator.

        Backwards-compatible: tests call ReportGenerator(report_dir) where the
        first positional argument is intended to be the reports directory.

        Args:
            reports_dir: Path to central reports directory. If None, defaults to
                         'artifacts/reports'.
            experiments_base_dir: Base dir for per-run experiment folders.
            logger: Optional logger.
        """
        self.experiments_base_dir = experiments_base_dir
        # If caller provided only one positional arg (reports_dir), use it.
        if reports_dir:
            self.reports_dir = reports_dir
        else:
            self.reports_dir = os.path.join("artifacts", "reports")

        ensure_dir(self.reports_dir)
        self.logger = logger

    def _save_markdown(self, lines: List[str], filename: str, run_dir: Optional[str] = None, run_id: Optional[str] = None) -> str:
        """Save markdown to central reports dir and optionally copy to run_dir.

        If run_id is provided, write two versions:
          - central: in artifacts/reports with image links pointing to ../experiments/<run_id>/figures/...
          - run-local: in the run_dir with image links pointing to figures/... (so images render when opening file inside run folder)
        Returns path to the run-local file if run_dir provided, else central file path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{filename}_{ts}.md"
        out_path = os.path.join(self.reports_dir, fname)

        # Prepare central_lines (with ../experiments/<run_id>/figures/... for images) and run_lines (figures/...)
        if run_dir and run_id:
            # Create a report-specific assets folder inside reports dir and copy figures there so
            # central report renders images locally (no ../ relative resolution issues).
            import shutil

            assets_dir_name = f"{filename}_{ts}_assets"
            assets_dir = os.path.join(self.reports_dir, assets_dir_name)
            # ensure fresh assets dir
            try:
                if os.path.exists(assets_dir):
                    shutil.rmtree(assets_dir)
            except Exception:
                pass
            ensure_dir(assets_dir)

            # Copy run's figures folder into assets_dir/figures (copy per-file to be robust)
            src_figures = os.path.join(run_dir, 'figures')
            dest_figures = os.path.join(assets_dir, 'figures')
            copied_count = 0
            if os.path.isdir(src_figures):
                if self.logger:
                    self.logger.info(f"Copying figures from {src_figures} to assets {dest_figures}")
                ensure_dir(dest_figures)
                for root, dirs, files in os.walk(src_figures):
                    rel_root = os.path.relpath(root, src_figures)
                    dest_root = os.path.join(dest_figures, rel_root) if rel_root != '.' else dest_figures
                    ensure_dir(dest_root)
                    for f_name in files:
                        src_file = os.path.join(root, f_name)
                        dest_file = os.path.join(dest_root, f_name)
                        try:
                            shutil.copy2(src_file, dest_file)
                            copied_count += 1
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Failed to copy figure {src_file} to assets: {e}")
                if self.logger:
                    self.logger.info(f"Copied {copied_count} figure files into assets folder {assets_dir}")
            else:
                if self.logger:
                    self.logger.info(f"No figures dir at {src_figures}; no assets copied.")

            # Build central lines that point to local assets folder so images render from reports
            central_lines = []
            for ln in lines:
                # replace several possible image link patterns to point to assets folder
                new_ln = ln.replace('(figures/', f'({assets_dir_name}/figures/')
                new_ln = new_ln.replace(f'(../experiments/{run_id}/figures/', f'({assets_dir_name}/figures/')
                central_lines.append(new_ln)

            # write central report (with local asset references)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(central_lines))

            # Now write run-local copy (keep original lines which use figures/...)
            ensure_dir(run_dir)
            run_path = os.path.join(run_dir, os.path.basename(out_path))
            with open(run_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            if self.logger:
                self.logger.info(f"Report written to run dir: {run_path}")
            return run_path

        # No run_dir/run_id: just write central file with given lines
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if self.logger:
            self.logger.info(f"Report written: {out_path}")
        return out_path

    def _save_json(self, data: Dict[str, Any], filename: str, run_dir: Optional[str] = None) -> str:
        out_path = os.path.join(self.reports_dir, f"{filename}.json")
        IOHandler.save_json(data, out_path)
        if run_dir:
            ensure_dir(run_dir)
            run_path = os.path.join(run_dir, os.path.basename(out_path))
            IOHandler.save_json(data, run_path)
            if self.logger:
                self.logger.info(f"JSON report written to run dir: {run_path}")
            return run_path
        if self.logger:
            self.logger.info(f"JSON report written: {out_path}")
        return out_path

    def _get_run_dir(self, run_id: Optional[str]) -> Optional[str]:
        if not run_id:
            return None
        run_dir = os.path.join(self.experiments_base_dir, run_id)
        if os.path.isdir(run_dir):
            return run_dir
        # If not exists, still create it (safe)
        ensure_dir(run_dir)
        return run_dir

    def _figure_insights(self, fname: str, best_metrics: Dict[str, float], all_metrics: Dict[str, Dict[str, float]], feature_importance: Optional[Any]) -> List[str]:
        """Generate short objective analysis lines for a figure filename using available metrics/feature importance."""
        notes: List[str] = []
        lname = fname.lower()
        try:
            if 'roc' in lname:
                auc = best_metrics.get('roc_auc') if best_metrics else None
                if auc:
                    notes.append(f"- ROC-AUC (best model): **{auc:.4f}** — indicates the model's discrimination ability (closer to 1 is better).")
                else:
                    notes.append("- ROC curve available — review AUC to assess separability.")

            elif 'confusion' in lname:
                rec = best_metrics.get('recall') if best_metrics else None
                prec = best_metrics.get('precision') if best_metrics else None
                f1 = best_metrics.get('f1') if best_metrics else None
                if rec is not None:
                    notes.append(f"- Recall (Churn coverage): **{rec:.4f}** — proportion of churn instances detected.")
                if prec is not None:
                    notes.append(f"- Precision: **{prec:.4f}** — among predicted churn, proportion correct.")
                if f1 is not None:
                    notes.append(f"- F1-score: **{f1:.4f}** — harmonic mean of precision & recall.")
                notes.append("- Actionable: prioritize reducing False Negatives (increase recall) for retention campaigns.")

            elif 'feature_importance' in lname or 'importance' in lname:
                # show top features from provided feature_importance
                lines = []
                if feature_importance is not None:
                    try:
                        if hasattr(feature_importance, 'head'):
                            top = feature_importance.head(5)
                            for _, r in top.iterrows():
                                lines.append(f"{r['feature']}: {r['importance']:.4f}")
                        elif isinstance(feature_importance, dict):
                            for k, v in list(feature_importance.items())[:5]:
                                lines.append(f"{k}: {v:.4f}")
                    except Exception:
                        pass
                if lines:
                    notes.append("- Top features (preview):")
                    for l in lines:
                        notes.append(f"  - {l}")
                    notes.append("- Business insight: address top drivers (e.g., Tenure, Complain) in retention playbooks.")
                else:
                    notes.append("- Feature importance plot available; review top features impacting predictions.")

            elif 'model_comparison' in lname:
                # compare top two models
                try:
                    sorted_models = sorted(all_metrics.items(), key=lambda kv: kv[1].get('f1', 0), reverse=True)
                    if sorted_models:
                        top_name, top_metrics = sorted_models[0]
                        notes.append(f"- Top model by F1: **{top_name}** (F1={top_metrics.get('f1',0):.4f}).")
                        if len(sorted_models) > 1:
                            second_name, second_metrics = sorted_models[1]
                            notes.append(f"- Gap to 2nd ({second_name}): {top_metrics.get('f1',0)-second_metrics.get('f1',0):.4f} in F1.")
                except Exception:
                    pass

            elif 'missing' in lname:
                notes.append("- Shows columns with missing values; consider imputation (median for numeric, mode for categorical) before modeling.")
            elif 'target_distribution' in lname or 'distribution' in lname:
                notes.append("- Shows class distribution; if highly imbalanced, consider resampling (SMOTE/Tomek) or class-weighting.")
            elif 'correlation' in lname:
                notes.append("- Correlation matrix highlights multicollinearity and features correlated with target — consider feature selection.")
            else:
                notes.append("- Figure available: review visual for domain insights.")
        except Exception:
            notes.append("- Unable to auto-generate figure insights; review manually.")
        return notes

    def generate_training_report(
        self,
        run_id: str,
        best_model_name: str,
        all_metrics: Dict[str, Dict[str, float]],
        feature_importance: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        eda_summary: Optional[Dict[str, Any]] = None,
        format: str = "markdown"
    ) -> str:
        """
        Generate training report and save into central reports folder and into run folder.

        Returns path to saved report file (in run dir if run_id provided).
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_dir = self._get_run_dir(run_id)
        filename = f"training_report_{run_id}"

        # JSON format
        if format == 'json':
            report = {
                'run_id': run_id,
                'timestamp': timestamp,
                'best_model': {
                    'name': best_model_name,
                    'metrics': all_metrics.get(best_model_name, {})
                },
                'all_models': all_metrics,
                'config': config or {}
            }
            return self._save_json(report, filename, run_dir=run_dir)

        # Markdown dashboard-style training report (no full config dump)
        lines: List[str] = []
        lines.append(f"# Training Dashboard — Run: {run_id}")
        lines.append("")
        lines.append(f"**Generated:** {timestamp}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Summary block
        lines.append("## Executive Summary")
        lines.append("")
        if best_model_name and best_model_name in all_metrics:
            bm = all_metrics[best_model_name]
            lines.append(f"**Best model:** **{best_model_name.upper()}** — F1={bm.get('f1', 0):.4f}, ROC_AUC={bm.get('roc_auc', 0):.4f}, Accuracy={bm.get('accuracy', 0):.4f}")
        else:
            lines.append("**Best model:** N/A")
        lines.append("")

        # Auto-analysis: quick observations
        lines.append("### Key Observations (auto-generated)")
        lines.append("")
        # find top and second model by f1
        try:
            sorted_models = sorted(all_metrics.items(), key=lambda kv: kv[1].get('f1', 0), reverse=True)
            if sorted_models:
                top_name, top_metrics = sorted_models[0]
                lines.append(f"- Top model by F1: **{top_name}** (F1={top_metrics.get('f1',0):.4f})")
            if len(sorted_models) > 1:
                second_name, second_metrics = sorted_models[1]
                diff = top_metrics.get('f1',0) - second_metrics.get('f1',0)
                lines.append(f"- Gap to 2nd: {diff:.4f} (top vs {second_name})")
        except Exception:
            pass
        lines.append("")

        # Metrics table
        lines.append("## Model Performance")
        lines.append("")
        # header
        all_metric_keys = set()
        for m in all_metrics.values():
            all_metric_keys.update(m.keys())
        keys = sorted(all_metric_keys)
        if keys:
            header = "| Model | " + " | ".join(k.upper() for k in keys) + " |"
            sep = "|---" + "|".join(["---"] * len(keys)) + "|"
            lines.append(header)
            lines.append(sep)
            for name, metrics in all_metrics.items():
                row = f"| {name} | " + " | ".join(f"{metrics.get(k,0):.4f}" for k in keys) + " |"
                lines.append(row)
            lines.append("")

        # Plots (EDA + Evaluation) - embed inline images with controlled width and auto analysis
        lines.append("## Visuals")
        lines.append("")
        # EDA figures (smaller thumbnails)
        eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
        if eda_fig_dir and os.path.isdir(eda_fig_dir):
            lines.append("### EDA Figures")
            lines.append("")
            for fname in sorted(os.listdir(eda_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/eda/{fname}"
                    # smaller width for EDA
                    lines.append(f"<img src=\"{rel}\" width=320 />")
                    lines.append("")
                    # auto analysis under figure
                    for note in self._figure_insights(fname, bm if 'bm' in locals() else {}, all_metrics, feature_importance):
                        lines.append(note)
                    lines.append("")

        # Evaluation figures (medium size)
        eval_fig_dir = os.path.join(run_dir or '', 'figures', 'evaluation') if run_dir else None
        if eval_fig_dir and os.path.isdir(eval_fig_dir):
            lines.append("### Evaluation Figures")
            lines.append("")
            # Insert Champion Metrics block before visuals
            if best_model_name and best_model_name in all_metrics:
                bm = all_metrics[best_model_name]
                lines.append("## 🥇 Kết Quả Đánh Giá Mô Hình Tốt Nhất")
                lines.append("")
                # ROC placeholder path
                roc_path = os.path.join('figures', 'evaluation', 'roc_curve.png')
                cm_path = os.path.join('figures', 'evaluation', 'confusion_matrix_{}.png'.format(best_model_name))
                lines.append(f"**ROC Curve:** `{roc_path}`  ")
                lines.append("")
                # Champion metrics table per template
                lines.append("### Champion Metrics ({})".format(best_model_name.upper()))
                lines.append("")
                lines.append("| Metric | Giá trị | Ngưỡng MLOps (Min) | Trạng Thái |")
                lines.append("| :--- | :---: | :---: | :---: |")
                # Thresholds
                acc = bm.get('accuracy', 0)
                f1v = bm.get('f1', 0)
                rec = bm.get('recall', 0)
                roc = bm.get('roc_auc', 0)
                acc_pass = '✅ PASS' if acc >= 0.75 else '❌ FAIL'
                f1_pass = '✅ PASS' if f1v >= 0.70 else '❌ FAIL'
                lines.append(f"| **Accuracy** | **{acc:.4f}** | > 0.75 | {acc_pass} |")
                lines.append(f"| **F1-Score** | **{f1v:.4f}** | > 0.70 | {f1_pass} |")
                lines.append(f"| **Recall (Churn Coverage)** | {rec:.4f} | - | - |")
                lines.append(f"| **ROC-AUC** | **{roc:.4f}** | - | - |")
                lines.append("")
            # Then evaluation figures
            for fname in sorted(os.listdir(eval_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/evaluation/{fname}"
                    lines.append(f"<img src=\"{rel}\" width=520 />")
                    lines.append("")
                    for note in self._figure_insights(fname, bm if 'bm' in locals() else {}, all_metrics, feature_importance):
                        lines.append(note)
                    lines.append("")

        # Feature importance
        if feature_importance is not None:
            lines.append("## Feature Importance (Top 20)")
            lines.append("")
            try:
                for idx, (_, r) in enumerate(feature_importance.head(20).iterrows()):
                    lines.append(f"{idx+1}. **{r['feature']}** — {r['importance']:.4f}")
                lines.append("")
            except Exception:
                lines.append(str(feature_importance))
                lines.append("")

        # Recommendations — simple rule-based
        lines.append("## Recommendations (auto)")
        lines.append("")
        try:
            if best_model_name and all_metrics.get(best_model_name, {}).get('f1', 0) < 0.7:
                lines.append("- Model performance is below the 0.70 F1 threshold: consider more features, hyperparam tuning, or rebalancing.")
            else:
                lines.append("- Model performance is acceptable — consider deploying and monitoring drift.")
        except Exception:
            pass
        lines.append("")

        # EDA Key Insights (optional)
        if eda_summary:
            lines.append("---")
            lines.append("")
            lines.append("## 🔍 EDA Key Insights")
            lines.append("")
            # churn rate
            churn_rate = eda_summary.get('churn_rate')
            if churn_rate is not None:
                lines.append(f"- **Churn rate (class 1):** **{churn_rate:.1%}**")
            # imbalance treatment
            if eda_summary.get('resampling'):
                lines.append(f"- **Resampling strategy applied:** {eda_summary.get('resampling')}")
            # top correlated features
            top_corr = eda_summary.get('top_correlated')
            if top_corr:
                lines.append(f"- **Top correlated with target:** {', '.join(top_corr)}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Report generated by ReportGenerator — summary dashboard (no full config dump).*)")

        return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)

    # Other report types can be implemented similarly (EDA, data, comparison)

    def generate_eda_report(self, df, filename: str = "eda_report", target_col: str = None, format: str = "markdown", run_id: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_dir = self._get_run_dir(run_id)
        n_rows, n_cols = df.shape
        missing_total = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())

        if format == 'json':
            report = {
                'timestamp': timestamp,
                'shape': {'rows': n_rows, 'cols': n_cols},
                'missing_total': missing_total,
                'duplicates': duplicates
            }
            return self._save_json(report, filename, run_dir=run_dir)

        lines: List[str] = []
        lines.append(f"# EDA Dashboard")
        lines.append("")
        lines.append(f"**Generated:** {timestamp}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Rows: **{n_rows:,}**")
        lines.append(f"- Columns: **{n_cols:,}**")
        lines.append(f"- Missing values (total): **{missing_total}**")
        lines.append(f"- Duplicate rows: **{duplicates}**")
        lines.append("")

        # Target distribution quick stats
        if target_col and target_col in df.columns:
            try:
                vc = df[target_col].value_counts()
                lines.append("## Target")
                lines.append("")
                for idx, cnt in vc.items():
                    lines.append(f"- `{idx}`: {cnt} ({cnt/len(df):.1%})")
                lines.append("")
            except Exception:
                pass

        # Top missing columns
        try:
            miss_perc = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            top_missing = miss_perc[miss_perc > 0].head(10)
            if not top_missing.empty:
                lines.append("## Top Missing Columns")
                lines.append("")
                for col, pct in top_missing.items():
                    lines.append(f"- {col}: {pct:.1f}%")
                lines.append("")
        except Exception:
            pass

        # Top correlations with target (if numeric target)
        try:
            if target_col and target_col in df.columns:
                num = df.select_dtypes(include=['number'])
                if target_col in num.columns:
                    corrs = num.corr()[target_col].abs().sort_values(ascending=False).drop(target_col)
                    top_corr = corrs.head(10)
                    if not top_corr.empty:
                        lines.append("## Top correlations with target")
                        lines.append("")
                        for c, v in top_corr.items():
                            lines.append(f"- {c}: {v:.3f}")
                        lines.append("")
        except Exception:
            pass

        # Figures — list EDA figure files if present under run_dir
        lines.append("## Figures")
        lines.append("")
        eda_fig_dir = os.path.join(run_dir or '', 'figures', 'eda') if run_dir else None
        if eda_fig_dir and os.path.isdir(eda_fig_dir):
            for fname in sorted(os.listdir(eda_fig_dir)):
                if fname.lower().endswith('.png'):
                    rel = f"figures/eda/{fname}"
                    lines.append(f"- ![]({rel}) — `{fname}`")
            lines.append("")

        # Automated insights
        lines.append("## Automated Insights")
        lines.append("")
        try:
            if missing_total > 0:
                lines.append(f"- There are **{missing_total}** missing cells; consider imputation strategies (median/mode) before modeling.")
            else:
                lines.append("- No missing values detected.")

            if duplicates > 0:
                lines.append(f"- {duplicates} duplicate rows were found — consider deduplication if appropriate.")
        except Exception:
            pass

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*EDA dashboard generated by ReportGenerator — figures and insights included.*")

        return self._save_markdown(lines, filename, run_dir=run_dir, run_id=run_id)


    def generate_data_report(self, version_info: Dict[str, Any], quality_info: Dict[str, Any] = None, lineage_info: Dict[str, Any] = None, filename: str = "data_report", format: str = "markdown", run_id: Optional[str] = None) -> str:
        run_dir = self._get_run_dir(run_id)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if format == 'json':
            payload = {
                'timestamp': ts,
                'version': version_info,
                'quality': quality_info,
                'lineage': lineage_info
            }
            return self._save_json(payload, filename, run_dir=run_dir)

        lines = [f"# Data Report", "", f"Generated: {ts}", "", "---", "", "## Version", ""]
        lines.append(f"Version ID: {version_info.get('version_id', 'N/A')}")
        lines.append(f"Hash: {version_info.get('hash', 'N/A')}")
        if quality_info:
            lines.append("")
            lines.append("## Quality")
            for k, v in quality_info.items():
                lines.append(f"- {k}: {v}")
        return self._save_markdown(lines, filename, run_dir=run_dir)
