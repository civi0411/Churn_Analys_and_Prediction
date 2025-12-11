"""
Module `ops.dataops.drift_detector` - phát hiện data drift giữa dữ liệu mới và reference (training).

Important keywords: Args, Returns, Methods, Notes
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from typing import Dict, Any, Optional, Tuple
from ...utils import IOHandler, ensure_dir, get_timestamp


class DataDriftDetector:
    """
    Phát hiện drift giữa `new_data` và `reference_data` (training baseline).

    Methods:
        detect_drift(new_data, threshold): tổng hợp schema/numerical/categorical drift
        _detect_numerical_drift, _detect_categorical_drift, _detect_schema_drift
    """

    def __init__(self, reference_data: pd.DataFrame, logger=None):
        """Khởi tạo với reference_data (baseline).

        Args:
            reference_data (pd.DataFrame): dữ liệu training làm baseline
            logger: Logger tùy chọn
        """
        self.reference_data = reference_data.copy()
        self.logger = logger

        # Compute reference statistics
        self.ref_stats = self._compute_statistics(reference_data)

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Tính statistics cơ bản cho reference data (numerical/categorical/schema)."""
        stats_out = {
            'numerical': {},
            'categorical': {},
            'schema': {
                'columns': set(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_rate': df.isnull().mean().to_dict()
            }
        }

        # Numerical features
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            stats_out['numerical'][col] = {
                'mean': float(df[col].mean()) if not df[col].dropna().empty else None,
                'std': float(df[col].std()) if not df[col].dropna().empty else None,
                'min': float(df[col].min()) if not df[col].dropna().empty else None,
                'max': float(df[col].max()) if not df[col].dropna().empty else None,
                'quantiles': {0.25: float(df[col].quantile(0.25)), 0.5: float(df[col].quantile(0.5)), 0.75: float(df[col].quantile(0.75))}
            }

        # Categorical features
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            stats_out['categorical'][col] = {
                'unique_values': set(df[col].dropna().unique()),
                'value_counts': df[col].value_counts(normalize=True).to_dict()
            }

        return stats_out

    def detect_drift(self, new_data: pd.DataFrame, threshold: float = 0.05, sample_frac: float = 1.0, max_rows: int = None) -> Dict[str, Any]:
        """
        Phát hiện toàn bộ drift: schema, numerical, categorical

        Args:
            new_data: Data mới cần kiểm tra
            threshold: P-value threshold cho statistical tests

        Returns:
            Dict chứa drift report
        """
        report: Dict[str, Any] = {
            'has_drift': False,
            'drift_details': {},
            'summary': {}
        }

        # Possibly sample for performance
        if sample_frac is not None and sample_frac < 1.0:
            new_data = new_data.sample(frac=sample_frac, random_state=42)
        if max_rows is not None and len(new_data) > max_rows:
            new_data = new_data.sample(n=max_rows, random_state=42)

        # 1. Schema Drift
        schema_drift = self._detect_schema_drift(new_data)
        report['drift_details']['schema'] = schema_drift
        if schema_drift.get('has_drift'):
            report['has_drift'] = True

        # 2. Numerical Drift
        num_drift = self._detect_numerical_drift(new_data, threshold)
        report['drift_details']['numerical'] = num_drift
        if num_drift.get('drifted_features'):
            report['has_drift'] = True

        # 3. Categorical Drift
        cat_drift = self._detect_categorical_drift(new_data, threshold)
        report['drift_details']['categorical'] = cat_drift
        if cat_drift.get('drifted_features'):
            report['has_drift'] = True

        # 4. Summary
        total_checked = len(num_drift.get('ks_results', {})) + len(cat_drift.get('chi2_results', {}))
        drifted_count = len(num_drift.get('drifted_features', [])) + len(cat_drift.get('drifted_features', []))
        report['summary'] = {
            'total_features_checked': total_checked,
            'drifted_features': drifted_count,
            'drift_severity': self._calculate_drift_severity({'summary': {'total_features_checked': total_checked, 'drifted_features': drifted_count}, 'drift_details': report['drift_details']})
        }

        if self.logger:
            severity = report['summary']['drift_severity']
            if severity == 'CRITICAL':
                self.logger.warning("CRITICAL DATA DRIFT DETECTED!")
            elif severity == 'MODERATE':
                self.logger.warning("Moderate data drift detected")
            else:
                self.logger.info(f"No significant drift detected")

        return report

    def _detect_schema_drift(self, new_data: pd.DataFrame) -> Dict:
        """Phát hiện thay đổi schema"""
        ref_cols = self.ref_stats['schema']['columns']
        new_cols = set(new_data.columns)

        return {
            'has_drift': ref_cols != new_cols,
            'missing_columns': list(ref_cols - new_cols),
            'new_columns': list(new_cols - ref_cols),
            'dtype_changes': self._detect_dtype_changes(new_data)
        }

    def _detect_dtype_changes(self, new_data: pd.DataFrame) -> Dict:
        """Phát hiện thay đổi data types"""
        changes = {}
        ref_dtypes = self.ref_stats['schema']['dtypes']

        for col in new_data.columns:
            if col in ref_dtypes:
                if str(new_data[col].dtype) != str(ref_dtypes[col]):
                    changes[col] = {
                        'reference': str(ref_dtypes[col]),
                        'current': str(new_data[col].dtype)
                    }
        return changes

    def _detect_numerical_drift(self, new_data: pd.DataFrame, threshold: float) -> Dict:
        """
        Kolmogorov-Smirnov test cho numerical features

        KS test: So sánh 2 distributions, p-value < threshold => drift
        """
        results: Dict[str, Any] = {}
        drifted = []

        for col, ref_stat in self.ref_stats['numerical'].items():
            if col not in new_data.columns:
                continue

            ref_values = self.reference_data[col].dropna()
            new_values = new_data[col].dropna()

            if len(new_values) == 0 or len(ref_values) == 0:
                # cannot test if one side is empty
                continue

            # KS Test
            try:
                ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
            except Exception:
                ks_stat, p_value = None, None

            # Compare distributions
            mean_diff = abs(float(new_values.mean()) - float(ref_stat.get('mean', 0))) if new_values.size > 0 else None
            std_ratio = (float(new_values.std()) / float(ref_stat.get('std'))) if ref_stat.get('std') and ref_stat.get('std') > 0 else None

            has_drift = (p_value is not None and p_value < threshold)

            results[col] = {
                'ks_statistic': float(ks_stat) if ks_stat is not None else None,
                'p_value': float(p_value) if p_value is not None else None,
                'mean_diff': mean_diff,
                'std_ratio': std_ratio,
                'has_drift': has_drift
            }

            if has_drift:
                drifted.append(col)

        return {
            'ks_results': results,
            'drifted_features': drifted
        }

    def _detect_categorical_drift(self, new_data: pd.DataFrame, threshold: float) -> Dict:
        """
        Chi-square test cho categorical features, bỏ qua cột có quá nhiều unique hoặc chuỗi dài bất thường.
        Không warning/exception ngoài logger, không ép kiểu, không chi-squared với dữ liệu không phù hợp.
        """
        results: Dict[str, Any] = {}
        drifted = []
        MAX_UNIQUE = 50  # Nếu unique > 50 thì bỏ qua drift check cho cột đó
        MAX_STRLEN = 200  # Nếu giá trị chuỗi quá dài thì bỏ qua
        for col, ref_stat in self.ref_stats['categorical'].items():
            if col not in new_data.columns:
                continue
            # Nếu số lượng unique quá lớn, bỏ qua
            n_unique = len(set(self.reference_data[col].dropna().unique()) | set(new_data[col].dropna().unique()))
            if n_unique > MAX_UNIQUE:
                if self.logger:
                    self.logger.info(f"[Drift] Column '{col}' skipped: too many unique values ({n_unique} > {MAX_UNIQUE})")
                results[col] = {
                    'skipped': True,
                    'reason': f"Too many unique values: {n_unique}"
                }
                continue
            # Nếu giá trị là chuỗi rất dài (có thể là lỗi dữ liệu), bỏ qua
            try:
                sample_val = str(self.reference_data[col].dropna().iloc[0]) if not self.reference_data[col].dropna().empty else ''
                if len(sample_val) > MAX_STRLEN:
                    if self.logger:
                        self.logger.info(f"[Drift] Column '{col}' skipped: value too long ({len(sample_val)} chars > {MAX_STRLEN})")
                    results[col] = {
                        'skipped': True,
                        'reason': f"Value too long: {len(sample_val)} chars"
                    }
                    continue
            except Exception:
                if self.logger:
                    self.logger.info(f"[Drift] Column '{col}' skipped: error when checking value length.")
                results[col] = {
                    'skipped': True,
                    'reason': "Error when checking value length"
                }
                continue
            ref_counts = self.reference_data[col].value_counts()
            new_counts = new_data[col].value_counts()
            # Check for new categories
            ref_cats = ref_stat.get('unique_values', set())
            new_cats = set(new_data[col].dropna().unique())
            new_categories = list(new_cats - ref_cats)
            # Align categories for chi-square
            all_cats = list(ref_cats | new_cats)
            ref_freq = [ref_counts.get(cat, 0) for cat in all_cats]
            new_freq = [new_counts.get(cat, 0) for cat in all_cats]
            # Build contingency table: rows = [ref_counts, new_counts]
            try:
                ref_arr = [int(x) for x in ref_freq]
                new_arr = [int(x) for x in new_freq]
                table = np.array([ref_arr, new_arr])
                if table.sum() == 0:
                    chi2_stat, p_value = None, None
                    has_chi2_drift = False
                else:
                    chi2_stat, p_value, dof, expected = chi2_contingency(table)
                    has_chi2_drift = (p_value is not None and p_value < threshold)
                has_drift = has_chi2_drift or (len(new_categories) > 0)
            except Exception as e:
                if self.logger:
                    self.logger.info(f"[Drift] Column '{col}' skipped: chi2 test failed ({e})")
                chi2_stat, p_value = None, None
                has_drift = len(new_categories) > 0
            results[col] = {
                'chi2_statistic': float(chi2_stat) if chi2_stat is not None else None,
                'p_value': float(p_value) if p_value is not None else None,
                'new_categories': new_categories,
                'has_drift': has_drift
            }
            if has_drift:
                drifted.append(col)
        return {
            'chi2_results': results,
            'drifted_features': drifted
        }

    def _calculate_drift_severity(self, report: Dict) -> str:
        """
        Tính mức độ nghiêm trọng của drift

        Returns:
            'NONE', 'LOW', 'MODERATE', 'CRITICAL'
        """
        total = report['summary'].get('total_features_checked', 0)
        drifted = report['summary'].get('drifted_features', 0)

        if total == 0:
            return 'NONE'

        drift_ratio = (drifted / total) if total > 0 else 0

        # Check schema drift (critical)
        schema_drift = report['drift_details'].get('schema', {})
        if schema_drift.get('missing_columns') or schema_drift.get('dtype_changes'):
            return 'CRITICAL'

        # Check drift ratio
        if drift_ratio > 0.3:
            return 'CRITICAL'
        elif drift_ratio > 0.15:
            return 'MODERATE'
        elif drift_ratio > 0:
            return 'LOW'
        else:
            return 'NONE'

def detect_drift_full(df_train, df_predict, logger=None, threshold=0.05, report_path=None):
    """
    Hàm drift detection tổng hợp cho cả numeric và categorical, tổng hợp severity, lưu report nếu cần.
    """
    detector = DataDriftDetector(df_train, logger)
    report = detector.detect_drift(df_predict, threshold=threshold)
    return report
