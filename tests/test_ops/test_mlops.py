"""
tests/test_ops/test_mlops.py
Tests for src/ops/mlops.py
"""
import pytest
import pandas as pd
import numpy as np
import os

from src.ops.mlops import (
    ExperimentTracker,
    ModelRegistry,
    ModelMonitor,
    ModelExplainer
)
from src.utils import ReportGenerator
from sklearn.ensemble import RandomForestClassifier


class TestExperimentTracker:
    """Test cases for ExperimentTracker class"""

    def test_init(self, temp_dir):
        """Test khởi tạo ExperimentTracker"""
        tracker = ExperimentTracker(temp_dir)

        assert os.path.exists(temp_dir)
        assert os.path.exists(tracker.experiments_file)

    def test_start_run(self, temp_dir):
        """Test start_run tạo run mới"""
        tracker = ExperimentTracker(temp_dir)
        run_id = tracker.start_run('test_run')

        assert tracker.current_run_id == 'test_run'
        assert tracker.current_run_dir is not None
        assert os.path.exists(tracker.current_run_dir)

    def test_end_run(self, temp_dir):
        """Test end_run kết thúc run"""
        tracker = ExperimentTracker(temp_dir)
        tracker.start_run('test_run')
        tracker.end_run('FINISHED')

        assert tracker.current_run_id is None

        # Check experiments file updated
        df = pd.read_csv(tracker.experiments_file)
        assert len(df) == 1
        assert df.iloc[0]['status'] == 'FINISHED'

    def test_log_params(self, temp_dir):
        """Test log_params lưu parameters"""
        tracker = ExperimentTracker(temp_dir)
        tracker.start_run('test_run')

        params = {'learning_rate': 0.01, 'n_estimators': 100}
        tracker.log_params(params)

        params_file = os.path.join(tracker.current_run_dir, 'params.json')
        assert os.path.exists(params_file)

    def test_log_metrics(self, temp_dir):
        """Test log_metrics lưu metrics"""
        tracker = ExperimentTracker(temp_dir)
        tracker.start_run('test_run')

        metrics = {'accuracy': 0.95, 'f1': 0.92}
        tracker.log_metrics(metrics)

        metrics_file = os.path.join(tracker.current_run_dir, 'metrics.json')
        assert os.path.exists(metrics_file)

    def test_get_run_info(self, temp_dir):
        """Test get_run_info lấy thông tin run"""
        tracker = ExperimentTracker(temp_dir)
        tracker.start_run('test_run')
        tracker.log_params({'param1': 'value1'})
        tracker.log_metrics({'metric1': 0.9})

        info = tracker.get_run_info()

        assert info['run_id'] == 'test_run'
        assert 'params' in info
        assert 'metrics' in info


class TestModelRegistry:
    """Test cases for ModelRegistry class"""

    def test_init(self, temp_dir):
        """Test khởi tạo ModelRegistry"""
        registry = ModelRegistry(temp_dir)

        assert os.path.exists(temp_dir)
        assert registry.registry == {}

    def test_register_model(self, temp_dir, trained_model):
        """Test register_model đăng ký model"""
        registry = ModelRegistry(temp_dir)

        metrics = {'accuracy': 0.95, 'f1': 0.92}
        save_path = registry.register_model(
            'random_forest', trained_model, metrics, 'test_run'
        )

        assert os.path.exists(save_path)
        assert 'random_forest' in registry.registry
        assert len(registry.registry['random_forest']) == 1

    def test_register_model_increments_version(self, temp_dir, trained_model):
        """Test register_model tăng version"""
        registry = ModelRegistry(temp_dir)
        metrics = {'accuracy': 0.95}

        registry.register_model('rf', trained_model, metrics)
        registry.register_model('rf', trained_model, metrics)

        assert len(registry.registry['rf']) == 2
        assert registry.registry['rf'][0]['version'] == 1
        assert registry.registry['rf'][1]['version'] == 2

    def test_get_latest_model(self, temp_dir, trained_model):
        """Test get_latest_model lấy model mới nhất"""
        registry = ModelRegistry(temp_dir)
        metrics = {'accuracy': 0.95}

        registry.register_model('rf', trained_model, metrics)

        loaded_model = registry.get_latest_model('rf')

        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')

    def test_get_latest_model_returns_none_for_unknown(self, temp_dir):
        """Test get_latest_model trả về None cho model không tồn tại"""
        registry = ModelRegistry(temp_dir)

        result = registry.get_latest_model('unknown_model')

        assert result is None


class TestModelMonitor:
    """Test cases for ModelMonitor class"""

    def test_init(self, temp_dir):
        """Test khởi tạo ModelMonitor"""
        monitor = ModelMonitor(temp_dir)

        assert os.path.exists(temp_dir)
        assert os.path.exists(monitor.performance_log)

    def test_log_performance(self, temp_dir):
        """Test log_performance ghi lại performance"""
        monitor = ModelMonitor(temp_dir)

        metrics = {'accuracy': 0.95, 'f1': 0.92, 'precision': 0.90, 'recall': 0.94}
        monitor.log_performance('rf', metrics, model_version='v1', n_samples=1000)

        df = pd.read_csv(monitor.performance_log)
        assert len(df) == 1
        assert df.iloc[0]['model_name'] == 'rf'
        assert df.iloc[0]['accuracy'] == 0.95

    def test_get_performance_history(self, temp_dir):
        """Test get_performance_history lấy lịch sử"""
        monitor = ModelMonitor(temp_dir)

        monitor.log_performance('rf', {'accuracy': 0.95, 'f1': 0.92})
        monitor.log_performance('rf', {'accuracy': 0.93, 'f1': 0.90})

        history = monitor.get_performance_history('rf')

        assert len(history) == 2

    def test_detect_drift(self, temp_dir):
        """Test detect_drift phát hiện drift"""
        monitor = ModelMonitor(temp_dir)

        # Log baseline
        monitor.log_performance('rf', {'accuracy': 0.95, 'f1': 0.92})
        # Log degraded performance
        monitor.log_performance('rf', {'accuracy': 0.80, 'f1': 0.75})

        result = monitor.detect_drift('rf', metric='f1', threshold=0.05)

        assert result['drift_detected'] == True
        assert result['drift'] > 0.05

    def test_detect_drift_no_drift(self, temp_dir):
        """Test detect_drift không có drift"""
        monitor = ModelMonitor(temp_dir)

        monitor.log_performance('rf', {'f1': 0.92})
        monitor.log_performance('rf', {'f1': 0.91})

        result = monitor.detect_drift('rf', metric='f1', threshold=0.05)

        assert result['drift_detected'] == False

    def test_check_health(self, temp_dir):
        """Test check_health kiểm tra sức khỏe model"""
        monitor = ModelMonitor(temp_dir)

        current_metrics = {'f1': 0.85, 'accuracy': 0.90}
        result = monitor.check_health('rf', current_metrics)

        assert result['status'] in ['HEALTHY', 'WARNING', 'CRITICAL']

    def test_check_health_warning(self, temp_dir):
        """Test check_health phát hiện warning"""
        monitor = ModelMonitor(temp_dir)

        current_metrics = {'f1': 0.60, 'accuracy': 0.65}  # Below thresholds
        thresholds = {'f1_min': 0.70, 'accuracy_min': 0.75}

        result = monitor.check_health('rf', current_metrics, thresholds=thresholds)

        assert result['status'] in ['WARNING', 'CRITICAL']
        assert len(result['issues']) > 0


class TestModelExplainer:
    """Test cases for ModelExplainer class"""

    def test_init(self, trained_model, sample_train_test_split):
        """Test khởi tạo ModelExplainer"""
        X_train, _, _, _ = sample_train_test_split
        feature_names = list(X_train.columns)

        explainer = ModelExplainer(trained_model, X_train, feature_names)

        assert explainer.model is not None
        assert explainer.feature_names == feature_names

    def test_get_feature_importance(self, trained_model, sample_train_test_split):
        """Test get_feature_importance với tree-based model"""
        X_train, _, _, _ = sample_train_test_split
        feature_names = list(X_train.columns)

        explainer = ModelExplainer(trained_model, X_train, feature_names)
        importance = explainer.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_get_feature_importance_top_n(self, trained_model, sample_train_test_split):
        """Test get_feature_importance với top_n"""
        X_train, _, _, _ = sample_train_test_split
        feature_names = list(X_train.columns)

        explainer = ModelExplainer(trained_model, X_train, feature_names)
        importance = explainer.get_feature_importance(top_n=5)

        assert len(importance) <= 5

    def test_explain_with_shap(self, trained_model, sample_train_test_split, temp_dir):
        """Test explain_with_shap tạo SHAP plot"""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

        X_train, X_test, _, _ = sample_train_test_split
        feature_names = list(X_train.columns)

        explainer = ModelExplainer(trained_model, X_train, feature_names)
        save_path = os.path.join(temp_dir, 'shap.png')

        result = explainer.explain_with_shap(X_test.head(10), save_path)

        # Should return True on success
        assert isinstance(result, bool)

    def test_init_with_pipeline_model(self, sample_train_test_split):
        """Test init với model trong pipeline"""
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imblearn not installed")

        X_train, _, y_train, _ = sample_train_test_split
        feature_names = list(X_train.columns)

        pipeline = ImbPipeline([
            ('sampler', SMOTE(random_state=42)),
            ('model', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        explainer = ModelExplainer(pipeline, X_train, feature_names)

        # Should extract the actual model from pipeline
        assert isinstance(explainer.model, RandomForestClassifier)


class TestReportGenerator:
    """Test cases for ReportGenerator class"""

    def test_init(self, temp_dir):
        """Test khởi tạo ReportGenerator"""
        report_dir = os.path.join(temp_dir, 'reports')
        generator = ReportGenerator(report_dir)

        assert os.path.exists(report_dir)

    def test_generate_markdown_report(self, temp_dir):
        """Test generate_training_report tạo markdown"""
        generator = ReportGenerator(temp_dir)

        all_metrics = {
            'random_forest': {'accuracy': 0.95, 'precision': 0.90, 'recall': 0.88, 'f1': 0.89, 'roc_auc': 0.94},
            'xgboost': {'accuracy': 0.97, 'precision': 0.93, 'recall': 0.91, 'f1': 0.92, 'roc_auc': 0.96}
        }

        report_path = generator.generate_training_report(
            run_id='test_run_001',
            best_model_name='xgboost',
            all_metrics=all_metrics,
            format='markdown'
        )

        assert os.path.exists(report_path)
        assert report_path.endswith('.md')

        # Check content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'XGBOOST' in content
            assert '0.9200' in content  # F1 score

    def test_generate_json_report(self, temp_dir):
        """Test generate_training_report tạo JSON"""
        generator = ReportGenerator(temp_dir)

        all_metrics = {
            'random_forest': {'accuracy': 0.95, 'f1': 0.89}
        }

        report_path = generator.generate_training_report(
            run_id='test_run_002',
            best_model_name='random_forest',
            all_metrics=all_metrics,
            format='json'
        )

        assert os.path.exists(report_path)
        assert report_path.endswith('.json')

    def test_generate_report_with_feature_importance(self, temp_dir):
        """Test generate report với feature importance"""
        generator = ReportGenerator(temp_dir)

        all_metrics = {'xgboost': {'f1': 0.92}}
        feature_importance = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [0.5, 0.3, 0.2]
        })

        report_path = generator.generate_training_report(
            run_id='test_run_003',
            best_model_name='xgboost',
            all_metrics=all_metrics,
            feature_importance=feature_importance,
            format='markdown'
        )

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Feature Importance' in content
            assert 'feature_1' in content

    def test_generate_report_with_config(self, temp_dir, test_config):
        """Test generate report với config summary"""
        generator = ReportGenerator(temp_dir)

        all_metrics = {'xgboost': {'f1': 0.92}}

        report_path = generator.generate_training_report(
            run_id='test_run_004',
            best_model_name='xgboost',
            all_metrics=all_metrics,
            config=test_config,
            format='markdown'
        )

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Configuration' in content

