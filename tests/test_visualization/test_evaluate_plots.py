"""
tests/test_visualization/test_evaluate_plots.py
Tests for src/visualization/evaluate_plots.py
"""
import pytest
import pandas as pd
import numpy as np
import os

from src.visualization.evaluate_plots import EvaluateVisualizer


class TestEvaluateVisualizer:
    """Test cases for EvaluateVisualizer class"""

    def test_init(self, test_config, temp_artifacts_dir, mock_logger):
        """Test khởi tạo EvaluateVisualizer"""
        viz = EvaluateVisualizer(test_config, mock_logger, temp_artifacts_dir)

        assert viz.config == test_config
        assert os.path.exists(viz.eval_dir)

    def test_plot_confusion_matrix(self, test_config, temp_artifacts_dir, sample_train_test_split, trained_model):
        """Test plot_confusion_matrix tạo plot"""
        X_train, X_test, y_train, y_test = sample_train_test_split
        y_pred = trained_model.predict(X_test)

        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_confusion_matrix(y_test, y_pred, 'random_forest')

        assert os.path.exists(os.path.join(viz.eval_dir, 'confusion_matrix_random_forest.png'))

    def test_plot_roc_curve(self, test_config, temp_artifacts_dir, sample_train_test_split, trained_model):
        """Test plot_roc_curve tạo ROC curve"""
        X_train, X_test, y_train, y_test = sample_train_test_split

        from sklearn.metrics import roc_curve, roc_auc_score
        y_prob = trained_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        roc_data = {
            'random_forest': (fpr, tpr, auc_score)
        }

        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_roc_curve(roc_data)

        assert os.path.exists(os.path.join(viz.eval_dir, 'roc_curve.png'))

    def test_plot_roc_curve_multiple_models(self, test_config, temp_artifacts_dir, sample_train_test_split):
        """Test plot_roc_curve với nhiều models"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_curve, roc_auc_score

        X_train, X_test, y_train, y_test = sample_train_test_split

        # Train multiple models
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        roc_data = {}
        for name, model in [('random_forest', rf), ('logistic_regression', lr)]:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            roc_data[name] = (fpr, tpr, auc_score)

        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_roc_curve(roc_data)

        assert os.path.exists(os.path.join(viz.eval_dir, 'roc_curve.png'))

    def test_plot_feature_importance(self, test_config, temp_artifacts_dir, trained_model, sample_train_test_split):
        """Test plot_feature_importance tạo bar chart"""
        X_train, _, _, _ = sample_train_test_split

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': trained_model.feature_importances_
        }).sort_values('importance', ascending=False)

        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_feature_importance(importance_df, top_n=10)

        assert os.path.exists(os.path.join(viz.eval_dir, 'feature_importance_top_10.png'))

    def test_plot_feature_importance_empty_df(self, test_config, temp_artifacts_dir):
        """Test plot_feature_importance với empty DataFrame"""
        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        # Should handle gracefully
        viz.plot_feature_importance(None)
        viz.plot_feature_importance(pd.DataFrame())

    def test_plot_model_comparison(self, test_config, temp_artifacts_dir):
        """Test plot_model_comparison tạo grouped bar chart"""
        metrics_dict = {
            'random_forest': {'accuracy': 0.95, 'f1': 0.92, 'recall': 0.90},
            'logistic_regression': {'accuracy': 0.88, 'f1': 0.85, 'recall': 0.82},
            'xgboost': {'accuracy': 0.97, 'f1': 0.95, 'recall': 0.94}
        }

        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_model_comparison(metrics_dict, ['accuracy', 'f1', 'recall'])

        assert os.path.exists(os.path.join(viz.eval_dir, 'model_comparison.png'))

    def test_plot_model_comparison_empty(self, test_config, temp_artifacts_dir):
        """Test plot_model_comparison với empty dict"""
        viz = EvaluateVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        # Should handle gracefully
        viz.plot_model_comparison({}, ['accuracy'])

    def test_save_plot_format(self, test_config, temp_artifacts_dir, mock_logger):
        """Test _save_plot lưu đúng format và log"""
        viz = EvaluateVisualizer(test_config, mock_logger, temp_artifacts_dir)

        metrics_dict = {
            'test_model': {'accuracy': 0.9, 'f1': 0.85}
        }
        viz.plot_model_comparison(metrics_dict, ['accuracy', 'f1'])

        saved_file = os.path.join(viz.eval_dir, 'model_comparison.png')
        assert os.path.exists(saved_file)

        # Check file is actually a valid image
        assert os.path.getsize(saved_file) > 0

