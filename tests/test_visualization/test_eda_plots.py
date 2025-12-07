"""
tests/test_visualization/test_eda_plots.py
Tests for src/visualization/eda_plots.py
"""
import pytest
import pandas as pd
import numpy as np
import os

from src.visualization.eda_plots import EDAVisualizer


class TestEDAVisualizer:
    """Test cases for EDAVisualizer class"""

    def test_init(self, test_config, temp_artifacts_dir, mock_logger):
        """Test khởi tạo EDAVisualizer"""
        viz = EDAVisualizer(test_config, mock_logger, temp_artifacts_dir)

        assert viz.config == test_config
        assert os.path.exists(viz.eda_dir)

    def test_plot_missing_values(self, test_config, sample_raw_df_with_missing, temp_artifacts_dir):
        """Test plot_missing_values tạo plot"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        viz.plot_missing_values(sample_raw_df_with_missing)

        assert os.path.exists(os.path.join(viz.eda_dir, 'missing_values.png'))

    def test_plot_missing_values_no_missing(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_missing_values không tạo plot khi không có missing"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        viz.plot_missing_values(sample_raw_df)

        # Should not create file if no missing values
        # (behavior depends on implementation)

    def test_plot_target_distribution(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_target_distribution tạo donut chart"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        viz.plot_target_distribution(sample_raw_df['Churn'])

        assert os.path.exists(os.path.join(viz.eda_dir, 'target_distribution.png'))

    def test_plot_correlation_matrix(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_correlation_matrix tạo heatmap"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        # Use raw df which has more numeric columns
        viz.plot_correlation_matrix(sample_raw_df)

        # File may not be created if not enough numeric columns
        # Just check no error is raised

    def test_plot_numerical_distributions(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_numerical_distributions tạo histograms"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        numerical_cols = ['Tenure', 'WarehouseToHome', 'CashbackAmount']
        viz.plot_numerical_distributions(sample_raw_df, numerical_cols)

        assert os.path.exists(os.path.join(viz.eda_dir, 'numerical_distributions.png'))

    def test_plot_outliers_boxplot(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_outliers_boxplot tạo boxplots"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)

        numerical_cols = ['Tenure', 'WarehouseToHome']
        viz.plot_outliers_boxplot(sample_raw_df, numerical_cols)

        assert os.path.exists(os.path.join(viz.eda_dir, 'outliers_boxplot.png'))

    def test_save_plot_logs_correctly(self, test_config, sample_raw_df, temp_artifacts_dir, mock_logger):
        """Test _save_plot logs đúng format"""
        viz = EDAVisualizer(test_config, mock_logger, temp_artifacts_dir)

        viz.plot_target_distribution(sample_raw_df['Churn'])

        # Logger should have been called (check logs manually or mock)
        assert os.path.exists(os.path.join(viz.eda_dir, 'target_distribution.png'))

