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
        """Test plot_missing_values"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_missing_values(sample_raw_df_with_missing)
        assert os.path.exists(os.path.join(viz.eda_dir, 'missing_values.png'))

    def test_plot_target_distribution(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_target_distribution"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_target_distribution(sample_raw_df['Churn'])
        assert os.path.exists(os.path.join(viz.eda_dir, 'target_distribution.png'))

    def test_plot_numerical_distributions(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_numerical_distributions"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        num_cols = ['Tenure', 'WarehouseToHome', 'CashbackAmount']
        viz.plot_numerical_distributions(sample_raw_df, num_cols)
        assert os.path.exists(os.path.join(viz.eda_dir, 'numerical_distributions.png'))

    def test_plot_boxplots(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_boxplots"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        num_cols = ['Tenure', 'WarehouseToHome']
        viz.plot_boxplots(sample_raw_df, num_cols)
        assert os.path.exists(os.path.join(viz.eda_dir, 'boxplots.png'))

    def test_plot_correlation_matrix(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_correlation_matrix"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_correlation_matrix(sample_raw_df)
        assert os.path.exists(os.path.join(viz.eda_dir, 'correlation_matrix.png'))

    def test_plot_correlation_with_target(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_correlation_with_target"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_correlation_with_target(sample_raw_df, 'Churn')
        assert os.path.exists(os.path.join(viz.eda_dir, 'correlation_with_target.png'))

    def test_plot_churn_by_category(self, test_config, sample_raw_df, temp_artifacts_dir):
        """Test plot_churn_by_category"""
        viz = EDAVisualizer(test_config, run_specific_dir=temp_artifacts_dir)
        viz.plot_churn_by_category(sample_raw_df, 'Churn')
        assert os.path.exists(os.path.join(viz.eda_dir, 'churn_by_category.png'))

    def test_run_full_eda(self, test_config, sample_raw_df, temp_artifacts_dir, mock_logger):
        """Test run_full_eda tạo tất cả plots"""
        viz = EDAVisualizer(test_config, mock_logger, temp_artifacts_dir)
        viz.run_full_eda(sample_raw_df, 'Churn')

        # Check plots created
        assert os.path.exists(os.path.join(viz.eda_dir, 'target_distribution.png'))
        assert os.path.exists(os.path.join(viz.eda_dir, 'numerical_distributions.png'))
        assert os.path.exists(os.path.join(viz.eda_dir, 'boxplots.png'))
        assert os.path.exists(os.path.join(viz.eda_dir, 'correlation_matrix.png'))

