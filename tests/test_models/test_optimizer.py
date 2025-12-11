"""
tests/test_models/test_optimizer.py

Tests cho optimizer (smoke tests cho search/tuning interface).
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.optimizer import ModelOptimizer


def test_optimizer_grid_search(sample_train_test_split, test_config, mock_logger):
    """Test cho grid search optimizer."""
    X_train, X_test, y_train, y_test = sample_train_test_split

    config = test_config.copy()
    config['tuning'] = {'method': 'grid', 'cv_folds': 2, 'scoring': 'accuracy', 'n_jobs': 1}

    optimizer = ModelOptimizer(config, mock_logger)

    param_grid = {'n_estimators': [10, 20]}
    estimator = RandomForestClassifier(random_state=42)

    best_model, best_params = optimizer.optimize(estimator, X_train, y_train, param_grid, 'rf')

    assert best_model is not None
    assert isinstance(best_params, dict)


def test_optimizer_randomized_search(sample_train_test_split, test_config, mock_logger):
    """Test cho randomized search optimizer."""
    X_train, X_test, y_train, y_test = sample_train_test_split

    config = test_config.copy()
    config['tuning'] = {'method': 'randomized', 'cv_folds': 2, 'scoring': 'accuracy', 'n_iter': 2, 'n_jobs': 1}

    optimizer = ModelOptimizer(config, mock_logger)

    param_grid = {'n_estimators': [10, 20, 30]}
    estimator = RandomForestClassifier(random_state=42)

    best_model, best_params = optimizer.optimize(estimator, X_train, y_train, param_grid, 'rf')

    assert best_model is not None
    assert isinstance(best_params, dict)
