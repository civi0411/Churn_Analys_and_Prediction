"""
tests/test_ops/test_explainer.py

Tests cho ModelExplainer (khởi tạo cơ bản với dummy model).
"""

import pytest
import pandas as pd
from src.ops.mlops.explainer import ModelExplainer


def test_model_explainer_init():
    """Kiểm tra khởi tạo `ModelExplainer` với dummy model và X_train nhỏ."""
    # Dummy model, X_train, feature_names
    class DummyModel:
        pass
    model = DummyModel()
    X_train = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    feature_names = ['a', 'b']
    explainer = ModelExplainer(model, X_train, feature_names)
    assert explainer is not None
