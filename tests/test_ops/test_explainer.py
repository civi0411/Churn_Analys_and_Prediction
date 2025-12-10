import pytest
import pandas as pd
from src.ops.mlops.explainer import ModelExplainer

def test_model_explainer_init():
    # Dummy model, X_train, feature_names
    class DummyModel:
        pass
    model = DummyModel()
    X_train = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    feature_names = ['a', 'b']
    explainer = ModelExplainer(model, X_train, feature_names)
    assert explainer is not None
