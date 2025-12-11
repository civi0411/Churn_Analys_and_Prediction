"""
tests/models/test_evaluator.py

Tests cho evaluator (metrics calculation functions).
"""

from src.models.evaluator import ModelEvaluator
import logging


def test_evaluate_with_predict_proba(trained_model, sample_train_test_split, mock_logger):
    """Test model evaluation with predict_proba method."""
    X_train, X_test, y_train, y_test = sample_train_test_split

    evaluator = ModelEvaluator(logger=mock_logger)
    res = evaluator.evaluate(trained_model, X_test, y_test, model_name='random_forest')

    # Basic structure
    assert isinstance(res, dict)
    assert 'metrics' in res
    assert 'y_pred' in res
    assert 'y_pred_proba' in res
    assert 'classification_report' in res
    assert 'confusion_matrix' in res
    assert 'roc_curve_data' in res

    metrics = res['metrics']
    # Metrics present and within valid range
    for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0

    # Predictions shape
    assert len(res['y_pred']) == len(y_test)
    assert res['y_pred_proba'] is not None

    # Confusion matrix shape and counts
    cm = res['confusion_matrix']
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_test)

    # roc_curve_data is a 3-tuple (fpr, tpr, auc)
    assert isinstance(res['roc_curve_data'], tuple)
    fpr, tpr, auc = res['roc_curve_data']
    assert hasattr(fpr, '__len__') and hasattr(tpr, '__len__')
    assert isinstance(auc, float)


def test_evaluate_without_predict_proba(sample_train_test_split):
    """Test model evaluation without predict_proba method."""
    from sklearn.svm import SVC

    X_train, X_test, y_train, y_test = sample_train_test_split

    # SVC with probability=False does not provide predict_proba
    model = SVC(kernel='linear', probability=False, random_state=42)
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()
    res = evaluator.evaluate(model, X_test, y_test, model_name='svc')

    # roc related outputs should be None / absent
    assert res['y_pred_proba'] is None
    assert res['roc_curve_data'] is None
    assert 'roc_auc' not in res['metrics']

    # Basic metrics still present
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        assert k in res['metrics']
        assert 0.0 <= res['metrics'][k] <= 1.0


def test_logging_on_evaluate(caplog, trained_model, sample_train_test_split):
    """Test logging during model evaluation."""
    X_train, X_test, y_train, y_test = sample_train_test_split

    # Use caplog to capture logs from the evaluator's logger
    caplog.set_level(logging.INFO)
    evaluator = ModelEvaluator(logger=logging.getLogger('test_logger'))

    with caplog.at_level(logging.INFO):
        _ = evaluator.evaluate(trained_model, X_test, y_test, model_name='rf')

    # Expect at least one INFO record containing '[EVALUATION]'
    messages = [r.getMessage() for r in caplog.records]
    assert any('[EVALUATION]' in m or 'EVALUATION' in m for m in messages)
