"""
tests/models/test_trainer.py

Unit tests cho `trainer` (smoke tests cơ bản cho luồng huấn luyện).
"""

import os

from src.models.trainer import ModelTrainer


def test_trainer_load_and_train(temp_dir, sample_processed_df, test_config, mock_logger):
    """Kiểm tra nhanh: tải dữ liệu vào trainer và chạy train_all_models (cơ sở, không tối ưu hóa)."""
    # Prepare train/test split files
    train_file = os.path.join(temp_dir, 'train.parquet')
    test_file = os.path.join(temp_dir, 'test.parquet')

    # Use processed df and create train/test with target
    df = sample_processed_df.copy()
    df.to_parquet(train_file)
    df.to_parquet(test_file)

    config = test_config.copy()
    config['data']['train_test_dir'] = temp_dir
    # Define a simple model config for training
    config['models'] = {'logistic_regression': {'C': [0.1, 1]}}

    trainer = ModelTrainer(config, mock_logger)
    trainer.load_train_test_data(train_path=train_file, test_path=test_file)

    # Run baseline training (no optimization)
    metrics = trainer.train_all_models(optimize=False)

    assert isinstance(metrics, dict)
    assert 'logistic_regression' in metrics
    for k in ('accuracy', 'precision', 'recall', 'f1'):
        assert k in metrics['logistic_regression']


def test_save_and_load_model(temp_dir, sample_processed_df, test_config, mock_logger):
    """Kiểm tra lưu_model và load_model vòng lặp"""
    train_file = os.path.join(temp_dir, 'train.parquet')
    test_file = os.path.join(temp_dir, 'test.parquet')

    df = sample_processed_df.copy()
    df.to_parquet(train_file)
    df.to_parquet(test_file)

    config = test_config.copy()
    config['data']['train_test_dir'] = temp_dir
    config['models'] = {'random_forest': {'n_estimators': [10]}}

    trainer = ModelTrainer(config, mock_logger)
    trainer.load_train_test_data(train_path=train_file, test_path=test_file)

    # Train a single model
    model = trainer.train_model('random_forest')
    trainer.models['random_forest'] = model
    trainer.best_model_name = 'random_forest'

    model_path = trainer.save_model(file_path=os.path.join(temp_dir, 'rf.joblib'))
    assert os.path.exists(model_path)

    loaded = trainer.load_model(model_path)
    assert loaded is not None


def test_optimize_params_and_select_best(temp_dir, sample_processed_df, test_config, mock_logger):
    """Kiểm tra optimize_params và select_best_model."""
    train_file = os.path.join(temp_dir, 'train.parquet')
    test_file = os.path.join(temp_dir, 'test.parquet')
    df = sample_processed_df.copy()
    df.to_parquet(train_file)
    df.to_parquet(test_file)
    config = test_config.copy()
    config['data']['train_test_dir'] = temp_dir
    config['models'] = {'logistic_regression': {'C': [0.1, 1]}, 'random_forest': {'n_estimators': [5, 10]}}
    config['tuning'] = {'method': 'grid', 'cv_folds': 2, 'scoring': 'accuracy', 'n_jobs': 1}
    trainer = ModelTrainer(config, mock_logger)
    trainer.load_train_test_data(train_path=train_file, test_path=test_file)
    # Optimize params cho logistic_regression
    best_model, best_params = trainer.optimize_params('logistic_regression')
    assert best_model is not None
    assert isinstance(best_params, dict)
    # Train all models và chọn best
    metrics = trainer.train_all_models(optimize=True)
    trainer.select_best_model(metrics)
    assert trainer.best_model_name in metrics
    assert trainer.best_model is not None


def test_evaluate_and_feature_importance(temp_dir, sample_processed_df, test_config, mock_logger):
    """Kiểm tra evaluate và get_feature_importance."""
    train_file = os.path.join(temp_dir, 'train.parquet')
    test_file = os.path.join(temp_dir, 'test.parquet')
    df = sample_processed_df.copy()
    df.to_parquet(train_file)
    df.to_parquet(test_file)
    config = test_config.copy()
    config['data']['train_test_dir'] = temp_dir
    config['models'] = {'random_forest': {'n_estimators': [5]}}
    trainer = ModelTrainer(config, mock_logger)
    trainer.load_train_test_data(train_path=train_file, test_path=test_file)
    trainer.train_all_models(optimize=False)
    # Evaluate lại model
    eval_result = trainer.evaluate('random_forest')
    assert isinstance(eval_result, dict)
    # Feature importance
    fi = trainer.get_feature_importance('random_forest', top_n=2)
    if fi is not None:
        assert 'feature' in fi.columns and 'importance' in fi.columns

