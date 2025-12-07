"""
src/models/trainer.py

Model Training Module - Quản lý toàn bộ quy trình training models

Module này là entry point chính cho việc training ML models:
    - Load train/test data
    - Factory pattern để tạo các loại model khác nhau
    - Training với hyperparameter optimization
    - Evaluation và chọn best model
    - Feature importance extraction

Supported Models:
    - Logistic Regression
    - SVM (Support Vector Machine)
    - Decision Tree
    - Random Forest
    - XGBoost
    - AdaBoost

Architecture:
    - ModelTrainer sở hữu ModelOptimizer và ModelEvaluator (Composition)
    - Hỗ trợ imbalanced data với SMOTE/SMOTETomek injection

Example:
    >>> trainer = ModelTrainer(config, logger)
    >>> trainer.load_train_test_data()
    >>> metrics = trainer.train_all_models(optimize=True, sampler=smote)
    >>> print(f"Best model: {trainer.best_model_name}")

Author: Churn Prediction Team
"""
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

# Scikit-learn Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Conditional Imports
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Local Imports
from ..utils import IOHandler, get_latest_train_test, get_timestamp, ensure_dir
from .optimizer import ModelOptimizer
from .evaluator import ModelEvaluator


class ModelTrainer:
    """
    Model Trainer - Quản lý toàn bộ quy trình training.

    Class này đóng vai trò là orchestrator cho việc training:
        - Load data từ train/test files
        - Tạo model instances với Factory pattern
        - Train với hyperparameter optimization
        - Evaluate và chọn best model

    Attributes:
        config (Dict): Configuration dictionary
        logger: Logger instance
        optimizer (ModelOptimizer): Hyperparameter optimizer
        evaluator (ModelEvaluator): Model evaluator
        models (Dict): Dictionary lưu các trained models
        best_model: Model tốt nhất
        best_model_name (str): Tên model tốt nhất
        results (Dict): Kết quả evaluation
        X_train, X_test, y_train, y_test: Data placeholders

    Example:
        >>> trainer = ModelTrainer(config, logger)
        >>> trainer.load_train_test_data()
        >>> sampler = SMOTETomek()
        >>> metrics = trainer.train_all_models(optimize=True, sampler=sampler)
        >>> print(f"Best: {trainer.best_model_name} - F1: {metrics[trainer.best_model_name]['f1']:.4f}")
    """

    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Khởi tạo ModelTrainer.

        Args:
            config (Dict[str, Any]): Configuration dictionary chứa:
                - data.target_col: Tên cột target
                - data.train_test_dir: Thư mục chứa train/test files
                - models: Dictionary các model configs
                - tuning: Hyperparameter tuning settings
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger

        # Composition: Trainer sở hữu Optimizer và Evaluator
        self.optimizer = ModelOptimizer(config, logger)
        self.evaluator = ModelEvaluator(logger)

        # State
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}

        # Data Placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if self.logger:
            self.logger.info("ModelTrainer Initialized")

    # ==================== DATA LOADING ====================

    def load_train_test_data(self, train_path: str = None, test_path: str = None) -> None:
        """
        Load dữ liệu train/test từ files và tách X, y.

        Nếu không cung cấp paths, sẽ tự động tìm files mới nhất
        trong thư mục train_test_dir.

        Args:
            train_path (str, optional): Đường dẫn file train
            test_path (str, optional): Đường dẫn file test

        Sets:
            self.X_train, self.y_train: Training features và target
            self.X_test, self.y_test: Test features và target

        Raises:
            FileNotFoundError: Nếu không tìm thấy train/test files
            KeyError: Nếu target column không tồn tại

        Example:
            >>> trainer.load_train_test_data()
            >>> print(trainer.X_train.shape)
            (4504, 19)
        """
        if train_path is None or test_path is None:
            train_test_dir = self.config['data']['train_test_dir']
            raw_filename = self.config['data'].get('raw_path')
            train_path, test_path = get_latest_train_test(train_test_dir, raw_filename)

        train_df = IOHandler.read_data(train_path)
        test_df = IOHandler.read_data(test_path)
        target_col = self.config['data']['target_col']

        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_test = test_df.drop(columns=[target_col])
        self.y_test = test_df[target_col]

        if self.logger:
            self.logger.info(f"Train Loaded  | {os.path.basename(train_path)}")
            self.logger.info(f"Test Loaded   | {os.path.basename(test_path)}")
            self.logger.info(f"Data Shape    | Train: {self.X_train.shape} | Test: {self.X_test.shape}")

    # ==================== MODEL FACTORY ====================

    def _get_model_instance(self, model_name: str, params: Dict = None) -> Any:
        """
        Factory method tạo model instance.

        Hỗ trợ các models:
            - random_forest: RandomForestClassifier
            - logistic_regression: LogisticRegression
            - svm: SVC
            - decision_tree: DecisionTreeClassifier
            - adaboost: AdaBoostClassifier
            - xgboost: XGBClassifier

        Args:
            model_name (str): Tên model
            params (Dict, optional): Parameters cho model

        Returns:
            Sklearn estimator instance

        Raises:
            ValueError: Nếu model_name không được hỗ trợ
        """
        params = params or {}
        rs = self.config.get('data', {}).get('random_state', 42)

        if model_name == 'random_forest':
            return RandomForestClassifier(random_state=rs, **params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(random_state=rs, max_iter=1000, **params)
        elif model_name == 'svm':
            return SVC(random_state=rs, probability=True, **params)
        elif model_name == 'decision_tree':
            return DecisionTreeClassifier(random_state=rs, **params)
        elif model_name == 'adaboost':
            return AdaBoostClassifier(random_state=rs, **params)
        elif model_name == 'xgboost':
            return XGBClassifier(random_state=rs, **params) if XGBOOST_AVAILABLE else None
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _build_pipeline(self, model_name: str, params: Dict = None, sampler: Any = None) -> Tuple[Any, Dict]:
        """
        Tạo Pipeline [Sampler -> Model] cho imbalanced data.

        Nếu có sampler (SMOTE/SMOTETomek), tạo imblearn Pipeline.
        Nếu không, trả về model đơn lẻ.

        Args:
            model_name (str): Tên model
            params (Dict, optional): Parameters cho model
            sampler: Resampler object (SMOTE, SMOTETomek, etc.)

        Returns:
            Tuple[estimator, Dict]: (Pipeline hoặc Model, pipeline_params)
        """
        base_model = self._get_model_instance(model_name)
        params = params or {}

        if sampler and IMBLEARN_AVAILABLE:
            pipeline = ImbPipeline([('sampler', sampler), ('model', base_model)])
            # Param mapping cho pipeline: model__param
            pipeline_params = {f"model__{k}": v for k, v in params.items()}
            return pipeline, pipeline_params
        else:
            base_model.set_params(**params)
            return base_model, {}

    # ==================== TRAINING & OPTIMIZATION ====================

    def train_model(self, model_name: str, params: Dict = None, sampler: Any = None) -> Any:
        """
        Train model với baseline parameters (không tuning).

        Args:
            model_name (str): Tên model
            params (Dict, optional): Custom parameters
            sampler: Resampler cho imbalanced data

        Returns:
            Trained model (estimator hoặc Pipeline)

        Example:
            >>> model = trainer.train_model('random_forest')
        """
        if self.logger: self.logger.info(f"\n[TRAINING] {model_name.upper()}")

        estimator, pipe_params = self._build_pipeline(model_name, params, sampler=sampler)
        estimator.set_params(**pipe_params)

        start_time = datetime.now()
        estimator.fit(self.X_train, self.y_train)

        if self.logger:
            self.logger.info(f"  Time: {(datetime.now() - start_time).total_seconds():.2f}s")

        self.models[model_name] = estimator
        return estimator

    def optimize_params(self, model_name: str, sampler: Any = None) -> Tuple[Any, Dict]:
        """
        Train model với hyperparameter optimization.

        Delegate cho ModelOptimizer để thực hiện GridSearch/RandomizedSearch.

        Args:
            model_name (str): Tên model
            sampler: Resampler cho imbalanced data

        Returns:
            Tuple[model, Dict]: (Best model, best parameters)

        Example:
            >>> best_model, best_params = trainer.optimize_params('xgboost', sampler=smote)
        """
        # 1. Build estimator (Pipeline hoặc Model trần)
        estimator, _ = self._build_pipeline(model_name, sampler=sampler)

        # 2. Lấy grid
        raw_grid = self.config.get('models', {}).get(model_name, {})
        if not raw_grid:
            if self.logger: self.logger.warning(f"No grid for {model_name}. Fallback to train_model.")
            return self.train_model(model_name), {}

        # 3. Delegate cho Optimizer
        best_model, best_params = self.optimizer.optimize(
            estimator, self.X_train, self.y_train, raw_grid, model_name
        )

        self.models[model_name] = best_model
        return best_model, best_params


    def train_all_models(self, optimize: bool = True, sampler: Any = None) -> Dict[str, Dict]:
        """
        Train tất cả models được định nghĩa trong config.

        Batch training với logging chi tiết.

        Args:
            optimize (bool): True để tuning hyperparameters, False cho baseline
            sampler: Resampler cho imbalanced data

        Returns:
            Dict[str, Dict]: Dictionary chứa metrics của tất cả models
                {model_name: {accuracy, precision, recall, f1, roc_auc}}

        Raises:
            ValueError: Nếu không có models trong config

        Example:
            >>> metrics = trainer.train_all_models(optimize=True, sampler=smote)
            >>> print(metrics['xgboost']['f1'])
            0.9347
        """
        model_names = list(self.config.get('models', {}).keys())
        if not model_names: raise ValueError("No models defined in config")


        if self.logger:
            sampler_name = type(sampler).__name__ if sampler else 'None'
            self.logger.info("=" * 70)
            self.logger.info(f"BATCH TRAINING | Models: {len(model_names)} | Optimize: {optimize} | Sampler: {sampler_name}")
            self.logger.info("=" * 70)

        all_metrics = {}
        for model_name in model_names:
            try:
                # 1. Train / Optimize
                if optimize:
                    model, _ = self.optimize_params(model_name, sampler=sampler)
                else:
                    model = self.train_model(model_name, sampler=sampler)

                # 2. Evaluate (Delegate cho Evaluator)
                eval_result = self.evaluator.evaluate(model, self.X_test, self.y_test, model_name)

                # Store results
                self.results[model_name] = eval_result
                all_metrics[model_name] = eval_result['metrics']

            except Exception as e:
                if self.logger: self.logger.error(f"Error training {model_name}: {e}")
                continue

        self.select_best_model(all_metrics)
        return all_metrics

    # ==================== HELPERS ====================

    def evaluate(self, model_name: str) -> Dict:
        """
        Evaluate một model đã train.

        Delegate cho ModelEvaluator để tính metrics.

        Args:
            model_name (str): Tên model đã train

        Returns:
            Dict: Kết quả evaluation chứa:
                - metrics: {accuracy, precision, recall, f1, roc_auc}
                - confusion_matrix: Confusion matrix array
                - roc_curve_data: (fpr, tpr, auc) tuple

        Raises:
            ValueError: Nếu model chưa được train
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        eval_result = self.evaluator.evaluate(model, self.X_test, self.y_test, model_name)
        self.results[model_name] = eval_result
        return eval_result

    def select_best_model(self, all_metrics: Dict[str, Dict]) -> None:
        """
        Chọn model tốt nhất dựa trên scoring metric.

        Args:
            all_metrics (Dict): Metrics của tất cả models

        Sets:
            self.best_model: Best trained model
            self.best_model_name: Tên model tốt nhất

        Config:
            tuning.scoring: Metric để so sánh (default: 'f1')
        """
        scoring_metric = self.config.get('tuning', {}).get('scoring', 'f1')
        best_score = -1
        best_name = None

        for model_name, metrics in all_metrics.items():
            score = metrics.get(scoring_metric, 0)
            if score > best_score:
                best_score = score
                best_name = model_name

        if best_name:
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            if self.logger:
                self.logger.info("-" * 70)
                self.logger.info(f"BEST MODEL: {best_name.upper()} | {scoring_metric.upper()}: {best_score:.4f}")
                self.logger.info("-" * 70)

    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Lấy feature importance từ tree-based model.

        Chỉ hoạt động với models có attribute feature_importances_:
            - RandomForest
            - XGBoost
            - DecisionTree
            - AdaBoost

        Args:
            model_name (str, optional): Tên model. Mặc định là best_model
            top_n (int): Số features cần lấy

        Returns:
            pd.DataFrame: DataFrame với columns ['feature', 'importance']
            None: Nếu model không hỗ trợ feature importance

        Example:
            >>> importance = trainer.get_feature_importance(top_n=10)
            >>> print(importance.head())
               feature  importance
            0  feature_1    0.25
            1  feature_2    0.18
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)

        # Lấy model gốc nếu nằm trong Pipeline
        actual_model = model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            actual_model = model.named_steps['model']

        if model is None or not hasattr(actual_model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': actual_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    # ==================== IO ====================

    def save_model(self, model_name: str = None, file_path: str = None, method: str = 'joblib') -> str:
        """Lưu model"""
        if model_name is None: model_name = self.best_model_name
        model = self.models.get(model_name)

        if not model: raise ValueError(f"Model {model_name} not found")

        if file_path is None:
            models_dir = self.config.get('artifacts', {}).get('models_dir', 'artifacts/models')
            ensure_dir(models_dir)
            timestamp = get_timestamp()
            file_path = os.path.join(models_dir, f"{model_name}_{timestamp}.{method}")

        IOHandler.save_model(model, file_path, method)
        if self.logger: self.logger.info(f"Model Saved | {file_path}")
        return file_path

    def load_model(self, file_path: str, method: str = 'joblib') -> Any:
        """Load model"""
        model = IOHandler.load_model(file_path, method)
        if self.logger: self.logger.info(f"Model Loaded | {file_path}")
        return model

    def save_results(self, file_path: str = None) -> str:
        """Lưu kết quả evaluation"""
        if file_path is None:
            results_dir = self.config.get('artifacts', {}).get('results_dir', 'artifacts/results')
            ensure_dir(results_dir)
            timestamp = get_timestamp()
            file_path = os.path.join(results_dir, f"results_{timestamp}.json")

        # Chuẩn bị JSON (convert numpy types to python types)
        results_json = {}
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix'].tolist()  # JSON serializable
            }

        IOHandler.save_json(results_json, file_path)
        if self.logger: self.logger.info(f"Results Saved | {file_path}")
        return file_path