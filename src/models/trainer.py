"""
src/models/trainer.py
Nhiệm vụ: Quản lý Model, Training loop, Save/Load
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
    """Class quản lý toàn bộ quy trình training (Main Entry Point for Models)"""

    def __init__(self, config: Dict[str, Any], logger=None):
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
        """Load dữ liệu train/test và tách X, y"""
        if train_path is None or test_path is None:
            train_test_dir = self.config['data']['train_test_dir']
            raw_filename = self.config['data'].get('raw_path')
            if self.logger: self.logger.info("Auto-detecting latest train/test files...")
            train_path, test_path = get_latest_train_test(train_test_dir, raw_filename)

        train_df = IOHandler.read_data(train_path)
        test_df = IOHandler.read_data(test_path)
        target_col = self.config['data']['target_col']

        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_test = test_df.drop(columns=[target_col])
        self.y_test = test_df[target_col]

        if self.logger:
            self.logger.info(f"Data Loaded | Train: {self.X_train.shape} | Test: {self.X_test.shape}")

    # ==================== MODEL FACTORY ====================

    def _get_model_instance(self, model_name: str, params: Dict = None) -> Any:
        """Factory tạo model instance"""
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
        """Tạo Pipeline [Sampler -> Model]"""
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
        """Train đơn giản (Baseline)"""
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
        """Gọi Optimizer để tuning"""
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
        """Train tất cả models trong config"""
        model_names = list(self.config.get('models', {}).keys())
        if not model_names: raise ValueError("No models defined in config")

        if self.logger:
            self.logger.info("=" * 70)
            self.logger.info(f"BATCH TRAINING | Models: {len(model_names)} | Optimize: {optimize}")
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

    def select_best_model(self, all_metrics: Dict[str, Dict]) -> None:
        """Chọn model tốt nhất"""
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
        """Lấy feature importance"""
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