"""
Module `models.trainer` - quản lý huấn luyện, tuning và đánh giá mô hình.

Important keywords: Args, Returns, Methods, Raises
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
    Class: ModelTrainer
    Quản lý toàn bộ vòng đời huấn luyện mô hình: từ load dữ liệu, huấn luyện, tuning đến đánh giá và lưu trữ.

    Methods:
        load_train_test_data: Tải dữ liệu huấn luyện và kiểm thử.
        train_model: Huấn luyện một mô hình đơn lẻ (không tuning).
        optimize_params: Tối ưu hóa tham số (Hyperparameter Tuning) cho mô hình.
        train_all_models: Huấn luyện hàng loạt mô hình được định nghĩa trong config.
        evaluate: Đánh giá hiệu năng của một mô hình cụ thể.
        select_best_model: Chọn ra mô hình tốt nhất dựa trên metrics.
        get_feature_importance: Trích xuất độ quan trọng của các đặc trưng (Feature Importance).
        save_model: Lưu mô hình xuống file.
        load_model: Tải mô hình từ file.
        save_results: Lưu kết quả đánh giá xuống file JSON.
    """

    def __init__(self, config: Dict[str, Any], logger=None):
        """Khởi tạo ModelTrainer.

        Args:
            config (Dict): cấu hình mô hình và tuning
            logger: logger (optional)
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
        """Load dữ liệu train/test từ files và tách X, y.

        Args:
            train_path (str, optional): Đường dẫn file train. Nếu None, tự động tìm file mới nhất.
            test_path (str, optional): Đường dẫn file test. Nếu None, tự động tìm file mới nhất.

        Raises:
            FileNotFoundError: Nếu không tìm thấy file train/test.
            KeyError: Nếu cột target không tồn tại trong dữ liệu.
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
        Factory method để tạo instance của model dựa trên tên.

        Args:
            model_name (str): Tên mô hình (vd: 'random_forest', 'xgboost').
            params (Dict, optional): Các tham số khởi tạo cho model.

        Returns:
            Any: Scikit-learn estimator object.

        Raises:
            ValueError: Nếu tên mô hình không được hỗ trợ.
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
        Tạo Pipeline kết hợp Sampler (nếu có) và Model để xử lý mất cân bằng dữ liệu.

        Args:
            model_name (str): Tên mô hình.
            params (Dict, optional): Tham số của model.
            sampler (Any, optional): Đối tượng sampler (vd: SMOTE) từ imblearn.

        Returns:
            Tuple[Any, Dict]: (Estimator object, Dictionary tham số đã được prefix đúng chuẩn pipeline).
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
        Huấn luyện một mô hình với tham số cố định (không tìm kiếm GridSearch).

        Args:
            model_name (str): Tên mô hình cần huấn luyện.
            params (Dict, optional): Tham số mô hình.
            sampler (Any, optional): Sampler xử lý mất cân bằng.

        Returns:
            Any: Mô hình đã được huấn luyện (Fitted Estimator).
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
        Tìm bộ tham số tốt nhất cho mô hình bằng Hyperparameter Optimization (Grid/Random Search).

        Args:
            model_name (str): Tên mô hình.
            sampler (Any, optional): Sampler xử lý mất cân bằng.

        Returns:
            Tuple[Any, Dict]: (Mô hình tốt nhất, Bộ tham số tốt nhất).
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
        Huấn luyện tất cả các mô hình được định nghĩa trong file cấu hình.

        Args:
            optimize (bool): Có thực hiện tuning tham số hay không. Defaults to True.
            sampler (Any, optional): Sampler xử lý mất cân bằng.

        Returns:
            Dict[str, Dict]: Dictionary chứa metrics của tất cả các mô hình đã huấn luyện.
        """
        model_names = list(self.config.get('models', {}).keys())
        if not model_names: raise ValueError("No models defined in config")


        if self.logger:
            sampler_name = type(sampler).__name__ if sampler else 'None'
            self.logger.info("=" * 70)
            self.logger.info(f"TRAINING RUN | Models: {len(model_names)} | Optimize: {optimize} | Sampler: {sampler_name}")
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
        Đánh giá lại một mô hình đã có trong danh sách self.models.

        Args:
            model_name (str): Tên mô hình cần đánh giá.

        Returns:
            Dict: Kết quả đánh giá từ ModelEvaluator.
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        eval_result = self.evaluator.evaluate(model, self.X_test, self.y_test, model_name)
        self.results[model_name] = eval_result
        return eval_result

    def select_best_model(self, all_metrics: Dict[str, Dict]) -> None:
        """
        So sánh metrics của các mô hình và chọn ra mô hình tốt nhất.

        Args:
            all_metrics (Dict): Dictionary chứa metrics của các mô hình.

        Notes:
            Metric dùng để so sánh được định nghĩa trong config['tuning']['scoring'] (mặc định là f1).
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
        Lấy danh sách các đặc trưng quan trọng nhất (Feature Importance) từ mô hình cây (Tree-based).

        Args:
            model_name (str, optional): Tên mô hình. Nếu None, dùng best model.
            top_n (int, optional): Số lượng đặc trưng lấy ra. Defaults to 20.

        Returns:
            Optional[pd.DataFrame]: DataFrame gồm 2 cột ['feature', 'importance'], hoặc None nếu model không hỗ trợ.
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
        """
        Lưu đối tượng mô hình xuống file.

        Args:
            model_name (str, optional): Tên mô hình cần lưu. Defaults to best model.
            file_path (str, optional): Đường dẫn lưu file. Nếu None, tự tạo tên file theo timestamp.
            method (str, optional): Phương thức lưu ('joblib' hoặc 'pickle').

        Returns:
            str: Đường dẫn file đã lưu.
        """
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
        """
        Tải mô hình từ file lên bộ nhớ.

        Args:
            file_path (str): Đường dẫn file mô hình.
            method (str, optional): Phương thức tải.

        Returns:
            Any: Đối tượng mô hình đã tải.
        """
        model = IOHandler.load_model(file_path, method)
        if self.logger: self.logger.info(f"Model Loaded | {file_path}")
        return model

