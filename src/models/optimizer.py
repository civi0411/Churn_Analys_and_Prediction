"""
src/models/optimizer.py

Hyperparameter Optimization Module

Module này chịu trách nhiệm tối ưu hóa hyperparameters cho ML models:
    - GridSearchCV: Tìm kiếm exhaustive trên toàn bộ parameter grid
    - RandomizedSearchCV: Tìm kiếm ngẫu nhiên (nhanh hơn với grid lớn)

Features:
    - Hỗ trợ cả sklearn estimators và imblearn Pipeline
    - Stratified K-Fold cross-validation
    - Tự động xử lý parameter prefix cho Pipeline
    - Logging chi tiết thời gian và best params

Example:
    >>> optimizer = ModelOptimizer(config, logger)
    >>> best_model, best_params = optimizer.optimize(
    ...     estimator=RandomForestClassifier(),
    ...     X_train=X_train, y_train=y_train,
    ...     param_grid={'n_estimators': [100, 200]},
    ...     model_name='random_forest'
    ... )

Author: Churn Prediction Team
"""
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

# Import để check type
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    ImbPipeline = None


class ModelOptimizer:
    """
    Model Optimizer - Tối ưu hóa hyperparameters.

    Sử dụng GridSearchCV hoặc RandomizedSearchCV với cross-validation
    để tìm bộ parameters tốt nhất.

    Attributes:
        config (Dict): Configuration dictionary
        logger: Logger instance

    Example:
        >>> optimizer = ModelOptimizer(config, logger)
        >>> best_model, best_params = optimizer.optimize(
        ...     estimator, X_train, y_train,
        ...     param_grid={'n_estimators': [100, 200]},
        ...     model_name='random_forest'
        ... )
    """

    def __init__(self, config: Dict, logger=None):
        """
        Khởi tạo ModelOptimizer.

        Args:
            config (Dict): Configuration dictionary chứa:
                - tuning.method: 'grid' hoặc 'randomized'
                - tuning.cv_folds: Số folds cho cross-validation
                - tuning.cv_strategy: 'stratified' hoặc 'kfold'
                - tuning.scoring: Metric để optimize
                - tuning.n_iter: Số iterations cho RandomizedSearch
                - tuning.n_jobs: Số parallel jobs (-1 = all CPUs)
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger

    def optimize(self, estimator: Any, X_train, y_train,
                 param_grid: Dict, model_name: str) -> Tuple[Any, Dict]:
        """
        Thực hiện hyperparameter search (Grid hoặc Random).

        Tự động phát hiện và xử lý imblearn Pipeline bằng cách
        thêm prefix 'model__' vào parameters.

        Args:
            estimator: Sklearn estimator hoặc imblearn Pipeline
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (Dict): Dictionary các parameters cần search
            model_name (str): Tên model (để logging)

        Returns:
            Tuple[Any, Dict]: (Best estimator, Best parameters)

        Config Used:
            - tuning.method: 'grid' hoặc 'randomized'
            - tuning.cv_folds: Số folds (default: 5)
            - tuning.scoring: Metric để optimize (default: 'roc_auc')
            - tuning.n_iter: Số iterations cho RandomizedSearch (default: 30)

        Example:
            >>> best_model, best_params = optimizer.optimize(
            ...     estimator=rf_model,
            ...     X_train=X_train, y_train=y_train,
            ...     param_grid={'n_estimators': [100, 200], 'max_depth': [10, 20]},
            ...     model_name='random_forest'
            ... )
            >>> print(f"Best params: {best_params}")
        """
        if self.logger:
            self.logger.info(f"\n[OPTIMIZING] {model_name.upper()}")

        # 1. Xử lý param grid cho ImbPipeline (thêm prefix model__)
        # Nếu estimator là ImbPipeline, các params của model cần có tiền tố 'model__'
        if ImbPipeline and isinstance(estimator, ImbPipeline):
            final_grid = {f"model__{k}": v for k, v in param_grid.items()}
        else:
            final_grid = param_grid

        # 2. Lấy config tuning
        tuning_config = self.config.get('tuning', {})
        method = tuning_config.get('method', 'randomized')
        cv_folds = tuning_config.get('cv_folds', 5)
        cv_strategy = tuning_config.get('cv_strategy', 'stratified')
        scoring = tuning_config.get('scoring', 'roc_auc')
        n_jobs = tuning_config.get('n_jobs', -1)

        # 3. Setup Cross-validation
        if cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = cv_folds

        # 4. Setup Search Class
        SearchClass = GridSearchCV if method == 'grid' else RandomizedSearchCV

        search_kwargs = {
            'estimator': estimator,
            'cv': cv,
            'scoring': scoring,
            'n_jobs': n_jobs,
            'verbose': 0,
        }

        if method == 'grid':
            search_kwargs['param_grid'] = final_grid
        else:
            search_kwargs['param_distributions'] = final_grid
            search_kwargs['n_iter'] = tuning_config.get('n_iter', 30)
            search_kwargs['random_state'] = 42

        # 5. Execute Search
        if self.logger:
            self.logger.info(f" Method: {method} | CV: {cv_folds} | Scoring: {scoring}")

        start_time = datetime.now()
        search = SearchClass(**search_kwargs)
        search.fit(X_train, y_train)
        optimization_time = (datetime.now() - start_time).total_seconds()

        # 6. Log & Return
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        if self.logger:
            self.logger.info(f" Time: {optimization_time:.2f}s | Best {scoring.upper()}: {best_score:.4f}")
            self.logger.info(f" Best Params: {best_params}")
            self.logger.info(f"{model_name} | Best Params: {best_params}")

        return best_model, best_params