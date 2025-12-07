"""
src/models/optimizer.py
Nhiệm vụ: Hyperparameter Tuning (GridSearch / RandomSearch)
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
    """Class chuyên trách tối ưu hóa tham số"""

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger

    def optimize(self, estimator: Any, X_train, y_train,
                 param_grid: Dict, model_name: str) -> Tuple[Any, Dict]:
        """
        Thực hiện Search (Grid hoặc Random) dựa trên config
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

        return best_model, best_params