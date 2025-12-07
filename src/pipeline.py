"""
src/pipeline.py
Main ML Pipeline Orchestrator (Refactored & Complete)
Điều phối: Data -> Ops -> Training -> Evaluation -> Visualization
"""
import os
import shutil
import logging
from typing import Dict, Any, Tuple, Optional

# --- NEW MODULAR IMPORTS ---
from .data import DataPreprocessor, DataTransformer
from .models import ModelTrainer
from .visualization import EDAVisualizer, EvaluateVisualizer
from .ops import (
    ExperimentTracker, ModelRegistry, DataValidator,
    DataVersioning, ModelMonitor, ModelExplainer
)
from .utils import IOHandler, set_random_seed, get_latest_train_test, get_timestamp, ensure_dir, Logger


class Pipeline:
    """
    Main ML Pipeline với luồng xử lý chuẩn công nghiệp:
    RAW -> CLEAN (Preprocessor) -> SPLIT -> FIT/TRANSFORM (Transformer) -> TRAIN -> EVAL
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.target_col = config['data']['target_col']

        # Set Global Seed
        set_random_seed(config['data'].get('random_state', 42))

        # --- 1. SETUP OPS ---
        self.tracker = ExperimentTracker(config['experiments']['base_dir'])
        self.registry = ModelRegistry(config['mlops']['registry_dir'])
        self.validator = DataValidator(logger)
        self.versioning = DataVersioning(config['dataops']['versions_dir'])
        self.monitor = ModelMonitor(config['monitoring']['base_dir'])

        # --- 2. SETUP CORE COMPONENTS ---
        # Data
        self.preprocessor = DataPreprocessor(config, logger)
        self.transformer = DataTransformer(config, logger)

        # Model
        self.trainer = ModelTrainer(config, logger)

        # Vis (sẽ khởi tạo lại trong run() để gắn với thư mục output cụ thể)
        self.eda_viz = EDAVisualizer(config, logger)
        self.eval_viz = EvaluateVisualizer(config, logger)

        self.logger.info("Pipeline Initialized (Modular Structure)")

    # =========================================================================
    # STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    def run_eda(self) -> None:
        """
        Exploratory Data Analysis trên raw data
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STAGE: EXPLORATORY DATA ANALYSIS")
        self.logger.info("=" * 70)

        # 1. Load Data (Support Batch Folder or Single File)
        raw_path = self.config['data']['raw_path']
        batch_folder = self.config['data'].get('batch_folder')

        if raw_path.lower() == 'full' and batch_folder:
            self.logger.info(f"Loading batch data from: {batch_folder}")
            df = IOHandler.read_batch_folder(batch_folder)
        else:
            df = self.preprocessor.load_data(raw_path)

        self.logger.info(f"Raw Data Loaded | Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        # 2. Visualization using EDAVisualizer
        # Cập nhật đường dẫn lưu nếu đang trong một Run
        if self.tracker.current_run_dir:
            eda_dir = os.path.join(self.tracker.current_run_dir, 'figures', 'eda')
            self.eda_viz = EDAVisualizer(self.config, self.logger, run_specific_dir=os.path.dirname(eda_dir))

        self.eda_viz.plot_missing_values(df)
        self.eda_viz.plot_correlation_matrix(df)

        if self.target_col in df.columns:
            self.eda_viz.plot_target_distribution(df[self.target_col])

        # Helper to identify types for plotting distributions
        self.preprocessor._identify_column_types(
            df)  # (Hàm này có trong code cũ của bạn, cần đảm bảo preprocessor có logic này)
        if hasattr(self.preprocessor, 'numerical_cols'):
            self.eda_viz.plot_numerical_distributions(df, self.preprocessor.numerical_cols)

        self.logger.info("EDA Completed.")

    # =========================================================================
    # STAGE 2: PREPROCESSING (Clean -> Split -> Transform)
    # =========================================================================
    def run_preprocessing(self) -> Tuple[str, str, str]:
        """
        Chạy toàn bộ preprocessing pipeline:
        RAW -> CLEAN -> SPLIT -> FIT(Train) -> TRANSFORM(Test)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STAGE: FULL PREPROCESSING PIPELINE")
        self.logger.info("=" * 70)

        raw_path = self.config['data']['raw_path']

        # 1. DataOps: Version tracking
        source_path = self.config['data'].get('batch_folder') if raw_path.lower() == 'full' else raw_path
        ver_id = self.versioning.create_version(source_path)
        self.logger.info(f"Data Version Created | ID: {ver_id}")

        # 2. Load
        if raw_path.lower() == 'full':
            df = IOHandler.read_batch_folder(source_path)
        else:
            df = self.preprocessor.load_data(raw_path)

        # 3. Clean (Stateless)
        df_clean = self.preprocessor.clean_data(df)

        # Save intermediate cleaned data
        processed_dir = self.config['data']['processed_dir']
        ensure_dir(processed_dir)
        processed_path = os.path.join(processed_dir,
                                      f"{os.path.splitext(os.path.basename(raw_path))[0]}_cleaned.parquet")
        IOHandler.save_data(df_clean, processed_path)

        # 4. Split
        train_df, test_df = self.preprocessor.split_data(df_clean)

        # 5. Transform (Stateful - Fit on Train, Transform on Test)
        self.logger.info("Fitting and transforming features...")
        X_train, y_train = self.transformer.fit_transform(train_df)
        X_test, y_test = self.transformer.transform(test_df)

        # 6. Reconstruct & Save Final Parquet
        train_processed = X_train.copy()
        if y_train is not None: train_processed[self.target_col] = y_train.values

        test_processed = X_test.copy()
        if y_test is not None: test_processed[self.target_col] = y_test.values

        train_test_dir = self.config['data']['train_test_dir']
        ensure_dir(train_test_dir)
        base_name = os.path.splitext(os.path.basename(raw_path))[0]
        train_path = os.path.join(train_test_dir, f"{base_name}_train.parquet")
        test_path = os.path.join(train_test_dir, f"{base_name}_test.parquet")

        IOHandler.save_data(train_processed, train_path)
        IOHandler.save_data(test_processed, test_path)

        # 7. Validate Output Quality
        self.logger.info("--- Data Quality Check (Train Set) ---")
        self.validator.validate_quality(train_processed)

        return processed_path, train_path, test_path

    # =========================================================================
    # STAGE 3: TRAINING
    # =========================================================================
    def run_training(self, train_path: str = None, test_path: str = None,
                     model_name: str = 'all', optimize: bool = True) -> Tuple[ModelTrainer, Dict]:
        """
        Training với dữ liệu đã được preprocess
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STAGE: MODEL TRAINING")
        self.logger.info("=" * 70)

        # 1. Load train/test data
        self.trainer.load_train_test_data(train_path, test_path)

        # 2. DEPENDENCY INJECTION: Lấy Sampler từ Transformer (SMOTE/Tomek)
        # Vì Transformer giữ logic tạo sampler (dựa trên config)
        resampler = self.transformer.get_resampler()
        if resampler and self.logger:
            self.logger.info(f"Resampler Injected | Type: {type(resampler).__name__}")

        # 3. Train models
        metrics = {}
        if model_name == 'all':
            metrics = self.trainer.train_all_models(optimize=optimize, sampler=resampler)
        else:
            if optimize:
                self.trainer.optimize_params(model_name, sampler=resampler)
            else:
                self.trainer.train_model(model_name, sampler=resampler)

            # Evaluate & Select
            eval_res = self.trainer.evaluate(model_name)
            metrics = {model_name: eval_res['metrics']}
            self.trainer.select_best_model({model_name: eval_res['metrics']})

        # 4. Register Best Model & Monitor
        if self.trainer.best_model_name:
            # Register
            best_metrics = metrics[self.trainer.best_model_name]
            reg_path = self.registry.register_model(
                self.trainer.best_model_name,
                self.trainer.best_model,
                best_metrics,
                run_id=self.tracker.current_run_id
            )
            self.logger.info(f"Model Registry | {reg_path}")

            # Monitor Logging
            self.monitor.log_performance(
                self.trainer.best_model_name,
                best_metrics,
                n_samples=len(self.trainer.X_test),
                notes=f"Train Run (Opt={optimize})"
            )

            # Health Check
            history = self.monitor.get_performance_history(self.trainer.best_model_name)
            baseline = None
            if len(history) > 1:
                baseline = {
                    'f1': history.iloc[0]['f1'],
                    'accuracy': history.iloc[0]['accuracy'],
                    'roc_auc': history.iloc[0]['roc_auc']
                }

            health = self.monitor.check_health(
                self.trainer.best_model_name,
                best_metrics,
                baseline_metrics=baseline
            )

            if health['status'] != 'HEALTHY':
                self.logger.warning(f"Model Health: {health['status']} - Issues: {health['issues']}")
            else:
                self.logger.info(f"Model Health: {health['status']}")

        return self.trainer, metrics

    # =========================================================================
    # STAGE 4: VISUALIZATION & EXPLAINABILITY
    # =========================================================================
    def run_visualization(self, trainer: ModelTrainer, metrics: Dict) -> None:
        """
        Visualization + Model Explainability
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STAGE: VISUALIZATION & EXPLAINABILITY")
        self.logger.info("=" * 70)

        # Setup Visualizer trỏ vào thư mục run hiện tại
        if self.tracker.current_run_dir:
            run_figures_dir = os.path.join(self.tracker.current_run_dir, 'figures')
            self.eval_viz = EvaluateVisualizer(self.config, self.logger, run_specific_dir=run_figures_dir)

        # 1. Model comparison
        self.eval_viz.plot_model_comparison(metrics, ['f1', 'roc_auc', 'recall', 'precision'])

        # 2. ROC Curve
        roc_data = {
            name: result['roc_curve_data']
            for name, result in trainer.results.items()
            if result.get('roc_curve_data')
        }
        if roc_data:
            self.eval_viz.plot_roc_curve(roc_data)

        # 3. Best Model Specifics
        if trainer.best_model_name:
            # Confusion Matrix
            y_pred = trainer.best_model.predict(trainer.X_test)
            self.eval_viz.plot_confusion_matrix(trainer.y_test, y_pred, trainer.best_model_name)

            # Feature Importance
            imp = trainer.get_feature_importance()
            if imp is not None:
                self.eval_viz.plot_feature_importance(imp)

            # 4. Model Explainability (SHAP)
            if self.config.get('explainability', {}).get('enabled', False):
                try:
                    self.logger.info("Running SHAP explanation...")
                    # Lấy feature names từ columns của X_train (đã transform)
                    feature_names = trainer.X_train.columns.tolist()

                    explainer = ModelExplainer(
                        trainer.best_model,
                        trainer.X_train,
                        feature_names
                    )

                    # SHAP explanation on sample
                    X_sample = trainer.X_test.head(100)
                    shap_path = os.path.join(self.eval_viz.eval_dir, 'shap_summary.png')
                    explainer.explain_with_shap(X_sample, shap_path)
                    self.logger.info(f"SHAP Plot Saved | {shap_path}")

                except Exception as e:
                    self.logger.warning(f"Explainability failed: {e}")

        self.logger.info("Visualization Completed")

    # =========================================================================
    # MAIN ENTRY POINT (Run Orchestrator)
    # =========================================================================
    def run(self, mode: str = 'full', model_name: str = 'all',
            optimize: bool = True) -> Optional[Tuple]:
        """
        Main pipeline execution
        """
        # 1. Start Tracking
        run_name = f"{get_timestamp()}_{mode.upper()}"
        run_id = self.tracker.start_run(run_name)

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"PIPELINE STARTED | Mode: {mode.upper()} | Run ID: {run_id}")
        self.logger.info(f"{'=' * 70}\n")

        # 2. Setup Run-specific Logging & Config Snapshot
        if self.tracker.current_run_dir:
            # Snapshot config
            config_snapshot_path = os.path.join(self.tracker.current_run_dir, "config_snapshot.yaml")
            IOHandler.save_yaml(self.config, config_snapshot_path)

            # Redirect Logger to Run Folder
            run_log_path = os.path.join(self.tracker.current_run_dir, "run.log")
            # Gọi hàm helper get_logger để add thêm handler vào file log riêng này
            Logger.get_logger('MAIN', run_log_path=run_log_path)
            self.logger.info("✓ Run-specific log initialized")

        # Log params
        self.tracker.log_params({
            'mode': mode, 'model_name': model_name, 'optimize': optimize,
            'test_size': self.config['data']['test_size']
        })

        try:
            # ========== MODE: EDA ==========
            if mode == 'eda':
                self.run_eda()
                self.tracker.end_run("FINISHED")
                return None

            # ========== MODE: PREPROCESS ==========
            elif mode == 'preprocess':
                processed_path, train_path, test_path = self.run_preprocessing()

                # Copy Artifacts to Run Dir
                if self.tracker.current_run_dir:
                    data_dir = os.path.join(self.tracker.current_run_dir, 'data')
                    ensure_dir(data_dir)
                    shutil.copy2(processed_path, os.path.join(data_dir, os.path.basename(processed_path)))
                    shutil.copy2(train_path, os.path.join(data_dir, os.path.basename(train_path)))
                    shutil.copy2(test_path, os.path.join(data_dir, os.path.basename(test_path)))

                self.tracker.end_run("FINISHED")
                return processed_path, train_path, test_path

            # ========== MODE: TRAIN ==========
            elif mode == 'train':
                train_test_dir = self.config['data']['train_test_dir']
                raw_filename = self.config['data']['raw_path']
                train_path, test_path = get_latest_train_test(train_test_dir, raw_filename)

                self.logger.info(f"Using Train | {train_path}")
                self.logger.info(f"Using Test  | {test_path}")

                trainer, metrics = self.run_training(train_path, test_path, model_name, optimize)

                # Save Artifacts
                self._save_training_artifacts(trainer)

                # Log metrics
                if trainer.best_model_name:
                    self.tracker.log_metrics(metrics[trainer.best_model_name])

                self.tracker.end_run("FINISHED")
                return trainer, metrics

            # ========== MODE: VISUALIZE ==========
            elif mode == 'visualize':
                # Quick load & train (no optimization) just to visualize
                train_test_dir = self.config['data']['train_test_dir']
                raw_filename = self.config['data']['raw_path']
                train_path, test_path = get_latest_train_test(train_test_dir, raw_filename)

                self.logger.info("Visualize mode: Running quick training (no-opt)...")
                trainer, metrics = self.run_training(train_path, test_path, model_name, optimize=False)
                self.run_visualization(trainer, metrics)

                self._save_training_artifacts(trainer)
                self.tracker.end_run("FINISHED")
                return None

            # ========== MODE: FULL ==========
            elif mode == 'full':
                self.run_eda()

                # 1. Preprocess
                processed_path, train_path, test_path = self.run_preprocessing()
                if self.tracker.current_run_dir:
                    data_dir = os.path.join(self.tracker.current_run_dir, 'data')
                    ensure_dir(data_dir)
                    shutil.copy2(train_path, os.path.join(data_dir, os.path.basename(train_path)))
                    shutil.copy2(test_path, os.path.join(data_dir, os.path.basename(test_path)))

                # 2. Train
                trainer, metrics = self.run_training(train_path, test_path, model_name, optimize)

                # 3. Visualize
                self.run_visualization(trainer, metrics)

                # 4. Save & Log
                self._save_training_artifacts(trainer)
                if trainer.best_model_name:
                    self.tracker.log_metrics(metrics[trainer.best_model_name])

                self.tracker.end_run("FINISHED")
                return trainer, metrics

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            self.logger.error(f"Pipeline Failed: {e}", exc_info=True)
            self.tracker.end_run("FAILED")
            raise

    def _save_training_artifacts(self, trainer: ModelTrainer):
        """Helper để lưu model và objects vào run_dir"""
        if self.tracker.current_run_dir:
            models_dir = os.path.join(self.tracker.current_run_dir, 'models')
            ensure_dir(models_dir)

            # Save Transformers (quan trọng để inference sau này)
            transformer_path = os.path.join(models_dir, "transformer.joblib")
            IOHandler.save_model(self.transformer, transformer_path)

            # Save Best Model
            if trainer.best_model:
                model_path = os.path.join(models_dir, f"{trainer.best_model_name}.joblib")
                IOHandler.save_model(trainer.best_model, model_path)