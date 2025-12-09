"""
src/pipeline.py

Main ML Pipeline Orchestrator

Module này là entry point chính điều phối toàn bộ ML pipeline:
    - Stage 1: EDA (Exploratory Data Analysis)
    - Stage 2: Preprocessing (Load → Clean → Split → Transform)
    - Stage 3: Training (Model training với hyperparameter optimization)
    - Stage 4: Visualization & Explainability (Plots, SHAP, Reports)

Supported Modes:
    - 'full': Chạy toàn bộ pipeline từ đầu đến cuối
    - 'eda': Chỉ chạy EDA
    - 'preprocess': Chỉ chạy preprocessing
    - 'train': Chỉ chạy training (yêu cầu đã có train/test data)
    - 'visualize': Chỉ chạy visualization

Architecture:
    Pipeline sử dụng Composition pattern, sở hữu các components:
    - DataPreprocessor, DataTransformer (Data module)
    - ModelTrainer (Models module)
    - EDAVisualizer, EvaluateVisualizer (Visualization module)
    - ExperimentTracker, ModelRegistry, etc. (Ops module)

Example:
    >>> from src.pipeline import Pipeline
    >>> from src.utils import ConfigLoader, Logger
    >>>
    >>> config = ConfigLoader.load_config('config/config.yaml')
    >>> logger = Logger.get_logger('MAIN')
    >>> pipeline = Pipeline(config, logger)
    >>> trainer, metrics = pipeline.run(mode='full', optimize=True)

Author: Churn Prediction Team
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
    DataVersioning, ModelMonitor, ModelExplainer, ReportGenerator
)
from .utils import IOHandler, set_random_seed, get_latest_train_test, get_timestamp, ensure_dir, Logger


class Pipeline:
    """
    Main ML Pipeline Orchestrator.
    Điều phối toàn bộ workflow từ raw data đến trained model:
        RAW → CLEAN → SPLIT → TRANSFORM → TRAIN → EVALUATE → VISUALIZE
    Attributes:
        config (Dict): Configuration dictionary
        logger: Logger instance
        target_col (str): Tên cột target
        tracker (ExperimentTracker): Experiment tracking
        registry (ModelRegistry): Model registry
        validator (DataValidator): Data validation
        versioning (DataVersioning): Data versioning
        monitor (ModelMonitor): Model monitoring
        report_generator (ReportGenerator): Report generation
        preprocessor (DataPreprocessor): Data preprocessing
        transformer (DataTransformer): Data transformation
        trainer (ModelTrainer): Model training
        eda_viz (EDAVisualizer): EDA visualization
        eval_viz (EvaluateVisualizer): Evaluation visualization
    Example:
        >>> pipeline = Pipeline(config, logger)
        >>> trainer, metrics = pipeline.run(mode='full', optimize=True)
        >>> print(f"Best model: {trainer.best_model_name}")
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo Pipeline với tất cả components.

        Args:
            config (Dict[str, Any]): Configuration dictionary từ config.yaml
            logger (logging.Logger): Logger instance
        """
        self.config = config
        self.logger = logger
        self.target_col = config['data']['target_col']

        # Set Global Seed
        set_random_seed(config['data'].get('random_state', 42))

        # --- 1. SETUP OPS ---
        self.tracker = ExperimentTracker(config['experiments']['base_dir'])
        self.registry = ModelRegistry(config['mlops']['registry_dir'])
        self.validator = DataValidator(logger)
        self.versioning = DataVersioning(config['dataops']['versions_dir'], logger)
        self.monitor = ModelMonitor(config['monitoring']['base_dir'])
        # Construct ReportGenerator with named args to avoid argument-order mistakes
        self.report_generator = ReportGenerator(experiments_base_dir=config['experiments']['base_dir'], logger=logger)

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
    def run_eda(self) -> dict:
        """
        Exploratory Data Analysis trên raw data
        Returns EDA summary for reporting.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STAGE: EXPLORATORY DATA ANALYSIS")
        self.logger.info("=" * 70)

        # 1. Load Data (Single file)
        raw_path = self.config['data']['raw_path']
        df = self.preprocessor.load_data(raw_path)

        self.logger.info(f"Raw Data Loaded | Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        # 2. Visualization using EDAVisualizer
        if self.tracker.current_run_dir:
            run_figures_dir = os.path.join(self.tracker.current_run_dir, 'figures')
            self.eda_viz = EDAVisualizer(self.config, self.logger, run_specific_dir=run_figures_dir)

        self.eda_viz.run_full_eda(df, self.target_col)

        # Generate EDA markdown report that will embed the EDA figures under the run dir
        try:
            eda_report_path = self.report_generator.generate_eda_report(
                df,
                filename=f"eda_report_{self.tracker.current_run_id}",
                target_col=self.target_col,
                format='markdown',
                run_id=self.tracker.current_run_id
            )
            self.logger.info(f"EDA Report written to: {eda_report_path}")
        except Exception as e:
            self.logger.warning(f"Failed to generate EDA report: {e}")

        self.logger.info(f"EDA Completed. Check {self.eda_viz.eda_dir}")

        # Build EDA summary for reporting
        summary = {}
        if self.target_col in df.columns:
            churn_rate = df[self.target_col].value_counts(normalize=True).get(1, 0)
            summary['churn_rate'] = churn_rate
        summary['missing'] = int(df.isnull().sum().sum())
        summary['n_rows'] = df.shape[0]
        summary['n_cols'] = df.shape[1]
        summary['duplicate'] = int(df.duplicated().sum())
        # Top correlated features (if correlation matrix available)
        try:
            corr = df.corr()[self.target_col].abs().sort_values(ascending=False)
            top_corr = [c for c in corr.index if c != self.target_col][:3]
            summary['top_correlated'] = top_corr
        except Exception:
            summary['top_correlated'] = []
        summary['resampling'] = 'Đã áp dụng SMOTE/Tomek để cân bằng lại tập huấn luyện.'
        return summary

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

        # 1. Load Data - Single file
        df = self.preprocessor.load_data(raw_path)
        source_path = raw_path

        # 2. DataOps: Version tracking for RAW data (với đầy đủ metadata)
        ver_id = self.versioning.create_version(
            file_path=source_path,
            dataset_name="raw_data",
            df=df,
            description="Raw input data",
            tags=["raw", "input"]
        )
        self.logger.info(f"Data Version  | {ver_id} (raw)")

        # 3. Clean (Stateless)
        df_clean = self.preprocessor.clean_data(df)

        # Save intermediate cleaned data
        processed_dir = self.config['data']['processed_dir']
        ensure_dir(processed_dir)
        processed_path = os.path.join(processed_dir,
                                      f"{os.path.splitext(os.path.basename(raw_path))[0]}_cleaned.parquet")
        IOHandler.save_data(df_clean, processed_path)

        # DataOps: Version tracking for CLEANED data
        self.versioning.create_version(
            file_path=processed_path,
            dataset_name="cleaned_data",
            df=df_clean,
            description="Data after cleaning",
            tags=["cleaned", "processed"]
        )
        # Add lineage: cleaned <- raw
        self.versioning.add_lineage("cleaned_data", "raw_data", "cleaning")

        # 4. Split
        train_df, test_df = self.preprocessor.split_data(df_clean)

        # 5. Transform (Stateful - Fit on Train, Transform on Test)
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: FIT & TRANSFORM")
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

        self.logger.info(f"Saved: {train_path} ({train_processed.shape})")
        self.logger.info(f"Saved: {test_path} ({test_processed.shape})")

        # DataOps: Version tracking for TRAIN/TEST data
        self.versioning.create_version(
            file_path=train_path,
            dataset_name="train_data",
            df=train_processed,
            description="Training dataset (transformed)",
            tags=["train", "final"]
        )
        self.versioning.create_version(
            file_path=test_path,
            dataset_name="test_data",
            df=test_processed,
            description="Test dataset (transformed)",
            tags=["test", "final"]
        )
        # Add lineage
        self.versioning.add_lineage("train_data", "cleaned_data", "split+transform")
        self.versioning.add_lineage("test_data", "cleaned_data", "split+transform")

        # 7. Validate Output Quality
        self.logger.info("=" * 60)
        self.logger.info("PREPROCESSING COMPLETED")
        quality = self.validator.validate_quality(train_processed)
        self.logger.info(f"Data Quality  | Null: {quality['null_ratio']:.2%} | Dup: {quality['duplicate_ratio']:.2%}")

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
                    import numpy as np
                    # Lấy feature names từ columns của X_train (đã transform)
                    X_train_numeric = trainer.X_train.select_dtypes(include=[np.number])
                    feature_names = X_train_numeric.columns.tolist()

                    explainer = ModelExplainer(
                        trainer.best_model,
                        trainer.X_train,
                        feature_names,
                        logger=self.logger  # Truyền logger
                    )

                    # SHAP explanation on sample
                    shap_samples = self.config.get('explainability', {}).get('shap_samples', 100)
                    X_sample = trainer.X_test.head(shap_samples).select_dtypes(include=[np.number])
                    shap_path = os.path.join(self.eval_viz.eval_dir, 'shap_summary.png')

                    success = explainer.explain_with_shap(X_sample, shap_path)
                    if not success:
                        self.logger.warning("SHAP plot was not generated")

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

            # Redirect Logger to Run Folder. Name per-run log by selected mode (e.g., 'train.log', 'full.log')
            run_log_filename = f"{mode.lower()}.log"
            run_log_path = os.path.join(self.tracker.current_run_dir, run_log_filename)
            # Add per-run handler (captures INFO+ for the run)
            Logger.get_logger('MAIN', run_log_path=run_log_path)
            self.logger.info("✓ Run-specific log initialized")

        # Log params
        self.tracker.log_params({
            'mode': mode, 'model_name': model_name, 'optimize': optimize,
            'test_size': self.config['data']['test_size']
        })

        # Report format: Markdown-first (hard-coded)
        report_format = 'markdown'

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

                # 2. Visualization (will create evaluation figures inside run dir)
                self.run_visualization(trainer, metrics)

                # Generate Training Report (embed evaluation figures)
                try:
                    feature_importance = trainer.get_feature_importance()
                    report_path = self.report_generator.generate_training_report(
                        run_id=self.tracker.current_run_id,
                        best_model_name=trainer.best_model_name,
                        all_metrics=metrics,
                        feature_importance=feature_importance,
                        config=self.config,
                        format='markdown',
                        mode=mode,
                        optimize=optimize,
                        args=None
                    )
                    self.logger.info(f"Report Generated | {os.path.basename(report_path)}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate training report: {e}")

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
                eda_summary = self.run_eda()

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

                # 5. Final Summary
                self._log_training_summary(trainer, metrics)

                # 6. Generate full report (EDA + Training)
                try:
                    feature_importance = trainer.get_feature_importance()
                    report_path = self.report_generator.generate_training_report(
                        run_id=self.tracker.current_run_id,
                        best_model_name=trainer.best_model_name,
                        all_metrics=metrics,
                        feature_importance=feature_importance,
                        config=self.config,
                        format='markdown',
                        mode=mode,
                        optimize=optimize,
                        args=None,
                        eda_summary=eda_summary
                    )
                    self.logger.info(f"Report Generated | {os.path.basename(report_path)}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate training report: {e}")

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

    def _log_training_summary(self, trainer: ModelTrainer, metrics: Dict) -> None:
        """Log training results summary theo format chuẩn"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING RESULTS")
        self.logger.info(f"   Best Model: {trainer.best_model_name}")
        for model_name, model_metrics in metrics.items():
            self.logger.info(f"   {model_name}: {model_metrics}")
        self.logger.info("=" * 60)
        self.logger.info("[SUCCESS] Pipeline Finished.")
