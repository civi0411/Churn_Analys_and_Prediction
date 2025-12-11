"""
Module `pipeline` - điều phối các bước ML pipeline (EDA, Preprocessing, Training, Visualization, Inference).

Important keywords: Args, Returns, Methods, Notes
"""

import os
import shutil
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# --- NEW MODULAR IMPORTS ---
from .data import DataPreprocessor, DataTransformer
from .models import ModelTrainer
from .visualization import EDAVisualizer, EvaluateVisualizer
from .ops import (
    ExperimentTracker, ModelRegistry,
    DataVersioning, ModelMonitor, ModelExplainer, ReportGenerator
)
from .ops.dataops.validator import DataValidator
from .utils import IOHandler, set_random_seed, get_latest_train_test, get_timestamp, ensure_dir, Logger


class Pipeline:
    """
    Lớp điều phối pipeline.

    Mục đích:
    - Thiết lập các component (preprocessor, transformer, trainer, ops)
    - Điều phối các stage: EDA, preprocessing, training, visualization, predict

    Methods:
        run_eda(), run_preprocessing(), run_training(), run_visualization(), run_predict()
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Khởi tạo Pipeline với config và logger.

        Args:
            config (Dict): cấu hình dự án
            logger (logging.Logger): logger instance
        """
        self.config = config
        self.logger = logger
        self.target_col = config['data']['target_col']
        # Đảm bảo khi predict với --data khác, ta vẫn tìm đúng file cleaned train
        self.train_raw_path = config['data'].get('original_raw_path', config['data']['raw_path'])

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
        # KHỞI TẠO ĐƯỜNG DẪN LƯU TRẠNG THÁI TRANSFORMER
        self.transformer_path = os.path.join(config['mlops']['registry_dir'], 'transformer_state.joblib')

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
        """Chạy Exploratory Data Analysis trên raw data và trả về summary.

        Returns:
            dict: EDA summary (churn_rate, missing, n_rows, n_cols, duplicate, top_correlated)
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

        return summary

    # Helper method để generate report
    def _generate_report_for_run(self, mode: str, optimize: bool = False, trainer: ModelTrainer = None,
                                 metrics: Dict = None, eda_summary: Dict = None):
        """
        Tạo báo cáo tổng hợp (Markdown) cho lần chạy hiện tại, bao gồm thông tin EDA, Training và Metrics.

        Args:
            mode (str): Chế độ chạy hiện tại.
            optimize (bool, optional): Có tối ưu hóa tham số không.
            trainer (ModelTrainer, optional): Đối tượng huấn luyện chứa thông tin model tốt nhất.
            metrics (Dict, optional): Kết quả đánh giá các model.
            eda_summary (Dict, optional): Thông tin tóm tắt từ bước EDA.
        """
        try:
            feature_importance = None
            best_model_name = None

            if trainer and trainer.best_model_name:
                best_model_name = trainer.best_model_name
                feature_importance = trainer.get_feature_importance()

            report_path = self.report_generator.generate_report(
                run_id=self.tracker.current_run_id,
                mode=mode,
                optimize=optimize,
                best_model_name=best_model_name,
                all_metrics=metrics,
                feature_importance=feature_importance,
                eda_summary=eda_summary,
                config=self.config
            )

            if report_path and self.logger:
                self.logger.info(f"Report Generated | {os.path.basename(report_path)}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to generate report: {e}")

    # Helper method để lưu trạng thái DataTransformer
    def _save_transformer_state(self) -> str:
        """
        Lưu trạng thái đã fit của `DataTransformer` vào registry.

        Returns:
            str: Đường dẫn file đã lưu.
        """
        ensure_dir(os.path.dirname(self.transformer_path))
        try:
            IOHandler.save_model(self.transformer, self.transformer_path)
            self.logger.info(f"Transformer State Saved | {self.transformer_path}")
            return self.transformer_path
        except Exception as e:
            self.logger.error(f"Failed to save DataTransformer state: {e}")
            raise

    # =========================================================================
    # STAGE 2: PREPROCESSING (Clean -> Split -> Transform)
    # =========================================================================
    def run_preprocessing(self) -> Tuple[str, str, str]:
        """
        Chạy toàn bộ preprocessing pipeline:
        1. Tải và làm sạch dữ liệu thô.
        2. Kiểm tra quy tắc nghiệp vụ (Business Rules).
        3. Chia tập Train/Test.
        4. Biến đổi dữ liệu (Fit trên Train, Transform trên Test).
        5. Lưu trữ dữ liệu đã xử lý dưới dạng Parquet.
        Returns:
            Tuple[str, str, str]: Đường dẫn tới file dữ liệu đã xử lý (Full, Train, Test).
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

        # --- BUSINESS RULES VALIDATION ---
        try:
            br_conf = self.config.get('dataops', {}).get('business_rules', {})
            br_report = self.validator.validate_business_rules(df_clean)

            # Log summary if violations present
            if br_report and not br_report.get('passed', True):
                self.logger.warning("Business rules violations detected:")
                for cat in ('range_violations', 'logic_violations', 'format_violations', 'referential_violations'):
                    items = br_report.get(cat, {})
                    if items:
                        for rule, info in items.items():
                            self.logger.warning(f"  {cat}.{rule}: {info.get('count', 0)} rows (examples: {info.get('examples', [])[:3]})")

            # Persist report if enabled and run dir available
            if br_conf.get('persist', True) and self.tracker.current_run_dir:
                try:
                    br_path = os.path.join(self.tracker.current_run_dir, 'data', 'business_rules_report.json')
                    ensure_dir(os.path.dirname(br_path))
                    IOHandler.save_json(br_report, br_path)
                    self.logger.info(f"Business rules report saved: {br_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to persist business rules report: {e}")

            # Abort pipeline if configured
            if not br_report.get('passed', True) and br_conf.get('fail_on_violation', False):
                raise ValueError("Business rule violations found; aborting pipeline (fail_on_violation=True)")
        except Exception as _br_e:
            # Do not crash pipeline for non-fatal errors in validation; log and continue
            self.logger.warning(f"Business rules validation failed: {_br_e}")

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
        # <<< BỔ SUNG: LƯU TRẠNG THÁI TRANSFORMER >>>
        self._save_transformer_state()
        # <<< KẾT THÚC BỔ SUNG >>>

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
        Thực hiện quy trình huấn luyện, tối ưu hóa và đánh giá mô hình.

        Args:
            train_path (str, optional): Đường dẫn dữ liệu train.
            test_path (str, optional): Đường dẫn dữ liệu test.
            model_name (str, optional): Tên mô hình cần train ('all' hoặc tên cụ thể). Defaults to 'all'.
            optimize (bool, optional): Có thực hiện tìm kiếm tham số (Hyperparameter Tuning) không. Defaults to True.

        Returns:
            Tuple[ModelTrainer, Dict]: Đối tượng Trainer và Dictionary chứa metrics của các mô hình.
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
        Tạo các biểu đồ đánh giá chuyên sâu (ROC, Confusion Matrix, Feature Importance) và giải thích mô hình (SHAP).

        Args:
            trainer (ModelTrainer): Đối tượng trainer chứa mô hình đã huấn luyện.
            metrics (Dict): Kết quả đánh giá mô hình.
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
            optimize: bool = True, **kwargs) -> Any:
        """
        Điểm điều phối chính (Main Entry Point) để thực thi pipeline theo chế độ được yêu cầu.

        Args:
            mode (str, optional): Chế độ chạy ('full', 'eda', 'preprocess', 'train', 'visualize', 'predict'). Defaults to 'full'.
            model_name (str, optional): Tên mô hình cụ thể. Defaults to 'all'.
            optimize (bool, optional): Có tối ưu tham số không. Defaults to True.
            **kwargs: Các tham số phụ (vd: input_path cho predict).

        Returns:
            Any: Kết quả trả về tùy thuộc vào mode (thường là trainer, metrics hoặc đường dẫn file dự đoán).
        """
        # 1. Start Tracking
        run_name = f"{get_timestamp()}_{mode.upper()}"
        run_id = self.tracker.start_run(run_name)

        # === Ensure run-specific logging is attached BEFORE we write the header ===
        if self.tracker.current_run_dir:
            # Snapshot config early so run folder exists and config saved before logs
            config_snapshot_path = os.path.join(self.tracker.current_run_dir, "config_snapshot.yaml")
            IOHandler.save_yaml(self.config, config_snapshot_path)

            # Redirect Logger to Run Folder (add run handler immediately)
            run_log_filename = f"{mode.lower()}.log"
            run_log_path = os.path.join(self.tracker.current_run_dir, run_log_filename)
            Logger.get_logger('MAIN', run_log_path=run_log_path)
            # Note: self.logger references the same Logger instance (by name) so handlers are updated in-place

        # Use single log record for header to avoid duplicate separator-only lines
        self._log_header(f"PIPELINE STARTED  Mode: {mode.upper()}  Run ID: {run_id}")

        # 2. Setup Run-specific Logging & Config Snapshot (remaining tasks handled above)
        if self.tracker.current_run_dir:
            self.logger.info("\u2713 Run-specific log initialized")

        # Log params
        self.tracker.log_params({
            'mode': mode, 'model_name': model_name, 'optimize': optimize,
            'test_size': self.config['data']['test_size']
        })

        try:
            # ========== MODE: EDA ==========
            if mode == 'eda':
                eda_summary = self.run_eda()
                self._generate_report_for_run(mode, optimize=False, eda_summary=eda_summary)
                self.tracker.end_run("FINISHED")
                return None

            # ========== MODE: PREPROCESS ==========
            elif mode == 'preprocess':
                processed_path, train_path, test_path = self.run_preprocessing()
                if self.tracker.current_run_dir:
                    data_dir = os.path.join(self.tracker.current_run_dir, 'data')
                    ensure_dir(data_dir)
                    shutil.copy2(processed_path, os.path.join(data_dir, os.path.basename(processed_path)))
                    shutil.copy2(train_path, os.path.join(data_dir, os.path.basename(train_path)))
                    shutil.copy2(test_path, os.path.join(data_dir, os.path.basename(test_path)))
                self._generate_report_for_run(mode, optimize=False)
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
                self._save_training_artifacts(trainer)
                self.run_visualization(trainer, metrics)
                self._generate_report_for_run(mode, optimize, trainer, metrics)
                if trainer.best_model_name:
                    self.tracker.log_metrics(metrics[trainer.best_model_name])
                self.tracker.end_run("FINISHED")
                return trainer, metrics

            # ========== MODE: VISUALIZE ==========
            elif mode == 'visualize':
                train_test_dir = self.config['data']['train_test_dir']
                raw_filename = self.config['data']['raw_path']
                train_path, test_path = get_latest_train_test(train_test_dir, raw_filename)
                self.logger.info("Visualize mode: Running quick training (no-opt)...")
                trainer, metrics = self.run_training(train_path, test_path, model_name, optimize=False)
                self.run_visualization(trainer, metrics)
                self._save_training_artifacts(trainer)
                self._generate_report_for_run(mode, False, trainer, metrics)
                self.tracker.end_run("FINISHED")
                return None

            # ========== MODE: FULL ==========
            elif mode == 'full':
                # 1. EDA
                eda_summary = self.run_eda()

                # 2. Preprocess
                processed_path, train_path, test_path = self.run_preprocessing()
                if self.tracker.current_run_dir:
                    data_dir = os.path.join(self.tracker.current_run_dir, 'data')
                    ensure_dir(data_dir)
                    shutil.copy2(train_path, os.path.join(data_dir, os.path.basename(train_path)))
                    shutil.copy2(test_path, os.path.join(data_dir, os.path.basename(test_path)))

                # 3. Train
                trainer, metrics = self.run_training(train_path, test_path, model_name, optimize)

                # 4. Visualize
                self.run_visualization(trainer, metrics)

                # 5. Save & Log
                self._save_training_artifacts(trainer)
                if trainer.best_model_name:
                    self.tracker.log_metrics(metrics[trainer.best_model_name])

                # 6. Generate Report
                self._generate_report_for_run(mode, optimize, trainer, metrics, eda_summary)

                # 7. Final Summary
                self._log_training_summary(trainer, metrics)
                self.tracker.end_run("FINISHED")
                return trainer, metrics

            # ========== MODE: PREDICT ==========
            elif mode == 'predict':
                input_path = kwargs.get('input_path', self.config['data']['raw_path'])
                model_path = kwargs.get('model_path', None)
                output_path = kwargs.get('output_path', None)
                pred_path = self.run_predict(input_path=input_path, model_path=model_path, output_path=output_path)
                self.tracker.end_run("FINISHED")
                return pred_path

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            self.logger.error(f"Pipeline Failed: {e}", exc_info=True)
            self.tracker.end_run("FAILED")
            raise

    def _save_training_artifacts(self, trainer):
        """Save model artifacts for the current run (placeholder)."""
        # Example: Save best model, params, etc. to run dir
        if self.tracker.current_run_dir and trainer.best_model_name:
            model_dir = os.path.join(self.tracker.current_run_dir, 'models')
            ensure_dir(model_dir)
            model_path = os.path.join(model_dir, f"{trainer.best_model_name}.joblib")
            try:
                import joblib
                joblib.dump(trainer.best_model, model_path)
                self.logger.info(f"Saved model artifact: {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save model artifact: {e}")

    def _log_training_summary(self, trainer, metrics):
        """Log training summary for the current run (placeholder)."""
        if trainer.best_model_name:
            self.logger.info(f"Best Model: {trainer.best_model_name}")
            self.logger.info(f"Metrics: {metrics.get(trainer.best_model_name, {})}")

    def run_predict(self, input_path: str = None, model_path: str = None, output_path: str = None, transformer_path: str = None) -> Optional[str]:
        """
        Thực hiện dự đoán trên dữ liệu mới, bao gồm kiểm tra Data Drift và áp dụng các bước biến đổi dữ liệu tương tự lúc huấn luyện.

        Args:
            input_path (str, optional): Đường dẫn file dữ liệu cần dự đoán.
            model_path (str, optional): Đường dẫn file mô hình đã lưu. Nếu None, lấy model mới nhất từ registry.
            output_path (str, optional): Đường dẫn lưu kết quả dự đoán.
            transformer_path (str, optional): Đường dẫn file transformer state.

        Returns:
            Optional[str]: Đường dẫn tới file kết quả dự đoán (hoặc None nếu bị hủy do lỗi Drift nghiêm trọng).
        """
        self.logger.info("=" * 70)
        self.logger.info("STAGE: PREDICT")

        # Default transformer state path (may be used if you want to load a saved transformer)
        if transformer_path is None:
            transformer_path = os.path.join(self.config['mlops']['registry_dir'], 'transformer_state.joblib')

        # If a saved transformer state exists, load it so transform() applies the same encoders/scaler
        try:
            if transformer_path and os.path.exists(transformer_path):
                try:
                    loaded_transformer = IOHandler.load_model(transformer_path)
                    # Replace the in-memory transformer with the saved fitted one
                    self.transformer = loaded_transformer
                    self.logger.info(f"Loaded transformer state from {transformer_path}")
                except Exception as _e:
                    self.logger.warning(f"Failed to load transformer state ({transformer_path}): {_e}. Using current transformer instance.")
        except Exception:
            # Defensive: if any path-handling weirdness occurs, continue with in-memory transformer
            pass

        # Resolve model_path: use provided path, otherwise try to find latest model in registry
        if model_path is None:
            try:
                models_info = self.registry.list_models()
                latest_info = None
                for name, infos in models_info.items():
                    for info in infos:
                        if latest_info is None or info.get('timestamp', '') > latest_info.get('timestamp', ''):
                            latest_info = info
                if latest_info is None:
                    raise FileNotFoundError("No model found in registry")
                model_path = latest_info['path']
            except Exception as e:
                self.logger.error(f"Model file not found: {e}")
                raise FileNotFoundError(f"Model file not found: {e}")

        # 3. Load
        df = self.preprocessor.load_data(input_path)

        # 1. CLEAN DATA TRƯỚC (giống như training)
        df_clean = self.preprocessor.clean_data(df)

        # 4. Business Validation
        br_report = self.validator.validate_business_rules(df_clean)
        if not br_report['passed']:
            self.logger.warning("Business rule violations detected")

        # 5. DRIFT CHECK
        drift_config = self.config.get('dataops', {}).get('drift_detection', {})
        if drift_config.get('enabled', True):
            from .ops.dataops.drift_detector import DataDriftDetector
            processed_dir = self.config['data']['processed_dir']
            train_raw_base = os.path.splitext(os.path.basename(self.train_raw_path))[0]
            cleaned_train_path = os.path.join(processed_dir, f"{train_raw_base}_cleaned.parquet")
            cleaned_train_path = os.path.normpath(cleaned_train_path)
            if not os.path.exists(cleaned_train_path):
                self.logger.error(f"Cleaned train data file not found for drift check: {cleaned_train_path}")
                raise FileNotFoundError(f"Cleaned train data file not found: {cleaned_train_path}")
            df_train_cleaned = IOHandler.read_data(cleaned_train_path)
            # Chuẩn hóa kiểu dữ liệu trước khi drift check
            df_train_cleaned = self._prepare_drift_data(df_train_cleaned)
            df_clean_drift = self._prepare_drift_data(df_clean)
            detector = DataDriftDetector(df_train_cleaned, self.logger)
            report = detector.detect_drift(
                df_clean_drift,
                threshold=drift_config.get('pvalue_threshold', 0.05)
            )
            # Do not persist drift report JSON under monitoring; only emit a concise log and rely on alerts_log.csv
            try:
                # Log a concise summary for observability (full report kept in-memory by caller if needed)
                summary = report.get('summary', {})
                self.logger.info(f"Drift report summary: severity={summary.get('drift_severity')} drifted_features={summary.get('drifted_features')}")
            except Exception:
                self.logger.debug("Failed to log drift report summary")

            severity = report['summary'].get('drift_severity')
            if severity and severity != 'NONE':
                msg = f"Drift severity={severity}; drifted_features={report['summary'].get('drifted_features', 0)}"
                model_name_for_alert = getattr(self.trainer, 'best_model_name', None) or 'prediction'
                try:
                    created = self.monitor.create_alert(
                        model_name=model_name_for_alert,
                        alert_type='data_drift',
                        message=msg,
                        severity=severity
                    )
                    if created:
                        self.logger.warning(f"Drift alert created: {severity}")
                    else:
                        self.logger.info("Drift alert already existed or was skipped")
                except Exception as _e:
                    self.logger.warning(f"Failed to create drift alert: {_e}")

            if severity == 'CRITICAL' and drift_config.get('abort_on_critical', True):
                self.logger.error("CRITICAL DRIFT DETECTED - Aborting prediction")
                return None
            # end drift handling

        # 6. Transform & Predict
        X, _ = self.transformer.transform(df_clean)

        # If transform left any object dtype columns, try to encode them using saved encoders;
        # fallback: convert to pandas.Categorical so xgboost accepts them (requires xgboost >= support)
        try:
            obj_cols = [c for c in X.columns if X[c].dtype == 'object']
            if obj_cols:
                # Try to use transformer encoders if available
                try:
                    encoders = getattr(self.transformer, '_learned_params', {}).get('encoders', {})
                    for col in list(obj_cols):
                        if col in encoders:
                            le = encoders[col]
                            # Map using encoder classes; unseen -> -1
                            try:
                                X[col] = X[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
                                # ensure numeric dtype
                                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1).astype(int)
                            except Exception:
                                # fallback: attempt mapping via categories
                                X[col] = pd.Categorical(X[col].astype(str)).codes
                            # remove from obj_cols list
                            obj_cols.remove(col)
                except Exception:
                    self.logger.debug("Encoders not available or failed; will fallback to categorical codes")

                # For any remaining object cols, convert to categorical codes (int)
                for col in obj_cols:
                    try:
                        X[col] = pd.Categorical(X[col]).codes
                    except Exception:
                        X[col] = X[col].astype(str).fillna('').astype('category').cat.codes
        except Exception as _e:
            self.logger.warning(f"Post-transform dtype adjustment failed: {_e}")

        model = IOHandler.load_model(model_path)
        y_pred = model.predict(X)

        # 7. Save
        result_df = df_clean.copy()
        result_df['prediction'] = y_pred
        if output_path is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join('artifacts', 'predictions', f'{base}_predicted.csv')
        ensure_dir(os.path.dirname(output_path))
        result_df.to_csv(output_path, index=False)
        self.logger.info(f"Prediction saved: {output_path}")
        return output_path

    def _prepare_drift_data(self, df):
        """
        Chuẩn hóa kiểu dữ liệu (ép kiểu số, xử lý object) để đảm bảo tính toán Drift chính xác.

        Args:
            df (pd.DataFrame): DataFrame cần chuẩn hóa.

        Returns:
            pd.DataFrame: DataFrame đã được xử lý kiểu dữ liệu.
        """
        import numpy as np
        import pandas as pd
        df = df.copy()
        for col in df.columns:
            # Nếu là object nhưng toàn số, ép về float
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception:
                    pass  # Nếu không chuyển được thì giữ nguyên
            # Nếu là numeric nhưng có giá trị không hợp lệ, log warning
            elif np.issubdtype(df[col].dtype, np.number):
                try:
                    _ = pd.to_numeric(df[col], errors='raise')
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[Drift] Numeric column '{col}' contains non-numeric values: {e}. Will skip this column in drift check.")
                    df.drop(columns=[col], inplace=True)
        return df

    def _log_header(self, title: str):
        """
        Ghi log header cho một run, bao gồm các separator và tiêu đề.
        """
        self.logger.info("=" * 70)
        self.logger.info(title)
        self.logger.info("=" * 70)

