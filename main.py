"""
main.py
Entry Point - Điểm khởi chạy duy nhất của dự án

Usage:
    python main.py                                    # Full pipeline
    python main.py --mode eda                         # Chỉ chạy EDA
    python main.py --mode train --optimize            # Train với tuning
    python main.py --data file.csv                    # File cụ thể
"""
import argparse
import sys
import os
from src.utils import ConfigLoader, Logger, set_random_seed
from src.pipeline import Pipeline


def main():
    """
    Entry point chính của pipeline.

    Workflow:
        1. Parse CLI arguments
        2. Load config và setup logger
        3. Chạy Pipeline theo mode
        4. Log kết quả

    Returns:
        None. Exit code 0 nếu thành công, 1 nếu lỗi.
    """
    # 1. Thiết lập tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description="E-commerce Churn Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                      # Full pipeline
  python main.py --mode eda                           # EDA only
  python main.py --mode train --optimize              # Train + tuning

  # Data selection
  python main.py --data path/to/file.csv              # Single file (absolute or relative to project)
  python main.py --data data/raw/file.csv             # Relative to project
        """
    )

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'preprocess', 'train', 'eda', 'visualize'],
                        help='Chế độ chạy pipeline (default: full)')

    parser.add_argument('--data', type=str, default=None,
                        help="Đường dẫn file dữ liệu (absolute hoặc relative to project). Nếu không cung cấp sẽ dùng config")

    parser.add_argument('--model', type=str, default='all',
                        help="Tên model cụ thể (default: all)")

    parser.add_argument('--optimize', action='store_true',
                        help="Bật hyperparameter tuning")

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help="Đường dẫn file config")

    args = parser.parse_args()

    # 2. Khởi tạo Config và Logger
    try:
        config = ConfigLoader.load_config(args.config)

        # Handle --data: accept a single file path (absolute or relative to project root or data/)
        if args.data:
            # Absolute path
            if os.path.isabs(args.data) and os.path.isfile(args.data):
                config['data']['raw_path'] = args.data
            else:
                # Try relative to project data/ folder
                rel = os.path.join('data', args.data)
                if os.path.isfile(rel):
                    config['data']['raw_path'] = rel
                elif os.path.isfile(args.data):
                    config['data']['raw_path'] = args.data
                else:
                    print(f"[ERROR] Data file not found: {args.data}")
                    sys.exit(1)

        log_dir = config.get('artifacts', {}).get('logs_dir', 'artifacts/logs')
        log_level = config.get('logging', {}).get('level', 'INFO')
        logger = Logger.get_logger(name='MAIN', log_dir=log_dir, level=log_level)

        seed = config['data'].get('random_state', 42)
        set_random_seed(seed)

    except Exception as e:
        print(f"[INIT ERROR] {e}")
        sys.exit(1)

    # Log thông tin khởi chạy
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"   Mode     : {args.mode.upper()}")

    data_path = config['data'].get('raw_path')
    logger.info(f"   Data     : {data_path}")
    logger.info(f"   Model    : {args.model.upper()}")
    logger.info(f"   Optimize : {args.optimize}")
    logger.info("=" * 60)

    # 3. Chạy Pipeline
    try:
        pipeline = Pipeline(config, logger)

        result = pipeline.run(
            mode=args.mode,
            model_name=args.model,
            optimize=args.optimize
        )

        if args.mode in ['train', 'full'] and result:
            trainer, metrics = result

            logger.info("\n" + "=" * 60)
            logger.info("TRAINING RESULTS")
            logger.info("=" * 60)

            if trainer.best_model_name:
                best_metric = metrics[trainer.best_model_name]
                logger.info(f"   Best Model : {trainer.best_model_name.upper()}")
                logger.info(f"   F1 Score   : {best_metric.get('f1', 0):.4f}")
                logger.info(f"   ROC-AUC    : {best_metric.get('roc_auc', 0):.4f}")
                logger.info(f"   Accuracy   : {best_metric.get('accuracy', 0):.4f}")
            else:
                logger.warning("   No best model selected.")

        logger.info("\n" + "=" * 60)
        logger.info("[SUCCESS] Pipeline Completed!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\n[STOP] Pipeline interrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.critical(f"\n[FAILURE] Pipeline Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()