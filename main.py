"""
main.py
Entry Point - Điểm khởi chạy duy nhất của dự án

Usage:
    python main.py                                    # Full pipeline
    python main.py --mode eda                         # Chỉ chạy EDA
    python main.py --mode train --optimize            # Train với tuning
    python main.py --data data/raw/new.csv            # File data khác
    python main.py --batch data/raw/monthly           # Batch tất cả files
    python main.py --batch data/raw/monthly --from 2024-01 --to 2024-06  # Batch theo thời gian
"""
import argparse
import sys
import os
from src.utils import ConfigLoader, Logger, set_random_seed, filter_files_by_date
from src.pipeline import Pipeline



def main():
    """
    Entry point chính của pipeline.

    Workflow:
        1. Parse CLI arguments
        2. Load config và setup logger
        3. Xử lý batch mode nếu có
        4. Chạy Pipeline theo mode
        5. Log kết quả

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
  python main.py --batch data/raw/monthly             # Batch all files
  python main.py --batch data/raw/monthly --from 2024-01 --to 2024-06
  python main.py --batch data/raw/quarterly --from 2024-Q1 --to 2024-Q2
        """
    )

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'preprocess', 'train', 'eda', 'visualize'],
                        help='Chế độ chạy pipeline (default: full)')

    parser.add_argument('--data', type=str, default=None,
                        help="Đường dẫn file data (ghi đè config)")

    parser.add_argument('--batch', type=str, default=None,
                        help="Thư mục batch data (load nhiều files)")

    parser.add_argument('--from', dest='from_date', type=str, default=None,
                        help="Batch từ ngày (format: YYYY-MM hoặc YYYY-Q#)")

    parser.add_argument('--to', dest='to_date', type=str, default=None,
                        help="Batch đến ngày (format: YYYY-MM hoặc YYYY-Q#)")

    parser.add_argument('--model', type=str, default='all',
                        help="Tên model cụ thể (vd: xgboost). Mặc định: all")

    parser.add_argument('--optimize', action='store_true',
                        help="Bật hyperparameter tuning")

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help="Đường dẫn file config")

    args = parser.parse_args()

    # 2. Khởi tạo Config và Logger
    try:
        config = ConfigLoader.load_config(args.config)

        if args.data:
            config['data']['raw_path'] = args.data

        # Batch mode với filter thời gian
        batch_files = None
        if args.batch:
            if not os.path.isdir(args.batch):
                print(f"[ERROR] Batch folder not found: {args.batch}")
                sys.exit(1)

            batch_files = filter_files_by_date(args.batch, args.from_date, args.to_date)

            if not batch_files:
                print(f"[ERROR] No files found in {args.batch}")
                if args.from_date or args.to_date:
                    print(f"        Filter: from={args.from_date}, to={args.to_date}")
                sys.exit(1)

            config['data']['raw_path'] = 'full'
            config['data']['batch_folder'] = args.batch
            config['data']['batch_files'] = batch_files

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

    if args.batch:
        logger.info(f"   Data     : BATCH MODE")
        logger.info(f"   Folder   : {args.batch}")
        if args.from_date or args.to_date:
            logger.info(f"   Period   : {args.from_date or 'start'} -> {args.to_date or 'end'}")
        logger.info(f"   Files    : {len(batch_files)}")
        for f in batch_files[:5]:
            logger.info(f"              - {os.path.basename(f)}")
        if len(batch_files) > 5:
            logger.info(f"              ... and {len(batch_files) - 5} more")
    elif args.data:
        logger.info(f"   Data     : {args.data}")
    else:
        logger.info(f"   Data     : From Config")

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