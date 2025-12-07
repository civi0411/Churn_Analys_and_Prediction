"""
main.py
Diem khoi chay duy nhat cua du an (Entry Point)
"""
import argparse
import sys
# Import tu cac module con trong package src
from src.utils import ConfigLoader, Logger, set_random_seed
from src.pipeline import Pipeline

def main():
    # 1. Thiet lap tham so dong lenh
    parser = argparse.ArgumentParser(description="E-commerce Churn Prediction Pipeline")

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'preprocess', 'train', 'eda', 'visualize'],
                        help='Che do chay pipeline')

    parser.add_argument('--data', type=str, default=None,
                        help="Duong dan file data hoac thu muc batch (ghi de config)")

    parser.add_argument('--model', type=str, default='all',
                        help="Ten model cu the de train (vd: xgboost). Mac dinh la 'all'")

    parser.add_argument('--optimize', action='store_true',
                        help="Bat che do toi uu hyperparameters (Tuning)")

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help="Duong dan file cau hinh")

    args = parser.parse_args()

    # 2. Khoi tao Config va Logger
    try:
        # Load Config
        config = ConfigLoader.load_config(args.config)

        # Override Data Path neu user nhap tu dong lenh
        if args.data:
            config['data']['raw_path'] = args.data

        # Khoi tao Logger chinh (Global)
        log_dir = config.get('artifacts', {}).get('logs_dir', 'artifacts/logs')
        log_level = config.get('logging', {}).get('level', 'INFO')

        logger = Logger.get_logger(name='MAIN', log_dir=log_dir, level=log_level)

        # Thiet lap seed ngau nhien
        seed = config['data'].get('random_state', 42)
        set_random_seed(seed)

    except Exception as e:
        print(f"[INIT ERROR] {e}")
        sys.exit(1)

    # Log thong tin khoi chay
    logger.info("=" * 60)
    logger.info("START PIPELINE")
    logger.info(f"   Mode     : {args.mode.upper()}")
    logger.info(f"   Data     : {args.data if args.data else 'From Config'}")
    logger.info(f"   Model    : {args.model.upper()}")
    logger.info(f"   Optimize : {args.optimize}")
    logger.info("=" * 60)

    # 3. Chay Pipeline
    try:
        # Khoi tao Pipeline Orchestrator
        pipeline = Pipeline(config, logger)

        # Chay pipeline voi cac tham so da parse
        result = pipeline.run(
            mode=args.mode,
            model_name=args.model,
            optimize=args.optimize
        )

        # 4. Xu ly ket qua tra ve
        if args.mode in ['train', 'full'] and result:
            trainer, metrics = result

            logger.info("\n" + "=" * 60)
            logger.info("TRAINING RESULTS SUMMARY")
            logger.info("=" * 60)

            if trainer.best_model_name:
                model_name = trainer.best_model_name.upper()
                logger.info(f"   Best Model : {model_name}")

                best_metric = metrics[trainer.best_model_name]
                f1 = best_metric.get('f1', 0)
                auc = best_metric.get('roc_auc', 0)
                logger.info(f"   Scores     : F1={f1:.4f} | AUC={auc:.4f}")
            else:
                logger.warning("   No best model selected.")

        logger.info("\n[SUCCESS] Pipeline Finished Successfully.")

    except KeyboardInterrupt:
        logger.warning("\n[STOP] Pipeline interrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.critical(f"\n[FAILURE] Critical Pipeline Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()