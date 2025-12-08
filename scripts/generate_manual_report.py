import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to sys.path so we can import package modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ops.reporting import ReportGenerator

# Find latest TRAIN run in artifacts/experiments
exp_base = os.path.join('artifacts', 'experiments')
if not os.path.isdir(exp_base):
    raise SystemExit('No experiments directory found: artifacts/experiments')

runs = [d for d in os.listdir(exp_base) if os.path.isdir(os.path.join(exp_base, d)) and d.endswith('_TRAIN')]
if not runs:
    raise SystemExit('No TRAIN runs found in artifacts/experiments')
# pick latest by name (timestamp prefix)
runs_sorted = sorted(runs)
run_id = runs_sorted[-1]
run_dir = os.path.join(exp_base, run_id)
print('Using run:', run_id)

# Provided metrics
best_model = 'xgboost'
metrics = {
    'xgboost': {
        'accuracy': 0.9787,
        'f1': 0.9365,
        'precision': 0.9415,
        'recall': 0.9316,
        'roc_auc': 0.9962,
    }
}
# Feature importance provided
fi = {
    'Tenure': 0.2275,
    'Complain': 0.1215,
    'NumberOfDeviceRegistered': 0.0790,
    'PreferedOrderCat': 0.0659,
    'SatisfactionScore': 0.0551
}
fi_df = pd.DataFrame([{'feature':k,'importance':v} for k,v in fi.items()])

# EDA summary to include
eda_summary = {
    'churn_rate': 0.168,
    'resampling': 'SMOTE + Tomek (SMOTETomek)',
    'top_correlated': ['Tenure', 'Complain']
}

rg = ReportGenerator(logger=None, experiments_base_dir=exp_base)
out = rg.generate_training_report(run_id=run_id, best_model_name=best_model, all_metrics=metrics, feature_importance=fi_df, config=None, eda_summary=eda_summary, format='markdown')
print('Report created:', out)
