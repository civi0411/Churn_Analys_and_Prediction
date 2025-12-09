from src.ops.report.generator import ReportGenerator
import pandas as pd
import os

g = ReportGenerator(experiments_base_dir='artifacts/experiments', reports_dir='artifacts/reports')
all_metrics = {
    'xgboost': {'accuracy': 0.9787, 'precision': 0.9415, 'recall': 0.9316, 'f1': 0.9365, 'roc_auc': 0.9962},
    'random_forest': {'accuracy': 0.9654, 'precision': 0.8832, 'recall': 0.9158, 'f1': 0.8992, 'roc_auc': 0.9923}
}
feature_importance = pd.DataFrame({'feature':['Tenure','Complain','NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore'],'importance':[0.2275,0.1215,0.0790,0.0659,0.0551]})

path = g.generate_training_report(run_id='test_run_tmp', best_model_name='xgboost', all_metrics=all_metrics, feature_importance=feature_importance, config={'a':1}, eda_summary={'churn_rate':0.168,'top_correlated':['Tenure','Complain']}, mode='full')
print('Report path:', path)
print('--- Report head ---')
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i > 50:
            break

