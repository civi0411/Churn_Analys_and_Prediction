"""
tests/conftest.py

Shared fixtures dùng chung cho toàn bộ test suite.
Bao gồm cấu hình mẫu, data fixtures, temporary directories, và model fixtures.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Fix matplotlib backend for headless testing
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== CONFIG FIXTURES ====================

@pytest.fixture(scope='session')
def test_config():
    """Cấu hình mẫu dùng cho tests: data, preprocessing, tuning, models, artifacts."""
    return {
        'data': {
            'target_col': 'Churn',
            'test_size': 0.2,
            'random_state': 42,
            'raw_path': 'data/raw/test.xlsx',
            'processed_dir': 'data/processed',
            'train_test_dir': 'data/train_test'
        },
        'preprocessing': {
            'missing_strategy': {
                'numerical': 'median',
                'categorical': 'mode'
            },
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'scaler_type': 'standard',
            'categorical_encoding': 'label',
            'use_smote': False
        },
        'tuning': {
            'method': 'randomized',
            'cv_folds': 3,
            'scoring': 'f1',
            'n_iter': 5
        },
        'models': {
            'logistic_regression': {
                'C': [0.1, 1],
                'penalty': ['l2']
            }
        },
        'artifacts': {
            'figures_dir': 'artifacts/figures',
            'models_dir': 'artifacts/models'
        },
        'explainability': {
            'enabled': False
        }
    }


# ==================== DATA FIXTURES ====================

@pytest.fixture
def sample_raw_df():
    """Sample RAW DataFrame cho testing (mô phỏng dữ liệu e-commerce)."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        'CustomerID': range(1, n_samples + 1),
        'Tenure': np.random.randint(1, 60, n_samples),
        'PreferredLoginDevice': np.random.choice(['Mobile Phone', 'Phone', 'Computer'], n_samples),
        'CityTier': np.random.randint(1, 4, n_samples),
        'WarehouseToHome': np.random.uniform(5, 50, n_samples),
        'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'CC', 'COD'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'HourSpendOnApp': np.random.uniform(0, 5, n_samples),
        'NumberOfDeviceRegistered': np.random.randint(1, 6, n_samples),
        'SatisfactionScore': np.random.randint(1, 6, n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'NumberOfAddress': np.random.randint(1, 10, n_samples),
        'Complain': np.random.choice([0, 1], n_samples),
        'OrderAmountHikeFromlastYear': np.random.uniform(10, 30, n_samples),
        'CouponUsed': np.random.randint(0, 10, n_samples),
        'OrderCount': np.random.randint(1, 20, n_samples),
        'DaySinceLastOrder': np.random.randint(0, 30, n_samples),
        'CashbackAmount': np.random.uniform(50, 300, n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })


@pytest.fixture
def sample_raw_df_with_missing(sample_raw_df):
    """Sample DataFrame có thêm missing values để test imputation."""
    df = sample_raw_df.copy()
    # Add missing values
    df.loc[0:5, 'Tenure'] = np.nan
    df.loc[10:15, 'WarehouseToHome'] = np.nan
    df.loc[20:22, 'PreferredLoginDevice'] = np.nan
    return df


@pytest.fixture
def sample_processed_df():
    """Sample processed DataFrame numeric-only dùng trong các test tính toán."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        'Tenure': np.random.randint(1, 60, n_samples),
        'CityTier': np.random.randint(1, 4, n_samples),
        'WarehouseToHome': np.random.uniform(5, 50, n_samples),
        'HourSpendOnApp': np.random.uniform(0, 5, n_samples),
        'NumberOfDeviceRegistered': np.random.randint(1, 6, n_samples),
        'SatisfactionScore': np.random.randint(1, 6, n_samples),
        'NumberOfAddress': np.random.randint(1, 10, n_samples),
        'Complain': np.random.choice([0, 1], n_samples),
        'OrderAmountHikeFromlastYear': np.random.uniform(10, 30, n_samples),
        'CouponUsed': np.random.randint(0, 10, n_samples),
        'OrderCount': np.random.randint(1, 20, n_samples),
        'DaySinceLastOrder': np.random.randint(0, 30, n_samples),
        'CashbackAmount': np.random.uniform(50, 300, n_samples),
        'PreferredLoginDevice_encoded': np.random.randint(0, 3, n_samples),
        'Gender_encoded': np.random.randint(0, 2, n_samples),
        'MaritalStatus_encoded': np.random.randint(0, 3, n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })


@pytest.fixture
def sample_train_test_split(sample_processed_df):
    """Chia sample processed data thành train/test để dùng cho các model fixtures."""
    from sklearn.model_selection import train_test_split

    target_col = 'Churn'
    X = sample_processed_df.drop(columns=[target_col])
    y = sample_processed_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==================== TEMP DIRECTORY FIXTURES ====================

@pytest.fixture
def temp_dir():
    """Tạo temp directory cho tests và xoá khi xong."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_artifacts_dir(temp_dir):
    """Tạo cấu trúc artifacts (figures, models, experiments) trong temp_dir."""
    artifacts_dir = os.path.join(temp_dir, 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'figures', 'eda'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'figures', 'evaluation'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'experiments'), exist_ok=True)
    return artifacts_dir


# ==================== MODEL FIXTURES ====================

@pytest.fixture
def trained_model(sample_train_test_split):
    """Một mô hình đơn giản đã train dùng cho test IOHandler và trainer."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = sample_train_test_split
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def mock_logger():
    """Mock logger đơn giản cho tests (logging basic)."""
    import logging
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    return logger


# ==================== FILE FIXTURES ====================

@pytest.fixture(scope='session')
def tests_data_dir():
    """Đường dẫn tới thư mục `tests/data` để load sample files khi cần."""
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='session')
def load_sample_raw_csv(tests_data_dir):
    """Load `sample_raw.csv` nếu tồn tại trong tests/data."""
    path = os.path.join(tests_data_dir, 'sample_raw.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@pytest.fixture(scope='session')
def load_sample_processed_csv(tests_data_dir):
    """Load `sample_processed.csv` nếu tồn tại trong tests/data."""
    path = os.path.join(tests_data_dir, 'sample_processed.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

