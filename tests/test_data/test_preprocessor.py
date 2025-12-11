"""
tests/test_data/test_preprocessor.py

Unit tests cho `DataPreprocessor`:
- Kiểm tra làm sạch domain values
- Kiểm tra chia train/test có stratify
- Kiểm tra load_data với file tồn tại và không tồn tại
"""

from src.data.preprocessor import DataPreprocessor


def test_clean_data_basic(sample_raw_df, mock_logger, test_config):
    """Kiểm tra `clean_data` chuẩn hoá một số giá trị domain (ví dụ 'Phone' -> 'Mobile Phone')."""
    pre = DataPreprocessor(test_config, mock_logger)
    df = sample_raw_df.copy()

    # Introduce known domain values
    df.loc[0, 'PreferredLoginDevice'] = 'Phone'
    df.loc[1, 'PreferredPaymentMode'] = 'CC'

    cleaned = pre.clean_data(df)

    assert 'PreferredLoginDevice' in cleaned.columns
    assert cleaned.loc[0, 'PreferredLoginDevice'] == 'Mobile Phone'
    assert cleaned.loc[1, 'PreferredPaymentMode'] == 'Credit Card'


def test_split_data_stratified(sample_processed_df, test_config, mock_logger):
    """Kiểm tra `split_data` chia dữ liệu có stratification theo target."""
    cfg = test_config.copy()
    cfg['data']['target_col'] = 'Churn'
    cfg['data']['test_size'] = 0.25

    pre = DataPreprocessor(cfg, mock_logger)

    train_df, test_df = pre.split_data(sample_processed_df)

    # Ensure sizes roughly match
    assert abs(len(test_df) - int(0.25 * len(sample_processed_df))) <= 2

    # Stratification: proportion of positive class should be similar
    p_all = sample_processed_df['Churn'].mean()
    p_test = test_df['Churn'].mean()
    assert abs(p_all - p_test) < 0.1


def test_load_data_success(tmp_path, test_config):
    """Kiểm tra load_data đọc file CSV thành công."""
    import pandas as pd
    # Tạo file CSV mẫu
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv_file = tmp_path / 'test.csv'
    df.to_csv(csv_file, index=False)
    pre = DataPreprocessor(test_config)
    loaded = pre.load_data(str(csv_file))
    pd.testing.assert_frame_equal(loaded, df)


def test_load_data_file_not_found(test_config):
    """Kiểm tra load_data raise FileNotFoundError khi file không tồn tại."""
    pre = DataPreprocessor(test_config)
    import pytest
    with pytest.raises(FileNotFoundError):
        pre.load_data('nonexistent_file.csv')
