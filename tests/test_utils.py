"""
tests/test_utils.py
Tests for src/utils.py
"""
import pytest
import pandas as pd
import numpy as np
import os
import yaml

from src.utils import (
    ensure_dir,
    set_random_seed,
    get_timestamp,
    extract_base_filename,
    build_processed_filename,
    build_split_filenames,
    get_latest_train_test,
    IOHandler,
    ConfigLoader,
    compute_file_hash
)


class TestHelperFunctions:
    """Test cases for helper functions"""

    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test ensure_dir tạo directory"""
        new_dir = os.path.join(temp_dir, 'new_folder')

        ensure_dir(new_dir)

        assert os.path.exists(new_dir)

    def test_ensure_dir_handles_existing(self, temp_dir):
        """Test ensure_dir không lỗi với dir đã tồn tại"""
        ensure_dir(temp_dir)  # Already exists

        assert os.path.exists(temp_dir)

    def test_ensure_dir_handles_empty_path(self):
        """Test ensure_dir xử lý empty path"""
        # Should not raise error
        ensure_dir('')
        ensure_dir(None)

    def test_set_random_seed(self):
        """Test set_random_seed đặt seed"""
        set_random_seed(42)

        result1 = np.random.rand()

        set_random_seed(42)
        result2 = np.random.rand()

        assert result1 == result2

    def test_get_timestamp_format(self):
        """Test get_timestamp trả về đúng format"""
        ts = get_timestamp()

        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert '_' in ts

    def test_extract_base_filename(self):
        """Test extract_base_filename"""
        assert extract_base_filename('data/raw/test.xlsx') == 'test'
        assert extract_base_filename('test.csv') == 'test'
        assert extract_base_filename('/path/to/my_data.parquet') == 'my_data'

    def test_build_processed_filename(self, temp_dir):
        """Test build_processed_filename"""
        result = build_processed_filename('data/raw/test.xlsx', temp_dir)

        assert result.endswith('test_processed.csv')
        assert temp_dir in result

    def test_build_split_filenames(self, temp_dir):
        """Test build_split_filenames"""
        train_path, test_path = build_split_filenames('data/raw/test.xlsx', temp_dir)

        assert 'train.parquet' in train_path
        assert 'test.parquet' in test_path

    def test_get_latest_train_test(self, temp_dir):
        """Test get_latest_train_test"""
        # Create train/test files
        train_file = os.path.join(temp_dir, 'data_train.parquet')
        test_file = os.path.join(temp_dir, 'data_test.parquet')

        pd.DataFrame({'a': [1, 2]}).to_parquet(train_file)
        pd.DataFrame({'a': [3, 4]}).to_parquet(test_file)

        train_path, test_path = get_latest_train_test(temp_dir, 'data')

        assert train_path == train_file
        assert test_path == test_file

    def test_get_latest_train_test_raises_for_missing_dir(self):
        """Test get_latest_train_test raises error cho dir không tồn tại"""
        with pytest.raises(FileNotFoundError):
            get_latest_train_test('/nonexistent/dir')

    def test_compute_file_hash(self, temp_dir):
        """Test compute_file_hash"""
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length


class TestIOHandler:
    """Test cases for IOHandler class"""

    def test_read_csv(self, temp_dir):
        """Test read_data với CSV"""
        csv_file = os.path.join(temp_dir, 'test.csv')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_csv(csv_file, index=False)

        result = IOHandler.read_data(csv_file)

        pd.testing.assert_frame_equal(result, df)

    def test_read_parquet(self, temp_dir):
        """Test read_data với Parquet"""
        parquet_file = os.path.join(temp_dir, 'test.parquet')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_parquet(parquet_file)

        result = IOHandler.read_data(parquet_file)

        pd.testing.assert_frame_equal(result, df)

    def test_read_excel(self, temp_dir):
        """Test read_data với Excel"""
        excel_file = os.path.join(temp_dir, 'test.xlsx')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_excel(excel_file, index=False)

        result = IOHandler.read_data(excel_file)

        pd.testing.assert_frame_equal(result, df)

    def test_save_data_csv(self, temp_dir):
        """Test save_data với CSV"""
        csv_file = os.path.join(temp_dir, 'test.csv')
        df = pd.DataFrame({'a': [1, 2, 3]})

        IOHandler.save_data(df, csv_file)

        assert os.path.exists(csv_file)

    def test_save_data_parquet(self, temp_dir):
        """Test save_data với Parquet"""
        parquet_file = os.path.join(temp_dir, 'test.parquet')
        df = pd.DataFrame({'a': [1, 2, 3]})

        IOHandler.save_data(df, parquet_file)

        assert os.path.exists(parquet_file)

    def test_save_and_load_model_joblib(self, temp_dir, trained_model):
        """Test save_model và load_model với joblib"""
        model_file = os.path.join(temp_dir, 'model.joblib')

        IOHandler.save_model(trained_model, model_file)
        loaded = IOHandler.load_model(model_file)

        assert loaded is not None
        assert hasattr(loaded, 'predict')

    def test_save_and_load_json(self, temp_dir):
        """Test save_json và load_json"""
        json_file = os.path.join(temp_dir, 'test.json')
        data = {'key1': 'value1', 'key2': [1, 2, 3]}

        IOHandler.save_json(data, json_file)
        loaded = IOHandler.load_json(json_file)

        assert loaded == data

    def test_load_config_yaml(self, temp_dir):
        """Test load_config với YAML"""
        yaml_file = os.path.join(temp_dir, 'config.yaml')
        config = {'param1': 'value1', 'param2': {'nested': 'value'}}

        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)

        loaded = ConfigLoader.load_config(yaml_file)

        assert loaded == config

    def test_read_unsupported_format_raises_error(self, temp_dir):
        """Test read_data raises error cho format không hỗ trợ"""
        unsupported_file = os.path.join(temp_dir, 'test.xyz')
        with open(unsupported_file, 'w') as f:
            f.write('test')

        with pytest.raises(ValueError, match="Unsupported"):
            IOHandler.read_data(unsupported_file)

