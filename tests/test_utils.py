"""
tests/test_utils.py

Các unit test cho `src/utils.py`.
Kiểm tra các helper: files, IO, config, timestamp, hash.
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
    """Các test cho helper functions (ensure_dir, timestamp, filename helpers, hash)."""

    def test_ensure_dir_creates_directory(self, temp_dir):
        """Kiểm tra `ensure_dir` tạo thư mục mới."""
        new_dir = os.path.join(temp_dir, 'new_folder')

        ensure_dir(new_dir)

        assert os.path.exists(new_dir)

    def test_ensure_dir_handles_existing(self, temp_dir):
        """Kiểm tra `ensure_dir` không lỗi khi thư mục đã tồn tại."""
        ensure_dir(temp_dir)  # Already exists

        assert os.path.exists(temp_dir)

    def test_ensure_dir_handles_empty_path(self):
        """Kiểm tra `ensure_dir` xử lý đường dẫn rỗng/None an toàn."""
        # Should not raise error
        ensure_dir('')
        ensure_dir(None)

    def test_set_random_seed(self):
        """Kiểm tra `set_random_seed` thiết lập seed thống nhất cho các random generator."""
        set_random_seed(42)

        result1 = np.random.rand()

        set_random_seed(42)
        result2 = np.random.rand()

        assert result1 == result2

    def test_get_timestamp_format(self):
        """Kiểm tra `get_timestamp` trả về chuỗi theo định dạng YYYYMMDD_HHMMSS."""
        ts = get_timestamp()

        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert '_' in ts

    def test_extract_base_filename(self):
        """Kiểm tra `extract_base_filename` trích tên base từ path."""
        assert extract_base_filename('data/raw/test.xlsx') == 'test'
        assert extract_base_filename('test.csv') == 'test'
        assert extract_base_filename('/path/to/my_data.parquet') == 'my_data'

    def test_build_processed_filename(self, temp_dir):
        """Kiểm tra `build_processed_filename` trả về tên file processed trong thư mục đích."""
        result = build_processed_filename('data/raw/test.xlsx', temp_dir)

        assert result.endswith('test_processed.csv')
        assert temp_dir in result

    def test_build_split_filenames(self, temp_dir):
        """Kiểm tra `build_split_filenames` sinh đường dẫn train/test."""
        train_path, test_path = build_split_filenames('data/raw/test.xlsx', temp_dir)

        assert 'train.parquet' in train_path
        assert 'test.parquet' in test_path

    def test_get_latest_train_test(self, temp_dir):
        """Kiểm tra `get_latest_train_test` trả về file train/test mới nhất."""
        # Create train/test files
        train_file = os.path.join(temp_dir, 'data_train.parquet')
        test_file = os.path.join(temp_dir, 'data_test.parquet')

        pd.DataFrame({'a': [1, 2]}).to_parquet(train_file)
        pd.DataFrame({'a': [3, 4]}).to_parquet(test_file)

        train_path, test_path = get_latest_train_test(temp_dir, 'data')

        assert train_path == train_file
        assert test_path == test_file

    def test_get_latest_train_test_raises_for_missing_dir(self):
        """Kiểm tra `get_latest_train_test` raise FileNotFoundError khi thư mục không tồn tại."""
        with pytest.raises(FileNotFoundError):
            get_latest_train_test('/nonexistent/dir')

    def test_compute_file_hash(self, temp_dir):
        """Kiểm tra `compute_file_hash` trả về MD5 cố định cho cùng file."""
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length


class TestIOHandler:
    """Các test cho IOHandler (đọc/ghi CSV/Parquet/Excel, model, JSON, config)."""

    def test_read_csv(self, temp_dir):
        """Kiểm tra `IOHandler.read_data` đọc file CSV đúng."""
        csv_file = os.path.join(temp_dir, 'test.csv')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_csv(csv_file, index=False)

        result = IOHandler.read_data(csv_file)

        pd.testing.assert_frame_equal(result, df)

    def test_read_parquet(self, temp_dir):
        """Kiểm tra `IOHandler.read_data` đọc file Parquet đúng."""
        parquet_file = os.path.join(temp_dir, 'test.parquet')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_parquet(parquet_file)

        result = IOHandler.read_data(parquet_file)

        pd.testing.assert_frame_equal(result, df)

    def test_read_excel(self, temp_dir):
        """Kiểm tra `IOHandler.read_data` đọc file Excel đúng."""
        excel_file = os.path.join(temp_dir, 'test.xlsx')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_excel(excel_file, index=False)

        result = IOHandler.read_data(excel_file)

        pd.testing.assert_frame_equal(result, df)

    def test_save_data_csv(self, temp_dir):
        """Kiểm tra `IOHandler.save_data` ghi CSV thành công."""
        csv_file = os.path.join(temp_dir, 'test.csv')
        df = pd.DataFrame({'a': [1, 2, 3]})

        IOHandler.save_data(df, csv_file)

        assert os.path.exists(csv_file)

    def test_save_data_parquet(self, temp_dir):
        """Kiểm tra `IOHandler.save_data` ghi Parquet thành công."""
        parquet_file = os.path.join(temp_dir, 'test.parquet')
        df = pd.DataFrame({'a': [1, 2, 3]})

        IOHandler.save_data(df, parquet_file)

        assert os.path.exists(parquet_file)

    def test_save_and_load_model_joblib(self, temp_dir, trained_model):
        """Kiểm tra save/load model bằng joblib."""
        model_file = os.path.join(temp_dir, 'model.joblib')

        IOHandler.save_model(trained_model, model_file)
        loaded = IOHandler.load_model(model_file)

        assert loaded is not None
        assert hasattr(loaded, 'predict')

    def test_save_and_load_json(self, temp_dir):
        """Kiểm tra save/load JSON bằng IOHandler."""
        json_file = os.path.join(temp_dir, 'test.json')
        data = {'key1': 'value1', 'key2': [1, 2, 3]}

        IOHandler.save_json(data, json_file)
        loaded = IOHandler.load_json(json_file)

        assert loaded == data

    def test_load_config_yaml(self, temp_dir):
        """Kiểm tra `ConfigLoader.load_config` load YAML đúng."""
        yaml_file = os.path.join(temp_dir, 'config.yaml')
        config = {'param1': 'value1', 'param2': {'nested': 'value'}}

        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)

        loaded = ConfigLoader.load_config(yaml_file)

        assert loaded == config

    def test_read_unsupported_format_raises_error(self, temp_dir):
        """Kiểm tra `IOHandler.read_data` raise ValueError cho định dạng không hỗ trợ."""
        unsupported_file = os.path.join(temp_dir, 'test.xyz')
        with open(unsupported_file, 'w') as f:
            f.write('test')

        with pytest.raises(ValueError, match="Unsupported"):
            IOHandler.read_data(unsupported_file)
