"""
tests/test_ops/test_dataops.py
Tests for src/ops/dataops.py
"""
import pytest
import pandas as pd
import os

from src.ops.dataops import DataValidator, DataVersioning


class TestDataValidator:
    """Test cases for DataValidator class"""

    def test_init(self, mock_logger):
        """Test khởi tạo DataValidator"""
        validator = DataValidator(mock_logger)
        assert validator.logger == mock_logger

    def test_validate_quality_returns_metrics(self, sample_raw_df):
        """Test validate_quality trả về đúng metrics"""
        validator = DataValidator()
        result = validator.validate_quality(sample_raw_df)

        assert 'null_ratio' in result
        assert 'duplicate_ratio' in result
        assert 'rows' in result

    def test_validate_quality_with_missing_data(self, sample_raw_df_with_missing, mock_logger):
        """Test validate_quality với missing data"""
        validator = DataValidator(mock_logger)
        result = validator.validate_quality(sample_raw_df_with_missing)

        assert result['null_ratio'] > 0

    def test_validate_quality_empty_df(self):
        """Test validate_quality với empty DataFrame"""
        validator = DataValidator()
        df = pd.DataFrame()
        result = validator.validate_quality(df)

        assert result['rows'] == 0

    def test_validate_quality_no_duplicates(self, sample_raw_df):
        """Test validate_quality không có duplicates"""
        validator = DataValidator()
        result = validator.validate_quality(sample_raw_df)

        assert result['duplicate_ratio'] == 0

    def test_validate_quality_with_duplicates(self, sample_raw_df):
        """Test validate_quality với duplicates"""
        validator = DataValidator()
        df_with_dups = pd.concat([sample_raw_df, sample_raw_df.head(10)], ignore_index=True)
        result = validator.validate_quality(df_with_dups)

        assert result['duplicate_ratio'] > 0


class TestDataVersioning:
    """Test cases for DataVersioning class"""

    def test_init(self, temp_dir):
        """Test khởi tạo DataVersioning"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        assert os.path.exists(versions_dir)
        assert versioning.history == {}

    def test_create_version(self, temp_dir):
        """Test create_version"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create a test file
        test_file = os.path.join(temp_dir, 'test_data.csv')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(test_file, index=False)

        version_id = versioning.create_version(test_file, 'test_dataset')

        assert version_id.startswith('v_')
        assert 'test_dataset' in versioning.history

    def test_create_version_idempotent(self, temp_dir):
        """Test create_version idempotent - cùng file trả về cùng version"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create a test file
        test_file = os.path.join(temp_dir, 'test_data.csv')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(test_file, index=False)

        version1 = versioning.create_version(test_file, 'test_dataset')
        version2 = versioning.create_version(test_file, 'test_dataset')

        assert version1 == version2

    def test_create_version_different_for_different_content(self, temp_dir):
        """Test create_version khác nhau khi nội dung file thay đổi"""
        import time
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create first version
        file1 = os.path.join(temp_dir, 'data.csv')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(file1, index=False)
        version1 = versioning.create_version(file1, 'dataset')
        hash1 = versioning.history['dataset'][-1]['hash']

        # Wait a moment and modify content
        time.sleep(1.1)  # Wait more than 1 second to get different timestamp
        pd.DataFrame({'a': [100, 200, 300]}).to_csv(file1, index=False)
        version2 = versioning.create_version(file1, 'dataset')
        hash2 = versioning.history['dataset'][-1]['hash']

        # Hashes should be different because content changed
        assert hash1 != hash2

    def test_create_version_missing_file_raises_error(self, temp_dir):
        """Test create_version với file không tồn tại raises error"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        with pytest.raises(FileNotFoundError):
            versioning.create_version('/nonexistent/file.csv', 'test')

    def test_version_history_persists(self, temp_dir):
        """Test version history được lưu và load đúng"""
        versions_dir = os.path.join(temp_dir, 'versions')

        # Create version
        versioning1 = DataVersioning(versions_dir)
        test_file = os.path.join(temp_dir, 'test.csv')
        pd.DataFrame({'a': [1]}).to_csv(test_file, index=False)
        version_id = versioning1.create_version(test_file, 'test')

        # Create new instance - should load history
        versioning2 = DataVersioning(versions_dir)

        assert 'test' in versioning2.history
        assert versioning2.history['test'][0]['version_id'] == version_id

    def test_create_version_with_metadata(self, temp_dir, sample_raw_df):
        """Test create_version với DataFrame để extract metadata"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Save DataFrame to file
        test_file = os.path.join(temp_dir, 'test_data.csv')
        sample_raw_df.to_csv(test_file, index=False)

        # Create version with DataFrame
        version_id = versioning.create_version(
            test_file,
            'test_dataset',
            df=sample_raw_df,
            description="Test dataset",
            tags=["test", "sample"]
        )

        info = versioning.get_version_info('test_dataset')

        assert 'metadata' in info
        assert info['metadata']['rows'] == len(sample_raw_df)
        assert info['metadata']['columns'] == len(sample_raw_df.columns)
        assert info['description'] == "Test dataset"
        assert info['tags'] == ["test", "sample"]

    def test_create_version_with_profile(self, temp_dir, sample_raw_df):
        """Test create_version extract profile statistics"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        test_file = os.path.join(temp_dir, 'test_data.csv')
        sample_raw_df.to_csv(test_file, index=False)

        versioning.create_version(test_file, 'test_dataset', df=sample_raw_df)

        info = versioning.get_version_info('test_dataset')

        assert 'profile' in info
        assert 'null_percentage' in info['profile']
        assert 'duplicate_rows' in info['profile']

    def test_add_lineage(self, temp_dir):
        """Test add_lineage thêm thông tin nguồn gốc data"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create raw version
        raw_file = os.path.join(temp_dir, 'raw.csv')
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(raw_file, index=False)
        versioning.create_version(raw_file, 'raw_data')

        # Create train version
        train_file = os.path.join(temp_dir, 'train.csv')
        pd.DataFrame({'a': [1, 2]}).to_csv(train_file, index=False)
        versioning.create_version(train_file, 'train')

        # Add lineage
        versioning.add_lineage('train', 'raw_data', 'split 80/20')

        info = versioning.get_version_info('train')

        assert 'lineage' in info
        assert info['lineage']['parent'] == 'raw_data'
        assert info['lineage']['transformation'] == 'split 80/20'

    def test_get_lineage_tree(self, temp_dir):
        """Test get_lineage_tree lấy cây nguồn gốc"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create chain: raw -> processed -> train
        for name in ['raw', 'processed', 'train']:
            file = os.path.join(temp_dir, f'{name}.csv')
            pd.DataFrame({'a': [1]}).to_csv(file, index=False)
            versioning.create_version(file, name)

        versioning.add_lineage('processed', 'raw', 'cleaning')
        versioning.add_lineage('train', 'processed', 'split')

        tree = versioning.get_lineage_tree('train')

        assert tree['dataset'] == 'train'
        assert len(tree['parents']) == 2
        assert tree['parents'][0]['name'] == 'processed'
        assert tree['parents'][1]['name'] == 'raw'

    def test_list_versions(self, temp_dir):
        """Test list_versions liệt kê tất cả versions"""
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create multiple datasets
        for name in ['raw', 'train', 'test']:
            file = os.path.join(temp_dir, f'{name}.csv')
            pd.DataFrame({'a': [1]}).to_csv(file, index=False)
            versioning.create_version(file, name)

        all_versions = versioning.list_versions()

        assert 'raw' in all_versions
        assert 'train' in all_versions
        assert 'test' in all_versions

    def test_compare_versions(self, temp_dir):
        """Test compare_versions so sánh 2 versions"""
        import time
        versions_dir = os.path.join(temp_dir, 'versions')
        versioning = DataVersioning(versions_dir)

        # Create v1
        file = os.path.join(temp_dir, 'data.csv')
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df1.to_csv(file, index=False)
        v1 = versioning.create_version(file, 'dataset', df=df1)

        # Wait and create v2 with different content
        time.sleep(1.1)
        df2 = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        df2.to_csv(file, index=False)
        v2 = versioning.create_version(file, 'dataset', df=df2)

        comparison = versioning.compare_versions('dataset', v1, v2)

        assert comparison['hash_changed'] == True
        assert comparison['changes']['rows']['diff'] == 2  # 5 - 3 = 2



