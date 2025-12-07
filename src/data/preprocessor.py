"""
src/data/preprocessor.py

Data Preprocessing Module - Giai đoạn 1: Data Engineering (Stateless)

Module này chịu trách nhiệm:
    - Load dữ liệu từ các nguồn khác nhau (Excel, CSV, Parquet)
    - Làm sạch dữ liệu cơ bản (loại bỏ duplicates, chuẩn hóa giá trị)
    - Chia dữ liệu thành Train/Test sets với stratified sampling

Đặc điểm:
    - Stateless: Không học tham số từ dữ liệu
    - Hard-coded rules: Các quy tắc làm sạch cố định
    - Tách biệt với Feature Engineering (Transformer)

Example:
    >>> preprocessor = DataPreprocessor(config, logger)
    >>> df = preprocessor.load_data('data/raw/dataset.xlsx')
    >>> df_clean = preprocessor.clean_data(df)
    >>> train_df, test_df = preprocessor.split_data(df_clean)

Author: Churn Prediction Team
"""
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
# Import từ utils nằm ở thư mục cha
from ..utils import IOHandler


class DataPreprocessor:
    """
    Data Preprocessor - Giai đoạn 1: Data Engineering (Stateless).

    Nhiệm vụ chính:
        1. Load Data: Đọc dữ liệu từ file (Excel, CSV, Parquet)
        2. Clean Basics: Làm sạch cơ bản (duplicates, standardize values)
        3. Split Train/Test: Chia dữ liệu với stratified sampling

    Attributes:
        config (Dict): Configuration dictionary từ config.yaml
        logger: Logger instance để ghi log
        target_col (str): Tên cột target (mặc định: 'Churn')

    Example:
        >>> config = {'data': {'target_col': 'Churn', 'test_size': 0.2}}
        >>> preprocessor = DataPreprocessor(config, logger)
        >>> df = preprocessor.load_data('data/raw/customers.xlsx')
        >>> train_df, test_df = preprocessor.split_data(df)
    """

    def __init__(self, config: Dict, logger=None):
        """
        Khởi tạo DataPreprocessor.

        Args:
            config (Dict): Configuration dictionary chứa các settings:
                - data.target_col: Tên cột target
                - data.test_size: Tỷ lệ test set (0.0 - 1.0)
                - data.random_state: Random seed cho reproducibility
                - data.sheet_name: Sheet name cho file Excel
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger  # Nhận logger từ Pipeline
        self.target_col = config.get('data', {}).get('target_col', 'Churn')

    def load_data(self, raw_path: str) -> pd.DataFrame:
        """
        Load dữ liệu từ file và thực hiện inspect cơ bản.

        Hỗ trợ các định dạng: .xlsx, .xls, .csv, .parquet, .json

        Args:
            raw_path (str): Đường dẫn đến file dữ liệu

        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu đã load

        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu định dạng file không được hỗ trợ
            IOError: Nếu có lỗi khi đọc file

        Example:
            >>> df = preprocessor.load_data('data/raw/E Commerce Dataset.xlsx')
            >>> print(df.shape)
            (5630, 20)
        """
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 1: LOAD & CLEAN")

        sheet = self.config.get('data', {}).get('sheet_name', 0) if raw_path.endswith('.xlsx') else None

        try:
            df = IOHandler.read_data(raw_path, sheet_name=sheet)
            if self.logger:
                self.logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            if self.logger:
                self.logger.error(f"Load Error: {e}")
            raise e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu cơ bản (Hard rules - Stateless).

        Các bước xử lý:
            1. Chuẩn hóa tên cột (strip whitespace)
            2. Loại bỏ duplicate rows
            3. Chuẩn hóa giá trị domain-specific:
               - 'Phone' → 'Mobile Phone'
               - 'CC' → 'Credit Card'
               - 'Cash on Delivery' → 'COD'

        Args:
            df (pd.DataFrame): DataFrame cần làm sạch

        Returns:
            pd.DataFrame: DataFrame đã được làm sạch

        Note:
            - Method này KHÔNG thay đổi DataFrame gốc (tạo copy)
            - KHÔNG xử lý missing values (để cho Transformer)
            - KHÔNG xử lý outliers (để cho Transformer)

        Example:
            >>> df_clean = preprocessor.clean_data(df)
            >>> print(df_clean.duplicated().sum())
            0
        """
        df = df.copy()

        # 1. Standardize columns
        df.columns = df.columns.str.strip()

        # 2. Calculate stats before cleaning
        missing_count = df.isnull().sum().sum()
        duplicates_count = df.duplicated().sum()

        if self.logger:
            self.logger.info(f"Shape: {df.shape} | Missing: {missing_count} | Duplicates: {duplicates_count}")

        # 3. Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()

        # 4. Domain Specific Cleaning (Hard-coded mappings)
        if 'PreferredLoginDevice' in df.columns:
            df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})

        if 'PreferredPaymentMode' in df.columns:
            df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
                'CC': 'Credit Card',
                'Cash on Delivery': 'COD'
            })

        if self.logger:
            self.logger.info("Data cleaning completed")

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia dữ liệu thành Train/Test sets với Stratified Sampling.

        Stratified sampling đảm bảo tỷ lệ target class được giữ nguyên
        trong cả train và test sets.

        Args:
            df (pd.DataFrame): DataFrame cần chia (phải chứa target column)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

        Raises:
            ValueError: Nếu target column không tồn tại trong DataFrame

        Configuration:
            - data.test_size: Tỷ lệ test set (default: 0.2)
            - data.random_state: Random seed (default: 42)

        Example:
            >>> train_df, test_df = preprocessor.split_data(df)
            >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
            Train: 4504, Test: 1126
        """
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 2: SPLIT")

        if self.target_col not in df.columns:
            raise ValueError(f"Target '{self.target_col}' not found for splitting")

        test_size = self.config.get('data', {}).get('test_size', 0.2)
        random_state = self.config.get('data', {}).get('random_state', 42)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[self.target_col]
        )

        train_pct = (1 - test_size) * 100
        if self.logger:
            self.logger.info(f"Split: Train={len(train_df)} ({train_pct:.1f}%) | Test={len(test_df)}")

        return train_df, test_df