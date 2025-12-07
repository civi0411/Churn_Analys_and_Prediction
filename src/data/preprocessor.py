import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
# Import từ utils nằm ở thư mục cha
from ..utils import IOHandler


class DataPreprocessor:
    """
    Giai đoạn 1: Data Engineering (Stateless)
    Nhiệm vụ: Load Data -> Clean Basics -> Split Train/Test
    """

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger  # Nhận logger từ Pipeline
        self.target_col = config.get('data', {}).get('target_col', 'Churn')

    def load_data(self, raw_path: str) -> pd.DataFrame:
        """Load dữ liệu và inspect cơ bản"""
        sheet = self.config.get('data', {}).get('sheet_name', 0) if raw_path.endswith('.xlsx') else None

        try:
            # Gọi IOHandler từ utils
            df = IOHandler.read_data(raw_path, sheet_name=sheet)
            if self.logger:
                self.logger.info(f"Data Loaded | Path: {raw_path} | Shape: {df.shape}")
            return df
        except Exception as e:
            if self.logger: self.logger.error(f"Load Error: {e}")
            raise e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu cơ bản (Hard rules).
        Không phụ thuộc vào phân phối dữ liệu (Stateless).
        """
        df = df.copy()

        # 1. Standardize columns
        df.columns = df.columns.str.strip()

        # 2. Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if self.logger and len(df) < initial_rows:
            self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # 3. Domain Specific Cleaning (Hard-coded mappings)
        # Sửa các lỗi nhập liệu thường gặp
        if 'PreferredLoginDevice' in df.columns:
            df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})

        if 'PreferredPaymentMode' in df.columns:
            df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
                'CC': 'Credit Card',
                'Cash on Delivery': 'COD'
            })

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia Train/Test Stratify theo target.
        """
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

        if self.logger:
            self.logger.info(f"Data Split | Train: {len(train_df)} | Test: {len(test_df)}")

        return train_df, test_df