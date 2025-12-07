import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import IsolationForest

# Import Imblearn an toàn (cho phương thức get_resampler)
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE, SMOTETomek = None, None
    IMBLEARN_AVAILABLE = False


class DataTransformer:
    """
    Giai đoạn 2: Feature Engineering & Transformation (Stateful)
    Nhiệm vụ: Impute -> Feature Eng -> Encode -> Scale
    """

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger
        self.target_col = config.get('data', {}).get('target_col', 'Churn')

        # Lưu trữ các tham số đã học từ tập Train
        self._learned_params = {
            'imputers': {},
            'outliers': {},  # (lower, upper) bound
            'encoders': {},  # LabelEncoders
            'scaler': None,  # Scaler object
            'selected_features': []
        }

        self.numerical_cols = []
        self.categorical_cols = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Dùng cho tập TRAIN: Học tham số và biến đổi"""
        if self.logger:
            self.logger.info(f"Fit Transform | Input Shape: {df.shape}")
        return self._process(df, is_training=True)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Dùng cho tập TEST: Áp dụng tham số đã học"""
        if self.logger:
            self.logger.info(f"Transform     | Input Shape: {df.shape}")
        return self._process(df, is_training=False)

    def _process(self, df: pd.DataFrame, is_training: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Core processing logic"""
        df = df.copy()

        # Tách X, y
        if self.target_col in df.columns:
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
        else:
            y = None
            X = df

        # 1. Identify Types (Initial)
        if is_training:
            self._identify_types(X)

        # 2. Handle Missing (Imputation)
        X = self._handle_missing(X, is_training)

        # 3. Feature Engineering (Tạo cột mới)
        X = self._feature_engineering(X)

        # Re-identify types sau khi tạo cột mới (chỉ cần làm ở bước train)
        if is_training:
            self._identify_types(X)

        # 4. Handle Outliers
        X = self._handle_outliers(X, is_training)

        # 5. Encode Categorical
        X = self._encode_categorical(X, is_training)

        # 6. Scale Features
        X = self._scale_features(X, is_training)

        # 7. Feature Selection
        if is_training and y is not None:
            X, selected = self._select_features(X, y)
            self._learned_params['selected_features'] = selected
        elif self._learned_params['selected_features']:
            # Đảm bảo cột tồn tại ở test, nếu thiếu fill 0
            for col in self._learned_params['selected_features']:
                if col not in X.columns: X[col] = 0
            X = X[self._learned_params['selected_features']]

        return X, y

    # ================= INTERNAL METHODS =================

    def _identify_types(self, df: pd.DataFrame):
        """Phân loại cột số và cột phân loại"""
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _handle_missing(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Xử lý missing values dùng SimpleImputer"""
        missing_cfg = self.config.get('missing_strategy', {})

        # Numerical
        num_strat = missing_cfg.get('numerical', 'median')
        for col in self.numerical_cols:
            if col not in df.columns: continue

            if is_training:
                imp = SimpleImputer(strategy=num_strat)
                df[col] = imp.fit_transform(df[[col]]).ravel()
                self._learned_params['imputers'][col] = imp
            elif col in self._learned_params['imputers']:
                df[col] = self._learned_params['imputers'][col].transform(df[[col]]).ravel()

        # Categorical
        cat_strat = missing_cfg.get('categorical', 'mode')
        sk_strat = 'most_frequent' if cat_strat == 'mode' else 'constant'
        fill_val = 'Unknown' if cat_strat == 'unknown' else None

        for col in self.categorical_cols:
            if col not in df.columns: continue

            if is_training:
                imp = SimpleImputer(strategy=sk_strat, fill_value=fill_val)
                df[col] = imp.fit_transform(df[[col]]).ravel()
                self._learned_params['imputers'][col] = imp
            elif col in self._learned_params['imputers']:
                df[col] = self._learned_params['imputers'][col].transform(df[[col]]).ravel()

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các đặc trưng mới (Domain Knowledge)"""
        if not self.config.get('create_features', False):
            return df

        # Các feature logic từ file cũ của bạn
        if 'CashbackAmount' in df.columns and 'OrderCount' in df.columns:
            df['avg_cashbk_per_order'] = df['CashbackAmount'] / np.maximum(df['OrderCount'], 1)

        if 'OrderCount' in df.columns and 'Tenure' in df.columns:
            df['order_frequency_mpm'] = df['OrderCount'] / np.maximum(df['Tenure'], 1)

        if 'CouponUsed' in df.columns and 'OrderCount' in df.columns:
            df['coupon_rate'] = df['CouponUsed'] / np.maximum(df['OrderCount'], 1)

        if 'HourSpendOnApp' in df.columns and 'NumberOfDeviceRegistered' in df.columns:
            df['engagement_score'] = df['HourSpendOnApp'] * np.log1p(df['NumberOfDeviceRegistered'])

        # Log transform cho các cột bị lệch
        skew_cols = ['DaySinceLastOrder', 'WarehouseToHome', 'OrderCount', 'CashbackAmount']
        for col in skew_cols:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])

        # Boolean features
        if 'CityTier' in df.columns:
            df['city_is_tier1'] = (df['CityTier'].astype(str) == '1').astype(int)

        if 'NumberOfAddress' in df.columns:
            df['multi_address'] = (df['NumberOfAddress'] > 1).astype(int)

        return df

    def _handle_outliers(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Kẹp (Clip) giá trị ngoại lai theo IQR"""
        method = self.config.get('outlier_method', 'iqr')
        threshold = self.config.get('outlier_threshold', 1.5)

        # Chỉ hỗ trợ IQR cho pipeline production an toàn (các method khác như IsolationForest thường dùng filter dòng, khó áp dụng transform)
        if method == 'iqr':
            for col in self.numerical_cols:
                if col not in df.columns: continue

                if is_training:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                    self._learned_params['outliers'][col] = (lower, upper)
                    df[col] = df[col].clip(lower=lower, upper=upper)
                elif col in self._learned_params['outliers']:
                    lower, upper = self._learned_params['outliers'][col]
                    df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _encode_categorical(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Label Encoding"""
        encoding_method = self.config.get('categorical_encoding', 'label')

        if encoding_method == 'label':
            for col in self.categorical_cols:
                if col not in df.columns: continue

                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._learned_params['encoders'][col] = le
                elif col in self._learned_params['encoders']:
                    le = self._learned_params['encoders'][col]
                    # Handle unseen labels safe way
                    df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        return df

    def _scale_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Scaling (Standard, MinMax, Robust)"""
        cols = [c for c in self.numerical_cols if c in df.columns]
        if not cols: return df

        scaler_type = self.config.get('scaler_type', 'standard')

        if is_training:
            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            df[cols] = scaler.fit_transform(df[cols])
            self._learned_params['scaler'] = scaler
        elif self._learned_params.get('scaler'):
            df[cols] = self._learned_params['scaler'].transform(df[cols])

        return df

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Feature Selection (Chỉ chạy trên Train)"""
        if not self.config.get('feature_selection', False):
            return X, list(X.columns)

        k = self.config.get('n_top_features', 15)
        k = min(k, X.shape[1])

        method = self.config.get('feature_selection_method', 'f_classif')
        score_func = mutual_info_classif if method == 'mutual_info' else f_classif

        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)

        selected_cols = [col for col, selected in zip(X.columns, selector.get_support()) if selected]

        if self.logger:
            self.logger.info(f"Feature Select | Method: {method} | Selected: {len(selected_cols)}/{X.shape[1]}")

        return pd.DataFrame(X_new, columns=selected_cols, index=X.index), selected_cols

    def get_resampler(self):
        """Factory: Trả về object SMOTE để sử dụng trong Pipeline Training (không dùng trong transform data)"""
        if not self.config.get('preprocessing', {}).get('use_smote', False):
            return None

        if not IMBLEARN_AVAILABLE:
            if self.logger:
                self.logger.warning("Imblearn not installed. Skipping SMOTE.")
            return None

        k = self.config.get('preprocessing', {}).get('k_neighbors', 5)
        use_tomek = self.config.get('preprocessing', {}).get('use_tomek', False)

        if use_tomek:
            resampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k))
        else:
            resampler = SMOTE(random_state=42, k_neighbors=k)

        return resampler
