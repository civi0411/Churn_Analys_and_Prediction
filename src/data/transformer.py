"""
src/data/transformer.py

DataTransformer: chuyển đổi, chuẩn hóa và chọn đặc trưng cho pipeline.

Tóm tắt (Summary):
- Stateful transformer: fit trên tập train để học các tham số (imputers, encoders, scaler, outlier bounds, selected_features) và áp dụng cùng các tham số đó lên dữ liệu mới (test/predict).
- Hỗ trợ: imputation, feature engineering, outlier clipping (IQR), label encoding, scaling, feature selection, và factory cho SMOTE.

Important keywords: Args, Returns, Raises, Notes, Attributes, Methods
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

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
    Quản lý các bước tiền xử lý dữ liệu có trạng thái (stateful).

    Mục đích:
    - Học các tham số cần thiết từ training set và lưu lại trong `_learned_params`.
    - Áp dụng chính xác cùng các bước lên dữ liệu mới (avoids data leakage).

    Attributes:
        config (Dict): Cấu hình pipeline.
        pp (Dict): Phần preprocessing của config (shorthand).
        logger: Logger tùy chọn.
        target_col (str): Tên cột target.
        _learned_params (dict): Lưu imputer, encoders, scaler, outliers, selected_features.
        numerical_cols (List[str]): Danh sách cột số.
        categorical_cols (List[str]): Danh sách cột phân loại.

    Methods:
        fit_transform(df): Fit trên training data và trả về (X_transformed, y).
        transform(df): Áp dụng các tham số đã học lên dữ liệu mới.
        get_resampler(): Trả về instance SMOTE/SMOTETomek nếu được bật.
    """

    def __init__(self, config: Dict, logger=None):
        """
        Khởi tạo DataTransformer.

        Args:
            config (Dict): Cấu hình dự án (phải có phần 'preprocessing' và 'data').
            logger: Logger tuỳ chọn để log thông tin.

        Notes:
            - Không thực hiện fit tại khởi tạo; gọi `fit_transform` để học tham số.
        """
        self.config = config
        # Preprocessing-specific config (shorthand)
        self.pp = config.get('preprocessing', {}) if isinstance(config, dict) else {}
        self.logger = logger
        self.target_col = config.get('data', {}).get('target_col', 'Churn')

        # Lưu trữ các tham số đã học từ tập Train
        self._learned_params: Dict[str, Any] = {
             'imputers': {},
             'outliers': {},  # (lower, upper) bound
             'encoders': {},  # LabelEncoders
             'scaler': None,  # Scaler object
             'selected_features': []
         }

        # Explicit attribute for scaler to help type-checkers and access
        self.scaler: Optional[Any] = None

        self.numerical_cols = []
        self.categorical_cols = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit transformer trên training data và trả về X, y đã được biến đổi.

        Args:
            df (pd.DataFrame): DataFrame training (phải chứa cột target).

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_transformed, y)

        Notes:
            - Phương thức sẽ cập nhật `_learned_params` (imputers, encoders, scaler, outliers, selected_features).
            - Chỉ dùng cho training set (sẽ học tham số).
        """
        if self.logger:
            self.logger.info(f"Fit Transform | Input Shape: {df.shape}")
        return self._process(df, is_training=True)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Áp dụng các tham số đã học lên dữ liệu mới (test/predict).

        Args:
            df (pd.DataFrame): DataFrame đầu vào (có thể có hoặc không có cột target).

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_transformed, y_or_None)

        Raises:
            Warning: Nếu chưa gọi `fit_transform` trước đó thì sẽ dùng tham số mặc định (không khuyến khích).

        Notes:
            - Không học tham số mới (tránh data leakage).
        """
        if self.logger:
            self.logger.info(f"Transform     | Input Shape: {df.shape}")
        return self._process(df, is_training=False)

    def _process(self, df: pd.DataFrame, is_training: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Luồng xử lý chính (shared between fit_transform and transform).

        Pipeline steps (ngắn gọn):
            1. Tách X và y
            2. Xác định kiểu cột (numerical/categorical)
            3. Imputation
            4. Feature engineering (tuỳ chọn)
            5. Outlier clipping (IQR)
            6. Encoding cho categorical
            7. Scaling cho numerical
            8. Feature selection (fit trên train, reuse trên test)

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu đang fit (training), False nếu apply (testing)

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_transformed, y_or_None)
        """
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

    def _identify_types(self, df: pd.DataFrame) -> None:
        """
        Xác định cột numerical và categorical.

        Args:
            df (pd.DataFrame): DataFrame đầu vào (features only).

        Sets:
            self.numerical_cols: List các cột kiểu số (int64, float64).
            self.categorical_cols: List các cột kiểu object/category.
        """
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _handle_missing(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Impute missing values bằng SimpleImputer theo cấu hình.

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu fit imputer từ dữ liệu

        Returns:
            pd.DataFrame: DataFrame sau imputation

        Config (pp):
            preprocessing.missing_strategy.numerical: 'median'|'mean' (default 'median')
            preprocessing.missing_strategy.categorical: 'mode'|'unknown' (default 'mode')

        Notes:
            - Lưu imputer vào `_learned_params['imputers']` khi is_training=True.
        """
        missing_cfg = self.pp.get('missing_strategy', {})

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
        """
        Tạo các feature mới theo quy tắc domain có sẵn (tuỳ chọn).

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame có thêm feature mới (nếu bật)

        Config:
            preprocessing.create_features: bool (default False)

        Notes:
            - Chỉ thêm các cột nếu dữ liệu có các nguồn cột cần thiết.
        """
        if not self.pp.get('create_features', False):
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
        """
        Giảm ảnh hưởng outliers bằng clipping theo IQR hoặc Z-score.

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu tính và lưu bounds từ train

        Returns:
            pd.DataFrame: DataFrame sau clipping

        Config:
            preprocessing.outlier_method: 'iqr' hoặc 'zscore'
            preprocessing.outlier_threshold: float (default 1.5 cho IQR, ví dụ 3.0 cho zscore)

        Notes:
            - Lưu (lower, upper) cho mỗi cột số vào `_learned_params['outliers']` khi is_training=True (IQR).
            - Với zscore, không cần lưu state, chỉ dùng mean/std của tập hiện tại.
        """
        method = self.pp.get('outlier_method', 'iqr')
        threshold = self.pp.get('outlier_threshold', 1.5)

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
        elif method == 'zscore':
            for col in self.numerical_cols:
                if col not in df.columns: continue
                mean = df[col].mean()
                std = df[col].std()
                if std == 0: continue
                z = (df[col] - mean) / std
                # Clip các giá trị vượt quá ngưỡng zscore
                df[col] = df[col].where(z.abs() <= threshold, mean)
        return df

    def _encode_categorical(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Encode categorical columns bằng LabelEncoder.

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu fit encoder

        Returns:
            pd.DataFrame: DataFrame đã encode

        Config:
            preprocessing.categorical_encoding: 'label' (hiện chỉ hỗ trợ label encoding)

        Notes:
            - Unseen labels trên dữ liệu mới được gán giá trị -1 (an toàn với scikit-learn LabelEncoder).
            - Lưu encoder vào `_learned_params['encoders']` khi is_training=True.
        """
        encoding_method = self.pp.get('categorical_encoding', 'label')

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
        """
        Scale numerical features theo scaler cấu hình.

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu fit scaler

        Returns:
            pd.DataFrame: DataFrame đã scale

        Config:
            preprocessing.scaler_type: 'standard'|'minmax'|'robust' (default 'standard')

        Notes:
            - Lưu scaler vào `_learned_params['scaler']` khi is_training=True.
        """
        cols = [c for c in self.numerical_cols if c in df.columns]
        if not cols: return df

        scaler_type = self.pp.get('scaler_type', 'standard')

        if is_training:
            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            df[cols] = scaler.fit_transform(df[cols])
            self._learned_params['scaler'] = scaler
            self.scaler = scaler
        elif self.scaler:
            df[cols] = self.scaler.transform(df[cols])

        return df

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chọn top-K features bằng SelectKBest (f_classif hoặc mutual_info).

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector

        Returns:
            Tuple[pd.DataFrame, List[str]]: (X_selected, selected_feature_names)

        Config:
            preprocessing.feature_selection: bool
            preprocessing.n_top_features: int
            preprocessing.feature_selection_method: 'f_classif'|'mutual_info'

        Notes:
            - Bỏ các cột constant trước khi chọn.
            - Chỉ chạy selector khi is_training=True; test chỉ reuse tên cột đã chọn.
        """
        # Remove constant columns (all values identical)
        nunique = X.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            if self.logger:
                self.logger.info(f"Feature Select | Dropping constant columns: {constant_cols}")
            X = X.drop(columns=constant_cols)

        if not self.pp.get('feature_selection', False):
            return X, list(X.columns)

        k = self.pp.get('n_top_features', 15)
        k = min(k, X.shape[1])

        method = self.pp.get('feature_selection_method', 'f_classif')
        score_func = mutual_info_classif if method == 'mutual_info' else f_classif

        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)

        # Use boolean mask and enumerate to build indices (avoids type-checker warnings)
        support_mask = selector.get_support(indices=False)  # type: ignore
        # Build a plain Python list from mask; ignore static type warnings for numpy array stubs
        support_list = list(support_mask)  # type: ignore
        selected_idx = [i for i, v in enumerate(support_list) if v]
        selected_cols = [X.columns[i] for i in selected_idx]

        if self.logger:
            self.logger.info(f"Feature Select | Method: {method} | Selected: {len(selected_cols)}/{X.shape[1]}")

        return pd.DataFrame(X_new, columns=selected_cols, index=X.index), selected_cols

    def get_resampler(self):
        """
        Factory: trả về SMOTE / SMOTETomek theo cấu hình.

        Returns:
            resampler instance or None

        Config:
            preprocessing.use_smote: bool
            preprocessing.k_neighbors: int
            preprocessing.use_tomek: bool

        Notes:
            - Nếu imblearn không cài đặt sẽ trả về None và log warning.
            - Resampler dùng ngoài transform (ví dụ trong trainer).
        """
        if not self.pp.get('use_smote', False):
            return None

        if not IMBLEARN_AVAILABLE:
            if self.logger:
                self.logger.warning("Imblearn not installed. Skipping SMOTE.")
            return None

        k = self.pp.get('k_neighbors', 5)
        use_tomek = self.pp.get('use_tomek', False)

        if use_tomek:
            resampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k))
        else:
            resampler = SMOTE(random_state=42, k_neighbors=k)

        return resampler
