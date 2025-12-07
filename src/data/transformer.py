"""
src/data/transformer.py

Data Transformation Module - Giai đoạn 2: Feature Engineering (Stateful)

Module này chịu trách nhiệm biến đổi dữ liệu với pattern Fit-Transform:
    - Fit trên Train: Học các tham số (mean, std, encoders, bounds...)
    - Transform trên Test: Áp dụng tham số đã học (không học mới)

Pipeline xử lý:
    1. Imputation: Điền missing values (median/mode)
    2. Feature Engineering: Tạo features mới từ domain knowledge
    3. Outlier Handling: Clip outliers theo IQR
    4. Encoding: Label encode categorical columns
    5. Scaling: Standardize/normalize numerical columns
    6. Feature Selection: Chọn top K features quan trọng

Đặc điểm:
    - Stateful: Học và lưu trữ tham số từ training data
    - Prevent Data Leakage: Test data không ảnh hưởng đến tham số
    - Reproducible: Có thể áp dụng lại cho new data

Example:
    >>> transformer = DataTransformer(config, logger)
    >>> X_train, y_train = transformer.fit_transform(train_df)
    >>> X_test, y_test = transformer.transform(test_df)

Author: Churn Prediction Team
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
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
    Data Transformer - Giai đoạn 2: Feature Engineering (Stateful).

    Thực hiện biến đổi dữ liệu với pattern Fit-Transform để tránh data leakage.
    Tất cả tham số được học từ training data và áp dụng cho test data.

    Attributes:
        config (Dict): Configuration dictionary
        logger: Logger instance
        target_col (str): Tên cột target
        numerical_cols (List[str]): Danh sách cột số
        categorical_cols (List[str]): Danh sách cột phân loại
        _learned_params (Dict): Các tham số đã học:
            - imputers: SimpleImputer cho từng cột
            - outliers: (lower, upper) bounds cho từng cột
            - encoders: LabelEncoder cho từng cột categorical
            - scaler: Scaler object (StandardScaler/MinMaxScaler/RobustScaler)
            - selected_features: Danh sách features được chọn

    Example:
        >>> transformer = DataTransformer(config, logger)
        >>> # Fit và transform trên train
        >>> X_train, y_train = transformer.fit_transform(train_df)
        >>> # Chỉ transform trên test (dùng params đã học)
        >>> X_test, y_test = transformer.transform(test_df)
    """

    def __init__(self, config: Dict, logger=None):
        """
        Khởi tạo DataTransformer.

        Args:
            config (Dict): Configuration dictionary chứa:
                - data.target_col: Tên cột target
                - preprocessing.missing_strategy: Strategy cho missing values
                - preprocessing.outlier_method: Phương pháp xử lý outliers
                - preprocessing.scaler_type: Loại scaler (standard/minmax/robust)
                - preprocessing.create_features: Có tạo features mới không
                - preprocessing.feature_selection: Có chọn features không
            logger: Logger instance (optional)
        """
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
        """
        Học tham số từ training data và biến đổi.

        Method này CHỈ dùng cho TRAINING DATA. Nó sẽ:
            1. Học các tham số (mean, std, encoders, bounds...)
            2. Lưu trữ tham số vào _learned_params
            3. Áp dụng transformation

        Args:
            df (pd.DataFrame): Training DataFrame (phải chứa target column)

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_train transformed, y_train)

        Example:
            >>> X_train, y_train = transformer.fit_transform(train_df)
            >>> print(X_train.shape, y_train.shape)
            (4504, 15) (4504,)
        """
        if self.logger:
            self.logger.info(f"Fit Transform | Input Shape: {df.shape}")
        return self._process(df, is_training=True)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Áp dụng transformation với tham số đã học.

        Method này CHỈ dùng cho TEST DATA. Nó sẽ:
            1. Sử dụng tham số đã học từ fit_transform()
            2. KHÔNG học tham số mới (tránh data leakage)

        Args:
            df (pd.DataFrame): Test DataFrame

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_test transformed, y_test)

        Raises:
            Warning: Nếu gọi trước fit_transform(), sẽ dùng default params

        Example:
            >>> X_test, y_test = transformer.transform(test_df)
        """
        if self.logger:
            self.logger.info(f"Transform     | Input Shape: {df.shape}")
        return self._process(df, is_training=False)

    def _process(self, df: pd.DataFrame, is_training: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Core processing logic - xử lý chung cho cả fit_transform và transform.

        Pipeline:
            1. Tách X và y
            2. Identify column types (numerical/categorical)
            3. Handle missing values
            4. Feature engineering
            5. Handle outliers
            6. Encode categorical
            7. Scale features
            8. Feature selection

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu đang training (học params), False nếu testing

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X transformed, y)
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
        Phân loại cột thành numerical và categorical.

        Args:
            df (pd.DataFrame): Input DataFrame

        Sets:
            self.numerical_cols: List các cột số (int64, float64)
            self.categorical_cols: List các cột phân loại (object, category)
        """
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _handle_missing(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Xử lý missing values sử dụng SimpleImputer.

        Strategy:
            - Numerical: median (robust với outliers)
            - Categorical: mode (most frequent) hoặc 'Unknown'

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu đang học imputer

        Returns:
            pd.DataFrame: DataFrame đã impute missing values
        """
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
        """
        Tạo các đặc trưng mới từ Domain Knowledge.

        Features được tạo:
            - avg_cashbk_per_order: Cashback trung bình mỗi đơn
            - order_frequency_mpm: Tần suất đặt hàng theo tháng
            - coupon_rate: Tỷ lệ sử dụng coupon
            - engagement_score: Điểm tương tác (giờ dùng app * log devices)
            - log_*: Log transform cho các cột bị lệch
            - city_is_tier1: Binary feature cho thành phố tier 1
            - multi_address: Có nhiều địa chỉ không

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame với features mới

        Config:
            preprocessing.create_features: Bật/tắt feature engineering
        """
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
        """
        Xử lý outliers bằng phương pháp IQR Clipping.

        Công thức:
            - Q1 = 25th percentile
            - Q3 = 75th percentile
            - IQR = Q3 - Q1
            - Lower bound = Q1 - threshold * IQR
            - Upper bound = Q3 + threshold * IQR

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu đang học bounds

        Returns:
            pd.DataFrame: DataFrame đã clip outliers

        Config:
            preprocessing.outlier_method: 'iqr' (chỉ hỗ trợ IQR)
            preprocessing.outlier_threshold: Hệ số IQR (default: 1.5)
        """
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
        """
        Encode categorical columns sử dụng LabelEncoder.

        Xử lý unseen labels:
            - Labels không có trong training sẽ được gán giá trị -1

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu đang fit encoder

        Returns:
            pd.DataFrame: DataFrame đã encode

        Config:
            preprocessing.categorical_encoding: 'label' (chỉ hỗ trợ label encoding)
        """
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
        """
        Scale numerical features.

        Scalers hỗ trợ:
            - StandardScaler: z-score normalization (mean=0, std=1)
            - MinMaxScaler: scale về [0, 1]
            - RobustScaler: dùng median và IQR (robust với outliers)

        Args:
            df (pd.DataFrame): Input DataFrame
            is_training (bool): True nếu đang fit scaler

        Returns:
            pd.DataFrame: DataFrame đã scale

        Config:
            preprocessing.scaler_type: 'standard', 'minmax', hoặc 'robust'
        """
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
        """
        Chọn top K features quan trọng nhất.

        Methods hỗ trợ:
            - f_classif: ANOVA F-value (nhanh, linear relationship)
            - mutual_info: Mutual Information (chậm hơn, nonlinear)

        Args:
            X (pd.DataFrame): Features DataFrame
            y (pd.Series): Target Series

        Returns:
            Tuple[pd.DataFrame, List[str]]: (X với features đã chọn, danh sách tên features)

        Config:
            preprocessing.feature_selection: Bật/tắt
            preprocessing.n_top_features: Số features cần chọn
            preprocessing.feature_selection_method: 'f_classif' hoặc 'mutual_info'

        Note:
            Method này CHỈ chạy trong training. Test data sẽ dùng
            danh sách features đã được chọn từ training.
        """
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
        """
        Factory method: Tạo và trả về SMOTE resampler.

        SMOTE (Synthetic Minority Over-sampling Technique) được dùng để
        cân bằng class trong imbalanced datasets.

        Returns:
            SMOTE hoặc SMOTETomek object nếu được cấu hình
            None nếu không dùng SMOTE hoặc imblearn không được cài

        Config:
            preprocessing.use_smote: Bật/tắt SMOTE
            preprocessing.k_neighbors: Số neighbors cho SMOTE
            preprocessing.use_tomek: Dùng SMOTETomek (SMOTE + Tomek Links cleaning)

        Note:
            Resampler này được inject vào ModelTrainer, KHÔNG được dùng
            trong transform() vì resampling chỉ áp dụng cho training data.

        Example:
            >>> resampler = transformer.get_resampler()
            >>> if resampler:
            ...     X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        """
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
